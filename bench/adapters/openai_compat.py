"""OpenAI-compatible async adapter for Adaption Labs and NVIDIA NIM."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

from bench.adapters.base import AdapterCapabilities, GenerationResult, ModelAdapter
from bench.metrics import strip_thinking


_TOKENIZER_CACHE: Dict[str, Any] = {}
_ADAPTION_PACE_LOCK: Optional[asyncio.Lock] = None
_ADAPTION_NEXT_TS: float = 0.0


async def _pace_adaption_request(min_interval_s: float = 1.32) -> None:
    """Global account-level pacing (~45 RPM) for Adaption Labs (50 RPM cap)."""
    global _ADAPTION_PACE_LOCK, _ADAPTION_NEXT_TS
    if _ADAPTION_PACE_LOCK is None:
        _ADAPTION_PACE_LOCK = asyncio.Lock()
    async with _ADAPTION_PACE_LOCK:
        now = time.monotonic()
        if now < _ADAPTION_NEXT_TS:
            await asyncio.sleep(_ADAPTION_NEXT_TS - now)
        _ADAPTION_NEXT_TS = max(time.monotonic(), _ADAPTION_NEXT_TS) + min_interval_s


async def _push_back_adaption_reset(reset_epoch: Optional[str]) -> float:
    """When 429 occurs, compute sleep seconds from x-ratelimit-reset and update global gate."""
    global _ADAPTION_PACE_LOCK, _ADAPTION_NEXT_TS
    if _ADAPTION_PACE_LOCK is None:
        _ADAPTION_PACE_LOCK = asyncio.Lock()
    sleep_s = 12.0
    if reset_epoch:
        try:
            delta = float(reset_epoch) - time.time() + 1.0
            sleep_s = max(2.0, min(65.0, delta))
        except ValueError:
            pass
    async with _ADAPTION_PACE_LOCK:
        _ADAPTION_NEXT_TS = max(_ADAPTION_NEXT_TS, time.monotonic() + sleep_s)
    return sleep_s


def _load_hf_tokenizer(hf_id: Optional[str]) -> Optional[Any]:
    if not hf_id:
        return None
    if hf_id in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[hf_id]
    try:
        from transformers import AutoTokenizer

        token = os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN")
        tok = AutoTokenizer.from_pretrained(hf_id, token=token, trust_remote_code=True)
        _TOKENIZER_CACHE[hf_id] = tok
        return tok
    except Exception:
        _TOKENIZER_CACHE[hf_id] = None
        return None


class OpenAICompatAdapter(ModelAdapter):
    """Async adapter for Adaption Labs (api.prod.adaptionlabs.ai/api/v1) and NVIDIA NIM."""

    def __init__(
        self,
        *,
        provider: str,
        model_id: str,
        api_key: str,
        base_url: str,
        hf_tokenizer_id: Optional[str] = None,
        active_params_b: Optional[float] = None,
        price_in_per_1m: float = 0.0,
        price_out_per_1m: float = 0.0,
        timeout_s: float = 90.0,
    ):
        self.provider = provider
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.hf_tokenizer_id = hf_tokenizer_id
        self.timeout_s = timeout_s
        self.capabilities = AdapterCapabilities(
            supports_logprobs=False,
            supports_tokenizer=True,
            supports_bpb=False,
            is_api=True,
            active_params_b=active_params_b,
            price_in_per_1m=price_in_per_1m,
            price_out_per_1m=price_out_per_1m,
        )
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout_s, connect=20.0),
        )
        self._tokenizer = None
        self._tokenizer_loaded = False
        self._use_reasoning_effort = True

    def _get_tokenizer(self) -> Optional[Any]:
        if not self._tokenizer_loaded:
            self._tokenizer = _load_hf_tokenizer(self.hf_tokenizer_id)
            if self._tokenizer is None and self.hf_tokenizer_id != "google/gemma-4-E2B":
                # Proxy open tokenizer if specific repo is gated/unavailable
                self._tokenizer = _load_hf_tokenizer("google/gemma-4-E2B")
                self._used_proxy_tok = True
            else:
                self._used_proxy_tok = False
            self._tokenizer_loaded = True
        return self._tokenizer

    async def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> GenerationResult:
        last_err: Optional[str] = None
        for attempt in range(5):
            payload: Dict[str, Any] = {
                "model": self.model_id,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if self._use_reasoning_effort and self.provider in {"adaption", "nim"}:
                payload["reasoning_effort"] = "low"
            if self.provider == "adaption":
                await _pace_adaption_request()
            t0 = time.perf_counter()
            try:
                resp = await self._client.post("/chat/completions", json=payload)
                elapsed = time.perf_counter() - t0
                if resp.status_code == 400 and self._use_reasoning_effort and "reasoning_effort" in resp.text:
                    self._use_reasoning_effort = False
                    continue
                if resp.status_code == 429 and self.provider == "adaption":
                    last_err = f"HTTP 429: {resp.text[:200]}"
                    sleep_s = await _push_back_adaption_reset(resp.headers.get("x-ratelimit-reset"))
                    await asyncio.sleep(sleep_s)
                    continue
                if resp.status_code in {429, 500, 502, 503, 504}:
                    last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    await asyncio.sleep(min(15.0, (1.8 ** attempt) + 0.5))
                    continue
                if resp.status_code != 200:
                    return GenerationResult(
                        text="",
                        raw_text=resp.text[:400],
                        latency_s=elapsed,
                        error=f"HTTP {resp.status_code}: {resp.text[:240]}",
                    )
                data = resp.json()
                choice = (data.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                raw_content = msg.get("content")
                if raw_content is None:
                    raw_content = msg.get("reasoning_content") or ""
                raw_text = str(raw_content)
                clean_text = strip_thinking(raw_text)
                if not clean_text and msg.get("reasoning_content") and msg.get("content") is None:
                    clean_text = strip_thinking(str(msg.get("reasoning_content")))
                usage = data.get("usage") or {}
                return GenerationResult(
                    text=clean_text,
                    raw_text=raw_text,
                    latency_s=elapsed,
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                last_err = f"{type(exc).__name__}: {str(exc)[:200]}"
                await asyncio.sleep(min(20.0, (1.8 ** attempt) + 0.5))

        return GenerationResult(text="", raw_text="", latency_s=0.0, error=last_err or "Unknown API error")

    async def count_tokens_batch(self, texts: Sequence[str]) -> Tuple[List[int], str]:
        tok = self._get_tokenizer()
        if tok is not None:
            counts = [len(tok.encode(t, add_special_tokens=False)) for t in texts]
            mode = "proxy-gemma4" if getattr(self, "_used_proxy_tok", False) else "exact-hf"
            return counts, mode
        # Conservative UTF-8 fallback if offline
        return [max(1, len(t.encode("utf-8")) // 2) for t in texts], "utf8-approx"

    def tokenize_pieces(self, word: str) -> Optional[List[str]]:
        tok = self._get_tokenizer()
        if tok is None:
            return None
        ids = tok.encode(word, add_special_tokens=False)
        pieces = []
        for tid in ids:
            piece = tok.decode([tid], skip_special_tokens=False)
            pieces.append(piece if piece else f"[{tid}]")
        return pieces

    async def close(self) -> None:
        await self._client.aclose()
