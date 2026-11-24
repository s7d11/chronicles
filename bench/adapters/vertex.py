"""Vertex AI Gemini adapter targeting GCP project gravityx-509611."""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

from bench.adapters.base import AdapterCapabilities, GenerationResult, ModelAdapter
from bench.metrics import strip_thinking


class VertexGeminiAdapter(ModelAdapter):
    """Async REST client for Vertex AI publishers/google models with thinking-part filtering."""

    def __init__(
        self,
        *,
        model_id: str,
        project_id: str = "gravityx-509611",
        location: str = "global",
        price_in_per_1m: float = 0.0,
        price_out_per_1m: float = 0.0,
        timeout_s: float = 120.0,
    ):
        self.model_id = model_id
        self.project_id = project_id
        self.location = location
        self.capabilities = AdapterCapabilities(
            supports_logprobs=False,
            supports_tokenizer=True,
            supports_bpb=False,
            is_api=True,
            active_params_b=None,
            price_in_per_1m=price_in_per_1m,
            price_out_per_1m=price_out_per_1m,
        )
        host = (
            "aiplatform.googleapis.com"
            if location == "global"
            else f"{location}-aiplatform.googleapis.com"
        )
        self.base_path = (
            f"https://{host}/v1/projects/{project_id}/locations/{location}"
            f"/publishers/google/models/{model_id}"
        )
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=20.0))
        self._token: Optional[str] = None
        self._token_ts: float = 0.0
        self._token_lock = asyncio.Lock()

    async def _get_access_token(self) -> str:
        async with self._token_lock:
            now = time.time()
            if self._token and (now - self._token_ts) < 2400:
                return self._token

            # Prefer service account / ADC only when explicitly requested (e.g. GitHub Actions CI)
            if os.environ.get("VERTEX_USE_SERVICE_ACCOUNT") == "1" and os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                try:
                    import google.auth
                    import google.auth.transport.requests

                    creds, _ = google.auth.default(
                        scopes=["https://www.googleapis.com/auth/cloud-platform"]
                    )
                    creds.refresh(google.auth.transport.requests.Request())
                    if creds.token:
                        self._token = creds.token
                        self._token_ts = now
                        return self._token
                except Exception:
                    pass

            tok = await asyncio.to_thread(
                lambda: subprocess.check_output(
                    ["gcloud", "auth", "print-access-token"], text=True
                ).strip()
            )
            self._token = tok
            self._token_ts = now
            return tok

    def _headers(self, token: str) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {token}",
            "x-goog-user-project": self.project_id,
            "Content-Type": "application/json",
        }

    async def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> GenerationResult:
        sys_parts = [m["content"] for m in messages if m["role"] == "system"]
        contents = [
            {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
            for m in messages
            if m["role"] != "system"
        ]
        gen_cfg: Dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max(max_tokens, 512),
            "thinkingConfig": {"thinkingLevel": "LOW"},
        }
        if json_mode:
            gen_cfg["responseMimeType"] = "application/json"

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": gen_cfg,
        }
        if sys_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(sys_parts)}]}

        last_err: Optional[str] = None
        for attempt in range(5):
            token = await self._get_access_token()
            t0 = time.perf_counter()
            try:
                resp = await self._client.post(
                    f"{self.base_path}:generateContent",
                    headers=self._headers(token),
                    json=payload,
                )
                elapsed = time.perf_counter() - t0
                if resp.status_code == 401:
                    self._token = None
                if resp.status_code in {401, 429, 500, 502, 503, 504}:
                    last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    await asyncio.sleep(min(20.0, (1.8 ** attempt) + 0.5))
                    continue
                if resp.status_code != 200:
                    return GenerationResult(
                        text="",
                        raw_text=resp.text[:400],
                        latency_s=elapsed,
                        error=f"HTTP {resp.status_code}: {resp.text[:240]}",
                    )

                data = resp.json()
                cands = data.get("candidates") or []
                parts = ((cands[0].get("content") or {}).get("parts") or []) if cands else []
                answer_chunks = [
                    str(p.get("text") or "")
                    for p in parts
                    if not p.get("thought") and p.get("text") is not None
                ]
                raw_text = "\n".join(str(p.get("text") or "") for p in parts if p.get("text") is not None)
                clean_text = strip_thinking("".join(answer_chunks))
                usage = data.get("usageMetadata") or {}
                in_toks = int(usage.get("promptTokenCount") or 0)
                out_toks = int(
                    usage.get("candidatesTokenCount")
                    or usage.get("totalTokenCount", 0) - in_toks
                    or 0
                )
                return GenerationResult(
                    text=clean_text,
                    raw_text=raw_text,
                    latency_s=elapsed,
                    prompt_tokens=in_toks,
                    completion_tokens=max(0, out_toks),
                )
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {str(exc)[:200]}"
                await asyncio.sleep(min(20.0, (1.8 ** attempt) + 0.5))

        return GenerationResult(text="", raw_text="", latency_s=0.0, error=last_err or "Vertex error")

    async def count_tokens_batch(self, texts: Sequence[str]) -> Tuple[List[int], str]:
        counts: List[int] = []
        sem = asyncio.Semaphore(6)

        async def _count_one(txt: str) -> int:
            async with sem:
                for attempt in range(4):
                    token = await self._get_access_token()
                    try:
                        resp = await self._client.post(
                            f"{self.base_path}:countTokens",
                            headers=self._headers(token),
                            json={"contents": [{"role": "user", "parts": [{"text": txt}]}]},
                        )
                        if resp.status_code == 200:
                            return int(resp.json().get("totalTokens") or 1)
                        if resp.status_code in {401, 429, 500, 502, 503}:
                            if resp.status_code == 401:
                                self._token = None
                            await asyncio.sleep(1.5 ** attempt)
                            continue
                        break
                    except Exception:
                        await asyncio.sleep(1.5 ** attempt)
                return max(1, len(txt.encode("utf-8")) // 2)

        counts = await asyncio.gather(*[_count_one(t) for t in texts])
        return list(counts), "exact-vertex"

    async def close(self) -> None:
        await self._client.aclose()
