"""Isolated LLM providers for Volume II: NVIDIA Build, OpenRouter (:free ONLY), and Vertex gemini-3.1-flash-lite."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from bench.metrics import strip_thinking
from conflict_lab import ALLOWED_OPENROUTER_FREE_MODELS

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
VERTEX_DEFAULT_PROJECT = "mlgramv1"

_VERTEX_TOKEN: Optional[str] = None
_VERTEX_TOKEN_TS: float = 0.0
_VERTEX_LOCK: Optional[asyncio.Lock] = None


def extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    text = strip_thinking(raw_text).strip()
    if not text:
        return None
    if text.startswith("```"):
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last > first:
        try:
            obj = json.loads(text[first : last + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass
    return None


def assert_openrouter_free_model(model_id: str) -> str:
    """Enforce strictly $0 OpenRouter free-tier usage; reject paid models."""
    clean = (model_id or "").strip()
    if clean not in ALLOWED_OPENROUTER_FREE_MODELS and not clean.endswith(":free"):
        raise ValueError(
            f"Refusing non-free OpenRouter model '{clean}'; only ALLOWED_OPENROUTER_FREE_MODELS / *:free allowed."
        )
    return clean


async def call_nvidia_model(
    client: httpx.AsyncClient,
    model_id: str,
    messages: List[Dict[str, str]],
    *,
    max_tokens: int = 2048,
    temperature: float = 0.15,
) -> Tuple[str, float, Optional[str]]:
    """Call an NVIDIA Build NIM endpoint (`NVIDIA_KEY`), preserving zero Edition 4 overlap."""
    api_key = os.environ.get("NVIDIA_KEY") or ""
    if not api_key:
        return "", 0.0, "NVIDIA_KEY missing"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    timeout = httpx.Timeout(connect=20.0, read=60.0, write=30.0, pool=20.0)
    last_err: Optional[str] = None
    for attempt in range(3):
        t0 = time.perf_counter()
        try:
            resp = await client.post(
                f"{NVIDIA_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            elapsed = time.perf_counter() - t0
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices") or []
                content = ""
                if choices:
                    msg = choices[0].get("message") or {}
                    content = str(msg.get("content") or "")
                    if not content.strip() and msg.get("reasoning_content"):
                        rc = str(msg.get("reasoning_content") or "")
                        if "{" in rc:
                            content = rc
                return strip_thinking(content), elapsed, None
            if resp.status_code in {429, 500, 502, 503, 504}:
                last_err = f"NVIDIA HTTP {resp.status_code}: {resp.text[:160]}"
                await asyncio.sleep(min(8.0, 1.7 ** attempt + 0.4))
                continue
            return "", elapsed, f"NVIDIA HTTP {resp.status_code}: {resp.text[:180]}"
        except Exception as exc:
            last_err = f"NVIDIA {type(exc).__name__}: {str(exc)[:140]}"
            await asyncio.sleep(min(8.0, 1.7 ** attempt + 0.4))
    return "", 0.0, last_err or "NVIDIA unknown failure"


async def call_openrouter_free(
    client: httpx.AsyncClient,
    model_id: str,
    messages: List[Dict[str, str]],
    *,
    max_tokens: int = 2048,
    temperature: float = 0.15,
) -> Tuple[str, float, Optional[str]]:
    """Call an OpenRouter `:free` model strictly enforced by `assert_openrouter_free_model`."""
    safe_model = assert_openrouter_free_model(model_id)
    api_key = os.environ.get("OPENROUTER") or os.environ.get("OPENROUTER_API_KEY") or ""
    if not api_key:
        return "", 0.0, "OPENROUTER missing"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://s7d11.github.io/chronicles/",
        "X-Title": "The Sebeni Chronicle Vol. 2",
        "Content-Type": "application/json",
    }
    payload = {
        "model": safe_model,
        "messages": messages,
        "max_tokens": max(1024, int(max_tokens)),
        "temperature": temperature,
    }
    timeout = httpx.Timeout(connect=20.0, read=60.0, write=30.0, pool=20.0)
    last_err: Optional[str] = None
    for attempt in range(3):
        t0 = time.perf_counter()
        try:
            resp = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            elapsed = time.perf_counter() - t0
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices") or []
                content = ""
                if choices:
                    msg = choices[0].get("message") or {}
                    content = str(msg.get("content") or "")
                    if not content.strip() and msg.get("reasoning_content"):
                        rc = str(msg.get("reasoning_content") or "")
                        if "{" in rc:
                            content = rc
                return strip_thinking(content), elapsed, None
            if resp.status_code in {429, 500, 502, 503, 504}:
                last_err = f"OpenRouter HTTP {resp.status_code}: {resp.text[:160]}"
                await asyncio.sleep(min(8.0, 2.0 ** attempt))
                continue
            return "", elapsed, f"OpenRouter HTTP {resp.status_code}: {resp.text[:180]}"
        except Exception as exc:
            last_err = f"OpenRouter {type(exc).__name__}: {str(exc)[:140]}"
            await asyncio.sleep(min(8.0, 2.0 ** attempt))
    return "", 0.0, last_err or "OpenRouter unknown failure"


async def _get_vertex_access_token(client: httpx.AsyncClient, *, force_refresh: bool = False) -> str:
    global _VERTEX_TOKEN, _VERTEX_TOKEN_TS, _VERTEX_LOCK
    if _VERTEX_LOCK is None:
        _VERTEX_LOCK = asyncio.Lock()
    async with _VERTEX_LOCK:
        now = time.time()
        if _VERTEX_TOKEN and not force_refresh and (now - _VERTEX_TOKEN_TS) < 2400:
            return _VERTEX_TOKEN
        try:
            resp = await client.get(
                "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
                headers={"Metadata-Flavor": "Google"},
                timeout=3.0,
            )
            if resp.status_code == 200:
                tok = str(resp.json().get("access_token") or "").strip()
                if tok:
                    _VERTEX_TOKEN = tok
                    _VERTEX_TOKEN_TS = now
                    return tok
        except Exception:
            pass
        try:
            import google.auth
            import google.auth.transport.requests

            creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            creds.refresh(google.auth.transport.requests.Request())
            if creds.token:
                _VERTEX_TOKEN = str(creds.token)
                _VERTEX_TOKEN_TS = now
                return _VERTEX_TOKEN
        except Exception:
            pass
        clean_env = {k: v for k, v in os.environ.items() if k != "GOOGLE_APPLICATION_CREDENTIALS"}
        clean_env["PATH"] = f"{os.path.expanduser('~')}/google-cloud-sdk/bin:{clean_env.get('PATH', '')}"
        tok = await asyncio.to_thread(
            lambda: subprocess.check_output(["gcloud", "auth", "print-access-token"], env=clean_env, text=True).strip()
        )
        _VERTEX_TOKEN = tok
        _VERTEX_TOKEN_TS = now
        return tok


async def call_gemini_flash_lite(
    client: httpx.AsyncClient,
    messages: List[Dict[str, str]],
    *,
    model_id: Optional[str] = None,
    project_id: Optional[str] = None,
    max_tokens: int = 8192,
    temperature: float = 0.1,
) -> Tuple[str, float, Optional[str]]:
    """Call ultra-low-cost Vertex AI `gemini-3.1-flash-lite` with automatic 401 token refresh."""
    global _VERTEX_TOKEN, _VERTEX_TOKEN_TS
    resolved = (model_id or os.environ.get("VOL2_CRITIC_MODEL") or "gemini-3.1-flash-lite").strip()
    proj = project_id or os.environ.get("GCP_VERTEX_PROJECT") or VERTEX_DEFAULT_PROJECT
    system_texts = [m["content"] for m in messages if m.get("role") == "system"]
    contents = [
        {"role": "model" if m.get("role") == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in messages
        if m.get("role") != "system"
    ]
    payload: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
    }
    if system_texts:
        payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_texts)}]}

    timeout = httpx.Timeout(connect=20.0, read=90.0, write=30.0, pool=20.0)
    last_err: Optional[str] = None
    for attempt in range(3):
        t0 = time.perf_counter()
        try:
            tok = await _get_vertex_access_token(client, force_refresh=(attempt > 0))
            url = (
                f"https://aiplatform.googleapis.com/v1/projects/{proj}/locations/global"
                f"/publishers/google/models/{resolved}:generateContent"
            )
            resp = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {tok}",
                    "x-goog-user-project": proj,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
            elapsed = time.perf_counter() - t0
            if resp.status_code == 200:
                cands = resp.json().get("candidates") or []
                parts = (cands[0].get("content") or {}).get("parts") or [] if cands else []
                texts = [str(p["text"]) for p in parts if not p.get("thought") and isinstance(p.get("text"), str)]
                if not texts:
                    texts = [str(p["text"]) for p in parts if isinstance(p.get("text"), str)]
                return "\n".join(texts).strip(), elapsed, None
            if resp.status_code == 401:
                _VERTEX_TOKEN = None
                _VERTEX_TOKEN_TS = 0.0
                last_err = f"Vertex 401 refreshing token"
                continue
            if resp.status_code in {429, 500, 502, 503, 504}:
                last_err = f"Vertex HTTP {resp.status_code}: {resp.text[:160]}"
                await asyncio.sleep(min(10.0, 2.0 ** attempt))
                continue
            return "", elapsed, f"Vertex HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as exc:
            last_err = f"Vertex {type(exc).__name__}: {str(exc)[:140]}"
            await asyncio.sleep(min(10.0, 2.0 ** attempt))
    return "", 0.0, last_err or "Vertex unknown error"


async def invoke_vol2_agent(
    client: httpx.AsyncClient,
    spec: Any,
    messages: List[Dict[str, str]],
    *,
    max_tokens: int = 2048,
    temperature: float = 0.15,
) -> Tuple[str, float, Optional[str]]:
    """Dispatch to NVIDIA, OpenRouter :free, or Vertex Flash-Lite with automatic fallback."""
    async def _call(prov: str, mid: str) -> Tuple[str, float, Optional[str]]:
        if prov == "openrouter_free":
            return await call_openrouter_free(client, mid, messages, max_tokens=max_tokens, temperature=temperature)
        if prov == "vertex":
            return await call_gemini_flash_lite(client, messages, model_id=mid, max_tokens=max_tokens, temperature=temperature)
        return await call_nvidia_model(client, mid, messages, max_tokens=max_tokens, temperature=temperature)

    raw, lat, err = await _call(spec.provider, spec.model_id)
    if (err or not raw.strip()) and getattr(spec, "fallback_model_id", "") and getattr(spec, "fallback_provider", ""):
        fb_raw, fb_lat, fb_err = await _call(spec.fallback_provider, spec.fallback_model_id)
        if not fb_err and fb_raw.strip():
            return fb_raw, fb_lat, None
    return raw, lat, err
