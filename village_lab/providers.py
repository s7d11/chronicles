"""Async LLM callers for Mande (NIM), 9 Adaption Labs models, and Gemini 4 Argon Critic."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from bench.adapters.openai_compat import _pace_adaption_request, _push_back_adaption_reset
from bench.metrics import _extract_json_flexible, strip_thinking

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
ADAPTION_BASE_URL = "https://api.prod.adaptionlabs.ai/api/v1"
VERTEX_DEFAULT_PROJECT = "mlgramv1"

_VERTEX_TOKEN: Optional[str] = None
_VERTEX_TOKEN_TS: float = 0.0
_VERTEX_LOCK: Optional[asyncio.Lock] = None


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract a clean JSON dict from LLM output, resilient to reasoning/code fences."""
    clean = strip_thinking(text or "")
    parsed = _extract_json_flexible(clean)
    if isinstance(parsed, dict):
        return parsed
    start = clean.find("{")
    end = clean.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(clean[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


async def call_nim_mande(
    client: httpx.AsyncClient,
    messages: List[Dict[str, str]],
    *,
    model_id: str = "nvidia/nemotron-3-ultra-550b-a55b",
    max_tokens: int = 2048,
    temperature: float = 0.3,
) -> Tuple[str, float, Optional[str]]:
    """Stream SSE from NVIDIA Build NIM for Mande orchestrator decisions."""
    api_key = os.environ.get("NVIDIA_KEY") or os.environ.get("NIM_API_KEY") or ""
    if not api_key:
        return "", 0.0, "NVIDIA_KEY missing"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    use_effort = True
    last_err: Optional[str] = None
    timeout = httpx.Timeout(connect=20.0, read=65.0, write=30.0, pool=20.0)

    for attempt in range(3):
        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if use_effort:
            payload["reasoning_effort"] = "low"

        t0 = time.perf_counter()
        try:
            async with client.stream(
                "POST",
                f"{NIM_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            ) as resp:
                if resp.status_code != 200:
                    err_txt = (await resp.aread()).decode("utf-8", errors="replace")
                    if resp.status_code == 400 and use_effort and "reasoning_effort" in err_txt:
                        use_effort = False
                        continue
                    if resp.status_code in {429, 500, 502, 503, 504}:
                        last_err = f"NIM HTTP {resp.status_code}: {err_txt[:160]}"
                        await asyncio.sleep(min(10.0, 1.8 ** attempt + 0.5))
                        continue
                    return "", time.perf_counter() - t0, f"NIM HTTP {resp.status_code}: {err_txt[:200]}"

                ct_parts: List[str] = []
                async for line in resp.aiter_lines():
                    if time.perf_counter() - t0 > 120.0:
                        break
                    s = line.strip()
                    if not s.startswith("data: "):
                        continue
                    body = s[6:].strip()
                    if body == "[DONE]":
                        break
                    try:
                        chunk = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        if delta.get("content"):
                            ct_parts.append(str(delta["content"]))

                return strip_thinking("".join(ct_parts)), time.perf_counter() - t0, None
        except Exception as exc:
            last_err = f"NIM {type(exc).__name__}: {str(exc)[:160]}"
            await asyncio.sleep(min(10.0, 1.8 ** attempt + 0.5))

    return "", 0.0, last_err or "NIM unknown failure"


async def call_adaption(
    client: httpx.AsyncClient,
    model_id: str,
    messages: List[Dict[str, str]],
    *,
    max_tokens: int = 4096,
    temperature: float = 0.25,
    json_mode: bool = True,
) -> Tuple[str, float, Optional[str]]:
    """Call an Adaption Labs model obeying the global 50 RPM pacing lock."""
    api_key = os.environ.get("ADOPTION_KEY") or os.environ.get("ADAPTION_KEY") or ""
    if not api_key:
        return "", 0.0, "ADOPTION_KEY missing"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    del json_mode  # Adaption endpoints put JSON cleanly in content; response_format can empty content on some models
    last_err: Optional[str] = None
    timeout = httpx.Timeout(connect=20.0, read=120.0, write=30.0, pool=20.0)
    try:
        parallel_tasks = max(
            1,
            int(os.environ.get("ED4_MAX_PARALLEL_TASKS") or os.environ.get("CLOUD_RUN_TASK_COUNT") or "1"),
        )
    except ValueError:
        parallel_tasks = 1
    default_interval = max(1.35, 1.35 * parallel_tasks)
    try:
        pace_interval = max(0.0, float(os.environ.get("ED4_ADAPTION_MIN_INTERVAL", default_interval)))
    except ValueError:
        pace_interval = default_interval

    for attempt in range(4):
        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        await _pace_adaption_request(min_interval_s=pace_interval)
        t0 = time.perf_counter()
        try:
            resp = await client.post(
                f"{ADAPTION_BASE_URL}/chat/completions",
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
            if resp.status_code == 429:
                wait_s = await _push_back_adaption_reset(resp.headers.get("x-ratelimit-reset"))
                last_err = f"Adaption 429 on {model_id}"
                await asyncio.sleep(min(25.0, wait_s))
                continue
            if resp.status_code in {500, 502, 503, 504}:
                last_err = f"Adaption HTTP {resp.status_code} on {model_id}: {resp.text[:140]}"
                await asyncio.sleep(min(10.0, 1.8 ** attempt + 0.5))
                continue
            return "", elapsed, f"Adaption HTTP {resp.status_code}: {resp.text[:180]}"
        except Exception as exc:
            last_err = f"Adaption {type(exc).__name__}: {str(exc)[:160]}"
            await asyncio.sleep(min(10.0, 1.8 ** attempt + 0.5))

    return "", 0.0, last_err or f"Adaption failure ({model_id})"


async def _get_vertex_access_token(client: httpx.AsyncClient) -> str:
    """Obtain GCP access token via Cloud Run GCE metadata server or gcloud CLI fallback."""
    global _VERTEX_TOKEN, _VERTEX_TOKEN_TS, _VERTEX_LOCK
    if _VERTEX_LOCK is None:
        _VERTEX_LOCK = asyncio.Lock()
    async with _VERTEX_LOCK:
        now = time.time()
        if _VERTEX_TOKEN and (now - _VERTEX_TOKEN_TS) < 2400:
            return _VERTEX_TOKEN

        # 1. Cloud Run / GCE Metadata server (automatic ADC with cloud-platform scope)
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

        # 2. Local google.auth / gcloud fallback
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

        gcloud_env = {k: v for k, v in os.environ.items() if k != "GOOGLE_APPLICATION_CREDENTIALS"}
        tok = await asyncio.to_thread(
            lambda: subprocess.check_output(
                ["gcloud", "auth", "print-access-token"], env=gcloud_env, text=True
            ).strip()
        )
        _VERTEX_TOKEN = tok
        _VERTEX_TOKEN_TS = now
        return tok


def _extract_gemini_text(data: Dict[str, Any]) -> str:
    cands = data.get("candidates") or []
    if not cands:
        return ""
    parts = (cands[0].get("content") or {}).get("parts") or []
    texts = [str(p["text"]) for p in parts if not p.get("thought") and isinstance(p.get("text"), str)]
    if not texts:
        texts = [str(p["text"]) for p in parts if isinstance(p.get("text"), str)]
    return "\n".join(texts).strip()


async def call_gemini_argon(
    client: httpx.AsyncClient,
    messages: List[Dict[str, str]],
    *,
    model_id: Optional[str] = None,
    project_id: Optional[str] = None,
    max_tokens: int = 1600,
    temperature: float = 0.1,
    thinking_budget: Optional[int] = 1024,
) -> Tuple[str, float, Optional[str]]:
    """Call Gemini critic (default Gemini 3.5 Flash) via GEMINI_API_KEY or Vertex AI."""
    global _VERTEX_TOKEN, _VERTEX_TOKEN_TS
    resolved_model = (model_id or os.environ.get("ED4_CRITIC_MODEL") or "gemini-3.5-flash").strip()
    system_texts = [m["content"] for m in messages if m.get("role") == "system"]
    contents = [
        {
            "role": "model" if m.get("role") == "assistant" else "user",
            "parts": [{"text": m["content"]}],
        }
        for m in messages
        if m.get("role") != "system"
    ]
    use_thinking_cfg = thinking_budget is not None and thinking_budget > 0 and max_tokens >= 4096

    timeout = httpx.Timeout(connect=20.0, read=90.0, write=30.0, pool=20.0)
    gemini_key = os.environ.get("GEMINI_API_KEY") or ""
    proj = project_id or os.environ.get("GCP_VERTEX_PROJECT") or VERTEX_DEFAULT_PROJECT
    last_err: Optional[str] = None

    for attempt in range(3):
        gen_cfg: Dict[str, Any] = {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
        }
        if use_thinking_cfg:
            gen_cfg["thinkingConfig"] = {"thinkingBudget": int(thinking_budget)}
        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": gen_cfg,
        }
        if system_texts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_texts)}]}
        t0 = time.perf_counter()
        try:
            if gemini_key:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{resolved_model}:generateContent?key={gemini_key}"
                headers = {"Content-Type": "application/json"}
            else:
                tok = await _get_vertex_access_token(client)
                url = (
                    f"https://aiplatform.googleapis.com/v1/projects/{proj}/locations/global"
                    f"/publishers/google/models/{resolved_model}:generateContent"
                )
                headers = {
                    "Authorization": f"Bearer {tok}",
                    "x-goog-user-project": proj,
                    "Content-Type": "application/json",
                }
            resp = await client.post(url, headers=headers, json=payload, timeout=timeout)
            elapsed = time.perf_counter() - t0
            if resp.status_code == 200:
                return _extract_gemini_text(resp.json()), elapsed, None
            if resp.status_code == 400 and use_thinking_cfg and "thinking" in resp.text.lower():
                use_thinking_cfg = False
                last_err = f"Argon HTTP 400 (disabling thinkingConfig): {resp.text[:160]}"
                continue
            if resp.status_code == 401 and not gemini_key:
                _VERTEX_TOKEN = None
                _VERTEX_TOKEN_TS = 0.0
                last_err = f"Argon HTTP 401 (refreshing token): {resp.text[:160]}"
                continue
            if resp.status_code in {429, 500, 502, 503, 504}:
                last_err = f"Argon HTTP {resp.status_code}: {resp.text[:180]}"
                await asyncio.sleep(min(10.0, 2.0 ** attempt))
                continue
            return "", elapsed, f"Argon HTTP {resp.status_code}: {resp.text[:220]}"
        except Exception as exc:
            last_err = f"Argon {type(exc).__name__}: {str(exc)[:160]}"
            await asyncio.sleep(min(10.0, 2.0 ** attempt))

    return "", 0.0, last_err or "Argon unknown failure"
