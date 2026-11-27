"""Real-time Cloud Firestore publisher for Edition No. 4 Cloud Run Jobs.

Writes public read-only telemetry to Cloud Firestore via REST API using GCP ADC
so GitHub Pages (`site/assets/village_world.js`) receives instant real-time
updates via Web SDK `onSnapshot` without polling Cloud Run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import httpx

from village_lab.schema import LanguageVillageState, VillageEvent

DEFAULT_FIRESTORE_PROJECT = "mlgramv1"
DEFAULT_FIRESTORE_DB = "chronicles-live"


def _to_fs_value(val: Any) -> Dict[str, Any]:
    """Convert a Python primitive/list/dict to a Firestore REST Value object."""
    if val is None:
        return {"nullValue": None}
    if isinstance(val, bool):
        return {"booleanValue": val}
    if isinstance(val, int):
        return {"integerValue": str(val)}
    if isinstance(val, float):
        return {"doubleValue": val}
    if isinstance(val, dict):
        return {"mapValue": {"fields": _to_fs_fields(val)}}
    if isinstance(val, (list, tuple)):
        return {"arrayValue": {"values": [_to_fs_value(x) for x in val]}}
    return {"stringValue": str(val)}


def _to_fs_fields(mapping: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Convert a Python dictionary into Firestore REST `fields` mapping."""
    return {str(k): _to_fs_value(v) for k, v in mapping.items()}


def _serialize_bounded_json_list(items: List[Dict[str, Any]], max_chars: int) -> str:
    """Serialize the newest suffix of `items` that fits within `max_chars` as valid JSON."""
    if not items or max_chars < 2:
        return "[]"
    # Keep newest items (tail of chronological list), trimming oldest first to avoid mid-JSON truncation.
    for start in range(len(items)):
        encoded = json.dumps(items[start:], ensure_ascii=False)
        if len(encoded) <= max_chars:
            return encoded
    return "[]"


class FirestoreLivePublisher:
    """Best-effort REST publisher pushing Edition 4 events & state to Cloud Firestore."""

    def __init__(
        self,
        *,
        project_id: str = DEFAULT_FIRESTORE_PROJECT,
        database_id: str = DEFAULT_FIRESTORE_DB,
    ) -> None:
        self.project_id = project_id
        self.database_id = database_id
        self.base_url = (
            f"https://firestore.googleapis.com/v1/projects/{self.project_id}"
            f"/databases/{self.database_id}/documents"
        )
        self._client = httpx.Client(timeout=15.0)
        self._token: Optional[str] = None
        self._token_ts: float = 0.0

    @classmethod
    def from_env(cls) -> Optional["FirestoreLivePublisher"]:
        db_id = (os.environ.get("ED4_FIRESTORE_DB") or "").strip()
        if not db_id:
            return None
        proj = (
            os.environ.get("ED4_FIRESTORE_PROJECT")
            or os.environ.get("GCP_VERTEX_PROJECT")
            or DEFAULT_FIRESTORE_PROJECT
        ).strip()
        return cls(project_id=proj, database_id=db_id)

    def _get_token(self, *, force_refresh: bool = False) -> str:
        now = time.time()
        if self._token and not force_refresh and (now - self._token_ts) < 2400:
            return self._token

        try:
            resp = self._client.get(
                "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
                headers={"Metadata-Flavor": "Google"},
                timeout=3.0,
            )
            if resp.status_code == 200:
                tok = str(resp.json().get("access_token") or "").strip()
                if tok:
                    self._token = tok
                    self._token_ts = now
                    return tok
        except Exception:
            pass

        try:
            import google.auth
            import google.auth.transport.requests

            creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            creds.refresh(google.auth.transport.requests.Request())
            if creds.token:
                self._token = str(creds.token)
                self._token_ts = now
                return self._token
        except Exception:
            pass

        gcloud_env = {k: v for k, v in os.environ.items() if k != "GOOGLE_APPLICATION_CREDENTIALS"}
        tok = subprocess.check_output(
            ["gcloud", "auth", "print-access-token"],
            env=gcloud_env,
            text=True,
        ).strip()
        self._token = tok
        self._token_ts = now
        return tok

    def _patch_doc(
        self,
        doc_path: str,
        data: Dict[str, Any],
        *,
        params: Optional[Dict[str, str]] = None,
        ignore_already_exists: bool = False,
    ) -> bool:
        url = f"{self.base_url}/{doc_path.lstrip('/')}"
        payload = {"fields": _to_fs_fields(data)}
        for attempt in range(2):
            try:
                tok = self._get_token(force_refresh=(attempt > 0))
                resp = self._client.patch(
                    url,
                    headers={
                        "Authorization": f"Bearer {tok}",
                        "x-goog-user-project": self.project_id,
                        "Content-Type": "application/json",
                    },
                    params=params,
                    json=payload,
                )
                if resp.status_code in (200, 201):
                    return True
                if ignore_already_exists and resp.status_code in (409, 412):
                    return False
                if resp.status_code == 401 and attempt == 0:
                    self._token = None
                    continue
                print(
                    f"[firestore-sync] PATCH {doc_path} HTTP {resp.status_code}: {resp.text[:180]}",
                    file=sys.stderr,
                    flush=True,
                )
                return False
            except Exception as exc:
                print(
                    f"[firestore-sync] PATCH {doc_path} error: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                return False
        return False

    def publish_event_sync(self, ev: VillageEvent) -> bool:
        doc_path = f"ed4_languages/{ev.lang}/events/{ev.seq:06d}"
        data = {
            "seq": int(ev.seq),
            "turn_id": int(ev.turn_id),
            "ts": ev.ts,
            "lang": ev.lang,
            "kind": ev.kind,
            "agent_id": ev.agent_id,
            "target_agent_id": ev.target_agent_id,
            "zone": ev.zone,
            "target_zone": ev.target_zone,
            "label": ev.label[:256],
            "summary": ev.summary[:2000],
            "path_json": json.dumps(ev.path, ensure_ascii=False),
            "meta_json": json.dumps(ev.meta, ensure_ascii=False)[:16000],
        }
        return self._patch_doc(doc_path, data)

    def get_highest_event_seq(self, lang: str) -> int:
        """Query highest existing event `seq` in Firestore for idempotent restart backfill."""
        url = (
            f"{self.base_url}/ed4_languages/{lang}/events"
            "?orderBy=seq%20desc&pageSize=1"
        )
        try:
            tok = self._get_token()
            resp = self._client.get(
                url,
                headers={"Authorization": f"Bearer {tok}", "x-goog-user-project": self.project_id},
            )
            if resp.status_code == 200:
                docs = resp.json().get("documents") or []
                if docs:
                    fields = docs[0].get("fields") or {}
                    seq_val = (fields.get("seq") or {}).get("integerValue")
                    if seq_val is not None:
                        return int(seq_val)
        except Exception:
            pass
        return 0

    @staticmethod
    def _build_state_fields(
        state: LanguageVillageState,
        *,
        running: bool,
        job_execution: str = "",
    ) -> Dict[str, Any]:
        exchanges = [x.model_dump() for x in state.recent_exchanges[-18:]]
        decisions = [d.model_dump() for d in state.recent_decisions[-20:]]
        agents = [a.model_dump() for a in state.active_agents[:64]]
        return {
            "lang": state.lang,
            "lang_name": state.lang_name,
            "target_entries": int(state.target_entries),
            "baseline_lx_count": int(state.baseline_lx_count),
            "added_lx_count": int(state.added_lx_count),
            "added_va_count": int(state.added_va_count),
            "baseline_gram_lines": int(state.baseline_gram_lines),
            "added_gram_rules_count": int(state.added_gram_rules_count),
            "total_turns": int(state.total_turns),
            "total_candidates_reviewed": int(state.total_candidates_reviewed),
            "total_rejected": int(state.total_rejected),
            "baseline_devmini_pct": float(state.baseline_devmini_pct),
            "latest_devmini_pct": float(state.latest_devmini_pct),
            "updated_at": state.updated_at,
            "orchestrator_model": state.orchestrator_model,
            "critic_model": state.critic_model,
            "running": bool(running),
            "job_execution": job_execution[:128],
            "active_agents_json": _serialize_bounded_json_list(agents, 32000),
            "recent_exchanges_json": _serialize_bounded_json_list(exchanges, 32000),
            "recent_decisions_json": _serialize_bounded_json_list(decisions, 48000),
        }

    def seed_baseline_if_missing(
        self,
        state: LanguageVillageState,
        *,
        running: bool = False,
        job_execution: str = "",
    ) -> bool:
        """Create `ed4_languages/<lang>` atomically only if it does not already exist."""
        doc_path = f"ed4_languages/{state.lang}"
        data = self._build_state_fields(state, running=running, job_execution=job_execution)
        return self._patch_doc(
            doc_path,
            data,
            params={"currentDocument.exists": "false"},
            ignore_already_exists=True,
        )

    def publish_state_sync(
        self,
        state: LanguageVillageState,
        *,
        running: bool,
        job_execution: str = "",
    ) -> bool:
        doc_path = f"ed4_languages/{state.lang}"
        data = self._build_state_fields(state, running=running, job_execution=job_execution)
        return self._patch_doc(doc_path, data)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
