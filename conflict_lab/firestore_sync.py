"""Real-time Cloud Firestore publisher for Volume II Mali OSINT Observatory."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import httpx

from . import DEFAULT_FIRESTORE_DB, DEFAULT_FIRESTORE_PROJECT
from .schema import (
    AttackCluster,
    CercleMathProofState,
    CerclePrediction,
    HypothesisProof,
    ObservatoryMeta,
    VerifiedIncident,
)


def to_fs_value(val: Any) -> Dict[str, Any]:
    """Convert Python value into Cloud Firestore REST API Typed Value."""
    if val is None:
        return {"nullValue": None}
    if isinstance(val, bool):
        return {"booleanValue": val}
    if isinstance(val, int):
        return {"integerValue": str(val)}
    if isinstance(val, float):
        return {"doubleValue": float(val)}
    if isinstance(val, dict):
        return {"mapValue": {"fields": to_fs_fields(val)}}
    if isinstance(val, (list, tuple)):
        return {"arrayValue": {"values": [to_fs_value(v) for v in val]}}
    return {"stringValue": str(val)}


def to_fs_fields(mapping: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(k): to_fs_value(v) for k, v in mapping.items()}


def sanitize_doc_id(raw_id: str, *, max_len: int = 76) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in str(raw_id))
    cleaned = cleaned.strip("_") or "doc_001"
    return cleaned[:max_len]


def clamp_incident_doc(inc: VerifiedIncident) -> Dict[str, Any]:
    d = inc.model_dump()
    d["incident_id"] = sanitize_doc_id(d["incident_id"], max_len=78)
    d["headline"] = (d.get("headline") or "")[:480]
    d["summary"] = (d.get("summary") or "")[:3800]
    d["adjudicator_note"] = (d.get("adjudicator_note") or "")[:1900]
    d["locality"] = (d.get("locality") or "")[:150]
    d["actor_primary"] = (d.get("actor_primary") or "Unattributed")[:150]
    d["actor_secondary"] = (d.get("actor_secondary") or "")[:150]
    d["sources"] = list(d.get("sources") or [])[:60]
    d["perspectives"] = list(d.get("perspectives") or [])[:12]
    return d


def clamp_cluster_doc(cl: AttackCluster) -> Dict[str, Any]:
    d = cl.model_dump()
    d["cluster_id"] = sanitize_doc_id(d["cluster_id"], max_len=78)
    d["label"] = (d.get("label") or "")[:240]
    d["sample_events"] = list(d.get("sample_events") or [])[:40]
    return d


def clamp_hypothesis_doc(h: HypothesisProof) -> Dict[str, Any]:
    d = h.model_dump()
    d["hyp_id"] = sanitize_doc_id(d["hyp_id"], max_len=78)
    d["cluster_id"] = sanitize_doc_id(d["cluster_id"], max_len=78)
    d["title"] = (d.get("title") or "")[:240]
    d["symbolic_formula"] = (d.get("symbolic_formula") or "")[:980]
    d["null_hypothesis"] = (d.get("null_hypothesis") or "")[:980]
    d["alternative_hypothesis"] = (d.get("alternative_hypothesis") or "")[:980]
    d["analyst_narrative"] = (d.get("analyst_narrative") or "")[:3800]
    d["derivation_steps"] = [str(s)[:400] for s in (d.get("derivation_steps") or [])[:16]]
    params = d.get("parameters") or {}
    # Keep parameters flat and bounded <= 32 keys
    bounded_params: Dict[str, Any] = {}
    for k, v in list(params.items())[:32]:
        if isinstance(v, (dict, list)):
            continue
        bounded_params[str(k)] = v
    d["parameters"] = bounded_params
    return d


class Vol2FirestorePublisher:
    """Publishes Volume II Mali telemetry & mathematical proof states to Cloud Firestore."""

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
    def from_env(cls) -> Optional["Vol2FirestorePublisher"]:
        db_id = (
            os.environ.get("VOL2_FIRESTORE_DB")
            or os.environ.get("ED4_FIRESTORE_DB")
            or DEFAULT_FIRESTORE_DB
        ).strip()
        if not db_id:
            return None
        proj = (
            os.environ.get("VOL2_FIRESTORE_PROJECT")
            or os.environ.get("ED4_FIRESTORE_PROJECT")
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
            gcloud_env = {k: v for k, v in os.environ.items() if k != "GOOGLE_APPLICATION_CREDENTIALS"}
            tok = subprocess.check_output(
                ["gcloud", "auth", "print-access-token"],
                env=gcloud_env,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if tok:
                self._token = tok
                self._token_ts = now
                return tok
        except Exception:
            pass

        import google.auth
        import google.auth.transport.requests

        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(google.auth.transport.requests.Request())
        self._token = str(creds.token)
        self._token_ts = now
        return self._token

    def patch_doc(self, doc_path: str, data: Dict[str, Any]) -> bool:
        url = f"{self.base_url}/{doc_path.lstrip('/')}"
        payload = {"fields": to_fs_fields(data)}
        for attempt in range(2):
            try:
                tok = self._get_token(force_refresh=(attempt > 0))
                resp = self._client.patch(
                    url,
                    headers={
                        "Authorization": f"Bearer {tok}",
                        "x-goog-user-project": self.project_id,
                    },
                    json=payload,
                )
                if resp.status_code == 401 and attempt == 0:
                    continue
                if resp.status_code in (200, 201):
                    return True
                print(
                    f"[vol2-firestore] WARN {resp.status_code} on {doc_path}: {resp.text[:180]}",
                    file=sys.stderr,
                )
                return False
            except Exception as exc:
                if attempt == 1:
                    print(f"[vol2-firestore] WARN patch {doc_path} failed: {exc}", file=sys.stderr)
                    return False
        return False

    def publish_bundle(
        self,
        *,
        meta: ObservatoryMeta,
        incidents: Sequence[VerifiedIncident],
        clusters: Sequence[AttackCluster],
        hypotheses: Sequence[HypothesisProof],
        proofs: Sequence[CercleMathProofState],
        predictions: Sequence[CerclePrediction],
        max_incidents_live: int = 120,
    ) -> Dict[str, int]:
        """Push Volume II observatory telemetry to Firestore collections using bounded thread pool."""
        from concurrent.futures import ThreadPoolExecutor

        counts = {"meta": 0, "incidents": 0, "clusters": 0, "hypotheses": 0, "proofs": 0, "predictions": 0}
        self._get_token()

        meta_dict = meta.model_dump()
        meta_dict["corridor_edges"] = list(meta_dict.get("corridor_edges") or [])[:40]
        meta_dict["active_agents"] = list(meta_dict.get("active_agents") or [])[:16]
        if self.patch_doc("vol2_meta/overview", meta_dict):
            counts["meta"] = 1

        tasks: List[tuple[str, str, Dict[str, Any]]] = []
        sorted_inc = sorted(incidents, key=lambda x: x.date_start, reverse=True)[:max_incidents_live]
        for inc in sorted_inc:
            cd = clamp_incident_doc(inc)
            tasks.append(("incidents", f"vol2_incidents/{cd['incident_id']}", cd))
        for cl in clusters[:30]:
            cd = clamp_cluster_doc(cl)
            tasks.append(("clusters", f"vol2_clusters/{cd['cluster_id']}", cd))
        for h in hypotheses[:90]:
            hd = clamp_hypothesis_doc(h)
            tasks.append(("hypotheses", f"vol2_hypotheses/{hd['hyp_id']}", hd))
        for st in proofs:
            sd = st.model_dump()
            sd["event_timestamps_days"] = list(sd.get("event_timestamps_days") or [])[-365:]
            tasks.append(("proofs", f"vol2_proofs/{sanitize_doc_id(st.cercle_id, max_len=64)}", sd))
        for pr in predictions:
            tasks.append(("predictions", f"vol2_predictions/{sanitize_doc_id(pr.cercle_id, max_len=64)}", pr.model_dump()))

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda item: (item[0], self.patch_doc(item[1], item[2])), tasks))
        for kind, ok in results:
            if ok:
                counts[kind] += 1

        return counts
