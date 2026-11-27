"""Static JSON bundle exporter & Hugging Face Hub dataset persistence for Volume II."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import VOL2_STATE_REPO
from .schema import (
    AgentExchangeRecord,
    AttackCluster,
    CercleMathProofState,
    CerclePrediction,
    HypothesisProof,
    ObservatoryMeta,
    VerifiedIncident,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _flatten_for_hf_jsonl(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize heterogeneous nested dicts/lists so Hugging Face Datasets Viewer infers clean Arrow schemas."""
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, dict):
            out[f"{k}_json"] = json.dumps(v, ensure_ascii=False, sort_keys=True)
        elif isinstance(v, list) and any(isinstance(el, (dict, list)) for el in v):
            out[f"{k}_json"] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = v
    return out


def _write_jsonl(path: Path, items: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(_flatten_for_hf_jsonl(item), ensure_ascii=False) for item in items]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def export_static_site_bundle(
    out_dir: str | Path,
    *,
    meta: ObservatoryMeta,
    incidents: Sequence[VerifiedIncident],
    clusters: Sequence[AttackCluster],
    hypotheses: Sequence[HypothesisProof],
    proofs: Sequence[CercleMathProofState],
    predictions: Sequence[CerclePrediction],
    exchanges: Optional[AgentExchangeRecord] = None,
    max_site_incidents: int = 450,
) -> Dict[str, str]:
    """Export all Volume II JSON replay files cleanly into `site/data/`."""
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)

    sorted_inc = sorted(incidents, key=lambda x: x.date_start, reverse=True)[:max_site_incidents]
    paths = {
        "meta": base / "vol2_meta.json",
        "incidents": base / "vol2_incidents.json",
        "clusters": base / "vol2_clusters.json",
        "hypotheses": base / "vol2_hypotheses.json",
        "proofs": base / "vol2_proofs.json",
        "predictions": base / "vol2_predictions.json",
        "exchanges": base / "vol2_exchanges.json",
    }

    _write_json(paths["meta"], meta.model_dump())
    _write_json(paths["incidents"], [inc.model_dump() for inc in sorted_inc])
    _write_json(paths["clusters"], [cl.model_dump() for cl in clusters])
    _write_json(paths["hypotheses"], [h.model_dump() for h in hypotheses])
    _write_json(paths["proofs"], [st.model_dump() for st in proofs])
    _write_json(paths["predictions"], [pr.model_dump() for pr in predictions])
    ex_payload = (
        exchanges.model_dump()
        if exchanges is not None
        else {
            "exchange_id": f"{meta.sweep_id}_exchange",
            "sweep_id": meta.sweep_id,
            "updated_at": meta.updated_at,
            "topic": "Autonomous Agent Mathematical Proof Exchange",
            "open_hyp_ids": [],
            "turns": meta.agent_exchanges,
        }
    )
    _write_json(paths["exchanges"], ex_payload)
    return {k: str(v) for k, v in paths.items()}


def _build_dataset_card(meta: ObservatoryMeta) -> str:
    return f"""---
license: cc-by-4.0
task_categories:
  - time-series-forecasting
  - text-classification
tags:
  - mali
  - osint
  - conflict-forecasting
  - hawkes-process
  - sebeni-chronicle
configs:
  - config_name: incidents
    data_files: "data/incidents.jsonl"
    default: true
  - config_name: clusters
    data_files: "data/clusters.jsonl"
  - config_name: hypotheses
    data_files: "data/hypotheses.jsonl"
  - config_name: proofs
    data_files: "data/proofs.jsonl"
  - config_name: predictions
    data_files: "data/predictions.jsonl"
  - config_name: exchanges
    data_files: "data/exchanges.jsonl"
---

# The Sebeni Chronicle · Volume II: Mali Conflict OSINT Observatory

Structured, multi-perspective, admin-2 (*cercle*)-level conflict pattern dataset, falsifiable mathematical hypothesis proofs, and strictly causal walk-forward temporal benchmarks across Mali (`38` canonical *cercles*).

- **Latest Sweep ID**: `{meta.sweep_id}` (`{meta.updated_at}`)
- **Verified Incidents**: `{meta.total_verified_incidents}` (deduplicated from `{meta.total_raw_records}` raw OSINT records)
- **Attack Clusters**: `{meta.total_clusters}`
- **Tested Mathematical Hypotheses**: `{meta.total_hypotheses}` (`{meta.proven_or_supported_count}` `PROVEN`/`SUPPORTED` at $\\alpha = 0.05$)
- **Walk-Forward Evaluation**: `{meta.walk_forward_folds}` causal folds · `AUROC={meta.global_auroc}` · `PR-AUC={meta.global_pr_auc}` · `Brier={meta.global_brier}` · `BSS={meta.global_bss}`

## Humanitarian & Academic Guardrail
All records are aggregated at Admin-2 (*cercle*) or public open-source locality precision from public humanitarian, multilateral, press, and official (`fama.mil.ml` DIRPA) reports—strictly for academic mathematical modeling and early-warning research. Zero personally identifiable information (PII) or non-public targeting data is stored.

## Credits
**The Sebeni Chronicle** · Antigravity Team · NVIDIA Build · OpenRouter Free Models · Vertex AI · `Gemini 4 Neon-High-Orchestrator`.
"""


def push_bundle_to_hf(
    *,
    meta: ObservatoryMeta,
    incidents: Sequence[VerifiedIncident],
    clusters: Sequence[AttackCluster],
    hypotheses: Sequence[HypothesisProof],
    proofs: Sequence[CercleMathProofState],
    predictions: Sequence[CerclePrediction],
    exchanges: Optional[AgentExchangeRecord] = None,
    staging_dir: str | Path = "/tmp/vol2_hf_staging",
    repo_id: str = VOL2_STATE_REPO,
) -> str:
    """Create/update public Hugging Face dataset repo `s7d11/chronicles-vol2-mali`."""
    from huggingface_hub import HfApi

    token = (os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("Missing HF_KEY / HF_TOKEN in environment")

    stage = Path(staging_dir)
    stage.mkdir(parents=True, exist_ok=True)
    export_static_site_bundle(
        stage / "json",
        meta=meta,
        incidents=incidents,
        clusters=clusters,
        hypotheses=hypotheses,
        proofs=proofs,
        predictions=predictions,
        exchanges=exchanges,
        max_site_incidents=1500,
    )
    _write_jsonl(stage / "data" / "incidents.jsonl", [i.model_dump() for i in incidents])
    _write_jsonl(stage / "data" / "clusters.jsonl", [c.model_dump() for c in clusters])
    _write_jsonl(stage / "data" / "hypotheses.jsonl", [h.model_dump() for h in hypotheses])
    _write_jsonl(stage / "data" / "proofs.jsonl", [p.model_dump() for p in proofs])
    _write_jsonl(stage / "data" / "predictions.jsonl", [p.model_dump() for p in predictions])
    if exchanges is not None:
        _write_jsonl(stage / "data" / "exchanges.jsonl", [t.model_dump() for t in exchanges.turns])
    (stage / "README.md").write_text(_build_dataset_card(meta), encoding="utf-8")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(stage),
        commit_message=f"Vol II Mali sweep {meta.sweep_id} ({meta.total_verified_incidents} incidents, AUROC {meta.global_auroc})",
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def pull_json_bundle_from_hf(
    out_dir: str | Path,
    *,
    repo_id: str = VOL2_STATE_REPO,
) -> List[str]:
    """Restore `site/data/vol2_*.json` from Hugging Face dataset snapshot."""
    from huggingface_hub import hf_hub_download

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for name in (
        "vol2_meta.json",
        "vol2_incidents.json",
        "vol2_clusters.json",
        "vol2_hypotheses.json",
        "vol2_proofs.json",
        "vol2_predictions.json",
        "vol2_exchanges.json",
    ):
        try:
            local = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=f"json/{name}",
                token=(os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN") or None),
            )
            dst = target / name
            dst.write_text(Path(local).read_text(encoding="utf-8"), encoding="utf-8")
            written.append(str(dst))
        except Exception:
            if name != "vol2_exchanges.json":
                raise
    return written
