"""CLI runner for Volume II 9-agent Mali OSINT retrieval, mathematical proofs, and live publishing."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import httpx

from . import AGENT_CAST
from .firestore_sync import Vol2FirestorePublisher
from .georesolve_dedup import adjudicate_recent_incidents_batched, deduplicate_and_georesolve
from .hf_sync import export_static_site_bundle, pull_json_bundle_from_hf, push_bundle_to_hf
from .hypothesis_proofs import synthesize_llm_hypothesis_notes
from .pattern_model import (
    build_attack_clusters,
    build_corridor_edges,
    run_walk_forward_and_predictions,
)
from .schema import ObservatoryMeta
from .sources import gather_all_sources


def load_repo_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


async def run_sweep_pipeline(
    *,
    out_dir: str | Path = "site/data",
    max_ucdp_rows: int = 1200,
    use_llm: bool = True,
    push_hf: bool = False,
    push_firestore: bool = False,
) -> Dict[str, Any]:
    load_repo_dotenv()
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sweep_id = datetime.now(timezone.utc).strftime("vol2_%Y%m%d_%H%M%SZ")

    agent_roster: List[Dict[str, Any]] = [
        {"agent_id": s.agent_id, "name": s.name, "provider": s.provider, "model": s.model_id, "role": s.role}
        for s in AGENT_CAST
    ]

    async with httpx.AsyncClient(timeout=35.0, follow_redirects=True) as client:
        print(f"[vol2] Gathering multi-perspective Mali OSINT across 6 live sources (use_llm={use_llm})...", flush=True)
        gathered = await gather_all_sources(client, max_ucdp_rows=max_ucdp_rows, use_llm=use_llm)
        records = gathered["records"]
        source_counts = gathered["source_counts"]
        print(
            f"[vol2] Raw records gathered: {len(records)} ({source_counts})",
            flush=True,
        )

        print("[vol2] Deduplicating (±2d, 15km) & georesolving across 38 Malian cercles...", flush=True)
        incidents = deduplicate_and_georesolve(records)
        if use_llm and incidents:
            print("[vol2] Adjudicating recent events with Gemini 3.1 Flash-Lite + OpenRouter Free...", flush=True)
            incidents = await adjudicate_recent_incidents_batched(client, incidents, batch_size=20, max_to_audit=40)

        print(f"[vol2] Verified incidents: {len(incidents)}. Clustering corridors & proving hypotheses...", flush=True)
        clusters, cluster_members = build_attack_clusters(incidents)
        corridor_edges = build_corridor_edges(incidents)

        model_out = run_walk_forward_and_predictions(incidents, clusters, cluster_members)
        proofs = model_out["proofs"]
        if use_llm and proofs:
            proofs = await synthesize_llm_hypothesis_notes(client, proofs, max_to_enrich=8, use_llm=True)

        cercle_states = model_out["cercle_states"]
        predictions = model_out["predictions"]

    perspective_counts = dict(Counter(p for inc in incidents for p in inc.perspectives))
    actor_counts = dict(Counter(inc.actor_primary for inc in incidents).most_common(12))
    tactic_counts = dict(Counter(inc.tactic for inc in incidents).most_common(10))
    proven_cnt = sum(1 for h in proofs if h.verdict in ("PROVEN", "SUPPORTED"))

    meta = ObservatoryMeta(
        sweep_id=sweep_id,
        updated_at=now_iso,
        total_raw_records=len(records),
        total_verified_incidents=len(incidents),
        total_clusters=len(clusters),
        total_hypotheses=len(proofs),
        proven_or_supported_count=proven_cnt,
        walk_forward_folds=model_out["walk_forward_folds"],
        global_auroc=model_out["global_auroc"],
        global_pr_auc=model_out["global_pr_auc"],
        global_brier=model_out["global_brier"],
        global_bss=model_out["global_bss"],
        source_counts=source_counts,
        perspective_counts=perspective_counts,
        actor_counts=actor_counts,
        tactic_counts=tactic_counts,
        corridor_edges=corridor_edges,
        active_agents=agent_roster,
        running=False,
    )

    exported = export_static_site_bundle(
        out_dir,
        meta=meta,
        incidents=incidents,
        clusters=clusters,
        hypotheses=proofs,
        proofs=cercle_states,
        predictions=predictions,
    )
    print(
        f"[vol2] Exported JSON bundle to {out_dir} | clusters={len(clusters)} "
        f"hypotheses={len(proofs)} (proven/supported={proven_cnt}) "
        f"AUROC={meta.global_auroc} Brier={meta.global_brier}",
        flush=True,
    )

    hf_url = None
    if push_hf:
        try:
            hf_url = push_bundle_to_hf(
                meta=meta,
                incidents=incidents,
                clusters=clusters,
                hypotheses=proofs,
                proofs=cercle_states,
                predictions=predictions,
            )
            print(f"[vol2] Synced HF Dataset: {hf_url}", flush=True)
        except Exception as exc:
            print(f"[vol2] WARN HF upload failed: {exc}", file=sys.stderr, flush=True)

    fs_counts = None
    if push_firestore:
        pub = Vol2FirestorePublisher.from_env()
        if pub is not None:
            fs_counts = pub.publish_bundle(
                meta=meta,
                incidents=incidents,
                clusters=clusters,
                hypotheses=proofs,
                proofs=cercle_states,
                predictions=predictions,
            )
            print(f"[vol2] Synced Cloud Firestore ({pub.database_id}): {fs_counts}", flush=True)

    return {
        "meta": meta.model_dump(),
        "exported": exported,
        "hf_url": hf_url,
        "firestore_counts": fs_counts,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sebeni Chronicle Vol II Mali OSINT & Math Proofs CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sw = sub.add_parser("sweep", help="Gather live Mali OSINT, run math proofs & predictions, export JSON")
    sw.add_argument("--out-dir", default="site/data")
    sw.add_argument("--max-ucdp-rows", type=int, default=1200)
    sw.add_argument("--no-llm", action="store_true", help="Skip external LLM adjudication/synthesis calls")
    sw.add_argument("--push-hf", action="store_true", help="Upload snapshots to Hugging Face Hub dataset")
    sw.add_argument("--push-firestore", action="store_true", help="Publish live states to Cloud Firestore")

    sync_cmd = sub.add_parser("sync-from-hf", help="Pull Vol II JSON files from Hugging Face Hub")
    sync_cmd.add_argument("--out-dir", default="site/data")

    args = parser.parse_args(argv)
    if args.command == "sweep":
        asyncio.run(
            run_sweep_pipeline(
                out_dir=args.out_dir,
                max_ucdp_rows=args.max_ucdp_rows,
                use_llm=not args.no_llm,
                push_hf=args.push_hf,
                push_firestore=args.push_firestore,
            )
        )
        return 0

    if args.command == "sync-from-hf":
        load_repo_dotenv()
        written = pull_json_bundle_from_hf(args.out_dir)
        print(f"[vol2] Restored {len(written)} files from HF into {args.out_dir}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
