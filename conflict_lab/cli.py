"""CLI runner for Volume II 9-agent Mali OSINT retrieval, mathematical proofs, and live publishing."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from . import AGENT_CAST
from .firestore_sync import Vol2FirestorePublisher
from .georesolve_dedup import adjudicate_recent_incidents_batched, deduplicate_and_georesolve
from .hf_sync import export_static_site_bundle, pull_json_bundle_from_hf, push_bundle_to_hf
from .hypothesis_proofs import run_autonomous_proof_debate, synthesize_llm_hypothesis_notes
from .pattern_model import (
    build_attack_clusters,
    build_corridor_edges,
    run_walk_forward_and_predictions,
)
from .schema import ObservatoryMeta, VerifiedIncident
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
    from_incidents: Optional[str | Path] = None,
) -> Dict[str, Any]:
    load_repo_dotenv()
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sweep_id = datetime.now(timezone.utc).strftime("vol2_%Y%m%d_%H%M%SZ")

    agent_roster: List[Dict[str, Any]] = [
        {"agent_id": s.agent_id, "name": s.name, "provider": s.provider, "model": s.model_id, "role": s.role}
        for s in AGENT_CAST
    ]
    fs_pub = Vol2FirestorePublisher.from_env() if push_firestore else None
    if fs_pub is not None:
        from .firestore_sync import clamp_exchange_doc
        from .schema import AgentExchangeRecord as _ExRec

        fs_pub.patch_doc(
            "vol2_meta/overview",
            {
                "sweep_id": sweep_id,
                "updated_at": now_iso,
                "running": True,
                "active_agents": agent_roster[:16],
            },
        )
        fs_pub.patch_doc(
            "vol2_exchanges/latest",
            clamp_exchange_doc(
                _ExRec(
                    exchange_id=f"{sweep_id}_exchange",
                    sweep_id=sweep_id,
                    updated_at=now_iso,
                    topic="Gathering live Mali OSINT across 6 perspectives & preparing proof debate...",
                    open_hyp_ids=[],
                    turns=[],
                )
            ),
        )

    async with httpx.AsyncClient(timeout=35.0, follow_redirects=True) as client:
        if from_incidents:
            src_path = Path(from_incidents)
            raw_list = json.loads(src_path.read_text(encoding="utf-8"))
            incidents = [VerifiedIncident.model_validate(item) for item in raw_list]
            existing_meta_path = Path(out_dir) / "vol2_meta.json"
            prev_meta: Dict[str, Any] = {}
            if existing_meta_path.is_file():
                try:
                    prev_meta = json.loads(existing_meta_path.read_text(encoding="utf-8"))
                except Exception:
                    prev_meta = {}
            records_count = int(prev_meta.get("total_raw_records") or len(incidents))
            source_counts = dict(prev_meta.get("source_counts") or {"snapshot_replay": len(incidents)})
            print(f"[vol2] Loaded {len(incidents)} verified incidents from {src_path} (use_llm={use_llm})", flush=True)
        else:
            print(f"[vol2] Gathering multi-perspective Mali OSINT across 6 live sources (use_llm={use_llm})...", flush=True)
            gathered = await gather_all_sources(client, max_ucdp_rows=max_ucdp_rows, use_llm=use_llm)
            records = gathered["records"]
            records_count = len(records)
            source_counts = gathered["source_counts"]
            print(
                f"[vol2] Raw records gathered: {records_count} ({source_counts})",
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
            print("[vol2] Synthesizing hypothesis mechanism narratives...", flush=True)
            proofs = await synthesize_llm_hypothesis_notes(client, proofs, max_to_enrich=8, use_llm=True)

        print("[vol2] Running autonomous Explorer -> Theorist -> Arbiter proof debate...", flush=True)

        def _stream_turn_to_firestore(partial_ex: Any) -> None:
            if fs_pub is not None:
                from .firestore_sync import clamp_exchange_doc

                ex_doc = clamp_exchange_doc(partial_ex)
                fs_pub.patch_doc("vol2_exchanges/latest", ex_doc)
                fs_pub.patch_doc(
                    "vol2_meta/overview",
                    {
                        "sweep_id": sweep_id,
                        "updated_at": partial_ex.updated_at,
                        "running": True,
                        "agent_exchanges": list(ex_doc.get("turns") or [])[:24],
                    },
                )

        open_proofs, exchange_record = await run_autonomous_proof_debate(
            client if use_llm else None,
            incidents,
            clusters,
            proofs,
            sweep_id=sweep_id,
            use_llm=use_llm,
            on_turn=_stream_turn_to_firestore if fs_pub is not None else None,
        )
        proofs.extend(open_proofs)

        cercle_states = model_out["cercle_states"]
        predictions = model_out["predictions"]

    perspective_counts = dict(Counter(p for inc in incidents for p in inc.perspectives))
    actor_counts = dict(Counter(inc.actor_primary for inc in incidents).most_common(12))
    tactic_counts = dict(Counter(inc.tactic for inc in incidents).most_common(10))
    proven_cnt = sum(1 for h in proofs if h.verdict in ("PROVEN", "SUPPORTED"))

    meta = ObservatoryMeta(
        sweep_id=sweep_id,
        updated_at=now_iso,
        total_raw_records=records_count,
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
        agent_exchanges=[t.model_dump() for t in exchange_record.turns[:24]],
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
        exchanges=exchange_record,
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
                exchanges=exchange_record,
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
                exchanges=exchange_record,
            )
            print(f"[vol2] Synced Cloud Firestore ({pub.database_id}): {fs_counts}", flush=True)

    return {
        "meta": meta.model_dump(),
        "exported": exported,
        "hf_url": hf_url,
        "firestore_counts": fs_counts,
    }


async def run_continuous_loop(
    *,
    out_dir: str | Path = "site/data",
    max_ucdp_rows: int = 1200,
    use_llm: bool = True,
    push_hf: bool = False,
    push_firestore: bool = False,
    max_cycles: int = 0,
    max_hours: float = 23.2,
    cycle_sleep: float = 45.0,
    osint_every: int = 6,
    checkpoint_every: int = 6,
) -> Dict[str, Any]:
    """Ongoing continuous proof discovery & multi-agent exchange stream until max_cycles or max_hours."""
    import time as _time

    load_repo_dotenv()
    t_start = _time.monotonic()
    max_secs = max(60.0, float(max_hours) * 3600.0)
    fs_pub = Vol2FirestorePublisher.from_env() if push_firestore else None

    agent_roster: List[Dict[str, Any]] = [
        {"agent_id": s.agent_id, "name": s.name, "provider": s.provider, "model": s.model_id, "role": s.role}
        for s in AGENT_CAST
    ]

    async with httpx.AsyncClient(timeout=35.0, follow_redirects=True) as client:
        cycle = 0
        incidents: List[VerifiedIncident] = []
        raw_count = 0
        source_counts: Dict[str, int] = {}
        clusters: List[Any] = []
        cluster_members: Dict[str, Any] = {}
        corridor_edges: List[Dict[str, Any]] = []
        model_out: Dict[str, Any] = {}
        base_proofs: List[Any] = []
        history_turns: List[Any] = []
        focus_order: List[str] = []
        last_meta: Optional[ObservatoryMeta] = None
        last_proofs: List[Any] = []
        last_exchange = None

        while True:
            cycle += 1
            if max_cycles > 0 and cycle > max_cycles:
                break
            if (_time.monotonic() - t_start) >= max_secs and cycle > 1:
                break

            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            sweep_id = f"vol2_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')}_c{cycle}"
            print(f"[vol2-ongoing] === Cycle #{cycle} ({sweep_id}) ===", flush=True)

            if cycle == 1 or (osint_every > 0 and (cycle - 1) % osint_every == 0):
                gathered = await gather_all_sources(client, max_ucdp_rows=max_ucdp_rows, use_llm=use_llm)
                records = gathered["records"]
                raw_count = len(records)
                source_counts = gathered["source_counts"]
                incidents = deduplicate_and_georesolve(records)
                if use_llm and incidents:
                    incidents = await adjudicate_recent_incidents_batched(client, incidents, batch_size=20, max_to_audit=40)
                clusters, cluster_members = build_attack_clusters(incidents)
                corridor_edges = build_corridor_edges(incidents)
                model_out = run_walk_forward_and_predictions(incidents, clusters, cluster_members)
                base_proofs = list(model_out["proofs"])
                if use_llm and base_proofs:
                    base_proofs = await synthesize_llm_hypothesis_notes(client, base_proofs, max_to_enrich=8, use_llm=True)
                # Build focus rotation from high-divergence / high-count active cercles
                focus_order = [
                    p.cercle_id for p in sorted(model_out["cercle_states"], key=lambda x: (x.kl_actor_nats + x.kl_tactic_nats, x.event_count), reverse=True)
                    if p.event_count >= 5
                ] or [c.dominant_cercle_id for c in clusters]

            focus_c = focus_order[(cycle - 1) % len(focus_order)] if focus_order else None

            def _stream_turn(partial_ex: Any) -> None:
                if fs_pub is not None:
                    from .firestore_sync import clamp_exchange_doc

                    ex_doc = clamp_exchange_doc(partial_ex)
                    fs_pub.patch_doc("vol2_exchanges/latest", ex_doc)
                    fs_pub.patch_doc(
                        "vol2_meta/overview",
                        {
                            "sweep_id": sweep_id,
                            "updated_at": partial_ex.updated_at,
                            "running": True,
                            "agent_exchanges": list(ex_doc.get("turns") or [])[:24],
                        },
                    )

            open_proofs, exchange_record = await run_autonomous_proof_debate(
                client if use_llm else None,
                incidents,
                clusters,
                base_proofs,
                sweep_id=sweep_id,
                use_llm=use_llm,
                on_turn=_stream_turn if fs_pub is not None else None,
                focus_cercle=focus_c,
                cycle_num=cycle,
                prior_turns_history=history_turns,
            )
            history_turns = list(exchange_record.turns)
            all_proofs = list(base_proofs) + list(open_proofs)
            last_proofs = all_proofs
            last_exchange = exchange_record

            perspective_counts = dict(Counter(p for inc in incidents for p in inc.perspectives))
            actor_counts = dict(Counter(inc.actor_primary for inc in incidents).most_common(12))
            tactic_counts = dict(Counter(inc.tactic for inc in incidents).most_common(10))
            proven_cnt = sum(1 for h in all_proofs if h.verdict in ("PROVEN", "SUPPORTED"))
            still_running = not (max_cycles > 0 and cycle >= max_cycles) and ((_time.monotonic() - t_start + cycle_sleep) < max_secs)

            meta = ObservatoryMeta(
                sweep_id=sweep_id,
                updated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                total_raw_records=raw_count,
                total_verified_incidents=len(incidents),
                total_clusters=len(clusters),
                total_hypotheses=len(all_proofs),
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
                agent_exchanges=[t.model_dump() for t in history_turns[-24:]],
                running=still_running,
            )
            last_meta = meta

            export_static_site_bundle(
                out_dir,
                meta=meta,
                incidents=incidents,
                clusters=clusters,
                hypotheses=all_proofs,
                proofs=model_out["cercle_states"],
                predictions=model_out["predictions"],
                exchanges=exchange_record,
            )

            if fs_pub is not None:
                if cycle == 1 or (osint_every > 0 and (cycle - 1) % osint_every == 0):
                    fs_pub.publish_bundle(
                        meta=meta,
                        incidents=incidents,
                        clusters=clusters,
                        hypotheses=all_proofs,
                        proofs=model_out["cercle_states"],
                        predictions=model_out["predictions"],
                        exchanges=exchange_record,
                    )
                else:
                    from .firestore_sync import clamp_exchange_doc, clamp_hypothesis_doc

                    fs_pub.patch_doc("vol2_meta/overview", meta.model_dump())
                    fs_pub.patch_doc("vol2_exchanges/latest", clamp_exchange_doc(exchange_record))
                    for oh in open_proofs:
                        hd = clamp_hypothesis_doc(oh)
                        fs_pub.patch_doc(f"vol2_hypotheses/{hd['hyp_id']}", hd)

            if push_hf and (cycle == 1 or (checkpoint_every > 0 and cycle % checkpoint_every == 0)):
                try:
                    hf_url = push_bundle_to_hf(
                        meta=meta,
                        incidents=incidents,
                        clusters=clusters,
                        hypotheses=all_proofs,
                        proofs=model_out["cercle_states"],
                        predictions=model_out["predictions"],
                        exchanges=exchange_record,
                    )
                    print(f"[vol2-ongoing] Cycle #{cycle} HF checkpoint: {hf_url}", flush=True)
                except Exception as exc:
                    print(f"[vol2-ongoing] WARN Cycle #{cycle} HF checkpoint failed: {exc}", file=sys.stderr, flush=True)

            if not still_running:
                break
            await asyncio.sleep(max(5.0, float(cycle_sleep)))

        # Finalize running=False if loop ends cleanly
        if fs_pub is not None and last_meta is not None:
            last_meta.running = False
            fs_pub.patch_doc("vol2_meta/overview", last_meta.model_dump())

    return {
        "cycles_completed": cycle - 1 if (max_cycles > 0 and cycle > max_cycles) else cycle,
        "meta": last_meta.model_dump() if last_meta else {},
        "total_proofs": len(last_proofs),
        "turns_tail": len(last_exchange.turns) if last_exchange else 0,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sebeni Chronicle Vol II Mali OSINT & Math Proofs CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sw = sub.add_parser("sweep", help="Gather live Mali OSINT, run math proofs & predictions, export JSON")
    sw.add_argument("--out-dir", default="site/data")
    sw.add_argument("--max-ucdp-rows", type=int, default=1200)
    sw.add_argument("--no-llm", action="store_true", help="Skip external LLM adjudication/synthesis calls")
    sw.add_argument("--from-incidents", default=None, help="Recompute proofs & exchanges from existing VerifiedIncident JSON")
    sw.add_argument("--push-hf", action="store_true", help="Upload snapshots to Hugging Face Hub dataset")
    sw.add_argument("--push-firestore", action="store_true", help="Publish live states to Cloud Firestore")

    cont = sub.add_parser("run-continuous", help="Run ongoing continuous proof exploration & debate loop")
    cont.add_argument("--out-dir", default="site/data")
    cont.add_argument("--max-ucdp-rows", type=int, default=1200)
    cont.add_argument("--no-llm", action="store_true")
    cont.add_argument("--max-cycles", type=int, default=0, help="0 = run up to max-hours continuously")
    cont.add_argument("--max-hours", type=float, default=23.2)
    cont.add_argument("--cycle-sleep", type=float, default=45.0)
    cont.add_argument("--osint-every", type=int, default=6)
    cont.add_argument("--checkpoint-every", type=int, default=6)
    cont.add_argument("--push-hf", action="store_true")
    cont.add_argument("--push-firestore", action="store_true")

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
                from_incidents=args.from_incidents,
            )
        )
        return 0

    if args.command == "run-continuous":
        asyncio.run(
            run_continuous_loop(
                out_dir=args.out_dir,
                max_ucdp_rows=args.max_ucdp_rows,
                use_llm=not args.no_llm,
                push_hf=args.push_hf,
                push_firestore=args.push_firestore,
                max_cycles=args.max_cycles,
                max_hours=args.max_hours,
                cycle_sleep=args.cycle_sleep,
                osint_every=args.osint_every,
                checkpoint_every=args.checkpoint_every,
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
