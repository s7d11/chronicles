"""CLI entrypoint for building MALI13 splits, running models, scoring, and publishing."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import List, Optional

from bench.adapters.registry import list_model_specs
from bench.dataset import DEFAULT_OUT_DIR, build_dev_mini, verify_manifest
from bench.env import load_env
from bench.publish import DEFAULT_SITE_DATA_DIR, publish_site_data, score_all_runs
from bench.runner import DEFAULT_RUNS_DIR, run_many_models


def main(argv: Optional[List[str]] = None) -> int:
    load_env()
    parser = argparse.ArgumentParser(prog="bench.cli", description="MALI13 Benchmark Harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build or verify frozen dev-mini splits")
    p_build.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p_build.add_argument("--seed", type=int, default=2026)
    p_build.add_argument("--verify-only", action="store_true")

    p_run = sub.add_parser("run", help="Run models across MALI13 dev-mini")
    p_run.add_argument("--models", default="all", help="all | all-api | provider | track | slug1,slug2")
    p_run.add_argument("--tasks", default="all", help="all or comma-separated morph,gen,lid,cloze,efficiency")
    p_run.add_argument("--data-dir", type=Path, default=DEFAULT_OUT_DIR)
    p_run.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    p_run.add_argument("--resume", action="store_true", default=True)
    p_run.add_argument("--no-resume", dest="resume", action="store_false")
    p_run.add_argument("--concurrency", type=int, default=4)
    p_run.add_argument("--per-lang", type=int, default=None, help="Stratified N samples per language per task")
    p_run.add_argument("--limit", type=int, default=None, help="Optional per-task sample cap for smoke tests")

    p_score = sub.add_parser("score", help="Offline re-scoring of cached runs")
    p_score.add_argument("--data-dir", type=Path, default=DEFAULT_OUT_DIR)
    p_score.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)

    p_pub = sub.add_parser("publish", help="Publish leaderboard.json & samples.json")
    p_pub.add_argument("--data-dir", type=Path, default=DEFAULT_OUT_DIR)
    p_pub.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    p_pub.add_argument("--out-dir", type=Path, default=DEFAULT_SITE_DATA_DIR)

    args = parser.parse_args(argv)

    if args.cmd == "build":
        if args.verify_only:
            m = verify_manifest(args.out_dir)
            print(f"Verified {m.tier}: {m.total_rows} rows across {len(m.langs)} langs (overlap={m.anti_leakage_overlap})")
        else:
            m = build_dev_mini(out_dir=args.out_dir, seed=args.seed)
            print(f"Built {m.tier}: {m.total_rows} rows across {len(m.langs)} langs (overlap={m.anti_leakage_overlap})")
        return 0

    if args.cmd == "run":
        specs = list_model_specs(args.models)
        tasks = (
            ("morph", "gen", "lid", "cloze", "efficiency")
            if args.tasks == "all"
            else tuple(t.strip() for t in args.tasks.split(",") if t.strip())
        )
        outcomes = asyncio.run(
            run_many_models(
                specs,
                data_dir=args.data_dir,
                runs_dir=args.runs_dir,
                tasks=tasks,
                resume=args.resume,
                concurrency_per_model=args.concurrency,
                per_lang=args.per_lang,
                limit_per_task=args.limit,
            )
        )
        for slug, err in outcomes:
            print(f"RUN {slug}: {'OK' if err is None else 'ERROR -> ' + err}")
        return 0

    if args.cmd == "score":
        items = score_all_runs(data_dir=args.data_dir, runs_dir=args.runs_dir)
        for item in items:
            c = item["card"]
            print(f"SCORED {c['id']}: Q={c['q']} Q_min3={c['q_min3']} M={c['morph']} G={c['gen']} L={c['lid']} C={c['cloze']} fert={c['fert']}")
        return 0

    if args.cmd == "publish":
        payload = publish_site_data(
            data_dir=args.data_dir,
            runs_dir=args.runs_dir,
            out_dir=args.out_dir,
        )
        print(f"Published {len(payload.models)} models ({len(payload.pending_models)} pending) to {args.out_dir}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
