"""CLI entrypoints for Edition No. 2 self-organizing swarms, HF sync, and Three.js theater."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from bench.env import load_env
from swarm_lab import ALL_LANGS, CLUSTER_SLUGS, DEFAULT_LANGS
from swarm_lab.hf_sync import pack_sebeni_runtime, restore_state_from_hf
from swarm_lab.runner import (
    DEFAULT_ED2_RUNS_DIR,
    DEFAULT_SITE_DATA_DIR,
    publish_edition2,
    run_edition2,
)
from swarm_lab.viz_server import serve_viz


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(prog="python -m swarm_lab.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run Edition No. 2 self-organizing clusters on test.json")
    p_run.add_argument("--langs", default=",".join(DEFAULT_LANGS), help="Comma-separated ISO codes or 'all'")
    p_run.add_argument("--clusters", default=",".join(CLUSTER_SLUGS), help="Comma-separated Adaption cluster slugs")
    p_run.add_argument("--runs-dir", type=Path, default=DEFAULT_ED2_RUNS_DIR)
    p_run.add_argument("--seed", type=int, default=2026)
    p_run.add_argument("--resume", action="store_true")

    p_pub = sub.add_parser("publish", help="Export site/data/edition2.json and edition2_events.json")
    p_pub.add_argument("--runs-dir", type=Path, default=DEFAULT_ED2_RUNS_DIR)
    p_pub.add_argument("--out-dir", type=Path, default=DEFAULT_SITE_DATA_DIR)

    p_viz = sub.add_parser("viz", help="Serve Three.js Swarm Theater & Proof Observatory with CORS SSE")
    p_viz.add_argument("--host", default="127.0.0.1")
    p_viz.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")))
    p_viz.add_argument("--runs-dir", type=Path, default=DEFAULT_ED2_RUNS_DIR)
    p_viz.add_argument("--ed3-runs-dir", type=Path, default=None)
    p_viz.add_argument("--auto-run-all-13", action="store_true")
    p_viz.add_argument("--auto-run-ed3", action="store_true")

    p_pack = sub.add_parser("pack-runtime", help="Pack minimal beni + daba + baselines into tar.gz")
    p_pack.add_argument("--out", type=Path, required=True)

    p_sync = sub.add_parser("sync-from-hf", help="Restore remote HF Dataset state and publish site/data")
    p_sync.add_argument("--state-repo", default=os.environ.get("SWARM_STATE_REPO", "s7d11/chronicles-edition2-state"))
    p_sync.add_argument("--runs-dir", type=Path, default=DEFAULT_ED2_RUNS_DIR)
    p_sync.add_argument("--out-dir", type=Path, default=DEFAULT_SITE_DATA_DIR)

    args = parser.parse_args()
    if args.cmd == "run":
        langs = list(ALL_LANGS) if args.langs.strip().lower() == "all" else [
            x.strip() for x in args.langs.split(",") if x.strip()
        ]
        clusters = [x.strip() for x in args.clusters.split(",") if x.strip()]
        summaries = asyncio.run(
            run_edition2(
                langs=langs,
                cluster_slugs=clusters,
                runs_dir=args.runs_dir,
                resume=args.resume,
                seed=args.seed,
            )
        )
        print(f"Completed Edition No. 2 run across {len(summaries)} languages.")
    elif args.cmd == "publish":
        payload = publish_edition2(runs_dir=args.runs_dir, out_dir=args.out_dir)
        print(f"Published Edition No. 2 ({len(payload.langs)} langs, {len(payload.leaderboard)} clusters) to {args.out_dir}")
    elif args.cmd == "viz":
        auto_run = bool(args.auto_run_all_13 or os.environ.get("AUTO_RUN_ALL_13") == "1")
        auto_ed3 = bool(args.auto_run_ed3 or os.environ.get("AUTO_RUN_ED3") == "1")
        serve_viz(
            host=args.host,
            port=args.port,
            runs_dir=args.runs_dir,
            auto_run_all_13=auto_run,
            ed3_runs_dir=args.ed3_runs_dir,
            auto_run_ed3=auto_ed3,
        )
    elif args.cmd == "pack-runtime":
        out = pack_sebeni_runtime(args.out)
        print(f"Packed minimal Sebeni runtime to {out} ({out.stat().st_size / (1024 * 1024):.2f} MB)")
    elif args.cmd == "sync-from-hf":
        tok = os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN")
        restored = restore_state_from_hf(args.state_repo, args.runs_dir, token=tok)
        payload = publish_edition2(runs_dir=args.runs_dir, out_dir=args.out_dir)
        print(f"Restored {len(restored)} languages ({', '.join(restored)}); published to {args.out_dir} ({len(payload.langs)} langs).")


if __name__ == "__main__":
    main()
