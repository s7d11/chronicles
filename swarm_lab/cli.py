"""CLI entrypoints for Edition No. 2 self-organizing swarms and Three.js live theater."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from swarm_lab import ALL_LANGS, CLUSTER_SLUGS, DEFAULT_LANGS
from swarm_lab.runner import (
    DEFAULT_ED2_RUNS_DIR,
    DEFAULT_SITE_DATA_DIR,
    publish_edition2,
    run_edition2,
)
from swarm_lab.viz_server import serve_viz


def main() -> None:
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

    p_viz = sub.add_parser("viz", help="Serve Three.js Swarm Theater with real-time SSE stream")
    p_viz.add_argument("--host", default="127.0.0.1")
    p_viz.add_argument("--port", type=int, default=8765)

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
        serve_viz(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
