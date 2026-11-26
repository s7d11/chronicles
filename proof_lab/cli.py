"""CLI entrypoints for Edition No. 3 SebeniProofs 4,200-sentence stress tests & HF sync."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from bench.env import load_env
from proof_lab import DATASET_LANGS
from proof_lab.hf_sync import restore_ed3_state_from_hf
from proof_lab.runner import (
    DEFAULT_ED3_RUNS_DIR,
    DEFAULT_SITE_DATA_DIR,
    publish_edition3,
    run_edition3,
)


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(prog="python -m proof_lab.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run 4,200-sentence stress tests + 4 NVIDIA models")
    p_run.add_argument("--langs", default=",".join(DATASET_LANGS), help="Comma-separated codes or 'all'")
    p_run.add_argument("--runs-dir", type=Path, default=DEFAULT_ED3_RUNS_DIR)
    p_run.add_argument("--max-rows", type=int, default=300)
    p_run.add_argument("--resume", action="store_true")

    p_pub = sub.add_parser("publish", help="Export site/data/edition3.json and edition3_events.json")
    p_pub.add_argument("--runs-dir", type=Path, default=DEFAULT_ED3_RUNS_DIR)
    p_pub.add_argument("--out-dir", type=Path, default=DEFAULT_SITE_DATA_DIR)

    p_sync = sub.add_parser("sync-from-hf", help="Restore HF Dataset state and publish site/data")
    p_sync.add_argument("--state-repo", default=os.environ.get("ED3_STATE_REPO", "s7d11/chronicles-edition3-state"))
    p_sync.add_argument("--runs-dir", type=Path, default=DEFAULT_ED3_RUNS_DIR)
    p_sync.add_argument("--out-dir", type=Path, default=DEFAULT_SITE_DATA_DIR)

    args = parser.parse_args()
    if args.cmd == "run":
        langs = list(DATASET_LANGS) if args.langs.strip().lower() == "all" else [
            x.strip().lower() for x in args.langs.split(",") if x.strip()
        ]
        results = asyncio.run(
            run_edition3(langs=langs, runs_dir=args.runs_dir, resume=args.resume, max_rows=args.max_rows)
        )
        print(f"Completed Edition No. 3 across {len(results)} languages.")
    elif args.cmd == "publish":
        pub = publish_edition3(runs_dir=args.runs_dir, out_dir=args.out_dir)
        print(f"Published Edition No. 3 ({len(pub.langs)} langs, {pub.total_sentences} sentences) to {args.out_dir}")
    elif args.cmd == "sync-from-hf":
        tok = os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN")
        restored = restore_ed3_state_from_hf(args.state_repo, args.runs_dir, token=tok)
        pub = publish_edition3(runs_dir=args.runs_dir, out_dir=args.out_dir)
        print(f"Restored {len(restored)} languages ({', '.join(restored)}); published {pub.total_sentences} sentences to {args.out_dir}.")


if __name__ == "__main__":
    main()
