"""CLI entrypoints for Edition No. 4 Living Lexicon Village."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from bench.env import load_env
from village_lab import DEFAULT_LAUNCH_LANG, ED4_STATE_REPO
from village_lab.hf_sync import publish_ed4_site_data, push_ed4_state_to_hf, restore_ed4_state_from_hf
from village_lab.world_engine import LanguageVillageEngine

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ED4_RUNS_DIR = REPO_ROOT / "runs" / "edition4"
DEFAULT_SITE_DATA_DIR = REPO_ROOT / "site" / "data"


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(prog="python -m village_lab.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pub = sub.add_parser("publish", help="Export site/data/edition4.json and edition4_events.json")
    p_pub.add_argument("--runs-dir", type=Path, default=DEFAULT_ED4_RUNS_DIR)
    p_pub.add_argument("--out-dir", type=Path, default=DEFAULT_SITE_DATA_DIR)

    p_run = sub.add_parser("run-epoch", help="Run Edition 4 epoch (intended for Cloud Run execution)")
    p_run.add_argument("--lang", default=DEFAULT_LAUNCH_LANG)
    p_run.add_argument("--max-turns", type=int, default=3)
    p_run.add_argument("--runs-dir", type=Path, default=DEFAULT_ED4_RUNS_DIR)
    p_run.add_argument("--out-dir", type=Path, default=DEFAULT_SITE_DATA_DIR)
    p_run.add_argument("--push-hf", action="store_true")

    p_sync = sub.add_parser("sync-from-hf", help="Restore remote HF Dataset state and publish site/data")
    p_sync.add_argument("--state-repo", default=os.environ.get("ED4_STATE_REPO", ED4_STATE_REPO))
    p_sync.add_argument("--runs-dir", type=Path, default=DEFAULT_ED4_RUNS_DIR)
    p_sync.add_argument("--out-dir", type=Path, default=DEFAULT_SITE_DATA_DIR)

    args = parser.parse_args()
    if args.cmd == "publish":
        summary = publish_ed4_site_data(args.runs_dir, args.out_dir)
        print(
            f"[village-cli] Published Edition 4 ({len(summary['langs'])} langs, {summary['total_events']} events) -> {args.out_dir}",
            flush=True,
        )
    elif args.cmd == "run-epoch":
        eng = LanguageVillageEngine(args.lang, runs_dir=args.runs_dir)
        st = asyncio.run(eng.run_epoch(max_turns=args.max_turns))
        publish_ed4_site_data(args.runs_dir, args.out_dir)
        if args.push_hf:
            tok = os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN") or ""
            if tok:
                push_ed4_state_to_hf(os.environ.get("ED4_STATE_REPO", ED4_STATE_REPO), args.runs_dir, st.lang, args.out_dir, tok)
        print(
            f"[village-cli] Epoch complete ({st.lang}): turns={st.total_turns}, +\\lx={st.added_lx_count}, +\\va={st.added_va_count}, +.gram={st.added_gram_rules_count}",
            flush=True,
        )
    elif args.cmd == "sync-from-hf":
        tok = os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN") or ""
        restored = restore_ed4_state_from_hf(args.state_repo, args.runs_dir, token=tok)
        summary = publish_ed4_site_data(args.runs_dir, args.out_dir)
        print(
            f"[village-cli] Restored {len(restored)} languages ({', '.join(restored)}); published {summary['total_events']} events to {args.out_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()
