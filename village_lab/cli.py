"""CLI entrypoints for Edition No. 4 Living Lexicon Village."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List

from bench.env import load_env
from swarm_lab import ALL_LANGS, normalize_lang_code
from village_lab import DEFAULT_LAUNCH_LANG, ED4_STATE_REPO
from village_lab.firestore_sync import FirestoreLivePublisher
from village_lab.hf_sync import publish_ed4_site_data, push_ed4_state_to_hf, restore_ed4_state_from_hf
from village_lab.schema import VillageEvent
from village_lab.world_engine import LanguageVillageEngine

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ED4_RUNS_DIR = REPO_ROOT / "runs" / "edition4"
DEFAULT_SITE_DATA_DIR = REPO_ROOT / "site" / "data"

_STATE_SYNC_EVENT_KINDS = frozenset({
    "village_init",
    "agent_spawn",
    "critic_review",
    "lexicon_commit",
    "gram_commit",
    "council_reject",
    "epoch_checkpoint",
})


def _parse_job_langs(raw: str) -> List[str]:
    txt = (raw or "").strip().lower()
    if txt in {"", "all", "*"}:
        return list(ALL_LANGS)
    seen = set()
    out: List[str] = []
    for part in txt.split(","):
        p = part.strip()
        if not p:
            continue
        c = normalize_lang_code(p)
        if c in ALL_LANGS and c not in seen:
            seen.add(c)
            out.append(c)
    return out or [DEFAULT_LAUNCH_LANG]


def _backfill_firestore_events(
    fs_pub: FirestoreLivePublisher,
    eng: LanguageVillageEngine,
) -> int:
    if not eng.events_path.is_file():
        return 0
    remote_max = fs_pub.get_highest_event_seq(eng.lang)
    pushed = 0
    for ln in eng.events_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            ev = VillageEvent.model_validate(json.loads(ln))
        except Exception:
            continue
        if ev.seq > remote_max:
            if fs_pub.publish_event_sync(ev):
                pushed += 1
    return pushed


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

    p_job = sub.add_parser("run-job", help="Unattended Cloud Run Job runner with live Firestore sync + HF checkpoints")
    p_job.add_argument("--langs", default=os.environ.get("ED4_JOB_LANGS", DEFAULT_LAUNCH_LANG))
    p_job.add_argument("--turns-per-lang", type=int, default=int(os.environ.get("ED4_JOB_TURNS", "40")))
    p_job.add_argument("--checkpoint-every", type=int, default=int(os.environ.get("ED4_CHECKPOINT_EVERY", "5")))
    p_job.add_argument("--runs-dir", type=Path, default=DEFAULT_ED4_RUNS_DIR)
    p_job.add_argument("--out-dir", type=Path, default=DEFAULT_SITE_DATA_DIR)
    p_job.add_argument("--push-hf", action="store_true")
    p_job.add_argument("--firestore-db", default=os.environ.get("ED4_FIRESTORE_DB", ""))
    p_job.add_argument("--firestore-project", default=os.environ.get("ED4_FIRESTORE_PROJECT") or os.environ.get("GCP_VERTEX_PROJECT") or "mlgramv1")

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
    elif args.cmd == "run-job":
        tok = os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN") or ""
        state_repo = os.environ.get("ED4_STATE_REPO", ED4_STATE_REPO)
        if state_repo:
            restored = restore_ed4_state_from_hf(state_repo, args.runs_dir, token=tok)
            print(f"[village-job] Restored {len(restored)} languages from HF: {restored}", flush=True)

        fs_pub = (
            FirestoreLivePublisher(project_id=args.firestore_project, database_id=args.firestore_db.strip())
            if (args.firestore_db or "").strip()
            else FirestoreLivePublisher.from_env()
        )
        exec_id = os.environ.get("CLOUD_RUN_EXECUTION", "local-job")
        target_langs = _parse_job_langs(args.langs)
        ckpt_step = max(1, int(args.checkpoint_every))
        print(
            f"[village-job] Starting unattended run-job: langs={target_langs}, turns_per_lang={args.turns_per_lang}, "
            f"checkpoint_every={ckpt_step}, firestore_db={fs_pub.database_id if fs_pub else 'disabled'}",
            flush=True,
        )

        try:
            for lang in target_langs:
                eng_holder: List[LanguageVillageEngine] = []

                def _on_ev(ev: VillageEvent, _holder: List[LanguageVillageEngine] = eng_holder) -> None:
                    if fs_pub is None:
                        return
                    fs_pub.publish_event_sync(ev)
                    if ev.kind in _STATE_SYNC_EVENT_KINDS and _holder:
                        fs_pub.publish_state_sync(_holder[0].state, running=True, job_execution=exec_id)

                eng = LanguageVillageEngine(lang, runs_dir=args.runs_dir, event_callback=_on_ev)
                eng_holder.append(eng)

                try:
                    if fs_pub is not None:
                        pushed_bf = _backfill_firestore_events(fs_pub, eng)
                        fs_pub.publish_state_sync(eng.state, running=True, job_execution=exec_id)
                        if pushed_bf:
                            print(f"[village-job] Backfilled {pushed_bf} prior {lang} events to Firestore", flush=True)

                    remaining = max(0, int(args.turns_per_lang))
                    while remaining > 0:
                        chunk = min(ckpt_step, remaining)

                        def _on_turn(_done: int, _eng: LanguageVillageEngine = eng) -> None:
                            if fs_pub is not None:
                                fs_pub.publish_state_sync(_eng.state, running=True, job_execution=exec_id)

                        st = asyncio.run(eng.run_epoch(max_turns=chunk, on_turn_complete=_on_turn))
                        publish_ed4_site_data(args.runs_dir, args.out_dir)
                        if args.push_hf and tok:
                            try:
                                push_ed4_state_to_hf(state_repo, args.runs_dir, st.lang, args.out_dir, tok)
                            except Exception as exc:
                                print(f"[village-job] HF checkpoint warning ({st.lang}): {exc}", file=sys.stderr, flush=True)
                        remaining -= chunk
                        print(
                            f"[village-job] Checkpoint ({st.lang}): turns={st.total_turns}, +\\lx={st.added_lx_count}, "
                            f"+\\va={st.added_va_count}, +.gram={st.added_gram_rules_count}, remaining={remaining}",
                            flush=True,
                        )
                finally:
                    if fs_pub is not None:
                        fs_pub.publish_state_sync(eng.state, running=False, job_execution=exec_id)
        finally:
            if fs_pub is not None:
                fs_pub.close()
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
