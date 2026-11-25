"""Resumable Edition No. 2 swarm orchestrator and publisher."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from bench import LANG_NAMES
from bench.adapters.base import ModelAdapter
from bench.adapters.registry import get_model_spec
from bench.env import load_env
from bench.runner import create_adapter
from swarm_lab import CLUSTER_SLUGS, COORDINATOR_SLUG, DEFAULT_LANGS, normalize_lang_code
from swarm_lab.coordinator import CoordinatorAgent, build_random_native_command
from swarm_lab.resources import diagnose_test_text, load_prompt_native, load_test_texts
from swarm_lab.schema import ClusterResult, Edition2Payload, LanguageSummary
from swarm_lab.topology import EventLogger, run_cluster_session

DEFAULT_ED2_RUNS_DIR = Path(__file__).resolve().parent.parent / "runs" / "edition2"
DEFAULT_SITE_DATA_DIR = Path(__file__).resolve().parent.parent / "site" / "data"


async def run_language_swarm(
    lang: str,
    *,
    cluster_slugs: Sequence[str] = CLUSTER_SLUGS,
    runs_dir: Path = DEFAULT_ED2_RUNS_DIR,
    resume: bool = True,
    seed: int = 2026,
    coordinator: Optional[CoordinatorAgent] = None,
    adapter_factory=create_adapter,
) -> LanguageSummary:
    """Run all self-organizing clusters on one test.json language text."""
    canon = normalize_lang_code(lang)
    test_texts = load_test_texts()
    prompt_native = load_prompt_native()
    if canon not in test_texts:
        raise KeyError(f"Language '{canon}' not found in test.json")
    test_text = test_texts[canon]
    native_labels = prompt_native.get(
        canon,
        {"Text": "Text", "Title": "Title", "Content": "Content", "Context": "Context", "Grammar": "Grammar"},
    )

    lang_dir = runs_dir / canon
    lang_dir.mkdir(parents=True, exist_ok=True)
    live_path = runs_dir / "live" / f"{canon}.jsonl"
    logger = EventLogger(live_path, reset=not resume)

    diag = diagnose_test_text(canon, test_text, runtime_dir=runs_dir / "_rt_base" / canon)
    native_cmd = build_random_native_command(canon, native_labels, seed=seed)

    own_coord = coordinator is None
    coord = coordinator or CoordinatorAgent()
    try:
        lang_summary_path = lang_dir / "language_summary.json"
        if resume and lang_summary_path.is_file():
            existing_ls = LanguageSummary.model_validate_json(lang_summary_path.read_text(encoding="utf-8"))
            existing_slugs = {c.cluster_slug for c in existing_ls.clusters}
            if set(cluster_slugs).issubset(existing_slugs):
                print(f"[edition2] {canon}: already complete ({len(existing_ls.clusters)} clusters), skipping.", flush=True)
                return existing_ls

        brief = await coord.brief_language(canon, diag, native_cmd, logger)
        print(f"[edition2] {canon} brief ready | baseline={diag['baseline_parse_pct']}% | directive={native_cmd['archetype']}", flush=True)

        results: List[ClusterResult] = []
        for slug in cluster_slugs:
            cdir = lang_dir / slug
            cdir.mkdir(parents=True, exist_ok=True)
            res_path = cdir / "result.json"
            if resume and res_path.is_file():
                try:
                    cached = ClusterResult.model_validate_json(res_path.read_text(encoding="utf-8"))
                    results.append(cached)
                    print(f"[edition2] {canon}/{slug}: cached ({cached.plan.n_agents}x {cached.plan.topology})", flush=True)
                    continue
                except Exception:
                    pass

            spec = get_model_spec(slug)
            adapter: ModelAdapter = adapter_factory(spec)
            try:
                cr = await run_cluster_session(
                    cluster_slug=slug,
                    model_name=spec.display_name,
                    adapter=adapter,
                    lang=canon,
                    test_text=test_text,
                    diag_packet=diag,
                    coordinator_brief=brief,
                    native_cmd=native_cmd,
                    event_logger=logger,
                    cluster_run_dir=cdir,
                )
            finally:
                await adapter.close()

            res_path.write_text(cr.model_dump_json(indent=2) + "\n", encoding="utf-8")
            results.append(cr)
            print(
                f"[edition2] {canon}/{slug}: {cr.plan.n_agents}x {cr.plan.topology} | "
                f"agree={cr.agreement_pct}% ({cr.rounds_used}R) | shadow={cr.shadow.test_delta_pct:+.1f}% | "
                f"native={cr.native_gen.shadow_parse_pct:.1f}%",
                flush=True,
            )

        audited, verdict = await coord.audit_language_clusters(canon, results, logger)
        for cr in audited:
            (lang_dir / cr.cluster_slug / "result.json").write_text(
                cr.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )

        ls = LanguageSummary(
            lang=canon,
            lang_name=LANG_NAMES.get(canon, canon),
            test_text=test_text,
            native_labels=native_labels,
            native_command=native_cmd["command"],
            coordinator_brief=brief,
            coordinator_verdict=verdict,
            target_words=diag["target_words"],
            clusters=audited,
        )
        lang_summary_path.write_text(ls.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return ls
    finally:
        if own_coord:
            await coord.close()


async def run_edition2(
    *,
    langs: Sequence[str] = DEFAULT_LANGS,
    cluster_slugs: Sequence[str] = CLUSTER_SLUGS,
    runs_dir: Path = DEFAULT_ED2_RUNS_DIR,
    resume: bool = True,
    seed: int = 2026,
    coordinator: Optional[CoordinatorAgent] = None,
    adapter_factory=create_adapter,
) -> List[LanguageSummary]:
    """Run languages sequentially one test text at a time."""
    load_env()
    own_coord = coordinator is None
    coord = coordinator or CoordinatorAgent()
    out: List[LanguageSummary] = []
    try:
        for lang in langs:
            ls = await run_language_swarm(
                lang,
                cluster_slugs=cluster_slugs,
                runs_dir=runs_dir,
                resume=resume,
                seed=seed,
                coordinator=coord,
                adapter_factory=adapter_factory,
            )
            out.append(ls)
        return out
    finally:
        if own_coord:
            await coord.close()


def _aggregate_leaderboard(languages: Dict[str, LanguageSummary]) -> List[Dict[str, Any]]:
    by_cluster: Dict[str, List[ClusterResult]] = {}
    for ls in languages.values():
        for cr in ls.clusters:
            by_cluster.setdefault(cr.cluster_slug, []).append(cr)

    rows: List[Dict[str, Any]] = []
    for slug, items in by_cluster.items():
        n = len(items)
        if not n:
            continue
        topos: Dict[str, int] = {}
        for cr in items:
            topos[cr.plan.topology] = topos.get(cr.plan.topology, 0) + 1
        dominant_topo = max(topos.items(), key=lambda kv: kv[1])[0]
        rows.append(
            {
                "cluster_slug": slug,
                "model_name": items[0].model_name,
                "langs_completed": n,
                "avg_agents": round(sum(c.plan.n_agents for c in items) / n, 2),
                "dominant_topology": dominant_topo,
                "topology_counts": topos,
                "unanimous_rate_pct": round(100.0 * sum(1 for c in items if c.unanimous) / n, 1),
                "avg_agreement_pct": round(sum(c.agreement_pct for c in items) / n, 1),
                "avg_rounds": round(sum(c.rounds_used for c in items) / n, 2),
                "total_dict_added": sum(c.shadow.dict_entries_added for c in items),
                "compile_ok_rate_pct": round(100.0 * sum(1 for c in items if c.shadow.dict_compile_ok) / n, 1),
                "avg_test_delta_pct": round(sum(c.shadow.test_delta_pct for c in items) / n, 2),
                "avg_devmini_delta_pct": round(sum(c.shadow.devmini_delta_pct for c in items) / n, 2),
                "avg_native_parse_pct": round(sum(c.native_gen.shadow_parse_pct for c in items) / n, 1),
                "avg_header_compliance_pct": round(sum(c.native_gen.header_compliance for c in items) / n, 1),
                "avg_coordinator_score": round(sum(c.coordinator_score for c in items) / n, 1),
            }
        )
    rows.sort(key=lambda r: (-r["avg_coordinator_score"], -r["avg_native_parse_pct"]))
    return rows


def publish_edition2(
    *,
    runs_dir: Path = DEFAULT_ED2_RUNS_DIR,
    out_dir: Path = DEFAULT_SITE_DATA_DIR,
) -> Edition2Payload:
    """Publish validated site/data/edition2.json and compact site/data/edition2_events.json."""
    languages: Dict[str, LanguageSummary] = {}
    events_by_lang: Dict[str, List[Dict[str, Any]]] = {}

    if runs_dir.is_dir():
        for lang_dir in sorted(runs_dir.iterdir()):
            if not lang_dir.is_dir() or lang_dir.name in {"live", "_rt_base"}:
                continue
            ls_path = lang_dir / "language_summary.json"
            if not ls_path.is_file():
                continue
            ls = LanguageSummary.model_validate_json(ls_path.read_text(encoding="utf-8"))
            languages[ls.lang] = ls

            live_file = runs_dir / "live" / f"{ls.lang}.jsonl"
            evs: List[Dict[str, Any]] = []
            if live_file.is_file():
                for raw_line in live_file.read_text(encoding="utf-8").splitlines():
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        evs.append(json.loads(raw_line))
                    except json.JSONDecodeError:
                        continue
            events_by_lang[ls.lang] = evs

    ordered_langs = [l for l in DEFAULT_LANGS if l in languages] + [
        l for l in sorted(languages) if l not in DEFAULT_LANGS
    ]
    payload = Edition2Payload(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        coordinator_model=f"nvidia/{COORDINATOR_SLUG} (NVIDIA Build)",
        generation_model="Gemini 4 Neon-High-Orchestrator (Antigravity)",
        langs=ordered_langs,
        languages={l: languages[l] for l in ordered_langs},
        leaderboard=_aggregate_leaderboard(languages),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "edition2.json").write_text(payload.model_dump_json(indent=2) + "\n", encoding="utf-8")

    events_payload = {
        "generated_at": payload.generated_at,
        "langs": ordered_langs,
        "events_by_lang": {l: events_by_lang.get(l, []) for l in ordered_langs},
    }
    (out_dir / "edition2_events.json").write_text(
        json.dumps(events_payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return payload
