"""Orchestrator & static bundle publisher for Edition No. 3."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from proof_lab import DATASET_LANGS, LEAN_THEOREMS, NIM_MODELS
from proof_lab.corpus import analyze_language_corpus
from proof_lab.deductions import run_nim_deduction_panel
from proof_lab.schema import Edition3Summary, LanguageProofResult, ProofEvent

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ED3_RUNS_DIR = REPO_ROOT / "runs" / "edition3"
DEFAULT_SITE_DATA_DIR = REPO_ROOT / "site" / "data"


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _write_live_event(live_path: Path, seq: int, lang: str, kind: str, model_slug: str, model_name: str, label: str, summary: str, reasoning_excerpt: str = "", metrics: Optional[Dict[str, Any]] = None) -> None:
    live_path.parent.mkdir(parents=True, exist_ok=True)
    ev = ProofEvent(
        seq=seq,
        ts=_utc_now_str(),
        lang=lang,
        kind=kind,
        model_slug=model_slug,
        model_name=model_name,
        label=label,
        summary=summary,
        reasoning_excerpt=reasoning_excerpt,
        metrics=metrics or {},
    )
    with live_path.open("a", encoding="utf-8") as f:
        f.write(ev.model_dump_json() + "\n")


async def run_language_proof(
    lang: str,
    *,
    runs_dir: Path = DEFAULT_ED3_RUNS_DIR,
    resume: bool = True,
    max_rows: int = 300,
    skip_llm: bool = False,
) -> LanguageProofResult:
    lang_dir = runs_dir / lang
    res_path = lang_dir / "result.json"
    if resume and res_path.is_file():
        return LanguageProofResult.model_validate_json(res_path.read_text(encoding="utf-8"))

    lang_dir.mkdir(parents=True, exist_ok=True)
    live_path = runs_dir / "live" / f"{lang}.jsonl"
    live_path.parent.mkdir(parents=True, exist_ok=True)
    live_path.write_text("", encoding="utf-8")
    seq_state = {"seq": 0}

    def emit(kind: str, m_slug: str, m_name: str, label: str, summary: str, reasoning: str = "", metrics: Optional[Dict[str, Any]] = None) -> None:
        seq_state["seq"] += 1
        _write_live_event(live_path, seq_state["seq"], lang, kind, m_slug, m_name, label, summary, reasoning, metrics)

    corpus = analyze_language_corpus(lang, max_rows=max_rows)
    emit(
        "corpus_stress_complete",
        "dabax-oracle",
        "DabaX(strict_parse=True)",
        f"{corpus.n_samples} rows · d(MER)={corpus.cohens_d_mer_corrupt}",
        (
            f"Evaluated {corpus.n_samples} rows ({corpus.n_tokens_total} tokens): "
            f"clean strict parse={corpus.clean_strict_parse_pct}%, "
            f"OOD gap={corpus.ood_strict_gap_pct} pp, "
            f"shift LCS recovery=+{corpus.shift_mcs_recovery_pct} pp, "
            f"MER>1 saturation prevented={corpus.mer_saturation_prevented_pct}%."
        ),
        metrics={
            "clean_strict_parse_pct": corpus.clean_strict_parse_pct,
            "cohens_d_mer_corrupt": corpus.cohens_d_mer_corrupt,
            "cohens_d_mcs_shift": corpus.cohens_d_mcs_shift,
            "cohens_d_uwec_ood": corpus.cohens_d_uwec_ood,
            "ood_strict_gap_pct": corpus.ood_strict_gap_pct,
        },
    )

    models = []
    if not skip_llm:
        models = await run_nim_deduction_panel(corpus, models=NIM_MODELS, on_event=emit)

    scored = [m for m in models if m.effectiveness_score > 0]
    consensus_eff = round(sum(m.effectiveness_score for m in scored) / len(scored), 1) if scored else 0.0
    verdict_counts = Counter(m.verdict for m in models if m.verdict and m.verdict != "ERROR")
    consensus_verdict = verdict_counts.most_common(1)[0][0] if verdict_counts else "DETERMINISTIC_ONLY"
    tot_p = sum(m.oracle_total_count for m in models)
    tot_v = sum(m.oracle_verified_count for m in models)
    oracle_rate = round(100.0 * tot_v / tot_p, 1) if tot_p else 0.0

    emit(
        "language_synthesis",
        "sebeni-proofs",
        "SebeniProofs Synthesis",
        f"{consensus_verdict} · {consensus_eff:.1f}/100",
        f"Consensus {consensus_verdict} ({consensus_eff:.1f}/100, oracle {tot_v}/{tot_p} verified) across {len(models)} NVIDIA Build models.",
        metrics={"consensus_effectiveness": consensus_eff, "oracle_pass_rate_pct": oracle_rate},
    )
    emit(
        "run_complete",
        "system",
        "Edition 3 Runner",
        f"{lang.upper()} COMPLETE",
        f"Completed {corpus.n_samples}-sentence stress suite and {len(models)} live model deductions for {lang}.",
    )

    out = LanguageProofResult(
        lang=lang,
        generated_at=_utc_now_str(),
        corpus=corpus,
        models=models,
        consensus_effectiveness=consensus_eff,
        consensus_verdict=consensus_verdict,
        oracle_pass_rate_pct=oracle_rate,
        n_events=seq_state["seq"],
    )
    res_path.write_text(out.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return out


async def run_edition3(
    *,
    langs: Sequence[str] = DATASET_LANGS,
    runs_dir: Path = DEFAULT_ED3_RUNS_DIR,
    resume: bool = True,
    max_rows: int = 300,
) -> List[LanguageProofResult]:
    out: List[LanguageProofResult] = []
    for lang in langs:
        res = await run_language_proof(lang, runs_dir=runs_dir, resume=resume, max_rows=max_rows)
        out.append(res)
    return out


def publish_edition3(
    *,
    runs_dir: Path = DEFAULT_ED3_RUNS_DIR,
    out_dir: Path = DEFAULT_SITE_DATA_DIR,
) -> Edition3Summary:
    out_dir.mkdir(parents=True, exist_ok=True)
    languages: Dict[str, LanguageProofResult] = {}
    ordered_langs: List[str] = []

    for lang in DATASET_LANGS:
        fp = runs_dir / lang / "result.json"
        if fp.is_file():
            languages[lang] = LanguageProofResult.model_validate_json(fp.read_text(encoding="utf-8"))
            ordered_langs.append(lang)

    # Aggregate 4-model NVIDIA Build standings
    by_model: Dict[str, Dict[str, Any]] = {}
    for lang in ordered_langs:
        lres = languages[lang]
        for m in lres.models:
            agg = by_model.setdefault(
                m.model_slug,
                {
                    "model_slug": m.model_slug,
                    "model_name": m.model_name,
                    "model_id": m.model_id,
                    "params_label": m.params_label,
                    "n_langs": 0,
                    "eff_scores": [],
                    "lat_r1": [],
                    "lat_r2": [],
                    "verified": 0,
                    "total_probes": 0,
                    "verdicts": Counter(),
                    "errors": 0,
                },
            )
            agg["n_langs"] += 1
            if m.effectiveness_score > 0:
                agg["eff_scores"].append(m.effectiveness_score)
            agg["lat_r1"].append(m.round1_latency_s)
            agg["lat_r2"].append(m.round2_latency_s)
            agg["verified"] += m.oracle_verified_count
            agg["total_probes"] += m.oracle_total_count
            if m.verdict:
                agg["verdicts"][m.verdict] += 1
            if m.round1_error or m.round2_error:
                agg["errors"] += 1

    leaderboard: List[Dict[str, Any]] = []
    for agg in by_model.values():
        n_l = max(1, agg["n_langs"])
        effs = agg["eff_scores"]
        top_v = agg["verdicts"].most_common(1)[0][0] if agg["verdicts"] else "PENDING"
        leaderboard.append(
            {
                "model_slug": agg["model_slug"],
                "model_name": agg["model_name"],
                "model_id": agg["model_id"],
                "params_label": agg["params_label"],
                "langs_completed": agg["n_langs"],
                "mean_effectiveness_score": round(sum(effs) / len(effs), 1) if effs else 0.0,
                "oracle_verified": agg["verified"],
                "oracle_total": agg["total_probes"],
                "oracle_pass_pct": round(100.0 * agg["verified"] / max(1, agg["total_probes"]), 1),
                "mean_r1_latency_s": round(sum(agg["lat_r1"]) / n_l, 2),
                "mean_r2_latency_s": round(sum(agg["lat_r2"]) / n_l, 2),
                "dominant_verdict": top_v,
                "error_count": agg["errors"],
            }
        )
    leaderboard.sort(key=lambda r: (-r["mean_effectiveness_score"], -r["oracle_pass_pct"], r["mean_r1_latency_s"]))

    # Global empirical proof summary across all completed corpora
    corps = [languages[l].corpus for l in ordered_langs]
    in_domain = [c for c in corps if not c.is_ood]
    ood_corps = [c for c in corps if c.is_ood]
    n_c = max(1, len(corps))
    n_in = max(1, len(in_domain))

    global_metrics = {
        "n_langs": len(corps),
        "n_in_domain_langs": len(in_domain),
        "n_ood_langs": len(ood_corps),
        "total_sentences": sum(c.n_samples for c in corps),
        "total_tokens": sum(c.n_tokens_total for c in corps),
        "mean_clean_strict_parse_pct": round(sum(c.clean_strict_parse_pct for c in in_domain) / n_in, 2),
        "ood_bbo_strict_parse_pct": round(sum(c.clean_strict_parse_pct for c in ood_corps) / max(1, len(ood_corps)), 2) if ood_corps else 0.0,
        "mean_cohens_d_mer_corrupt": round(sum(c.cohens_d_mer_corrupt for c in corps) / n_c, 2),
        "mean_cohens_d_mcs_shift": round(sum(c.cohens_d_mcs_shift for c in corps) / n_c, 2),
        "mean_cohens_d_uwec_ood": round(sum(c.cohens_d_uwec_ood for c in corps) / n_c, 2),
        "mean_ood_strict_gap_pct": round(sum(c.ood_strict_gap_pct for c in in_domain) / n_in, 2),
        "mean_shift_mcs_recovery_pct": round(sum(c.shift_mcs_recovery_pct for c in corps) / n_c, 2),
        "mean_mer_saturation_prevented_pct": round(sum(c.mer_saturation_prevented_pct for c in corps) / n_c, 2),
        "mean_truncation_denom_bias_pct": round(sum(c.truncation_denom_bias_pct for c in corps) / n_c, 2),
    }

    # Events stream payload for interactive live/replay inspection
    events_by_lang: Dict[str, List[Dict[str, Any]]] = {}
    total_events = 0
    for lang in ordered_langs:
        ev_file = runs_dir / "live" / f"{lang}.jsonl"
        ev_rows: List[Dict[str, Any]] = []
        if ev_file.is_file():
            for ln in ev_file.read_text(encoding="utf-8").splitlines():
                s = ln.strip()
                if s:
                    ev_rows.append(json.loads(s))
        events_by_lang[lang] = ev_rows
        total_events += len(ev_rows)

    summary = Edition3Summary(
        generated_at=_utc_now_str(),
        langs=ordered_langs,
        total_sentences=sum(c.n_samples for c in corps),
        total_events=total_events,
        muse_eol_note=(
            "Requested model meta/muse-glimmer-30b reached NVIDIA Build end-of-life on 2026-11-24T09:00:00Z "
            "(HTTP 410 Gone); substituted with live models minimaxai/minimax-m3.1-flash (from models.list) "
            "and nvidia/nemotron-3.5-super-vl-120b-a12b alongside mistralai/mistral-large-4-1t-a52b and "
            "nvidia/nemotron-3-ultra-550b-a55b."
        ),
        lean_theorems=LEAN_THEOREMS,
        model_leaderboard=leaderboard,
        global_metrics=global_metrics,
        languages=languages,
    )

    (out_dir / "edition3.json").write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (out_dir / "edition3_events.json").write_text(
        json.dumps(
            {
                "generated_at": summary.generated_at,
                "langs": ordered_langs,
                "events_by_lang": events_by_lang,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary
