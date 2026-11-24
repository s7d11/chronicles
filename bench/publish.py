"""Score cached runs offline with DabaX(strict_parse=True) and publish site/data JSON."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from bench import LANGS, LANG_NAMES
from bench.adapters.registry import MODEL_REGISTRY
from bench.dataset import DEFAULT_OUT_DIR, load_split, verify_manifest
from bench.metrics import Mali13Scorer, strip_thinking
from bench.runner import DEFAULT_RUNS_DIR, load_cached_records
from bench.schema import (
    CompletionRecord,
    EfficiencyStats,
    LeaderboardPayload,
    ModelScorecard,
    ModelSpec,
)

DEFAULT_SITE_DATA_DIR = Path(__file__).resolve().parent.parent / "site" / "data"

SHATTER_WORDS = {
    "taq": {
        "word": "dăɣ-ăddunya-ta-dăɣ",
        "oracle": ["dăɣ-", "ăddunya", "-ta", "-dăɣ"],
    },
    "ful": {
        "word": "suɓaaɓe",
        "oracle": ["suɓaa", "-ɓe"],
    },
    "spp": {
        "word": "SARIYAƝUNGI",
        "oracle": ["sariyaɲun", "-gi"],
    },
    "mey": {
        "word": "ƏLWAAJIBAAT",
        "oracle": ["əl-", "wājib", "-aat"],
    },
    "dtm": {
        "word": "PÀŊÁBE",
        "oracle": ["pàŋá", "-be"],
    },
}


def score_all_runs(
    *,
    data_dir: Path = DEFAULT_OUT_DIR,
    runs_dir: Path = DEFAULT_RUNS_DIR,
) -> List[Dict[str, Any]]:
    """Score every cached model directory under runs_dir and write scorecard.json."""
    morph_samples = load_split(data_dir, "morph")
    gen_samples = load_split(data_dir, "gen")
    lid_samples = load_split(data_dir, "lid")
    cloze_samples = load_split(data_dir, "cloze")

    scorer = Mali13Scorer()
    scored_items: List[Dict[str, Any]] = []

    if not runs_dir.is_dir():
        return scored_items

    for slug_dir in sorted(runs_dir.iterdir()):
        if not slug_dir.is_dir():
            continue
        spec_path = slug_dir / "spec.json"
        if spec_path.is_file():
            spec = ModelSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
        elif slug_dir.name in MODEL_REGISTRY:
            spec = MODEL_REGISTRY[slug_dir.name]
        else:
            continue

        records: List[CompletionRecord] = []
        valid_tasks = 0
        for task in ("morph", "gen", "lid", "cloze"):
            recs = load_cached_records(slug_dir / f"{task}.completions.jsonl")
            if any(r.error is None for r in recs.values()):
                valid_tasks += 1
            records.extend(recs.values())
        eff_path = slug_dir / "efficiency.json"
        if not records or not eff_path.is_file() or valid_tasks < 3:
            continue

        eff = EfficiencyStats.model_validate_json(eff_path.read_text(encoding="utf-8"))

        card, per_lang_detail, lid_matrix = scorer.build_scorecard(
            spec,
            morph_samples,
            gen_samples,
            lid_samples,
            cloze_samples,
            records,
            eff,
        )

        telem_path = slug_dir / "telemetry.json"
        if telem_path.is_file():
            try:
                t = json.loads(telem_path.read_text(encoding="utf-8"))
                if t.get("vram_mb") is not None:
                    card.vram_mb = float(t["vram_mb"])
                    card.compute = f"{card.compute} · {card.vram_mb / 1024:.1f} GB VRAM"
            except Exception:
                pass

        shatter = {}
        shatter_path = slug_dir / "shatter.json"
        if shatter_path.is_file():
            try:
                shatter = json.loads(shatter_path.read_text(encoding="utf-8"))
            except Exception:
                shatter = {}

        bundle = {
            "card": card.model_dump(),
            "per_lang": per_lang_detail,
            "lid_matrix": lid_matrix,
            "shatter": shatter,
        }
        (slug_dir / "scorecard.json").write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        scored_items.append(bundle)

    return scored_items


def publish_site_data(
    *,
    data_dir: Path = DEFAULT_OUT_DIR,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    out_dir: Path = DEFAULT_SITE_DATA_DIR,
    registry: Optional[Dict[str, ModelSpec]] = None,
) -> LeaderboardPayload:
    """Aggregate scored runs into site/data/leaderboard.json and site/data/samples.json."""
    manifest = verify_manifest(data_dir)
    scored = score_all_runs(data_dir=data_dir, runs_dir=runs_dir)

    reg = MODEL_REGISTRY if registry is None else registry
    cards: List[ModelScorecard] = []
    incomplete_reasons: Dict[str, str] = {}

    per_lang: Dict[str, Dict[str, List[float]]] = {
        axis: {} for axis in ("q", "morph", "gen", "lid", "cloze", "fert")
    }
    lid_confusion: Dict[str, List[List[int]]] = {}
    shatter_by_model: Dict[str, Dict[str, List[str]]] = {}

    for item in scored:
        card = ModelScorecard.model_validate(item["card"])
        # Require at least 40 valid completions (or all 4 tasks present) and >=75% success rate
        # unless a smaller custom test registry is passed.
        slug_dir = runs_dir / card.slug
        n_tasks_with_ok = sum(
            1 for t in ("morph", "gen", "lid", "cloze")
            if any(not r.error for r in load_cached_records(slug_dir / f"{t}.completions.jsonl").values())
        )
        min_ok_threshold = 1 if registry is not None else 4
        if n_tasks_with_ok < min_ok_threshold or card.completion_rate < 0.60:
            incomplete_reasons[card.slug] = f"incomplete ({n_tasks_with_ok}/4 tasks, {int(card.completion_rate*100)}% ok)"
            continue

        cards.append(card)
        mid = card.id
        pl = item["per_lang"]
        for axis in ("q", "morph", "gen", "lid", "cloze", "fert"):
            per_lang[axis][mid] = [float(pl[axis].get(l, 0.0)) for l in LANGS]
        lid_confusion[mid] = item["lid_matrix"]
        if item.get("shatter"):
            shatter_by_model[mid] = item["shatter"]

    cards.sort(key=lambda c: (-c.q, -c.q_min3))

    shatter_samples: Dict[str, Any] = {}
    for lang, base in SHATTER_WORDS.items():
        models_cuts: Dict[str, List[str]] = {}
        for mid, mdict in shatter_by_model.items():
            if lang in mdict:
                models_cuts[mid] = mdict[lang]
        shatter_samples[lang] = {
            "word": base["word"],
            "oracle": base["oracle"],
            "tokenizers": models_cuts,
        }

    evaluated_slugs = {c.slug for c in cards}
    pending = [
        {
            "slug": s.slug,
            "id": s.display_name,
            "provider": s.provider + (f" ({incomplete_reasons[s.slug]})" if s.slug in incomplete_reasons else ""),
            "track": s.track,
        }
        for s in reg.values()
        if s.slug not in evaluated_slugs
    ]

    manifest_sha = manifest.files.get("morph.jsonl", {}).get("sha256", "")[:12]
    payload = LeaderboardPayload(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        tier=manifest.tier,
        manifest_sha256=manifest_sha,
        total_samples=manifest.total_rows,
        langs=list(LANGS),
        lang_names=dict(LANG_NAMES),
        models=cards,
        per_lang=per_lang,
        lid_confusion=lid_confusion,
        shatter_samples=shatter_samples,
        pending_models=pending,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "leaderboard.json").write_text(
        payload.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    # Publish real qualitative outputs for Sample Output Inspector
    morph_samples = load_split(data_dir, "morph")
    by_lang_sample = {}
    for s in morph_samples:
        if s.lang not in by_lang_sample:
            by_lang_sample[s.lang] = s

    # Pick representative models across tracks for side-by-side inspection
    rep_cards: List[ModelScorecard] = []
    for trk in ("Sebeni", "Google", "Open Frontier"):
        match = [c for c in cards if c.track == trk]
        if match:
            rep_cards.append(match[0])
    if len(rep_cards) < 3:
        for c in cards:
            if c not in rep_cards and len(rep_cards) < 3:
                rep_cards.append(c)

    samples_payload: Dict[str, Any] = {}
    for lang in ("taq", "ful", "mey", "dtm", "bam"):
        s = by_lang_sample.get(lang)
        if not s:
            continue
        oracle_json = json.dumps(
            {
                "lang": s.lang,
                "tokens": [t.model_dump() for t in s.ref_tokens],
            },
            ensure_ascii=False,
            indent=2,
        )
        model_outputs = []
        for c in rep_cards:
            recs = load_cached_records(runs_dir / c.slug / "morph.completions.jsonl")
            rec = recs.get(s.id)
            txt = strip_thinking(rec.completion) if rec and not rec.error else "[No completion recorded]"
            if len(txt) > 900:
                txt = txt[:900] + "\n... [truncated]"
            model_outputs.append(
                {
                    "id": c.id,
                    "track": c.track,
                    "trackCls": c.trackCls,
                    "output": txt,
                }
            )
        samples_payload[lang] = {
            "lang": lang,
            "lang_name": LANG_NAMES[lang],
            "sample_id": s.id,
            "prompt": s.text,
            "oracle": oracle_json,
            "models": model_outputs,
        }

    (out_dir / "samples.json").write_text(
        json.dumps(samples_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload
