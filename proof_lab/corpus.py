"""Deterministic 4,200-sentence MER/MCS/UWEC stress-test suite on dataset_300_samples.jsonl."""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import beni
from beni.core.compute.helpers import is_empr_token
from beni.core.compute.metrics import MorphologyScorer
from beni.core.morphotactic.dabax import DabaX

from bench.dataset import _batch_dabax_parse
from proof_lab import DATASET_LANGS, LANG_SPECS
from proof_lab.metrics_core import (
    aligned_mcs,
    allow_promote_strict_ood,
    bounded_mer,
    cohens_d,
    corpus_mcs_ratio,
    decomposed_uwec,
    exact_mer,
    fsa_accepts,
    index_mcs,
)
from proof_lab.schema import CorpusStressResult, RegimeStats, SpotlightExample

_NON_ASCII_MALI = set("ɛɔɲŋɓɗƴăəɣḍḥṭẓʼèéàòùìîôûâêëïü")


def locate_dataset_300() -> Path:
    candidates: List[Path] = []
    env_dir = os.environ.get("SEBENI_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir) / "raw" / "dataset_300_samples.jsonl")
    candidates.append(Path(beni.__file__).resolve().parent / "data" / "raw" / "dataset_300_samples.jsonl")
    candidates.append(Path("/home/seb/sebenx/sebeni/beni/data/raw/dataset_300_samples.jsonl"))
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Could not locate dataset_300_samples.jsonl in SEBENI_DATA_DIR, installed beni, or /home/seb/sebenx/sebeni."
    )


def load_language_rows(lang: str, *, max_rows: int = 300) -> List[str]:
    ds_path = locate_dataset_300()
    rows: List[str] = []
    with ds_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if str(obj.get("lang", "")).strip() == lang:
                txt = str(obj.get("text", "")).strip()
                if txt:
                    rows.append(txt)
                if len(rows) >= max_rows:
                    break
    if not rows:
        raise ValueError(f"No rows found for language '{lang}' in {ds_path}")
    return rows


def _extract_sentence_features(
    sent: Any,
    fallback_text: str,
    scorer: MorphologyScorer,
) -> Dict[str, Any]:
    tokens = getattr(sent, "tokens", None) or []
    morphemes: List[str] = []
    stages: List[int] = []
    fsa_tags: List[str] = []
    n_empr = 0
    n_tokens = len(tokens)
    tok_previews: List[Dict[str, Any]] = []

    for tok in tokens:
        st_raw = str(getattr(tok, "stage", "-1"))
        try:
            st_int = int(st_raw)
        except ValueError:
            st_int = -1 if st_raw == "-1" else 0
        surf = str(getattr(tok, "surface", ""))
        empr = bool(is_empr_token(tok))
        if empr:
            n_empr += 1
        try:
            bm = [str(m) for m in tok.best_morphemes() if str(m).strip()]
        except Exception:
            bm = []
        if not bm and surf:
            bm = [surf]
        morphemes.extend(bm)

        if st_int < 0:
            stages.append(-1)
            fsa_tags.append("unk")
        elif empr:
            stages.append(st_int)
            fsa_tags.append("empr")
        else:
            stages.append(st_int)
            fsa_tags.append("stem")
            for _ in range(max(0, len(bm) - 1)):
                fsa_tags.append("infl")

        if len(tok_previews) < 10:
            gl = ""
            analyses = getattr(tok, "analyses", None) or []
            if analyses:
                gl = str(getattr(analyses[0], "gloss", "") or "")
            tok_previews.append(
                {
                    "surface": surf,
                    "stage": st_int,
                    "morphemes": bm,
                    "gloss": gl[:48],
                    "empr": empr,
                }
            )

    if not morphemes:
        morphemes = fallback_text.split() or ["?"]
        stages = [-1] * len(morphemes)
        fsa_tags = ["unk"] * len(morphemes)

    phi = float(scorer.phi(sent)) if tokens else 0.0
    strict = float(scorer.strict_parsed_rate(sent)) if tokens else 0.0
    return {
        "morphemes": morphemes,
        "stages": stages,
        "fsa_tags": fsa_tags,
        "phi": phi,
        "strict": strict,
        "n_tokens": max(1, n_tokens),
        "n_empr": n_empr,
        "tok_previews": tok_previews,
    }


def _build_regime_stats(
    *,
    regime_id: str,
    label: str,
    theorem_ref: str,
    ref_morphs_list: Sequence[Sequence[str]],
    hyp_morphs_list: Sequence[Sequence[str]],
    ref_stages_list: Sequence[Sequence[int]],
    pred_stages_list: Sequence[Sequence[int]],
    phis: Sequence[float],
    stricts: Sequence[float],
    pi_thetas: Sequence[Sequence[float]],
    pi_refs: Sequence[Sequence[float]],
    reject_flags: Sequence[bool],
) -> Tuple[RegimeStats, Dict[str, List[float]]]:
    mers_raw: List[float] = []
    mers_bnd: List[float] = []
    mcs_idx: List[float] = []
    mcs_aln: List[float] = []
    uw_orig: List[float] = []
    uw_inv: List[float] = []

    for rm, hm, rs, ps, pt, pr in zip(
        ref_morphs_list, hyp_morphs_list, ref_stages_list, pred_stages_list, pi_thetas, pi_refs
    ):
        mers_raw.append(exact_mer(rm, hm))
        mers_bnd.append(bounded_mer(rm, hm))
        mcs_idx.append(index_mcs(ps, rs))
        mcs_aln.append(aligned_mcs(ps, rs))
        u0, _, _ = decomposed_uwec(ps, pt, pr, invert_indicator=False)
        u1, _, _ = decomposed_uwec(ps, pt, pr, invert_indicator=True)
        uw_orig.append(u0)
        uw_inv.append(u1)

    n = len(ref_morphs_list)
    corpus_m = corpus_mcs_ratio(pred_stages_list, ref_stages_list)
    gt_one_pct = (100.0 * sum(1 for x in mers_raw if x > 1.0) / n) if n else 0.0
    stats = RegimeStats(
        regime_id=regime_id,
        label=label,
        theorem_ref=theorem_ref,
        n_sentences=n,
        mean_mer_raw=round(sum(mers_raw) / max(1, n), 4),
        mean_mer_bounded=round(sum(mers_bnd) / max(1, n), 4),
        mer_gt_one_pct=round(gt_one_pct, 2),
        mean_mcs_index=round(sum(mcs_idx) / max(1, n), 4),
        mean_mcs_aligned=round(sum(mcs_aln) / max(1, n), 4),
        mean_mcs_corpus=round(corpus_m, 4),
        mean_uwec_orig=round(sum(uw_orig) / max(1, n), 4),
        mean_uwec_inv=round(sum(uw_inv) / max(1, n), 4),
        mean_phi=round(sum(phis) / max(1, n), 4),
        mean_strict_parse=round(sum(stricts) / max(1, n), 4),
        strict_ood_reject_pct=round(100.0 * sum(1 for f in reject_flags if f) / max(1, n), 2),
    )
    return stats, {"mer_bnd": mers_bnd, "mcs_idx": mcs_idx, "uwec_inv": uw_inv}


def analyze_language_corpus(lang_code: str, *, max_rows: int = 300) -> CorpusStressResult:
    if lang_code not in LANG_SPECS:
        raise KeyError(f"Unknown dataset_300 language '{lang_code}'. Expected one of {DATASET_LANGS}")
    spec = LANG_SPECS[lang_code]
    rows = load_language_rows(lang_code, max_rows=max_rows)
    scorer = MorphologyScorer(beta=0.1, eps=1e-8, empr=0.5)

    with tempfile.TemporaryDirectory() as tmp:
        dx_main = DabaX(spec.dabax_baseline, runtime_dir=tmp, strict_parse=True, normalize_homoglyphs=True)
        parsed_main = _batch_dabax_parse(dx_main, rows, chunk_size=40)

        dx_ood = DabaX(spec.ood_cross_baseline, runtime_dir=tmp, strict_parse=True, normalize_homoglyphs=True)
        parsed_ood = _batch_dabax_parse(dx_ood, rows, chunk_size=40)

    clean_feats = [
        _extract_sentence_features(sent, txt, scorer)
        for sent, txt in zip(parsed_main, rows)
    ]
    ood_feats = [
        _extract_sentence_features(sent, txt, scorer)
        for sent, txt in zip(parsed_ood, rows)
    ]

    ref_m = [f["morphemes"] for f in clean_feats]
    ref_s = [f["stages"] for f in clean_feats]
    clean_phis = [f["phi"] for f in clean_feats]
    clean_strict = [f["strict"] for f in clean_feats]
    n_tokens_total = sum(f["n_tokens"] for f in clean_feats)
    empr_total = sum(f["n_empr"] for f in clean_feats)

    words_all = [w for txt in rows for w in txt.split()]
    diac_words = sum(1 for w in words_all if any(ch in _NON_ASCII_MALI for ch in w.lower()))
    diac_pct = 100.0 * diac_words / max(1, len(words_all))

    rng = random.Random(2026 + sum(ord(c) for c in lang_code))
    pi_id = [[0.85 if s >= 0 else 0.15 for s in rs] for rs in ref_s]
    pi_ref_mid = [[0.50] * len(rs) for rs in ref_s]

    regimes: Dict[str, RegimeStats] = {}
    vecs: Dict[str, Dict[str, List[float]]] = {}

    # 1. Clean identity
    regimes["clean_identity"], vecs["clean"] = _build_regime_stats(
        regime_id="clean_identity",
        label="Clean Identity (In-Domain)",
        theorem_ref="Basic.lean::stageToPhi_nonneg_bound",
        ref_morphs_list=ref_m,
        hyp_morphs_list=ref_m,
        ref_stages_list=ref_s,
        pred_stages_list=ref_s,
        phis=clean_phis,
        stricts=clean_strict,
        pi_thetas=pi_id,
        pi_refs=pi_ref_mid,
        reject_flags=[False] * len(rows),
    )

    # 2. Leading shift / clitic insertion
    shift_m = [["<clitic>"] + rm for rm in ref_m]
    shift_s = [[-1] + rs for rs in ref_s]
    regimes["leading_shift"], vecs["shift"] = _build_regime_stats(
        regime_id="leading_shift",
        label="Leading Token / Clitic Shift",
        theorem_ref="Metrics.lean::stageLcs_resists_shift_witness",
        ref_morphs_list=ref_m,
        hyp_morphs_list=shift_m,
        ref_stages_list=ref_s,
        pred_stages_list=shift_s,
        phis=clean_phis,
        stricts=[0.85 * s for s in clean_strict],
        pi_thetas=[[0.5] + p for p in pi_id],
        pi_refs=[[0.5] * len(ss) for ss in shift_s],
        reject_flags=[
            not allow_promote_strict_ood(
                mcs=1.0,
                mcs_prime=index_mcs(ps, rs),
                phi=p0,
                phi_prime=p0 + 0.05,
                mer=bounded_mer(rm, hm),
            )
            for rm, hm, rs, ps, p0 in zip(ref_m, shift_m, ref_s, shift_s, clean_phis)
        ],
    )

    # 3. Severe over-generation drift (|hyp| > 2*|ref| => raw MER > 1.0)
    over_m = [rm + [f"x{k}" for k in range(max(3, len(rm) + 2))] for rm in ref_m]
    over_s = [rs + [-1] * (len(om) - len(rs)) for rs, om in zip(ref_s, over_m)]
    regimes["overgeneration_drift"], vecs["overgen"] = _build_regime_stats(
        regime_id="overgeneration_drift",
        label="Severe Over-Generation Drift",
        theorem_ref="Metrics.lean::bounded_mer_always_unit_bounded",
        ref_morphs_list=ref_m,
        hyp_morphs_list=over_m,
        ref_stages_list=ref_s,
        pred_stages_list=over_s,
        phis=[p * (len(rs) / max(1, len(os_))) for p, rs, os_ in zip(clean_phis, ref_s, over_s)],
        stricts=[s * 0.45 for s in clean_strict],
        pi_thetas=[[0.90] * len(os_) for os_ in over_s],
        pi_refs=[[0.20] * len(os_) for os_ in over_s],
        reject_flags=[True] * len(rows),
    )

    # 4. Morphotactic corruption & bare-affix FSA break
    corr_m: List[List[str]] = []
    corr_s: List[List[int]] = []
    for rm, rs in zip(ref_m, ref_s):
        if len(rm) >= 2:
            cm = list(reversed(rm[1:])) if len(rm) >= 3 else [rm[-1], "-sfx"]
            cs = [-1 if k % 2 == 0 else (rs[min(k, len(rs) - 1)] + 1) for k in range(len(cm))]
        else:
            cm = ["-sfx", "unk_stem"]
            cs = [-1, 0]
        if rng.random() < 0.35 and len(cm) >= 1:
            cm[0] = cm[0] + "-na?"
        corr_m.append(cm)
        corr_s.append(cs)
    regimes["morphotactic_corruption"], vecs["corrupt"] = _build_regime_stats(
        regime_id="morphotactic_corruption",
        label="Morphotactic / Affix Corruption",
        theorem_ref="Morphotactics.lean::phi_inflation_without_fsa_validity",
        ref_morphs_list=ref_m,
        hyp_morphs_list=corr_m,
        ref_stages_list=ref_s,
        pred_stages_list=corr_s,
        phis=[0.95] * len(rows),  # inflated naive Phi despite invalid FSA sequences
        stricts=[0.05] * len(rows),
        pi_thetas=[[0.75] * len(cs) for cs in corr_s],
        pi_refs=[[0.45] * len(cs) for cs in corr_s],
        reject_flags=[
            not allow_promote_strict_ood(
                mcs=1.0,
                mcs_prime=aligned_mcs(ps, rs),
                phi=p0,
                phi_prime=0.95,
                mer=bounded_mer(rm, cm),
            )
            for rm, cm, rs, ps, p0 in zip(ref_m, corr_m, ref_s, corr_s, clean_phis)
        ],
    )

    # 5. Cross-language / OOD detector check (DabaX under mismatched family grammar)
    ood_m = [f["morphemes"] for f in ood_feats]
    ood_s = [f["stages"] for f in ood_feats]
    ood_phis = [f["phi"] for f in ood_feats]
    ood_strict = [f["strict"] for f in ood_feats]
    regimes["ood_cross_family"], vecs["ood"] = _build_regime_stats(
        regime_id="ood_cross_family",
        label=f"Cross-Family / OOD ({spec.ood_cross_baseline} parser)",
        theorem_ref="Metrics.lean::uwec_current_indicator_prefers_unknown",
        ref_morphs_list=ref_m,
        hyp_morphs_list=ood_m,
        ref_stages_list=ref_s,
        pred_stages_list=ood_s,
        phis=ood_phis,
        stricts=ood_strict,
        pi_thetas=[[0.25 if s < 0 else 0.75 for s in os_] for os_ in ood_s],
        pi_refs=[[0.75] * len(os_) for os_ in ood_s],
        reject_flags=[
            not allow_promote_strict_ood(
                mcs=1.0,
                mcs_prime=index_mcs(os_, rs),
                phi=p0,
                phi_prime=p1,
                mer=bounded_mer(rm, om),
            )
            for rm, om, rs, os_, p0, p1 in zip(ref_m, ood_m, ref_s, ood_s, clean_phis, ood_phis)
        ],
    )

    # 6. Truncation asymmetry (|pred| < |ref| => mcs_corpus > mcs_sentence)
    trunc_m = [rm[: max(1, len(rm) // 2)] for rm in ref_m]
    trunc_s = [rs[: max(1, len(rs) // 2)] for rs in ref_s]
    regimes["truncation_asymmetry"], _ = _build_regime_stats(
        regime_id="truncation_asymmetry",
        label="Half-Sequence Truncation Asymmetry",
        theorem_ref="Metrics.lean::mcs_corpus_denom_asymmetry",
        ref_morphs_list=ref_m,
        hyp_morphs_list=trunc_m,
        ref_stages_list=ref_s,
        pred_stages_list=trunc_s,
        phis=clean_phis,
        stricts=clean_strict,
        pi_thetas=[[0.75] * len(ts) for ts in trunc_s],
        pi_refs=[[0.50] * len(ts) for ts in trunc_s],
        reject_flags=[
            not allow_promote_strict_ood(
                mcs=1.0,
                mcs_prime=index_mcs(ts, rs),
                phi=0.5,
                phi_prime=0.6,
                mer=bounded_mer(rm, tm),
            )
            for rm, tm, rs, ts in zip(ref_m, trunc_m, ref_s, trunc_s)
        ],
    )

    # Spotlight 3 diverse sentences (short/medium/complex) for the Proof Workbench
    scored_indices = sorted(
        range(len(rows)),
        key=lambda i: (abs(len(ref_m[i]) - 6), -clean_strict[i], i),
    )
    pick_indices = []
    for candidate in scored_indices:
        if len(rows[candidate]) <= 220 and len(ref_m[candidate]) >= 3:
            pick_indices.append(candidate)
        if len(pick_indices) == 3:
            break
    if not pick_indices:
        pick_indices = list(range(min(3, len(rows))))

    spotlights: List[SpotlightExample] = []
    for idx in pick_indices:
        cf = clean_feats[idx]
        of = ood_feats[idx]
        spotlights.append(
            SpotlightExample(
                idx=idx,
                text=rows[idx],
                tokens_preview=cf["tok_previews"],
                ref_morphemes=cf["morphemes"][:14],
                ref_stages=cf["stages"][:14],
                fsa_tags=cf["fsa_tags"][:14],
                fsa_valid=fsa_accepts(cf["fsa_tags"]),
                ood_cross_stages=of["stages"][:14],
                ood_cross_morphemes=of["morphemes"][:14],
                clean_phi=round(cf["phi"], 4),
                clean_strict_parse=round(cf["strict"], 4),
                ood_strict_parse=round(of["strict"], 4),
            )
        )

    shift_stat = regimes["leading_shift"]
    shift_rec_pct = 100.0 * max(0.0, shift_stat.mean_mcs_aligned - shift_stat.mean_mcs_index)
    trunc_stat = regimes["truncation_asymmetry"]
    trunc_bias_pct = 100.0 * max(0.0, trunc_stat.mean_mcs_corpus - trunc_stat.mean_mcs_index)
    clean_st_pct = 100.0 * sum(clean_strict) / max(1, len(clean_strict))
    ood_st_pct = 100.0 * sum(ood_strict) / max(1, len(ood_strict))

    return CorpusStressResult(
        lang=lang_code,
        lang_name=spec.name,
        family=spec.family,
        dabax_baseline=spec.dabax_baseline,
        ood_cross_baseline=spec.ood_cross_baseline,
        is_ood=spec.is_ood,
        alias_note=spec.alias_note,
        n_samples=len(rows),
        n_tokens_total=n_tokens_total,
        clean_phi_pct=round(100.0 * sum(clean_phis) / max(1, len(clean_phis)), 2),
        clean_strict_parse_pct=round(clean_st_pct, 2),
        empr_token_pct=round(100.0 * empr_total / max(1, n_tokens_total), 2),
        diacritic_word_pct=round(diac_pct, 2),
        cohens_d_mer_corrupt=round(cohens_d(vecs["clean"]["mer_bnd"], vecs["corrupt"]["mer_bnd"]), 2),
        cohens_d_mcs_shift=round(cohens_d(vecs["clean"]["mcs_idx"], vecs["shift"]["mcs_idx"]), 2),
        cohens_d_uwec_ood=round(cohens_d(vecs["clean"]["uwec_inv"], vecs["ood"]["uwec_inv"]), 2),
        ood_strict_gap_pct=round(clean_st_pct - ood_st_pct, 2),
        shift_mcs_recovery_pct=round(shift_rec_pct, 2),
        mer_saturation_prevented_pct=round(regimes["overgeneration_drift"].mer_gt_one_pct, 2),
        truncation_denom_bias_pct=round(trunc_bias_pct, 2),
        regimes=regimes,
        spotlights=spotlights,
    )
