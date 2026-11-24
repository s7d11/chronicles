"""MALI13 scoring engine: DabaX(strict_parse=True), M/G/L/C, Equity, Fertility, BPB & Pareto."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from beni.core.compute.helpers import is_empr_token, mer_micro
from beni.core.compute.metrics import MorphologyScorer
from beni.core.compute.rewards import RewardManager
from beni.core.morphotactic.dabax import DabaX
from beni.utils.normalize import is_clean_corpus_line, normalize_homoglyphs

from bench import LANGS
from bench.schema import (
    ClozeSample,
    CompletionRecord,
    EfficiencyStats,
    GenSample,
    LidSample,
    ModelScorecard,
    ModelSpec,
    MorphSample,
)

_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?(?:</think>|$)", re.IGNORECASE)


def _morpheme_edit_rate(ref_forms: Sequence[str], hyp_forms: Sequence[str]) -> float:
    """Exact Levenshtein Morpheme Error Rate (S + D + I) / N_ref."""
    ref = [str(x) for x in ref_forms]
    hyp = [str(x) for x in hyp_forms]
    m, n = len(ref), len(hyp)
    if m == 0:
        return 0.0 if n == 0 else 1.0
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        ri = ref[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n] / m
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_ISO_CODE_RE = re.compile(r"\b(" + "|".join(LANGS) + r")\b", re.IGNORECASE)
_CHOICE_JSON_RE = re.compile(r'"choice"\s*:\s*"([ABCD])"', re.IGNORECASE)
_CHOICE_LETTER_RE = re.compile(r"(?:^|[\s:(])([ABCD])(?:$|[\s).,:])")
_WORD_RE = re.compile(r"[^\s.,;:!?\"'()\[\]{}<>]+")

_EN_FR_NOISE = frozenset(
    {
        "the", "and", "for", "that", "with", "this", "from", "are", "was", "were",
        "have", "has", "been", "which", "their", "will", "would", "could", "should",
        "meaning", "translation", "sentence", "language", "here", "continuation",
        "les", "des", "une", "dans", "pour", "pas", "qui", "que", "sur", "est",
        "sont", "avec", "par", "plus", "tout", "mais", "cette", "comme", "aussi",
        "signifie", "phrase", "traduction", "voici",
    }
)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning traces before parsing model outputs."""
    if not text:
        return ""
    cleaned = _THINK_BLOCK_RE.sub("", text).strip()
    return cleaned


def _extract_json_flexible(raw_text: str) -> Dict[str, Any]:
    cleaned = strip_thinking(raw_text)
    if not cleaned:
        return {}
    parsed = RewardManager.extract_json(cleaned)
    if parsed:
        return parsed
    for m in _CODE_FENCE_RE.finditer(cleaned):
        parsed = RewardManager.extract_json(m.group(1))
        if parsed:
            return parsed
    return {}


def _extract_predicted_tokens(parsed: Dict[str, Any]) -> Tuple[bool, List[Tuple[str, List[str]]]]:
    """Return (format_ok, [(surface, [morpheme_forms])]) supporting both full and compact schemas."""
    tokens = parsed.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        return False, []
    out: List[Tuple[str, List[str]]] = []
    for t in tokens:
        if not isinstance(t, dict):
            continue
        surf = str(t.get("surface") or t.get("token") or "").strip()
        morphs: List[str] = []
        if isinstance(t.get("analyses"), list) and t["analyses"]:
            a0 = t["analyses"][0]
            if isinstance(a0, dict):
                raw_m = a0.get("morphemes") or []
                for m in raw_m:
                    if isinstance(m, dict):
                        nested = m.get("morphemes")
                        if isinstance(nested, list) and nested:
                            for sub in nested:
                                if isinstance(sub, dict) and sub.get("form"):
                                    morphs.append(str(sub["form"]))
                                elif isinstance(sub, str) and sub.strip():
                                    morphs.append(sub.strip())
                        elif m.get("form"):
                            morphs.append(str(m["form"]))
                    elif isinstance(m, str) and m.strip():
                        morphs.append(m.strip())
                if not morphs and a0.get("form"):
                    morphs = [str(a0["form"])]
        elif isinstance(t.get("morphemes"), list):
            for m in t["morphemes"]:
                if isinstance(m, dict) and m.get("form"):
                    morphs.append(str(m["form"]))
                elif isinstance(m, str) and m.strip():
                    morphs.append(m.strip())
        if surf or morphs:
            if not surf:
                surf = "".join(morphs)
            if not morphs:
                morphs = [surf]
            out.append((surf, morphs))
    return bool(out), out


class Mali13Scorer:
    """Thread-safe/reusable evaluator wrapping DabaX(strict_parse=True) across MULTI13."""

    def __init__(self, runtime_dir: Optional[Path] = None):
        self.runtime_dir = runtime_dir or Path("/tmp/chronicles_dabax_runtime")
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.morph_scorer = MorphologyScorer()
        self._dabax_cache: Dict[str, DabaX] = {}

    def dabax(self, lang: str) -> DabaX:
        if lang not in self._dabax_cache:
            self._dabax_cache[lang] = DabaX(
                lang,
                runtime_dir=self.runtime_dir,
                strict_parse=True,
                normalize_homoglyphs=True,
            )
        return self._dabax_cache[lang]

    def _reparse_tokens(self, lang: str, surfaces: Sequence[str]):
        if not surfaces:
            return []
        dx = self.dabax(lang)
        joined = " ".join(s.replace("\n", " ").strip() for s in surfaces if s.strip())
        if not joined:
            return []
        try:
            sents = dx.loader(joined)
            toks = []
            for s in sents:
                toks.extend(s.tokens)
            return toks
        except Exception:
            return []

    def score_morph(
        self,
        samples: Sequence[MorphSample],
        records_by_id: Dict[str, CompletionRecord],
    ) -> Dict[str, float]:
        """Compute M_lang per language (0-100)."""
        by_lang: Dict[str, List[MorphSample]] = defaultdict(list)
        for s in samples:
            by_lang[s.lang].append(s)

        scores: Dict[str, float] = {}
        for lang in LANGS:
            all_rows = by_lang.get(lang, [])
            rows = [s for s in all_rows if s.id in records_by_id] if records_by_id else all_rows
            if not rows:
                scores[lang] = 0.0
                continue
            fmt_ok_count = 0
            phis: List[float] = []
            stricts: List[float] = []
            mer_quals: List[float] = []
            mcs_vals: List[float] = []

            for sample in rows:
                rec = records_by_id.get(sample.id)
                if rec is None or rec.error or not rec.completion:
                    continue
                parsed = _extract_json_flexible(rec.completion)
                ok, pred_pairs = _extract_predicted_tokens(parsed)
                if not ok:
                    continue
                fmt_ok_count += 1

                pred_surfaces = [p[0] for p in pred_pairs]
                pred_morphemes = [m for _, ms in pred_pairs for m in ms]
                reparsed_toks = self._reparse_tokens(lang, pred_surfaces)

                # Guard against gaming Phi/StrictRate by hallucinating unrelated valid words:
                # only tokens matching a reference surface token earn Phi/StrictRate credit.
                ref_surf_set = {
                    normalize_homoglyphs(t.surface).strip().lower()
                    for t in sample.ref_tokens
                }
                if reparsed_toks:
                    credited = [
                        t for t in reparsed_toks
                        if not ref_surf_set
                        or normalize_homoglyphs(str(t.surface)).strip().lower() in ref_surf_set
                    ]
                    denom_toks = max(len(reparsed_toks), len(sample.ref_tokens), 1)
                    phi_val = (
                        sum(
                            float(self.morph_scorer.phi([t]))
                            for t in credited
                        )
                        / denom_toks
                    )
                    strict_val = (
                        sum(
                            float(self.morph_scorer.strict_parsed_rate([t]))
                            for t in credited
                        )
                        / denom_toks
                    )
                    pred_stages = [str(t.stage) for t in reparsed_toks]
                else:
                    phi_val = 0.0
                    strict_val = 0.0
                    pred_stages = ["-1"] * len(pred_surfaces)

                mer = _morpheme_edit_rate(sample.ref_morphemes, pred_morphemes)
                mer_qual = 0.0 if not math.isfinite(mer) else max(0.0, 1.0 - min(1.0, mer))

                denom = max(len(pred_stages), len(sample.ref_stages), 1)
                matches = sum(
                    1 for p, r in zip(pred_stages, sample.ref_stages) if str(p) == str(r)
                )
                mcs_val = matches / denom

                phis.append(phi_val)
                stricts.append(strict_val)
                mer_quals.append(mer_qual)
                mcs_vals.append(mcs_val)

            if fmt_ok_count == 0:
                scores[lang] = 0.0
                continue
            fmt_rate = fmt_ok_count / len(rows)
            avg_phi = sum(phis) / fmt_ok_count
            avg_strict = sum(stricts) / fmt_ok_count
            avg_mer_q = sum(mer_quals) / fmt_ok_count
            avg_mcs = sum(mcs_vals) / fmt_ok_count
            m_lang = 100.0 * fmt_rate * (
                0.30 * avg_phi + 0.30 * avg_strict + 0.25 * avg_mer_q + 0.15 * avg_mcs
            )
            scores[lang] = round(m_lang, 2)
        return scores

    def score_gen(
        self,
        samples: Sequence[GenSample],
        records_by_id: Dict[str, CompletionRecord],
    ) -> Dict[str, float]:
        """Compute G_lang per language (0-100)."""
        by_lang: Dict[str, List[GenSample]] = defaultdict(list)
        for s in samples:
            by_lang[s.lang].append(s)

        scores: Dict[str, float] = {}
        for lang in LANGS:
            all_rows = by_lang.get(lang, [])
            rows = [s for s in all_rows if s.id in records_by_id] if records_by_id else all_rows
            if not rows:
                scores[lang] = 0.0
                continue
            row_scores: List[float] = []
            for sample in rows:
                rec = records_by_id.get(sample.id)
                if rec is None or rec.error or not rec.completion:
                    row_scores.append(0.0)
                    continue
                cont = strip_thinking(rec.completion).strip()
                # Remove echoed prefix if the model repeated it
                norm_pref = normalize_homoglyphs(sample.prompt_prefix).strip()
                norm_cont = normalize_homoglyphs(cont).strip()
                if norm_cont.lower().startswith(norm_pref.lower()):
                    cont = norm_cont[len(norm_pref) :].strip()
                words = _WORD_RE.findall(cont)
                if len(words) < 3:
                    row_scores.append(0.0)
                    continue

                toks = self._reparse_tokens(lang, words[:60])
                if not toks:
                    row_scores.append(0.0)
                    continue

                phi_g = float(self.morph_scorer.phi(toks))
                strict_g = float(self.morph_scorer.strict_parsed_rate(toks))
                empr_rate = sum(1 for t in toks if is_empr_token(t)) / len(toks)
                low_words = [w.lower() for w in words[:60]]
                noise_hits = sum(1 for w in low_words if w in _EN_FR_NOISE)
                foreign_rate = noise_hits / len(low_words)
                if not is_clean_corpus_line(cont):
                    foreign_rate = max(foreign_rate, 0.35)
                purity = max(0.0, 1.0 - empr_rate - foreign_rate)

                bigrams = list(zip(low_words, low_words[1:]))
                distinct_2 = len(set(bigrams)) / max(1, len(bigrams))
                rep_pen = min(1.0, distinct_2 / 0.45)

                g = 100.0 * (0.45 * phi_g + 0.35 * strict_g + 0.20 * purity) * rep_pen
                row_scores.append(g)
            scores[lang] = round(sum(row_scores) / len(row_scores), 2)
        return scores

    @staticmethod
    def extract_lid_pred(text: str) -> Optional[str]:
        cleaned = strip_thinking(text)
        if not cleaned:
            return None
        parsed = _extract_json_flexible(cleaned)
        if isinstance(parsed.get("lang"), str):
            code = parsed["lang"].strip().lower()
            if code in LANGS:
                return code
        m = _ISO_CODE_RE.search(cleaned)
        if m:
            return m.group(1).lower()
        return None

    def score_lid(
        self,
        samples: Sequence[LidSample],
        records_by_id: Dict[str, CompletionRecord],
    ) -> Tuple[Dict[str, float], List[List[int]]]:
        """Return (per_lang_F1_0_100, 13x13 confusion matrix)."""
        idx_map = {code: i for i, code in enumerate(LANGS)}
        matrix = [[0 for _ in LANGS] for _ in LANGS]
        tp = [0] * len(LANGS)
        fp = [0] * len(LANGS)
        fn = [0] * len(LANGS)

        eval_samples = [s for s in samples if s.id in records_by_id] if records_by_id else list(samples)
        for sample in eval_samples:
            t_idx = idx_map[sample.lang]
            rec = records_by_id.get(sample.id)
            pred = self.extract_lid_pred(rec.completion) if rec and not rec.error else None
            if pred in idx_map:
                p_idx = idx_map[pred]
                matrix[t_idx][p_idx] += 1
                if p_idx == t_idx:
                    tp[t_idx] += 1
                else:
                    fn[t_idx] += 1
                    fp[p_idx] += 1
            else:
                fn[t_idx] += 1

        f1_scores: Dict[str, float] = {}
        for i, code in enumerate(LANGS):
            denom = 2 * tp[i] + fp[i] + fn[i]
            f1 = (2 * tp[i] / denom) if denom > 0 else 0.0
            f1_scores[code] = round(100.0 * f1, 2)
        return f1_scores, matrix

    @staticmethod
    def extract_cloze_pred(rec: Optional[CompletionRecord]) -> Optional[str]:
        if rec is None or rec.error:
            return None
        if rec.logprob_scores:
            best = max(rec.logprob_scores.items(), key=lambda kv: kv[1])[0].upper()
            if best in {"A", "B", "C", "D"}:
                return best
        cleaned = strip_thinking(rec.completion)
        if not cleaned:
            return None
        parsed = _extract_json_flexible(cleaned)
        ch = str(parsed.get("choice") or parsed.get("answer") or "").strip().upper()
        if ch in {"A", "B", "C", "D"}:
            return ch
        m = _CHOICE_JSON_RE.search(cleaned)
        if m:
            return m.group(1).upper()
        m2 = _CHOICE_LETTER_RE.search(cleaned)
        if m2:
            return m2.group(1).upper()
        return None

    def score_cloze(
        self,
        samples: Sequence[ClozeSample],
        records_by_id: Dict[str, CompletionRecord],
    ) -> Dict[str, float]:
        by_lang: Dict[str, List[ClozeSample]] = defaultdict(list)
        for s in samples:
            by_lang[s.lang].append(s)
        scores: Dict[str, float] = {}
        for lang in LANGS:
            all_rows = by_lang.get(lang, [])
            rows = [s for s in all_rows if s.id in records_by_id] if records_by_id else all_rows
            if not rows:
                scores[lang] = 0.0
                continue
            ok = sum(
                1 for s in rows if self.extract_cloze_pred(records_by_id.get(s.id)) == s.answer
            )
            scores[lang] = round(100.0 * ok / len(rows), 2)
        return scores

    def build_scorecard(
        self,
        spec: ModelSpec,
        morph_samples: Sequence[MorphSample],
        gen_samples: Sequence[GenSample],
        lid_samples: Sequence[LidSample],
        cloze_samples: Sequence[ClozeSample],
        records: Sequence[CompletionRecord],
        efficiency: EfficiencyStats,
    ) -> Tuple[ModelScorecard, Dict[str, Dict[str, float]], List[List[int]]]:
        rec_map = {r.sample_id: r for r in records}
        m_lang = self.score_morph(morph_samples, rec_map)
        g_lang = self.score_gen(gen_samples, rec_map)
        l_lang, lid_matrix = self.score_lid(lid_samples, rec_map)
        c_lang = self.score_cloze(cloze_samples, rec_map)

        q_lang: Dict[str, float] = {}
        for code in LANGS:
            q_lang[code] = round(
                0.25 * (m_lang[code] + g_lang[code] + l_lang[code] + c_lang[code]), 2
            )

        q_vals = sorted(q_lang.values())
        mali13_q = round(sum(q_vals) / len(LANGS), 2)
        q_min3 = round(sum(q_vals[:3]) / 3.0, 2)
        q_gap = round(q_vals[-1] - q_vals[0], 2)

        macro = lambda d: round(sum(d.values()) / len(LANGS), 2)
        valid_recs = [r for r in records if not r.error]
        attempted_count = max(len(records), 1)
        n_errors = sum(1 for r in records if r.error)

        latencies = [r.latency_s for r in valid_recs if r.latency_s > 0]
        mean_lat = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
        total_gen_toks = sum(r.completion_tokens for r in valid_recs)
        total_gen_s = sum(r.latency_s for r in valid_recs)
        tok_s = round(total_gen_toks / total_gen_s, 1) if total_gen_s > 0 else 0.0

        total_in_toks = sum(r.prompt_tokens for r in valid_recs)
        est_cost = round(
            (total_in_toks / 1_000_000.0) * spec.price_in_per_1m
            + (total_gen_toks / 1_000_000.0) * spec.price_out_per_1m,
            4,
        )

        track_cls = (
            "sebeni" if spec.track == "Sebeni"
            else ("api" if spec.track == "Google" else "slm")
        )
        base_compute = spec.compute_label or spec.provider.upper()
        n_valid = len(valid_recs)
        compute_str = f"{base_compute} · {n_valid} evals" if n_valid else base_compute

        card = ModelScorecard(
            slug=spec.slug,
            id=spec.display_name,
            provider=spec.provider,
            model_id=spec.model_id,
            track=spec.track,
            trackCls=track_cls,
            q=mali13_q,
            q_min3=q_min3,
            q_gap=q_gap,
            morph=macro(m_lang),
            gen=macro(g_lang),
            lid=macro(l_lang),
            cloze=macro(c_lang),
            fert=round(efficiency.word_fertility, 3),
            morph_fert=round(efficiency.morpheme_fertility, 3),
            bytes_per_tok=round(efficiency.bytes_per_token, 3),
            diac=round(efficiency.diacritic_penalty, 3),
            ascii_fert=round(efficiency.ascii_fertility, 3),
            ext_fert=round(efficiency.diacritic_fertility, 3),
            bpb=round(efficiency.bpb, 3) if efficiency.bpb is not None else None,
            lat=mean_lat,
            tok_s=tok_s,
            params=spec.active_params_b,
            vram_mb=None,
            cost=est_cost,
            compute=compute_str,
            tokenizer_mode=efficiency.tokenizer_mode,
            completion_rate=round(n_valid / attempted_count, 4),
            n_errors=n_errors,
        )

        per_lang_detail = {
            "q": q_lang,
            "morph": m_lang,
            "gen": g_lang,
            "lid": l_lang,
            "cloze": c_lang,
            "fert": efficiency.per_lang_fertility,
        }
        return card, per_lang_detail, lid_matrix


def pareto_frontier(
    cards: Sequence[ModelScorecard],
    x_attr: str,
) -> List[str]:
    """Return slugs on the non-dominated Pareto envelope (minimize x_attr, maximize q)."""
    pts = []
    for c in cards:
        xv = getattr(c, x_attr, None)
        if xv is None:
            continue
        pts.append((float(xv), -float(c.q), c))
    pts.sort(key=lambda t: (t[0], t[1]))
    frontier: List[str] = []
    best_q = -1.0
    for _xv, _neg_q, card in pts:
        if card.q > best_q:
            frontier.append(card.slug)
            best_q = card.q
    return frontier
