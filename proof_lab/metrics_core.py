"""Executable Python counterparts of SebeniProofs Lean 4 theorems & stress metrics."""

from __future__ import annotations

import math
from enum import Enum
from typing import Dict, List, Sequence, Tuple

import numpy as np


class MorphTag(str, Enum):
    STEM = "stem"
    DERIV = "deriv"
    INFL = "infl"
    EMPR = "empr"
    UNK = "unk"


def fsa_accepts(tags: Sequence[MorphTag | str]) -> bool:
    """Finite-state morphotactic acceptor matching Morphotactics.lean::fsaAccepts."""
    has_stem = False
    for raw in tags:
        t = MorphTag(raw) if not isinstance(raw, MorphTag) else raw
        if t == MorphTag.STEM:
            has_stem = True
        elif t in (MorphTag.DERIV, MorphTag.INFL):
            if not has_stem:
                return False
        elif t == MorphTag.EMPR:
            if has_stem:
                return False
            has_stem = True
        elif t == MorphTag.UNK:
            return False
    return has_stem


def pfv_sandhi_licensed(tail_class: str, suffix: str) -> bool:
    """Bambara PFV.INTR sandhi conditioning (-na/-la/-ra) from Morphotactics.lean."""
    expected = {
        "nasal": "na",
        "liquid_or_glide": "la",
        "oral_non_liquid": "ra",
    }
    return expected.get(tail_class) == suffix


def exact_levenshtein_matrix(ref: Sequence[str], hyp: Sequence[str]) -> List[List[int]]:
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        ri = ref[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp


def exact_levenshtein_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    return exact_levenshtein_matrix(ref, hyp)[len(ref)][len(hyp)]


def exact_mer(ref: Sequence[str], hyp: Sequence[str]) -> float:
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else math.inf
    return exact_levenshtein_distance(ref, hyp) / len(ref)


def bounded_mer(ref: Sequence[str], hyp: Sequence[str]) -> float:
    denom = max(len(ref), len(hyp))
    if denom == 0:
        return 0.0
    return exact_levenshtein_distance(ref, hyp) / denom


def index_mcs(pred_stages: Sequence[int | str], ref_stages: Sequence[int | str]) -> float:
    denom = max(len(pred_stages), len(ref_stages))
    if denom == 0:
        return 0.0
    matches = sum(1 for p, r in zip(pred_stages, ref_stages) if str(p) == str(r))
    return matches / denom


def aligned_mcs(pred_stages: Sequence[int | str], ref_stages: Sequence[int | str]) -> float:
    p = [str(x) for x in pred_stages]
    r = [str(x) for x in ref_stages]
    denom = max(len(p), len(r))
    if denom == 0:
        return 0.0
    m, n = len(p), len(r)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        pi = p[i - 1]
        for j in range(1, n + 1):
            if pi == r[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n] / denom


def corpus_mcs_ratio(pred_seqs: Sequence[Sequence[int | str]], ref_seqs: Sequence[Sequence[int | str]]) -> float:
    """Corpus MCS using len(pred) denominator from MorphologyScorer.mcs_corpus."""
    correct = 0
    pred_total = 0
    for ps, rs in zip(pred_seqs, ref_seqs):
        correct += sum(1 for p, r in zip(ps, rs) if str(p) == str(r))
        pred_total += len(ps)
    return correct / pred_total if pred_total > 0 else 0.0


def decomposed_uwec(
    stages: Sequence[int],
    pi_theta: Sequence[float],
    pi_ref: Sequence[float],
    *,
    beta: float = 0.1,
    eps: float = 1e-8,
    invert_indicator: bool = False,
) -> Tuple[float, float, float]:
    """Return `(total_uwec, indicator_mean, kl_abs_mean)`."""
    s = np.asarray(stages, dtype=float)
    if s.size == 0:
        return 0.0, 0.0, 0.0
    theta = np.clip(np.asarray(pi_theta, dtype=float), 0.0, 1.0)
    ref = np.clip(np.asarray(pi_ref, dtype=float), 0.0, 1.0)
    if theta.size != s.size:
        theta = np.full(s.shape, float(np.mean(theta)) if theta.size else 1.0)
    if ref.size != s.size:
        ref = np.full(s.shape, float(np.mean(ref)) if ref.size else 1.0)
    ind = np.where(s == -1.0, 1.0, 0.0) if invert_indicator else np.where(s != -1.0, 1.0, 0.0)
    abs_lr = np.abs(np.log((theta + eps) / (ref + eps)))
    ind_mean = float(np.mean(ind))
    kl_mean = float(np.mean(beta * abs_lr))
    return ind_mean + kl_mean, ind_mean, kl_mean


def allow_promote(
    *,
    enabled: bool = True,
    language_ok: bool = True,
    parseable: bool = True,
    first_create: bool = False,
    phi: float = 0.0,
    phi_prime: float = 0.0,
    mer: float = 0.0,
    max_mer: float = 0.25,
) -> bool:
    if not enabled:
        return True
    if not language_ok:
        return False
    if first_create:
        return parseable
    if not parseable:
        return False
    if not (phi_prime > phi):
        return False
    if mer > max_mer:
        return False
    return True


def allow_promote_strict_ood(*, mcs: float, mcs_prime: float, **promote_kwargs: object) -> bool:
    return allow_promote(**promote_kwargs) and (mcs_prime >= mcs)


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Standardized mean separation (|mean(a) - mean(b)| / pooled_sd)."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size < 2 or y.size < 2:
        return 0.0
    vx = float(np.var(x, ddof=1))
    vy = float(np.var(y, ddof=1))
    pooled = math.sqrt(max(1e-9, ((x.size - 1) * vx + (y.size - 1) * vy) / (x.size + y.size - 2)))
    return float(abs(np.mean(x) - np.mean(y)) / pooled)
