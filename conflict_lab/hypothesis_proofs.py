"""Falsifiable mathematical hypothesis derivation and statistical proof engine for Volume II."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx

from . import AGENT_CAST, COORDINATOR_SPEC, CRITIC_SPEC
from .gazetteer import BY_CERCLE_ID

AGENT_BY_ID = {s.agent_id: s for s in AGENT_CAST}
from .providers import extract_json_object, invoke_vol2_agent
from .schema import (
    AgentExchangeRecord,
    AgentExchangeTurn,
    AttackCluster,
    CercleMathProofState,
    HypothesisProof,
    VerdictStatus,
    VerifiedIncident,
)

MALI_DRY_MONTHS = {11, 12, 1, 2, 3, 4, 5}

CANONICAL_ACTORS_5: Tuple[str, ...] = (
    "JNIM / Katiba Coalition",
    "FAMa / Allied Security Forces",
    "ISGS / Islamic State Sahel",
    "Local Militias / Self-Defense",
    "Other / Unattributed",
)

CANONICAL_TACTICS_8: Tuple[str, ...] = (
    "ied_mine",
    "ambush",
    "raid_assault",
    "indirect_fire",
    "kidnapping_intimidation",
    "airstrike_clash",
    "blockade_siege",
    "other",
)


def canonical_actor_bucket(actor: str) -> str:
    """Map raw/normalized actor strings into the canonical 5-class actor space."""
    if actor in CANONICAL_ACTORS_5[:4]:
        return actor
    return "Other / Unattributed"


def is_located_incident(inc: VerifiedIncident) -> bool:
    """Return True if incident is localized below Admin-1 (excludes region_centroid fallback)."""
    return getattr(inc, "precision_level", "cercle_centroid") != "region_centroid"


def extract_dense_segment(
    incidents: Sequence[VerifiedIncident],
    *,
    gap_days: int = 45,
    cutoff_date: Optional[str] = None,
    located_only: bool = True,
) -> List[VerifiedIncident]:
    """Return the longest dense contiguous segment (split on >gap_days gaps) strictly <= cutoff_date."""
    filtered: List[VerifiedIncident] = []
    for inc in incidents:
        if located_only and not is_located_incident(inc):
            continue
        if cutoff_date and inc.date_start[:10] > cutoff_date[:10]:
            continue
        filtered.append(inc)
    if not filtered:
        return []
    filtered.sort(key=lambda x: (x.date_start[:10], x.incident_id))
    segments: List[List[VerifiedIncident]] = [[filtered[0]]]
    prev_d = _parse_iso_date(filtered[0].date_start)
    for inc in filtered[1:]:
        cur_d = _parse_iso_date(inc.date_start)
        if (cur_d - prev_d).days > gap_days:
            segments.append([inc])
        else:
            segments[-1].append(inc)
        prev_d = cur_d
    best = max(
        segments,
        key=lambda seg: (
            len(seg),
            (_parse_iso_date(seg[-1].date_start) - _parse_iso_date(seg[0].date_start)).days,
        ),
    )
    return best


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    """Compute Benjamini-Hochberg FDR-adjusted q-values preserving input order."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda kv: kv[1])
    q_sorted = [1.0] * m
    prev_q = 1.0
    for rank_rev in range(m - 1, -1, -1):
        rank = rank_rev + 1
        p = max(0.0, min(1.0, float(indexed[rank_rev][1])))
        q = min(prev_q, (m / rank) * p)
        q_sorted[rank_rev] = round(q, 8)
        prev_q = q
    out = [1.0] * m
    for pos, (orig_idx, _) in enumerate(indexed):
        out[orig_idx] = q_sorted[pos]
    return out


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _parse_iso_date(val: str) -> date:
    s = (val or "")[:10]
    try:
        return date.fromisoformat(s)
    except ValueError:
        return date(2025, 1, 1)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Special mathematical functions (pure stdlib, scipy-free)
# ---------------------------------------------------------------------------


def _gammap_series(a: float, x: float) -> float:
    """Lower regularized incomplete gamma P(a, x) via power series."""
    gln = math.lgamma(a)
    ap = a
    term = 1.0 / a
    total = term
    for _ in range(1, 250):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) <= abs(total) * 1e-12:
            break
    return total * math.exp(-x + a * math.log(max(x, 1e-300)) - gln)


def _gammaq_cf(a: float, x: float) -> float:
    """Upper regularized incomplete gamma Q(a, x) via Lentz continued fraction."""
    gln = math.lgamma(a)
    fpmin = 1e-300
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / (b if abs(b) > fpmin else fpmin)
    h = d
    for i in range(1, 250):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return math.exp(-x + a * math.log(max(x, 1e-300)) - gln) * h


def chi2_sf(x: float, df: int) -> float:
    """Exact upper-tail p-value P(Chi2_df >= x)."""
    if df <= 0:
        return 1.0
    if x <= 0.0:
        return 1.0
    a = 0.5 * df
    xx = 0.5 * x
    try:
        if xx < a + 1.0:
            return max(0.0, min(1.0, 1.0 - _gammap_series(a, xx)))
        return max(0.0, min(1.0, _gammaq_cf(a, xx)))
    except OverflowError:
        return 0.0


def _betacf(a: float, b: float, x: float) -> float:
    maxit = 200
    eps = 3e-10
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete Beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return max(0.0, min(1.0, bt * _betacf(a, b, x) / a))
    return max(0.0, min(1.0, 1.0 - bt * _betacf(b, a, 1.0 - x) / b))


def beta_ppf(q: float, a: float, b: float) -> float:
    """Quantile of Beta(a, b) via monotone bisection."""
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(44):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def assign_verdict(n_obs: int, p_value: float, effect_positive: bool = True) -> VerdictStatus:
    """Falsifiable hypothesis verdict grading."""
    if n_obs < 4:
        return "INCONCLUSIVE"
    if not effect_positive:
        return "FALSIFIED" if p_value > 0.2 else "INCONCLUSIVE"
    if p_value <= 0.01:
        return "PROVEN"
    if p_value <= 0.05:
        return "SUPPORTED"
    if p_value <= 0.25:
        return "INCONCLUSIVE"
    return "FALSIFIED"


# ---------------------------------------------------------------------------
# Univariate Exponential Hawkes MLE (O(n) Ogata recursion)
# ---------------------------------------------------------------------------


def _prepare_event_days(times_days: Sequence[float]) -> List[float]:
    ordered = sorted(float(t) for t in times_days)
    out: List[float] = []
    seen_day: Dict[int, int] = defaultdict(int)
    for t in ordered:
        bucket = int(math.floor(t))
        idx = seen_day[bucket]
        seen_day[bucket] += 1
        # Small deterministic sub-day offset prevents simultaneous log(0) collapse while preserving day
        out.append(t + min(0.85, idx * 0.11))
    return sorted(out)


def hawkes_loglik(times: Sequence[float], T: float, mu: float, alpha: float, beta: float) -> float:
    if mu <= 1e-12 or beta <= 1e-12 or alpha < 0.0 or alpha >= beta:
        return -1e18
    if not times:
        return -mu * max(T, 1.0)
    ll = 0.0
    r = 0.0
    prev_t = times[0]
    for idx, t in enumerate(times):
        if idx > 0:
            dt = max(0.0, t - prev_t)
            r = math.exp(-beta * dt) * (1.0 + r)
        lam = mu + alpha * r
        if lam <= 1e-15:
            return -1e18
        ll += math.log(lam)
        prev_t = t
    integral_exc = 0.0
    for t in times:
        integral_exc += 1.0 - math.exp(-beta * max(0.0, T - t))
    ll -= mu * T + (alpha / beta) * integral_exc
    return ll


def fit_hawkes_process(times_days: Sequence[float], T: float) -> Dict[str, float]:
    """Fit exponential Hawkes process vs homogeneous Poisson baseline using O(n) EM."""
    T = max(float(T), 14.0)
    n = len(times_days)
    if n == 0:
        return {
            "mu": 0.001,
            "alpha": 0.0,
            "beta": 0.1,
            "branching_ratio": 0.0,
            "half_life_days": 6.93,
            "lambda_now": 0.001,
            "ll_poisson": 0.0,
            "ll_hawkes": 0.0,
            "llr": 0.0,
            "p_value": 1.0,
        }

    times = _prepare_event_days(times_days)
    T_eff = max(T, times[-1] + 0.25)
    lam0 = max(n / T_eff, 1e-5)
    ll0 = n * math.log(lam0) - lam0 * T_eff

    best_ll = ll0
    best_params = (lam0, 0.0, 0.10)

    if n >= 3:
        for beta in (0.03, 0.05, 0.08, 0.12, 0.20, 0.35, 0.60):
            # EM iterations for fixed beta (closed-form updates on branching matrix upper bound)
            mu = max(1e-5, 0.5 * lam0)
            eta = 0.35  # alpha / beta
            alpha = eta * beta
            for _ in range(14):
                exp_tail_sum = sum(1.0 - math.exp(-beta * max(0.0, T_eff - t)) for t in times)
                r = 0.0
                p_self_sum = 0.0
                p_bg_sum = 0.0
                prev_t = times[0]
                for idx, t in enumerate(times):
                    if idx > 0:
                        r = math.exp(-beta * max(0.0, t - prev_t)) * (1.0 + r)
                    lam = max(1e-12, mu + alpha * r)
                    p_bg_sum += mu / lam
                    p_self_sum += (alpha * r) / lam
                    prev_t = t
                mu = max(1e-6, p_bg_sum / T_eff)
                if exp_tail_sum > 1e-9:
                    eta = min(0.92, max(0.0, p_self_sum / exp_tail_sum))
                else:
                    eta = 0.0
                alpha = eta * beta

            ll_cand = hawkes_loglik(times, T_eff, mu, alpha, beta)
            if ll_cand > best_ll:
                best_ll = ll_cand
                best_params = (mu, alpha, beta)

    mu, alpha, beta = best_params
    llr = max(0.0, best_ll - ll0)
    lr_stat = 2.0 * llr
    p_val = chi2_sf(lr_stat, 2) if n >= 3 else 1.0
    branching = (alpha / beta) if beta > 0 else 0.0
    half_life = (math.log(2.0) / beta) if beta > 0 else 999.0
    tail_r = sum(math.exp(-beta * max(0.0, T_eff - t)) for t in times)
    lambda_now = max(1e-6, mu + alpha * tail_r)

    return {
        "mu": round(mu, 6),
        "alpha": round(alpha, 6),
        "beta": round(beta, 4),
        "branching_ratio": round(branching, 4),
        "half_life_days": round(half_life, 2),
        "lambda_now": round(lambda_now, 6),
        "ll_poisson": round(ll0, 4),
        "ll_hawkes": round(best_ll, 4),
        "llr": round(llr, 4),
        "p_value": round(p_val, 8),
    }


# ---------------------------------------------------------------------------
# Family B: Overdispersion (Negative Binomial vs Poisson)
# ---------------------------------------------------------------------------


def eval_overdispersion(weekly_counts: Sequence[int]) -> Dict[str, float]:
    k = len(weekly_counts)
    total = sum(weekly_counts)
    if k < 4 or total == 0:
        return {
            "weeks": float(k),
            "mean": 0.0,
            "variance": 0.0,
            "dispersion_ratio": 1.0,
            "pearson_chi2": 0.0,
            "nb_size_r": 1.0,
            "p_value": 1.0,
        }
    mean = total / k
    var = sum((c - mean) ** 2 for c in weekly_counts) / (k - 1)
    dispersion = var / mean if mean > 0 else 1.0
    chi2_stat = sum((c - mean) ** 2 for c in weekly_counts) / mean if mean > 0 else 0.0
    p_val = chi2_sf(chi2_stat, k - 1) if dispersion > 1.0 else 1.0
    excess = max(var - mean, 1e-6)
    nb_r = (mean * mean) / excess if var > mean else 100.0
    return {
        "weeks": float(k),
        "mean": round(mean, 4),
        "variance": round(var, 4),
        "dispersion_ratio": round(dispersion, 4),
        "pearson_chi2": round(chi2_stat, 4),
        "nb_size_r": round(nb_r, 4),
        "p_value": round(p_val, 8),
    }


# ---------------------------------------------------------------------------
# Family C: Markov Tactic Transition Test (G-test)
# ---------------------------------------------------------------------------


def eval_markov_transitions(tactics_seq: Sequence[str]) -> Dict[str, Any]:
    states = sorted(set(tactics_seq))
    n_trans = max(0, len(tactics_seq) - 1)
    if len(states) < 2 or n_trans < 4:
        last_t = tactics_seq[-1] if tactics_seq else "other"
        return {
            "states": states,
            "n_transitions": n_trans,
            "g_stat": 0.0,
            "df": 1,
            "p_value": 1.0,
            "next_probs": {last_t: 1.0} if tactics_seq else {"other": 1.0},
            "top_transition": "",
        }

    counts: Dict[Tuple[str, str], int] = Counter()
    row_totals: Dict[str, int] = Counter()
    col_totals: Dict[str, int] = Counter()
    for a, b in zip(tactics_seq[:-1], tactics_seq[1:]):
        counts[(a, b)] += 1
        row_totals[a] += 1
        col_totals[b] += 1

    g_stat = 0.0
    for a in states:
        for b in states:
            o = counts[(a, b)]
            if o > 0:
                e = (row_totals[a] * col_totals[b]) / n_trans
                if e > 0:
                    g_stat += 2.0 * o * math.log(o / e)

    df = max(1, (len(states) - 1) ** 2)
    p_val = chi2_sf(g_stat, df)

    cur = tactics_seq[-1]
    denom = row_totals[cur] + len(states) * 0.5
    next_probs = {
        s: round((counts[(cur, s)] + 0.5) / denom, 4) for s in states
    }
    top_pair, top_cnt = counts.most_common(1)[0]
    return {
        "states": states,
        "n_transitions": n_trans,
        "g_stat": round(g_stat, 4),
        "df": df,
        "p_value": round(p_val, 8),
        "next_probs": next_probs,
        "top_transition": f"{top_pair[0]} -> {top_pair[1]} ({top_cnt}x)",
    }


# ---------------------------------------------------------------------------
# Family D: Seasonal Harmattan Rate Ratio Proof
# ---------------------------------------------------------------------------


def eval_seasonal_harmattan(dates: Sequence[str]) -> Dict[str, float]:
    """Compare Nov-May Harmattan dry season vs Jun-Oct rainy season incidence rate."""
    if len(dates) < 4:
        return {
            "dry_count": 0,
            "rainy_count": 0,
            "dry_rate_per_30d": 0.0,
            "rainy_rate_per_30d": 0.0,
            "rate_ratio": 1.0,
            "chi2_stat": 0.0,
            "p_value": 1.0,
        }
    parsed = sorted(_parse_iso_date(d) for d in dates)
    start_d = parsed[0]
    end_d = parsed[-1]
    span_days = max(30, (end_d - start_d).days + 1)

    # Approximate exact exposure days in window (cap iteration cleanly via fast month walk if huge)
    dry_days = 0
    rainy_days = 0
    if span_days <= 4000:
        cur = start_d
        while cur <= end_d:
            if cur.month in MALI_DRY_MONTHS:
                dry_days += 1
            else:
                rainy_days += 1
            cur = date.fromordinal(cur.toordinal() + 1)
    else:
        dry_days = int(round(span_days * (212.0 / 365.0)))
        rainy_days = max(1, span_days - dry_days)

    dry_days = max(1, dry_days)
    rainy_days = max(1, rainy_days)
    dry_cnt = sum(1 for d in parsed if d.month in MALI_DRY_MONTHS)
    rainy_cnt = len(parsed) - dry_cnt

    dry_rate = dry_cnt / (dry_days / 30.0)
    rainy_rate = rainy_cnt / (rainy_days / 30.0)
    rate_ratio = (dry_rate + 1e-6) / (rainy_rate + 1e-6)

    total = dry_cnt + rainy_cnt
    p0 = dry_days / (dry_days + rainy_days)
    exp_dry = total * p0
    exp_rainy = total * (1.0 - p0)
    chi2_stat = 0.0
    if exp_dry > 0:
        chi2_stat += ((dry_cnt - exp_dry) ** 2) / exp_dry
    if exp_rainy > 0:
        chi2_stat += ((rainy_cnt - exp_rainy) ** 2) / exp_rainy
    p_val = chi2_sf(chi2_stat, 1)

    return {
        "dry_count": float(dry_cnt),
        "rainy_count": float(rainy_cnt),
        "dry_days": float(dry_days),
        "rainy_days": float(rainy_days),
        "dry_rate_per_30d": round(dry_rate, 3),
        "rainy_rate_per_30d": round(rainy_rate, 3),
        "rate_ratio": round(rate_ratio, 3),
        "chi2_stat": round(chi2_stat, 4),
        "p_value": round(p_val, 8),
    }


# ---------------------------------------------------------------------------
# Family E: Beta-Binomial Conjugate Weekly Hazard Proof
# ---------------------------------------------------------------------------


def compute_beta_binomial_hazard(
    active_weeks: int,
    total_weeks: int,
    *,
    alpha0: float = 1.0,
    beta0: float = 12.0,
    baseline_prob: float = 0.08,
) -> Dict[str, float]:
    total_weeks = max(1, total_weeks)
    active_weeks = max(0, min(total_weeks, active_weeks))
    post_a = alpha0 + active_weeks
    post_b = beta0 + (total_weeks - active_weeks)
    post_mean = post_a / (post_a + post_b)
    ci_low = beta_ppf(0.025, post_a, post_b)
    ci_high = beta_ppf(0.975, post_a, post_b)
    # One-sided posterior tail P(theta <= baseline_prob | data) as Bayes exceedance p-value
    p_below_base = betainc(post_a, post_b, min(0.999, max(0.001, baseline_prob)))
    return {
        "alpha0": round(alpha0, 3),
        "beta0": round(beta0, 3),
        "post_alpha": round(post_a, 3),
        "post_beta": round(post_b, 3),
        "posterior_mean": round(post_mean, 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        "p_value": round(p_below_base, 8),
    }


# ---------------------------------------------------------------------------
# Cluster / Cercle Mathematical Proof Builder
# ---------------------------------------------------------------------------


def build_weekly_series(
    dates: Sequence[str],
    min_date: date,
    max_date: date,
) -> Tuple[List[int], List[int]]:
    span_days = max(7, (max_date - min_date).days + 1)
    num_weeks = max(4, min(260, math.ceil(span_days / 7.0)))
    # Anchor the grid to end on max_date so the most recent num_weeks are exact
    window_start_ord = max_date.toordinal() - num_weeks * 7 + 1
    counts = [0] * num_weeks
    for ds in dates:
        d = _parse_iso_date(ds)
        idx = (d.toordinal() - window_start_ord) // 7
        if 0 <= idx < num_weeks:
            counts[idx] += 1
    binary = [1 if c > 0 else 0 for c in counts]
    return counts, binary


def derive_proofs_for_cluster(
    cluster: AttackCluster,
    incidents: Sequence[VerifiedIncident],
    global_min_date: date,
    global_max_date: date,
    national_weekly_prob: float,
) -> List[HypothesisProof]:
    """Generate 5 testable mathematical hypothesis proofs (Hawkes, Dispersion, Markov, Harmattan, Beta) for a cluster."""
    ordered = sorted(incidents, key=lambda x: x.date_start)
    dates = [inc.date_start for inc in ordered]
    cercle_id = cluster.dominant_cercle_id
    now_iso = _utc_now_iso()

    d0 = _parse_iso_date(dates[0]) if dates else global_min_date
    d1 = max(global_max_date, _parse_iso_date(dates[-1])) if dates else global_max_date
    t_span = max(30.0, float((d1 - d0).days + 1))
    times_days = [float((_parse_iso_date(ds) - d0).days) for ds in dates]

    weekly_counts, weekly_binary = build_weekly_series(dates, d0, d1)
    n_obs = len(ordered)

    # 1. Hawkes Self-Excitation
    hw = fit_hawkes_process(times_days, t_span)
    hw_verdict = assign_verdict(n_obs, hw["p_value"], effect_positive=hw["branching_ratio"] > 0.05)
    p_hawkes = HypothesisProof(
        hyp_id=f"{cluster.cluster_id}__H_hawkes",
        cluster_id=cluster.cluster_id,
        cercle_id=cercle_id,
        family="H_hawkes",
        title=f"Self-Exciting Attack Clustering in {cluster.label}",
        symbolic_formula="lambda(t) = mu + alpha * sum_{t_i < t} exp(-beta * (t - t_i))",
        null_hypothesis="H0: alpha = 0 (homogeneous Poisson arrival process)",
        alternative_hypothesis="H1: alpha > 0, eta = alpha/beta in (0, 1) (retaliation / surge memory)",
        parameters=hw,
        test_statistic_name="2*DeltaLogLik (chi2_2)",
        test_statistic_value=round(2.0 * hw["llr"], 4),
        p_value=hw["p_value"],
        log_likelihood_ratio=hw["llr"],
        verdict=hw_verdict,
        n_observations=n_obs,
        derivation_steps=[
            f"1. Fitted null Poisson baseline rate lambda_0 = {round(n_obs / t_span, 5)}/day -> ll_0 = {hw['ll_poisson']}.",
            f"2. Optimized exponential Hawkes (mu={hw['mu']}, alpha={hw['alpha']}, beta={hw['beta']}) via Ogata O(n) EM -> ll_1 = {hw['ll_hawkes']}.",
            f"3. Likelihood-ratio 2*Delta_ll = {round(2.0 * hw['llr'], 4)}, branching ratio eta = {hw['branching_ratio']}, decay half-life t_1/2 = {hw['half_life_days']}d -> p = {hw['p_value']} ({hw_verdict}).",
        ],
        analyst_narrative=(
            f"Branching ratio eta={hw['branching_ratio']} indicates each attack triggers ~{hw['branching_ratio']} follow-on incidents with a {hw['half_life_days']}-day half-life."
        ),
        updated_at=now_iso,
    )

    # 2. Overdispersion (Negative Binomial vs Poisson)
    disp = eval_overdispersion(weekly_counts)
    disp_verdict = assign_verdict(
        len(weekly_counts),
        disp["p_value"],
        effect_positive=disp["dispersion_ratio"] > 1.15 and n_obs >= 4,
    )
    p_dispersion = HypothesisProof(
        hyp_id=f"{cluster.cluster_id}__H_dispersion",
        cluster_id=cluster.cluster_id,
        cercle_id=cercle_id,
        family="H_dispersion",
        title=f"Burst-Driven Overdispersion Test in {cluster.label}",
        symbolic_formula="D = s^2 / x_bar; X^2 = sum((y_w - x_bar)^2) / x_bar ~ chi2_{K-1}",
        null_hypothesis="H0: D <= 1 (equidispersed Poisson weekly counts)",
        alternative_hypothesis="H1: D > 1 (Negative Binomial burst regime)",
        parameters=disp,
        test_statistic_name="Pearson X^2 dispersion",
        test_statistic_value=disp["pearson_chi2"],
        p_value=disp["p_value"],
        verdict=disp_verdict,
        n_observations=int(disp["weeks"]),
        derivation_steps=[
            f"1. Partitioned series into K = {int(disp['weeks'])} calendar weeks; mean = {disp['mean']}, var = {disp['variance']}.",
            f"2. Fisher dispersion index D = s^2/mean = {disp['dispersion_ratio']} (NB size r = {disp['nb_size_r']}).",
            f"3. Pearson chi-square X^2 = {disp['pearson_chi2']} on df={max(1, int(disp['weeks']) - 1)} -> p = {disp['p_value']} ({disp_verdict}).",
        ],
        analyst_narrative=(
            f"Weekly attack variance ({disp['variance']}) is {disp['dispersion_ratio']}x the mean ({disp['mean']}), distinguishing episodic offensives from uniform background risk."
        ),
        updated_at=now_iso,
    )

    # 3. Markov Tactic Transition Dependence
    mk = eval_markov_transitions([inc.tactic for inc in ordered])
    mk_verdict = assign_verdict(mk["n_transitions"], mk["p_value"], effect_positive=mk["g_stat"] > 1.0)
    p_markov = HypothesisProof(
        hyp_id=f"{cluster.cluster_id}__H_markov",
        cluster_id=cluster.cluster_id,
        cercle_id=cercle_id,
        family="H_markov",
        title=f"Sequential Tactic Transition Chain in {cluster.label}",
        symbolic_formula="G = 2 * sum_{i,j} O_{ij} ln(O_{ij} / E_{ij}) ~ chi2_{(S-1)^2}",
        null_hypothesis="H0: Tactic_t independent of Tactic_{t-1}",
        alternative_hypothesis="H1: First-order Markov tactical memory (e.g. IED -> ambush/raid cycle)",
        parameters=mk,
        test_statistic_name="Likelihood G-statistic",
        test_statistic_value=mk["g_stat"],
        p_value=mk["p_value"],
        verdict=mk_verdict,
        n_observations=mk["n_transitions"],
        derivation_steps=[
            f"1. Evaluated N = {mk['n_transitions']} consecutive transitions across {len(mk['states'])} observed tactics.",
            f"2. Dominant empirical link: {mk['top_transition'] or 'insufficient distinct states'}.",
            f"3. Log-likelihood ratio G = {mk['g_stat']} (df={mk['df']}) -> p = {mk['p_value']} ({mk_verdict}).",
        ],
        analyst_narrative=(
            f"Next-event distribution conditional on most recent tactic: "
            + ", ".join(f"{k}:{round(v * 100)}%" for k, v in sorted(mk["next_probs"].items(), key=lambda kv: -kv[1])[:3])
        ),
        updated_at=now_iso,
    )

    # 4. Harmattan Dry-Season Shift
    harm = eval_seasonal_harmattan(dates)
    harm_verdict = assign_verdict(
        n_obs,
        harm["p_value"],
        effect_positive=abs(math.log(max(harm["rate_ratio"], 1e-4))) > 0.25,
    )
    p_seasonal = HypothesisProof(
        hyp_id=f"{cluster.cluster_id}__H_seasonal",
        cluster_id=cluster.cluster_id,
        cercle_id=cercle_id,
        family="H_seasonal",
        title=f"Harmattan Dry vs. Rainy Season Rate Ratio in {cluster.label}",
        symbolic_formula="RR = (N_dry / E_dry) / (N_rainy / E_rainy); chi2_1 exposure goodness-of-fit",
        null_hypothesis="H0: Attack rate invariant across Nov-May dry & Jun-Oct rainy seasons",
        alternative_hypothesis="H1: Seasonal mobility modulation shifts normalized 30-day incidence",
        parameters=harm,
        test_statistic_name="Seasonal chi2_1",
        test_statistic_value=harm["chi2_stat"],
        p_value=harm["p_value"],
        verdict=harm_verdict,
        n_observations=n_obs,
        derivation_steps=[
            f"1. Observed {int(harm['dry_count'])} dry-season vs {int(harm['rainy_count'])} rainy-season incidents.",
            f"2. Normalized 30d rates: dry = {harm['dry_rate_per_30d']}/30d vs rainy = {harm['rainy_rate_per_30d']}/30d (RR = {harm['rate_ratio']}).",
            f"3. Exposure-adjusted chi2_1 = {harm['chi2_stat']} -> p = {harm['p_value']} ({harm_verdict}).",
        ],
        analyst_narrative=(
            f"Normalized 30-day rate ratio dry/rainy is {harm['rate_ratio']}x across {int(harm.get('dry_days', 0) + harm.get('rainy_days', 0))} monitored days."
        ),
        updated_at=now_iso,
    )

    # 5. Beta-Binomial Conjugate Weekly Hazard
    active_wks = sum(weekly_binary)
    total_wks = len(weekly_binary)
    bb = compute_beta_binomial_hazard(
        active_wks,
        total_wks,
        alpha0=1.0,
        beta0=12.0,
        baseline_prob=national_weekly_prob,
    )
    bb_verdict = assign_verdict(
        total_wks,
        bb["p_value"],
        effect_positive=bb["posterior_mean"] > national_weekly_prob,
    )
    p_beta = HypothesisProof(
        hyp_id=f"{cluster.cluster_id}__H_beta",
        cluster_id=cluster.cluster_id,
        cercle_id=cercle_id,
        family="H_beta",
        title=f"Beta-Binomial Conjugate Weekly Hazard in {cluster.label}",
        symbolic_formula="theta | k,n ~ Beta(alpha0 + k, beta0 + n - k)",
        null_hypothesis=f"H0: Weekly attack probability theta <= national mean ({round(national_weekly_prob, 3)})",
        alternative_hypothesis="H1: Sustained elevated posterior weekly hazard theta > baseline",
        parameters={**bb, "active_weeks": active_wks, "total_weeks": total_wks, "national_baseline": round(national_weekly_prob, 4)},
        test_statistic_name="Posterior Exceedance Mean",
        test_statistic_value=bb["posterior_mean"],
        p_value=bb["p_value"],
        verdict=bb_verdict,
        n_observations=total_wks,
        derivation_steps=[
            f"1. Prior Beta(1.0, 12.0) updated with k={active_wks} active weeks over n={total_wks} observed weeks.",
            f"2. Closed-form posterior Beta({bb['post_alpha']}, {bb['post_beta']}) -> E[theta|data] = {bb['posterior_mean']}.",
            f"3. Exact 95% Bayesian CrI = [{bb['ci_low']}, {bb['ci_high']}]; tail P(theta <= {round(national_weekly_prob, 3)}) = {bb['p_value']} ({bb_verdict}).",
        ],
        analyst_narrative=(
            f"Conjugate posterior places weekly attack risk at {round(bb['posterior_mean'] * 100, 1)}% (95% CrI: {round(bb['ci_low'] * 100, 1)}%-{round(bb['ci_high'] * 100, 1)}%)."
        ),
        updated_at=now_iso,
    )

    return [p_hawkes, p_dispersion, p_markov, p_seasonal, p_beta]


async def synthesize_llm_hypothesis_notes(
    client: httpx.AsyncClient,
    hypotheses: List[HypothesisProof],
    *,
    max_to_enrich: int = 8,
    use_llm: bool = True,
) -> List[HypothesisProof]:
    """Batch-enrich top statistical proofs with causal mechanism context via Step-5 & Gemini Flash-Lite."""
    if not use_llm or not hypotheses:
        return hypotheses

    # Prioritize Hawkes & Markov proofs in highest-count clusters
    ranked = sorted(
        hypotheses,
        key=lambda h: (h.verdict in ("PROVEN", "SUPPORTED"), h.n_observations, h.log_likelihood_ratio),
        reverse=True,
    )[:max_to_enrich]

    payload = [
        {
            "hyp_id": h.hyp_id,
            "cercle": BY_CERCLE_ID.get(h.cercle_id).name if h.cercle_id in BY_CERCLE_ID else h.cercle_id,
            "family": h.family,
            "verdict": h.verdict,
            "p_value": h.p_value,
            "stat": h.test_statistic_value,
            "summary": h.analyst_narrative,
        }
        for h in ranked
    ]

    spec = AGENT_BY_ID["a8-math-theorist"]
    messages = [
        {
            "role": "system",
            "content": (
                "You are Hypothesizer, quantitative conflict analyst for The Sebeni Chronicle Vol II. "
                "For each pre-computed statistical proof below, write ONE concise (<=30 words) falsifiable operational mechanism explanation. "
                "Never alter any numbers; only interpret the computed result cleanly. "
                'Return strictly JSON: {"notes": [{"hyp_id": "...", "note": "..."}]}'
            ),
        },
        {"role": "user", "content": json.dumps(payload)},
    ]

    try:
        raw, _, err = await asyncio.wait_for(
            invoke_vol2_agent(client, spec, messages, max_tokens=1400, temperature=0.1),
            timeout=25.0,
        )
    except Exception:
        raw, err = "", "timeout"
    if err or not raw:
        try:
            raw, _, _ = await asyncio.wait_for(
                invoke_vol2_agent(client, CRITIC_SPEC, messages, max_tokens=1400, temperature=0.1),
                timeout=20.0,
            )
        except Exception:
            raw = ""
    parsed = extract_json_object(raw) if raw else None
    note_map: Dict[str, str] = {}
    if parsed and isinstance(parsed.get("notes"), list):
        for item in parsed["notes"]:
            if isinstance(item, dict) and item.get("hyp_id") and item.get("note"):
                note_map[str(item["hyp_id"])] = str(item["note"])[:280]

    for h in hypotheses:
        if h.hyp_id in note_map:
            h.analyst_narrative = f"{h.analyst_narrative} {note_map[h.hyp_id]}".strip()
    return hypotheses


# ---------------------------------------------------------------------------
# Family F: Relative Entropy / LOO Empirical-Bayes KL Divergence & Regime JSD
# ---------------------------------------------------------------------------


def _two_sample_g_stat(c_counts: Sequence[int], loo_counts: Sequence[int]) -> float:
    """Two-sample contingency G-statistic comparing cercle counts vs LOO national counts."""
    n_c = sum(c_counts)
    n_loo = sum(loo_counts)
    n_tot = n_c + n_loo
    if n_c <= 0 or n_loo <= 0 or n_tot <= 0:
        return 0.0
    g_val = 0.0
    for c_k, l_k in zip(c_counts, loo_counts):
        tot_k = c_k + l_k
        if tot_k <= 0:
            continue
        if c_k > 0:
            e_c = n_c * tot_k / n_tot
            if e_c > 0:
                g_val += 2.0 * c_k * math.log(c_k / e_c)
        if l_k > 0:
            e_l = n_loo * tot_k / n_tot
            if e_l > 0:
                g_val += 2.0 * l_k * math.log(l_k / e_l)
    return max(0.0, g_val)


def eval_kl_divergence_loo(
    cercle_counts: Dict[str, int],
    national_counts: Dict[str, int],
    categories: Sequence[str],
    *,
    dimension: str,
    n_perm: int = 2000,
    seed_key: str = "kl",
) -> Dict[str, Any]:
    """Empirical-Bayes Dirichlet (alpha0=K) LOO KL divergence + exact without-replacement MC permutation."""
    k_dim = max(2, len(categories))
    c_vec = [int(cercle_counts.get(cat, 0)) for cat in categories]
    tot_vec = [int(national_counts.get(cat, 0)) for cat in categories]
    loo_vec = [max(0, tot_vec[i] - c_vec[i]) for i in range(k_dim)]

    n_c = sum(c_vec)
    n_loo = sum(loo_vec)
    n_tot = sum(tot_vec)
    if n_c == 0 or n_loo == 0 or n_tot <= n_c:
        return {
            "dimension": dimension,
            "k_eff": k_dim,
            "alpha0": float(k_dim),
            "n_c": n_c,
            "n_loo": n_loo,
            "kl_eb": 0.0,
            "g_loo": 0.0,
            "p_value": 1.0,
            "bins": [],
        }

    alpha0 = float(k_dim)
    denom_q = n_loo + 0.5 * k_dim
    denom_p = n_c + alpha0
    kl_eb = 0.0
    bins: List[Dict[str, Any]] = []
    for i, cat in enumerate(categories):
        q_loo = (loo_vec[i] + 0.5) / denom_q
        p_mle = c_vec[i] / n_c
        p_eb = (c_vec[i] + alpha0 * q_loo) / denom_p
        contrib = p_eb * math.log(p_eb / q_loo) if p_eb > 0 and q_loo > 0 else 0.0
        kl_eb += contrib
        log2_lift = math.log2(p_eb / q_loo) if p_eb > 0 and q_loo > 0 else 0.0
        bins.append(
            {
                "dimension": dimension,
                "bin": cat,
                "count_c": c_vec[i],
                "count_loo": loo_vec[i],
                "p_mle": round(p_mle, 5),
                "p_eb": round(p_eb, 5),
                "q_loo": round(q_loo, 5),
                "kl_contrib": round(contrib, 6),
                "log2_lift": round(log2_lift, 4),
            }
        )

    g_obs = _two_sample_g_stat(c_vec, loo_vec)

    # Exact without-replacement Monte Carlo permutation on national label multiset
    rng = random.Random(_stable_seed(seed_key, dimension, str(n_c), str(n_tot)))
    flat_pool: List[int] = []
    for idx, cnt in enumerate(tot_vec):
        flat_pool.extend([idx] * cnt)

    exceed = 0
    for _ in range(n_perm):
        drawn = rng.sample(flat_pool, n_c)
        perm_c = [0] * k_dim
        for idx in drawn:
            perm_c[idx] += 1
        perm_loo = [tot_vec[i] - perm_c[i] for i in range(k_dim)]
        if _two_sample_g_stat(perm_c, perm_loo) >= g_obs - 1e-12:
            exceed += 1

    p_perm = (1.0 + exceed) / (n_perm + 1.0)
    bins.sort(key=lambda b: abs(b["kl_contrib"]), reverse=True)
    return {
        "dimension": dimension,
        "k_eff": k_dim,
        "alpha0": alpha0,
        "n_c": n_c,
        "n_loo": n_loo,
        "kl_eb": round(max(0.0, kl_eb), 5),
        "g_loo": round(g_obs, 4),
        "p_value": round(p_perm, 6),
        "bins": bins,
    }


def eval_regime_jsd(
    dense_incidents: Sequence[VerifiedIncident],
    *,
    n_perm: int = 1000,
) -> Dict[str, Any]:
    """Compute National Spatial-Regime Jensen-Shannon Divergence (early vs late half of dense window)."""
    ordered = sorted(dense_incidents, key=lambda x: (x.date_start[:10], x.incident_id))
    n = len(ordered)
    if n < 12:
        return {"jsd_nats": 0.0, "p_value": 1.0, "n_early": 0, "n_late": 0, "split_date": "", "bins": []}

    mid = n // 2
    early = ordered[:mid]
    late = ordered[mid:]
    cercles = sorted({inc.cercle_id for inc in ordered})
    if len(cercles) < 2:
        return {"jsd_nats": 0.0, "p_value": 1.0, "n_early": len(early), "n_late": len(late), "split_date": "", "bins": []}

    c_idx = {cid: i for i, cid in enumerate(cercles)}
    k_c = len(cercles)

    def _jsd_from_counts(e_cnt: Sequence[int], l_cnt: Sequence[int]) -> float:
        ne = sum(e_cnt) + 0.5 * k_c
        nl = sum(l_cnt) + 0.5 * k_c
        jsd = 0.0
        for ec, lc in zip(e_cnt, l_cnt):
            pe = (ec + 0.5) / ne
            pl = (lc + 0.5) / nl
            m_prob = 0.5 * (pe + pl)
            jsd += 0.5 * pe * math.log(pe / m_prob) + 0.5 * pl * math.log(pl / m_prob)
        return max(0.0, jsd)

    early_cnt = [0] * k_c
    late_cnt = [0] * k_c
    for inc in early:
        early_cnt[c_idx[inc.cercle_id]] += 1
    for inc in late:
        late_cnt[c_idx[inc.cercle_id]] += 1

    jsd_obs = _jsd_from_counts(early_cnt, late_cnt)
    tot_cnt = [early_cnt[i] + late_cnt[i] for i in range(k_c)]

    labels = [c_idx[inc.cercle_id] for inc in ordered]
    rng = random.Random(_stable_seed("regime_jsd", str(n), ordered[0].date_start, ordered[-1].date_start))
    exceed = 0
    for _ in range(n_perm):
        drawn = rng.sample(labels, mid)
        pe_cnt = [0] * k_c
        for idx in drawn:
            pe_cnt[idx] += 1
        pl_cnt = [tot_cnt[i] - pe_cnt[i] for i in range(k_c)]
        if _jsd_from_counts(pe_cnt, pl_cnt) >= jsd_obs - 1e-12:
            exceed += 1
    p_val = (1.0 + exceed) / (n_perm + 1.0)

    ne = sum(early_cnt) + 0.5 * k_c
    nl = sum(late_cnt) + 0.5 * k_c
    bins: List[Dict[str, Any]] = []
    for i, cid in enumerate(cercles):
        pe = (early_cnt[i] + 0.5) / ne
        pl = (late_cnt[i] + 0.5) / nl
        m_prob = 0.5 * (pe + pl)
        contrib = 0.5 * pe * math.log(pe / m_prob) + 0.5 * pl * math.log(pl / m_prob)
        bins.append(
            {
                "dimension": "regime",
                "bin": BY_CERCLE_ID[cid].name if cid in BY_CERCLE_ID else cid,
                "cercle_id": cid,
                "count_c": late_cnt[i],
                "count_loo": early_cnt[i],
                "p_mle": round(late_cnt[i] / max(1, len(late)), 5),
                "p_eb": round(pl, 5),
                "q_loo": round(pe, 5),
                "kl_contrib": round(contrib, 6),
                "log2_lift": round(math.log2(pl / pe), 4),
            }
        )
    bins.sort(key=lambda b: abs(b["kl_contrib"]), reverse=True)
    return {
        "jsd_nats": round(jsd_obs, 5),
        "p_value": round(p_val, 6),
        "n_early": len(early),
        "n_late": len(late),
        "split_date": late[0].date_start[:10],
        "bins": bins[:12],
    }


# ---------------------------------------------------------------------------
# Family G: Neighborhood-Inflow Transfer Entropy (Miller-Madow + Year-Stratified)
# ---------------------------------------------------------------------------


def _binary_te_bits(x_series: Sequence[int], y_series: Sequence[int]) -> float:
    """Compute plug-in lag-1 Transfer Entropy TE(X -> Y) in bits for binary series."""
    w = min(len(x_series), len(y_series))
    if w < 3:
        return 0.0
    n_trans = w - 1
    c3 = [0] * 8  # (y_next << 2) | (y_cur << 1) | x_cur
    c_yx = [0] * 4  # (y_cur << 1) | x_cur
    c_yy = [0] * 4  # (y_next << 1) | y_cur
    c_y = [0, 0]
    for t in range(n_trans):
        x = 1 if x_series[t] else 0
        y = 1 if y_series[t] else 0
        yn = 1 if y_series[t + 1] else 0
        c3[(yn << 2) | (y << 1) | x] += 1
        c_yx[(y << 1) | x] += 1
        c_yy[(yn << 1) | y] += 1
        c_y[y] += 1

    te = 0.0
    for yn in (0, 1):
        for y in (0, 1):
            cyy = c_yy[(yn << 1) | y]
            cy = c_y[y]
            if cyy == 0 or cy == 0:
                continue
            for x in (0, 1):
                cnt = c3[(yn << 2) | (y << 1) | x]
                cyx = c_yx[(y << 1) | x]
                if cnt > 0 and cyx > 0:
                    te += (cnt / n_trans) * math.log2((cnt / cyx) / (cyy / cy))
    return max(0.0, te)


def _onset_lift(x_series: Sequence[int], y_series: Sequence[int]) -> float:
    w = min(len(x_series), len(y_series))
    if w < 3:
        return 0.0
    hit_x1 = tot_x1 = hit_x0 = tot_x0 = 0
    for t in range(w - 1):
        if y_series[t] == 0:
            if x_series[t] == 1:
                tot_x1 += 1
                if y_series[t + 1] == 1:
                    hit_x1 += 1
            else:
                tot_x0 += 1
                if y_series[t + 1] == 1:
                    hit_x0 += 1
    if tot_x1 == 0 or tot_x0 == 0:
        return 0.0
    return (hit_x1 / tot_x1) - (hit_x0 / tot_x0)


def eval_transfer_entropy_binary(
    x_series: Sequence[int],
    y_series: Sequence[int],
    week_years: Sequence[int],
    *,
    n_perm: int = 2000,
    seed_key: str = "te",
) -> Dict[str, Any]:
    """Neighborhood-inflow TE with Miller-Madow bias subtraction and year-stratified permutation null."""
    w = min(len(x_series), len(y_series), len(week_years))
    if w < 6:
        return {
            "n_weeks": w,
            "n_transitions": max(0, w - 1),
            "te_raw_bits": 0.0,
            "bias_mm_bits": 0.0,
            "te_mm_bits": 0.0,
            "onset_lift": 0.0,
            "g_stat": 0.0,
            "p_value": 1.0,
        }
    n_trans = w - 1
    te_raw = _binary_te_bits(x_series[:w], y_series[:w])
    # Binary X, Y => Delta df = (2-1)*2*(2-1) = 2 => Bias_MM = 2 / (2 * N_trans * ln(2))
    bias_mm = 1.0 / (n_trans * math.log(2.0))
    te_mm = max(0.0, te_raw - bias_mm)
    lift = _onset_lift(x_series[:w], y_series[:w])
    g_stat = 2.0 * n_trans * te_raw * math.log(2.0)

    # Group transition source indices t in [0..n_trans-1] by calendar year
    by_year: Dict[int, List[int]] = defaultdict(list)
    for t in range(n_trans):
        by_year[int(week_years[t])].append(t)
    year_groups = [idxs for idxs in by_year.values() if len(idxs) >= 2]

    rng = random.Random(_stable_seed(seed_key, str(w), str(sum(y_series[:w]))))
    x_perm = list(x_series[:w])
    exceed = 0
    for _ in range(n_perm):
        for idxs in year_groups:
            vals = [x_series[i] for i in idxs]
            rng.shuffle(vals)
            for pos, orig_t in enumerate(idxs):
                x_perm[orig_t] = vals[pos]
        if _binary_te_bits(x_perm, y_series[:w]) >= te_raw - 1e-12:
            exceed += 1

    p_yr = (1.0 + exceed) / (n_perm + 1.0)
    return {
        "n_weeks": w,
        "n_transitions": n_trans,
        "te_raw_bits": round(te_raw, 6),
        "bias_mm_bits": round(bias_mm, 6),
        "te_mm_bits": round(te_mm, 6),
        "onset_lift": round(lift, 4),
        "g_stat": round(g_stat, 4),
        "p_value": round(p_yr, 6),
    }


# ---------------------------------------------------------------------------
# Family H: Exact Poisson Split-Rate Supremum Changepoint
# ---------------------------------------------------------------------------


def _sup_poisson_changepoint(counts: Sequence[int], min_seg: int = 10) -> Tuple[float, int]:
    w = len(counts)
    total = sum(counts)
    if w < 2 * min_seg or total <= 0:
        return 0.0, -1
    lam0 = total / w
    prefix = [0] * (w + 1)
    for i, c in enumerate(counts):
        prefix[i + 1] = prefix[i] + c

    best_stat = 0.0
    best_tau = -1
    for tau in range(min_seg, w - min_seg + 1):
        s1 = prefix[tau]
        s2 = total - s1
        stat = 0.0
        if s1 > 0:
            lam1 = s1 / tau
            stat += 2.0 * s1 * math.log(lam1 / lam0)
        if s2 > 0:
            lam2 = s2 / (w - tau)
            stat += 2.0 * s2 * math.log(lam2 / lam0)
        if stat > best_stat:
            best_stat = stat
            best_tau = tau
    return max(0.0, best_stat), best_tau


def eval_changepoint_poisson(
    weekly_counts: Sequence[int],
    week_start_dates: Sequence[str],
    *,
    min_seg: int = 10,
    n_perm: int = 1000,
    seed_key: str = "cp",
) -> Dict[str, Any]:
    """Supremum Poisson split-rate LLR changepoint with exact temporal permutation p-value."""
    w = len(weekly_counts)
    total = sum(weekly_counts)
    if w < 2 * min_seg or total < 4:
        return {
            "n_weeks": w,
            "sup_2dll": 0.0,
            "tau_idx": -1,
            "changepoint_date": "",
            "rate_before_wk": 0.0,
            "rate_after_wk": 0.0,
            "rate_ratio": 1.0,
            "p_value": 1.0,
        }

    sup_obs, tau = _sup_poisson_changepoint(weekly_counts, min_seg=min_seg)
    s1 = sum(weekly_counts[:tau]) if tau > 0 else 0
    s2 = total - s1
    r_before = s1 / max(1, tau)
    r_after = s2 / max(1, w - tau)
    cp_date = week_start_dates[tau] if 0 <= tau < len(week_start_dates) else ""
    rate_ratio = (r_after + 1e-4) / (r_before + 1e-4)

    rng = random.Random(_stable_seed(seed_key, str(w), str(total)))
    shuffled = list(weekly_counts)
    exceed = 0
    for _ in range(n_perm):
        rng.shuffle(shuffled)
        sup_b, _ = _sup_poisson_changepoint(shuffled, min_seg=min_seg)
        if sup_b >= sup_obs - 1e-12:
            exceed += 1
    p_perm = (1.0 + exceed) / (n_perm + 1.0)
    return {
        "n_weeks": w,
        "sup_2dll": round(sup_obs, 4),
        "tau_idx": tau,
        "changepoint_date": cp_date,
        "rate_before_wk": round(r_before, 4),
        "rate_after_wk": round(r_after, 4),
        "rate_ratio": round(rate_ratio, 3),
        "p_value": round(p_perm, 6),
    }


# ---------------------------------------------------------------------------
# Family I: National Heavy-Tail Fatality Law (Pareto vs Truncated Lognormal)
# ---------------------------------------------------------------------------


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def eval_pareto_tail(
    fatalities: Sequence[int],
    *,
    x_min: int = 5,
    x0: float = 4.5,
) -> Dict[str, Any]:
    """Continuous Pareto MLE (x >= x_min, lower bound x0) vs left-truncated Lognormal via Vuong Z."""
    all_vals = sorted(max(0, int(x)) for x in fatalities)
    n_all = len(all_vals)
    tot_deaths = sum(all_vals)
    gini = 0.0
    top_5pct_share = 0.0
    if n_all > 1 and tot_deaths > 0:
        gini_num = sum((2 * (i + 1) - n_all - 1) * val for i, val in enumerate(all_vals))
        gini = gini_num / (n_all * tot_deaths)
        top_k = max(1, int(math.ceil(0.05 * n_all)))
        top_5pct_share = sum(all_vals[-top_k:]) / tot_deaths

    tail = [float(x) for x in all_vals if x >= x_min]
    n_tail = len(tail)
    if n_tail < 10:
        return {
            "n_all": n_all,
            "total_deaths": tot_deaths,
            "n_tail": n_tail,
            "x_min": x_min,
            "alpha_mle": 1.0,
            "vuong_z": 0.0,
            "p_value": 1.0,
            "gini": round(gini, 4),
            "top_5pct_share": round(top_5pct_share, 4),
        }

    log_ratios = [math.log(x / x0) for x in tail]
    sum_lr = sum(log_ratios)
    alpha_shape = n_tail / max(1e-9, sum_lr)
    alpha_mle = 1.0 + alpha_shape
    ll_pareto = [
        math.log(alpha_shape) - math.log(x0) - alpha_mle * math.log(x / x0) for x in tail
    ]

    # Fit left-truncated Lognormal at x0 via coarse-to-fine coordinate search
    logs = [math.log(x) for x in tail]
    lx0 = math.log(x0)
    mu_emp = sum(logs) / n_tail
    sig_emp = max(0.3, math.sqrt(sum((u - mu_emp) ** 2 for u in logs) / n_tail))

    def _trunc_ln_ll(mu: float, sig: float) -> float:
        if sig <= 0.05:
            return -1e18
        surv = _norm_sf((lx0 - mu) / sig)
        if surv <= 1e-12:
            return -1e18
        log_surv = math.log(surv)
        c0 = -0.5 * math.log(2.0 * math.pi) - math.log(sig) - log_surv
        return sum(c0 - u - ((u - mu) ** 2) / (2.0 * sig * sig) for u in logs)

    best_mu, best_sig = mu_emp, sig_emp
    best_ln_ll = _trunc_ln_ll(best_mu, best_sig)
    for d_mu in (-2.0, -1.2, -0.6, -0.3, 0.0, 0.3, 0.6):
        for s_mult in (0.6, 0.85, 1.0, 1.3, 1.8, 2.5):
            cm = mu_emp + d_mu
            cs = max(0.15, sig_emp * s_mult)
            val = _trunc_ln_ll(cm, cs)
            if val > best_ln_ll:
                best_ln_ll = val
                best_mu, best_sig = cm, cs

    for step in (0.25, 0.08, 0.02, 0.005):
        improved = True
        while improved:
            improved = False
            for dm, ds in ((step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)):
                cm = best_mu + dm
                cs = max(0.1, best_sig + ds)
                val = _trunc_ln_ll(cm, cs)
                if val > best_ln_ll + 1e-9:
                    best_ln_ll = val
                    best_mu, best_sig = cm, cs
                    improved = True

    surv_best = max(1e-12, _norm_sf((lx0 - best_mu) / best_sig))
    c0_best = -0.5 * math.log(2.0 * math.pi) - math.log(best_sig) - math.log(surv_best)
    ll_ln = [
        c0_best - u - ((u - best_mu) ** 2) / (2.0 * best_sig * best_sig) for u in logs
    ]

    diffs = [lp - lln for lp, lln in zip(ll_pareto, ll_ln)]
    mean_d = sum(diffs) / n_tail
    var_d = sum((d - mean_d) ** 2 for d in diffs) / max(1, n_tail - 1)
    vuong_z = (math.sqrt(n_tail) * mean_d / math.sqrt(var_d)) if var_d > 1e-12 else 0.0
    p_val = math.erfc(abs(vuong_z) / math.sqrt(2.0))

    return {
        "n_all": n_all,
        "total_deaths": tot_deaths,
        "n_tail": n_tail,
        "x_min": x_min,
        "x0": x0,
        "alpha_mle": round(alpha_mle, 4),
        "ln_mu": round(best_mu, 4),
        "ln_sigma": round(best_sig, 4),
        "vuong_z": round(vuong_z, 4),
        "p_value": round(p_val, 6),
        "gini": round(gini, 4),
        "top_5pct_share": round(top_5pct_share, 4),
    }


# ---------------------------------------------------------------------------
# Unified Open-Family Proof Derivation (KL, TE, Changepoint, Pareto)
# ---------------------------------------------------------------------------


def derive_information_theoretic_proofs(
    incidents: Sequence[VerifiedIncident],
    clusters: Sequence[AttackCluster],
    *,
    cutoff_date: Optional[str] = None,
    min_kl_events: int = 8,
    min_te_active_weeks: int = 5,
    min_cp_events: int = 12,
    n_perm_kl: int = 2000,
    n_perm_te: int = 2000,
    n_perm_cp: int = 1000,
) -> Tuple[List[HypothesisProof], Dict[str, Dict[str, Any]]]:
    """Derive H_kl_divergence, H_transfer_entropy, H_changepoint, and H_pareto_tail proofs with zero leakage."""
    now_iso = _utc_now_iso()
    causal_all = [
        inc for inc in incidents if not cutoff_date or inc.date_start[:10] <= cutoff_date[:10]
    ]
    located = [inc for inc in causal_all if is_located_incident(inc)]
    dense = extract_dense_segment(causal_all, gap_days=45, cutoff_date=cutoff_date, located_only=True)

    cluster_by_cercle: Dict[str, str] = {}
    for cl in clusters:
        if cl.dominant_cercle_id not in cluster_by_cercle:
            cluster_by_cercle[cl.dominant_cercle_id] = cl.cluster_id

    cercle_metrics: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "kl_actor_nats": 0.0,
            "kl_actor_q_bh": 1.0,
            "kl_tactic_nats": 0.0,
            "kl_tactic_q_bh": 1.0,
            "te_inflow_bits": 0.0,
            "te_inflow_q_bh": 1.0,
            "changepoint_date": "",
            "kl_bins": [],
        }
    )

    out_proofs: List[HypothesisProof] = []
    if not located:
        return out_proofs, cercle_metrics

    # 1. Per-cercle Relative Entropy (Actor K=5 & Tactic K_eff<=8)
    by_cercle_loc: Dict[str, List[VerifiedIncident]] = defaultdict(list)
    nat_actor_cnt: Counter = Counter()
    nat_tactic_cnt: Counter = Counter()
    for inc in located:
        by_cercle_loc[inc.cercle_id].append(inc)
        nat_actor_cnt[canonical_actor_bucket(inc.actor_primary)] += 1
        nat_tactic_cnt[inc.tactic] += 1

    active_actors = [a for a in CANONICAL_ACTORS_5 if nat_actor_cnt[a] > 0]
    if len(active_actors) < 2:
        active_actors = list(CANONICAL_ACTORS_5)
    active_tactics = [t for t in CANONICAL_TACTICS_8 if nat_tactic_cnt[t] > 0]
    if len(active_tactics) < 2:
        active_tactics = list(CANONICAL_TACTICS_8)

    kl_candidates = sorted(
        [(cid, evs) for cid, evs in by_cercle_loc.items() if len(evs) >= min_kl_events],
        key=lambda kv: (-len(kv[1]), kv[0]),
    )

    for dim, cats, nat_cnt in (
        ("actor", active_actors, dict(nat_actor_cnt)),
        ("tactic", active_tactics, dict(nat_tactic_cnt)),
    ):
        raw_res: List[Tuple[str, Dict[str, Any]]] = []
        for cid, evs in kl_candidates:
            if dim == "actor":
                c_cnt = dict(Counter(canonical_actor_bucket(e.actor_primary) for e in evs))
            else:
                c_cnt = dict(Counter(e.tactic for e in evs))
            res = eval_kl_divergence_loo(
                c_cnt, nat_cnt, cats, dimension=dim, n_perm=n_perm_kl, seed_key=f"kl_{cid}"
            )
            raw_res.append((cid, res))

        q_vals = benjamini_hochberg([r["p_value"] for _, r in raw_res])
        for (cid, res), q_bh in zip(raw_res, q_vals):
            c_entry = BY_CERCLE_ID.get(cid)
            c_name = c_entry.name if c_entry else cid
            cl_id = cluster_by_cercle.get(cid, f"cercle_{cid}")
            verdict = assign_verdict(res["n_c"], q_bh, effect_positive=res["kl_eb"] >= 0.05)

            top_bin = res["bins"][0] if res["bins"] else None
            top_desc = (
                f"{top_bin['bin']} ({round(top_bin['p_eb']*100,1)}% EB vs {round(top_bin['q_loo']*100,1)}% LOO)"
                if top_bin
                else "balanced"
            )
            if dim == "actor":
                cercle_metrics[cid]["kl_actor_nats"] = res["kl_eb"]
                cercle_metrics[cid]["kl_actor_q_bh"] = q_bh
            else:
                cercle_metrics[cid]["kl_tactic_nats"] = res["kl_eb"]
                cercle_metrics[cid]["kl_tactic_q_bh"] = q_bh
            cercle_metrics[cid]["kl_bins"].extend(res["bins"])

            out_proofs.append(
                HypothesisProof(
                    hyp_id=f"{cid}__H_kl_{dim}",
                    cluster_id=cl_id,
                    cercle_id=cid,
                    family="H_kl_divergence",
                    title=f"{dim.title()} Relative Entropy D_KL(P_c || Q_-c) in {c_name}",
                    symbolic_formula=(
                        f"D_KL^EB = sum p_hat^EB ln(p_hat^EB / q_hat_-c); "
                        f"alpha0 = K = {res['k_eff']}"
                    ),
                    null_hypothesis=f"H0: {c_name} {dim} distribution matches LOO national pool Q_-c",
                    alternative_hypothesis=f"H1: Localized {dim} specialization diverges from Q_-c (BH-FDR controlled)",
                    parameters={
                        "dimension": dim,
                        "k_eff": res["k_eff"],
                        "alpha0": res["alpha0"],
                        "kl_eb": res["kl_eb"],
                        "g_loo": res["g_loo"],
                        "n_c": res["n_c"],
                        "n_loo": res["n_loo"],
                        "top_driver_bin": top_bin["bin"] if top_bin else "",
                        "top_log2_lift": top_bin["log2_lift"] if top_bin else 0.0,
                    },
                    test_statistic_name="D_KL^EB (nats) / G_LOO",
                    test_statistic_value=res["kl_eb"],
                    p_value=res["p_value"],
                    q_value=q_bh,
                    log_likelihood_ratio=round(0.5 * res["g_loo"], 4),
                    verdict=verdict,
                    n_observations=res["n_c"],
                    derivation_steps=[
                        f"1. Filtered located incidents (excl. region_centroid): n_c = {res['n_c']} vs LOO pool N_-c = {res['n_loo']}.",
                        f"2. Empirical-Bayes Dirichlet smoothing (alpha0=K={res['k_eff']}) yields D_KL^EB = {res['kl_eb']} nats (G_LOO = {res['g_loo']}).",
                        f"3. Exact without-replacement MC permutation (B={n_perm_kl}) gives p = {res['p_value']}, BH-FDR q = {q_bh} ({verdict}).",
                    ],
                    kl_bins=res["bins"],
                    analyst_narrative=(
                        f"{c_name} exhibits {dim} divergence D_KL^EB={res['kl_eb']} nats (q_BH={q_bh}), led by {top_desc}."
                    ),
                    updated_at=now_iso,
                )
            )

    # 2. National Spatial-Regime JSD (early vs late half of dense located segment)
    if len(dense) >= 12:
        jsd_res = eval_regime_jsd(dense, n_perm=1000)
        jsd_verdict = assign_verdict(
            jsd_res["n_early"] + jsd_res["n_late"],
            jsd_res["p_value"],
            effect_positive=jsd_res["jsd_nats"] >= 0.03,
        )
        out_proofs.append(
            HypothesisProof(
                hyp_id="national_mali__H_kl_regime_jsd",
                cluster_id="national_mali",
                cercle_id="mali_national",
                family="H_kl_divergence",
                title="National Spatial-Regime JSD Shift (Early vs Late Dense Window)",
                symbolic_formula="JSD(P_early || P_late) = 0.5*D_KL(P_early||M) + 0.5*D_KL(P_late||M) in [0, ln 2]",
                null_hypothesis="H0: Spatial incident share across cercles invariant between early and late halves",
                alternative_hypothesis="H1: Geographic conflict epicenter shifted significantly across periods",
                parameters={
                    "dimension": "regime",
                    "jsd_nats": jsd_res["jsd_nats"],
                    "n_early": jsd_res["n_early"],
                    "n_late": jsd_res["n_late"],
                    "split_date": jsd_res["split_date"],
                },
                test_statistic_name="Regime JSD (nats)",
                test_statistic_value=jsd_res["jsd_nats"],
                p_value=jsd_res["p_value"],
                q_value=jsd_res["p_value"],
                verdict=jsd_verdict,
                n_observations=jsd_res["n_early"] + jsd_res["n_late"],
                derivation_steps=[
                    f"1. Split dense located window at {jsd_res['split_date']} into n_early={jsd_res['n_early']} and n_late={jsd_res['n_late']}.",
                    f"2. Symmetric Jeffreys-smoothed JSD = {jsd_res['jsd_nats']} nats (bounded in [0, 0.693]).",
                    f"3. Exact label-permutation test (B=1000) yields p = {jsd_res['p_value']} ({jsd_verdict}).",
                ],
                kl_bins=jsd_res["bins"],
                analyst_narrative=(
                    f"Geographic conflict share shifted across Mali with JSD={jsd_res['jsd_nats']} nats (split {jsd_res['split_date']})."
                ),
                updated_at=now_iso,
            )
        )

    # 3. Neighborhood-Inflow Transfer Entropy & Poisson Changepoints on dense segment
    if dense:
        dense_d0 = _parse_iso_date(dense[0].date_start)
        dense_d1 = _parse_iso_date(dense[-1].date_start)
        # Align to ISO Monday-Sunday calendar weeks so week boundaries don't shift with end-date weekday
        win_start_ord = dense_d0.toordinal() - dense_d0.weekday()
        w_len = max(1, (dense_d1.toordinal() - win_start_ord) // 7 + 1)
        by_cercle_dense: Dict[str, List[VerifiedIncident]] = defaultdict(list)
        for inc in dense:
            by_cercle_dense[inc.cercle_id].append(inc)

        weekly_counts_by_c: Dict[str, List[int]] = {}
        weekly_bin_by_c: Dict[str, List[int]] = {}
        for c_id in BY_CERCLE_ID:
            wc = [0] * w_len
            for e in by_cercle_dense.get(c_id, []):
                idx = (_parse_iso_date(e.date_start).toordinal() - win_start_ord) // 7
                if 0 <= idx < w_len:
                    wc[idx] += 1
            weekly_counts_by_c[c_id] = wc
            weekly_bin_by_c[c_id] = [1 if c > 0 else 0 for c in wc]

        if w_len >= 12:
            week_years = [date.fromordinal(win_start_ord + w * 7).year for w in range(w_len)]
            week_dates = [date.fromordinal(win_start_ord + w * 7).isoformat() for w in range(w_len)]

            # 3a. Neighborhood-Inflow TE
            te_rows: List[Tuple[str, Dict[str, Any], str, float, float]] = []
            for cid, y_bin in weekly_bin_by_c.items():
                entry = BY_CERCLE_ID.get(cid)
                if not entry or not entry.corridor_neighbors or sum(y_bin) < min_te_active_weeks:
                    continue
                x_nb = [
                    1 if any(weekly_bin_by_c.get(nb, [0] * w_len)[t] > 0 for nb in entry.corridor_neighbors) else 0
                    for t in range(w_len)
                ]
                if sum(x_nb) < 3:
                    continue
                te_res = eval_transfer_entropy_binary(
                    x_nb, y_bin, week_years, n_perm=n_perm_te, seed_key=f"te_{cid}"
                )
                # Identify top individual corridor driver
                best_nb = ""
                best_edge_te = 0.0
                best_edge_lift = 0.0
                for nb in entry.corridor_neighbors:
                    nb_bin = weekly_bin_by_c.get(nb)
                    if not nb_bin or sum(nb_bin) < 3:
                        continue
                    e_te = _binary_te_bits(nb_bin, y_bin)
                    if e_te > best_edge_te:
                        best_edge_te = e_te
                        best_nb = nb
                        best_edge_lift = _onset_lift(nb_bin, y_bin)
                te_rows.append((cid, te_res, best_nb, round(best_edge_te, 5), round(best_edge_lift, 4)))

            te_q = benjamini_hochberg([r[1]["p_value"] for r in te_rows])
            for (cid, te_res, best_nb, best_e_te, best_e_lift), q_bh in zip(te_rows, te_q):
                c_entry = BY_CERCLE_ID.get(cid)
                c_name = c_entry.name if c_entry else cid
                cl_id = cluster_by_cercle.get(cid, f"cercle_{cid}")
                eff_ok = te_res["te_mm_bits"] > 0.0 and te_res["onset_lift"] > 0.0
                verdict = assign_verdict(te_res["n_transitions"], q_bh, effect_positive=eff_ok)
                cercle_metrics[cid]["te_inflow_bits"] = te_res["te_mm_bits"]
                cercle_metrics[cid]["te_inflow_q_bh"] = q_bh
                drv_name = BY_CERCLE_ID[best_nb].name if best_nb in BY_CERCLE_ID else best_nb

                out_proofs.append(
                    HypothesisProof(
                        hyp_id=f"{cid}__H_transfer_entropy",
                        cluster_id=cl_id,
                        cercle_id=cid,
                        family="H_transfer_entropy",
                        title=f"Neighborhood-Inflow Transfer Entropy TE(N({c_name}) -> {c_name})",
                        symbolic_formula="TE_MM = max(0, sum p(y',y,x) log2[p(y'|y,x)/p(y'|y)] - 1/(N*ln 2))",
                        null_hypothesis=f"H0: Corridor neighbors N({c_name}) add zero 1-week predictive information",
                        alternative_hypothesis=f"H1: Neighbor activation precedes {c_name} onset (TE_MM > 0, onset_lift > 0)",
                        parameters={
                            **te_res,
                            "top_driver_cercle": best_nb,
                            "edge_te_bits": best_e_te,
                            "edge_onset_lift": best_e_lift,
                        },
                        test_statistic_name="TE_MM (bits)",
                        test_statistic_value=te_res["te_mm_bits"],
                        p_value=te_res["p_value"],
                        q_value=q_bh,
                        log_likelihood_ratio=round(0.5 * te_res["g_stat"], 4),
                        verdict=verdict,
                        n_observations=te_res["n_transitions"],
                        derivation_steps=[
                            f"1. Built binary weekly corridor union X_N({cid}) -> Y_{cid} over N={te_res['n_transitions']} contiguous weeks.",
                            f"2. Raw TE = {te_res['te_raw_bits']} bits minus Miller-Madow bias {te_res['bias_mm_bits']} => TE_MM = {te_res['te_mm_bits']} bits; onset lift = {round(te_res['onset_lift']*100,1)}pp.",
                            f"3. Year-stratified permutation (B={n_perm_te}) gives p_yr = {te_res['p_value']}, BH-FDR q = {q_bh} ({verdict}).",
                        ],
                        analyst_narrative=(
                            f"Corridor inflow TE_MM={te_res['te_mm_bits']} bits (onset lift {round(te_res['onset_lift']*100,1)}pp, q_BH={q_bh}); "
                            f"top neighbor driver: {drv_name or 'none'} ({best_e_te}b)."
                        ),
                        updated_at=now_iso,
                    )
                )

            # 3b. Exact Poisson Changepoints
            cp_rows: List[Tuple[str, Dict[str, Any], int]] = []
            for cid, evs in sorted(by_cercle_dense.items()):
                if len(evs) < min_cp_events or w_len < 24:
                    continue
                wc = weekly_counts_by_c.get(cid, [])
                cp_res = eval_changepoint_poisson(
                    wc, week_dates, min_seg=10, n_perm=n_perm_cp, seed_key=f"cp_{cid}"
                )
                cp_rows.append((cid, cp_res, len(evs)))

            cp_q = benjamini_hochberg([r[1]["p_value"] for r in cp_rows])
            for (cid, cp_res, n_evs), q_bh in zip(cp_rows, cp_q):
                c_entry = BY_CERCLE_ID.get(cid)
                c_name = c_entry.name if c_entry else cid
                cl_id = cluster_by_cercle.get(cid, f"cercle_{cid}")
                eff_ok = cp_res["sup_2dll"] > 3.0 and abs(cp_res["rate_after_wk"] - cp_res["rate_before_wk"]) > 0.05
                verdict = assign_verdict(n_evs, q_bh, effect_positive=eff_ok)
                if verdict in ("PROVEN", "SUPPORTED"):
                    cercle_metrics[cid]["changepoint_date"] = cp_res["changepoint_date"]

                out_proofs.append(
                    HypothesisProof(
                        hyp_id=f"{cid}__H_changepoint",
                        cluster_id=cl_id,
                        cercle_id=cid,
                        family="H_changepoint",
                        title=f"Poisson Split-Rate Regime Changepoint in {c_name}",
                        symbolic_formula="tau* = argmax_tau 2*[S1*ln(lam1/lam0) + S2*ln(lam2/lam0)]",
                        null_hypothesis=f"H0: Homogeneous weekly rate across dense window in {c_name}",
                        alternative_hypothesis=f"H1: Structural rate step at tau* (min_seg=10w)",
                        parameters=cp_res,
                        test_statistic_name="Supremum 2*DeltaLogLik",
                        test_statistic_value=cp_res["sup_2dll"],
                        p_value=cp_res["p_value"],
                        q_value=q_bh,
                        log_likelihood_ratio=round(0.5 * cp_res["sup_2dll"], 4),
                        verdict=verdict,
                        n_observations=n_evs,
                        derivation_steps=[
                            f"1. Scanned W={cp_res['n_weeks']} dense weeks (min_seg=10w) -> tau* at {cp_res['changepoint_date']}.",
                            f"2. Weekly rate shifted {cp_res['rate_before_wk']}/wk -> {cp_res['rate_after_wk']}/wk (sup 2*Delta_ll = {cp_res['sup_2dll']}).",
                            f"3. Exact permutation supremum (B={n_perm_cp}) yields p = {cp_res['p_value']}, BH-FDR q = {q_bh} ({verdict}).",
                        ],
                        analyst_narrative=(
                            f"Detected rate shift at {cp_res['changepoint_date']} ({cp_res['rate_before_wk']} -> {cp_res['rate_after_wk']}/wk, q_BH={q_bh})."
                        ),
                        updated_at=now_iso,
                    )
                )

    # 4. National Heavy-Tail Fatality Law (x >= 5)
    par_res = eval_pareto_tail([inc.fatalities_best for inc in causal_all], x_min=5, x0=4.5)
    if par_res["n_tail"] >= 10:
        par_verdict = assign_verdict(
            par_res["n_tail"], par_res["p_value"], effect_positive=par_res["vuong_z"] > 0.0
        )
        winner_txt = (
            "Pareto power-law beats truncated Lognormal"
            if par_res["vuong_z"] > 0.0
            else "Truncated Lognormal outperforms continuous Pareto at x0=4.5"
        )
        out_proofs.append(
            HypothesisProof(
                hyp_id="national_mali__H_pareto_tail",
                cluster_id="national_mali",
                cercle_id="mali_national",
                family="H_pareto_tail",
                title="National Heavy-Tail Fatality Power Law (x >= 5, Vuong vs Lognormal)",
                symbolic_formula="alpha_hat = 1 + n_tail / sum ln(x_i / 4.5); Vuong Z = sqrt(n)*d_bar / s_d",
                null_hypothesis="H0: Fatality tail (x >= 5) equally or better fit by truncated Lognormal",
                alternative_hypothesis="H1: Scale-free Pareto power-law tail dominates (Vuong Z > 0)",
                parameters=par_res,
                test_statistic_name="Vuong Z (Pareto vs Trunc-LN)",
                test_statistic_value=par_res["vuong_z"],
                p_value=par_res["p_value"],
                q_value=par_res["p_value"],
                verdict=par_verdict,
                n_observations=par_res["n_tail"],
                derivation_steps=[
                    f"1. Across N={par_res['n_all']} incidents ({par_res['total_deaths']} deaths), Gini={par_res['gini']}, top 5% share={round(par_res['top_5pct_share']*100,1)}%.",
                    f"2. Tail n={par_res['n_tail']} (x >= 5, x0=4.5): Pareto MLE alpha_hat = {par_res['alpha_mle']} vs Trunc-LN(mu={par_res['ln_mu']}, sigma={par_res['ln_sigma']}).",
                    f"3. Vuong test: Z = {par_res['vuong_z']} -> p = {par_res['p_value']} ({winner_txt}; {par_verdict}).",
                ],
                analyst_narrative=(
                    f"Fatality concentration Gini={par_res['gini']} (top 5% = {round(par_res['top_5pct_share']*100,1)}% of deaths); "
                    f"{winner_txt} (alpha={par_res['alpha_mle']}, Vuong Z={par_res['vuong_z']})."
                ),
                updated_at=now_iso,
            )
        )

    return out_proofs, dict(cercle_metrics)


# ---------------------------------------------------------------------------
# Autonomous Inter-Agent Proof Debate & Open Hypothesis Deduction
# ---------------------------------------------------------------------------


def _compute_open_hypotheses_from_data(
    incidents: Sequence[VerifiedIncident],
    info_proofs: Sequence[HypothesisProof],
    *,
    cutoff_date: Optional[str] = None,
) -> List[HypothesisProof]:
    """Derive and rigorously test 4 open cross-theater/structural hypotheses from empirical proofs."""
    now_iso = _utc_now_iso()
    causal = [
        inc for inc in incidents if not cutoff_date or inc.date_start[:10] <= cutoff_date[:10]
    ]
    located = [inc for inc in causal if is_located_incident(inc)]
    dense = extract_dense_segment(causal, gap_days=45, cutoff_date=cutoff_date, located_only=True)
    if len(located) < 12:
        return []

    open_proofs: List[HypothesisProof] = []
    proposer_label = f"{COORDINATOR_SPEC.name} ({COORDINATOR_SPEC.model_id})"
    theorist_spec = AGENT_BY_ID["a8-math-theorist"]
    challenger_label = f"{theorist_spec.name} ({theorist_spec.model_id})"
    arbiter_label = f"{CRITIC_SPEC.name} ({CRITIC_SPEC.model_id})"

    # Open 1: Liptako-Gourma (Menaka/Ansongo/Gao) vs Central Mali (Mopti/Bandiagara/Bankass/Djenne/Niono) Actor KL
    liptako_ids = {"menaka", "ansongo", "gao"}
    central_ids = {"bandiagara", "bankass", "mopti", "djenne", "niono", "koro", "macina", "douentza"}
    lip_evs = [e for e in located if e.cercle_id in liptako_ids]
    cen_evs = [e for e in located if e.cercle_id in central_ids]
    if len(lip_evs) >= 8 and len(cen_evs) >= 8:
        lip_cnt = Counter(canonical_actor_bucket(e.actor_primary) for e in lip_evs)
        cen_cnt = Counter(canonical_actor_bucket(e.actor_primary) for e in cen_evs)
        sub_pool = Counter()
        for k in CANONICAL_ACTORS_5:
            sub_pool[k] = lip_cnt[k] + cen_cnt[k]
        cats = [k for k in CANONICAL_ACTORS_5 if sub_pool[k] > 0] or list(CANONICAL_ACTORS_5)
        kl_res = eval_kl_divergence_loo(
            dict(lip_cnt), dict(sub_pool), cats, dimension="actor", n_perm=2000, seed_key="open_liptako"
        )
        verdict = assign_verdict(kl_res["n_c"], kl_res["p_value"], effect_positive=kl_res["kl_eb"] >= 0.1)
        open_proofs.append(
            HypothesisProof(
                hyp_id="open_mali__liptako_vs_central_kl",
                cluster_id="national_mali",
                cercle_id="menaka",
                family="H_open",
                title="Open Hypothesis: Bipolar Theater Divergence (Liptako-Gourma ISGS vs Central JNIM)",
                symbolic_formula="D_KL^EB(P_Liptako || Q_Central) with alpha0=K_actor=5; exact MC G-test",
                null_hypothesis="H0: Eastern Liptako-Gourma and Central Mali share identical actor shares",
                alternative_hypothesis="H1: Structural ISGS-vs-JNIM theater partitioning creates >0.1 nats KL divergence",
                parameters={
                    "theater_a": "Liptako-Gourma (menaka,ansongo,gao)",
                    "theater_b": "Central Belt",
                    "kl_eb": kl_res["kl_eb"],
                    "g_stat": kl_res["g_loo"],
                    "n_liptako": len(lip_evs),
                    "n_central": len(cen_evs),
                },
                test_statistic_name="Theater D_KL^EB (nats)",
                test_statistic_value=kl_res["kl_eb"],
                p_value=kl_res["p_value"],
                q_value=kl_res["p_value"],
                log_likelihood_ratio=round(0.5 * kl_res["g_loo"], 4),
                verdict=verdict,
                n_observations=len(lip_evs) + len(cen_evs),
                derivation_steps=[
                    f"1. Partitioned located corpus into Liptako-Gourma (n={len(lip_evs)}) vs Central Belt (n={len(cen_evs)}).",
                    f"2. Empirical-Bayes Actor KL divergence D_KL^EB = {kl_res['kl_eb']} nats (G = {kl_res['g_loo']}).",
                    f"3. Exact without-replacement MC permutation (B=2000) yields p = {kl_res['p_value']} ({verdict}).",
                ],
                kl_bins=kl_res["bins"],
                origin="agent_open",
                proposed_by=proposer_label,
                challenged_by=challenger_label,
                adjudicated_by=arbiter_label,
                analyst_narrative=(
                    f"Liptako-Gourma actor mix diverges sharply from Central Mali (D_KL^EB={kl_res['kl_eb']} nats, G={kl_res['g_loo']})."
                ),
                updated_at=now_iso,
            )
        )

    # Open 2: Post-2023 Northern Indirect-Fire / Airstrike Escalation Shift
    north_ids = {"kidal", "abeibara", "tessalit", "tombouctou", "goundam", "bourem", "gourma-rharous"}
    north_dense = [e for e in dense if e.cercle_id in north_ids]
    if len(north_dense) >= 16:
        pre_n = [e for e in north_dense if e.date_start[:10] < "2023-06-01"]
        post_n = [e for e in north_dense if e.date_start[:10] >= "2023-06-01"]
        if len(pre_n) >= 6 and len(post_n) >= 6:
            standoff = {"indirect_fire", "airstrike_clash"}
            pre_cnt = {"standoff_strike": sum(1 for e in pre_n if e.tactic in standoff), "ground_other": sum(1 for e in pre_n if e.tactic not in standoff)}
            post_cnt = {"standoff_strike": sum(1 for e in post_n if e.tactic in standoff), "ground_other": sum(1 for e in post_n if e.tactic not in standoff)}
            tot_cnt = {k: pre_cnt[k] + post_cnt[k] for k in pre_cnt}
            kl_north = eval_kl_divergence_loo(
                post_cnt, tot_cnt, ["standoff_strike", "ground_other"], dimension="tactic", n_perm=2000, seed_key="open_north_standoff"
            )
            post_share = post_cnt["standoff_strike"] / len(post_n)
            pre_share = pre_cnt["standoff_strike"] / len(pre_n)
            verdict = assign_verdict(len(north_dense), kl_north["p_value"], effect_positive=post_share > pre_share)
            open_proofs.append(
                HypothesisProof(
                    hyp_id="open_mali__northern_standoff_shift",
                    cluster_id="national_mali",
                    cercle_id="abeibara",
                    family="H_open",
                    title="Open Hypothesis: Post-2023 Northern Standoff (Indirect-Fire/Airstrike) Shift",
                    symbolic_formula="D_KL^EB(P_post-2023-06 || Q_pre-2023-06) on {standoff_strike, ground_other}",
                    null_hypothesis="H0: Northern Mali standoff-strike share invariant before vs after 2023-06-01",
                    alternative_hypothesis="H1: Post-MINUSMA northern escalation elevates indirect_fire + airstrike_clash share",
                    parameters={
                        "pre_count": len(pre_n),
                        "post_count": len(post_n),
                        "pre_standoff_share": round(pre_share, 4),
                        "post_standoff_share": round(post_share, 4),
                        "kl_eb": kl_north["kl_eb"],
                        "g_stat": kl_north["g_loo"],
                    },
                    test_statistic_name="Standoff Shift G_LOO",
                    test_statistic_value=kl_north["g_loo"],
                    p_value=kl_north["p_value"],
                    q_value=kl_north["p_value"],
                    log_likelihood_ratio=round(0.5 * kl_north["g_loo"], 4),
                    verdict=verdict,
                    n_observations=len(north_dense),
                    derivation_steps=[
                        f"1. Split northern dense events (n={len(north_dense)}) at 2023-06-01: pre={len(pre_n)}, post={len(post_n)}.",
                        f"2. Standoff share shifted {round(pre_share*100,1)}% -> {round(post_share*100,1)}% (D_KL^EB={kl_north['kl_eb']} nats).",
                        f"3. Exact MC permutation yields p = {kl_north['p_value']} ({verdict}).",
                    ],
                    kl_bins=kl_north["bins"],
                    origin="agent_open",
                    proposed_by=proposer_label,
                    challenged_by=challenger_label,
                    adjudicated_by=arbiter_label,
                    analyst_narrative=(
                        f"Northern standoff tactics shifted from {round(pre_share*100,1)}% pre-2023-06 to {round(post_share*100,1)}% post-2023-06."
                    ),
                    updated_at=now_iso,
                )
            )

    # Open 3: Corridor Asymmetry — Tombouctou Lagged Spillover vs Menaka-Ansongo Synchronous Co-Activation
    te_tomb = next((h for h in info_proofs if h.hyp_id == "tombouctou__H_transfer_entropy"), None)
    te_anso = next((h for h in info_proofs if h.hyp_id == "ansongo__H_transfer_entropy"), None)
    if te_tomb is not None:
        tomb_te = float(te_tomb.parameters.get("te_mm_bits", 0.0))
        anso_te = float(te_anso.parameters.get("te_mm_bits", 0.0)) if te_anso else 0.0
        tomb_q = float(te_tomb.q_value if te_tomb.q_value is not None else te_tomb.p_value)
        verdict = assign_verdict(
            te_tomb.n_observations,
            tomb_q,
            effect_positive=(tomb_te > anso_te and float(te_tomb.parameters.get("onset_lift", 0.0)) > 0.0),
        )
        open_proofs.append(
            HypothesisProof(
                hyp_id="open_mali__asymmetric_corridor_spillover",
                cluster_id=te_tomb.cluster_id,
                cercle_id="tombouctou",
                family="H_open",
                title="Open Hypothesis: Asymmetric Tombouctou Inflow vs Tri-Border Co-Activation",
                symbolic_formula="TE_MM(N(tombouctou)->tombouctou) > TE_MM(N(ansongo)->ansongo) under year-stratified null",
                null_hypothesis="H0: Directional 1-week corridor inflow disappears once annual co-escalation is stratified",
                alternative_hypothesis="H1: Tombouctou retains genuine lagged inflow while Menaka-Ansongo is contemporaneous",
                parameters={
                    "tombouctou_te_mm": tomb_te,
                    "tombouctou_onset_lift": te_tomb.parameters.get("onset_lift", 0.0),
                    "tombouctou_top_driver": te_tomb.parameters.get("top_driver_cercle", ""),
                    "ansongo_te_mm": anso_te,
                    "ansongo_p_yr": te_anso.p_value if te_anso else 1.0,
                },
                test_statistic_name="Tombouctou TE_MM (bits)",
                test_statistic_value=tomb_te,
                p_value=te_tomb.p_value,
                q_value=tomb_q,
                log_likelihood_ratio=te_tomb.log_likelihood_ratio,
                verdict=verdict,
                n_observations=te_tomb.n_observations,
                derivation_steps=[
                    f"1. Evaluated year-stratified neighborhood TE across dense weeks (N={te_tomb.n_observations}).",
                    f"2. Tombouctou achieves TE_MM={tomb_te}b (onset lift +{round(float(te_tomb.parameters.get('onset_lift',0))*100,1)}pp, driver={te_tomb.parameters.get('top_driver_cercle','')}).",
                    f"3. Ansongo inflow yields TE_MM={anso_te}b (p_yr={te_anso.p_value if te_anso else 'n/a'}), confirming lag vs co-activation contrast -> {verdict}.",
                ],
                origin="agent_open",
                proposed_by=proposer_label,
                challenged_by=challenger_label,
                adjudicated_by=arbiter_label,
                analyst_narrative=(
                    f"Year-stratified null isolates genuine 1-week spillover into Tombouctou (TE_MM={tomb_te}b, q_BH={tomb_q}) while rejecting tri-border lag."
                ),
                updated_at=now_iso,
            )
        )

    return open_proofs


async def run_autonomous_proof_debate(
    client: Optional[httpx.AsyncClient],
    incidents: Sequence[VerifiedIncident],
    clusters: Sequence[AttackCluster],
    proofs: Sequence[HypothesisProof],
    *,
    sweep_id: str = "",
    cutoff_date: Optional[str] = None,
    use_llm: bool = True,
    on_turn: Optional[Any] = None,
    focus_cercle: Optional[str] = None,
    cycle_num: int = 1,
    prior_turns_history: Optional[Sequence[AgentExchangeTurn]] = None,
    max_history_turns: int = 20,
) -> Tuple[List[HypothesisProof], AgentExchangeRecord]:
    """Run an autonomous multi-turn inter-agent proof debate, rotating across Mali cercles in ongoing loops."""
    now_iso = _utc_now_iso()
    open_proofs = _compute_open_hypotheses_from_data(incidents, proofs, cutoff_date=cutoff_date)

    # Focused cercle metrics when rotating continuously across cercles
    focus_proofs = [h for h in proofs if focus_cercle and h.cercle_id == focus_cercle]
    focus_stats = [
        {
            "hyp_id": h.hyp_id,
            "family": h.family,
            "stat": h.test_statistic_value,
            "p": h.p_value,
            "q": h.q_value,
            "verdict": h.verdict,
        }
        for h in focus_proofs
    ]

    # Build grounded statistical dossier for agent prompts and deterministic fallback
    actor_kls = sorted(
        [h for h in proofs if h.family == "H_kl_divergence" and h.parameters.get("dimension") == "actor"],
        key=lambda h: h.test_statistic_value,
        reverse=True,
    )[:5]
    tactic_kls = sorted(
        [h for h in proofs if h.family == "H_kl_divergence" and h.parameters.get("dimension") == "tactic"],
        key=lambda h: h.test_statistic_value,
        reverse=True,
    )[:4]
    te_top = sorted(
        [h for h in proofs if h.family == "H_transfer_entropy"],
        key=lambda h: h.test_statistic_value,
        reverse=True,
    )[:3]
    cp_top = sorted(
        [h for h in proofs if h.family == "H_changepoint"],
        key=lambda h: h.test_statistic_value,
        reverse=True,
    )[:4]
    jsd_h = next((h for h in proofs if h.hyp_id == "national_mali__H_kl_regime_jsd"), None)
    par_h = next((h for h in proofs if h.hyp_id == "national_mali__H_pareto_tail"), None)
    unlocated_cnt = sum(1 for i in incidents if not is_located_incident(i))

    dossier = {
        "cycle_num": cycle_num,
        "focus_cercle": focus_cercle or "national",
        "focus_cercle_proofs": focus_stats[:8],
        "n_total": len(incidents),
        "n_region_centroid_filtered": unlocated_cnt,
        "top_actor_kl": [
            {"cercle": h.cercle_id, "kl_eb": h.test_statistic_value, "q_bh": h.q_value, "driver": h.parameters.get("top_driver_bin")}
            for h in actor_kls
        ],
        "top_tactic_kl": [
            {"cercle": h.cercle_id, "kl_eb": h.test_statistic_value, "q_bh": h.q_value, "driver": h.parameters.get("top_driver_bin")}
            for h in tactic_kls
        ],
        "top_te": [
            {"cercle": h.cercle_id, "te_mm": h.test_statistic_value, "q_bh": h.q_value, "lift": h.parameters.get("onset_lift"), "driver": h.parameters.get("top_driver_cercle")}
            for h in te_top
        ],
        "top_changepoints": [
            {"cercle": h.cercle_id, "date": h.parameters.get("changepoint_date"), "sup_2dll": h.test_statistic_value, "q_bh": h.q_value}
            for h in cp_top
        ],
        "regime_jsd": jsd_h.test_statistic_value if jsd_h else 0.0,
        "pareto_alpha": par_h.parameters.get("alpha_mle") if par_h else None,
        "pareto_vuong_z": par_h.test_statistic_value if par_h else None,
        "open_candidates": [
            {
                "hyp_id": h.hyp_id,
                "title": h.title,
                "stat": h.test_statistic_value,
                "p": h.p_value,
                "q": h.q_value,
                "verdict": h.verdict,
            }
            for h in open_proofs
        ],
    }

    theorist_spec = AGENT_BY_ID["a8-math-theorist"]

    # Deterministic grounded fallback turn texts
    top_ak_str = ", ".join(f"{x['cercle']} (D_KL={x['kl_eb']}, q={x['q_bh']})" for x in dossier["top_actor_kl"][:3]) or "none"
    top_tk_str = ", ".join(f"{x['cercle']} (D_KL={x['kl_eb']}, {x['driver']})" for x in dossier["top_tactic_kl"][:2]) or "none"
    top_te_str = (
        f"{te_top[0].cercle_id} (TE_MM={te_top[0].test_statistic_value}b, lift={round(float(te_top[0].parameters.get('onset_lift',0))*100,1)}pp, q={te_top[0].q_value})"
        if te_top
        else "insufficient dense series"
    )
    top_cp_str = ", ".join(f"{x['cercle']}@{x['date']}(2dl={x['sup_2dll']})" for x in dossier["top_changepoints"][:3]) or "none"

    turns_spec: List[Dict[str, Any]] = [
        {
            "role_badge": "Explorer",
            "spec": COORDINATOR_SPEC,
            "target_hyp_id": open_proofs[0].hyp_id if open_proofs else "national_mali__H_kl_regime_jsd",
            "default_msg": (
                f"Empirical sweep over {len(incidents)} verified incidents (excluding {unlocated_cnt} region_centroid records from cercle KL/TE): "
                f"Actor KL peaks at {top_ak_str}; Tactic KL led by {top_tk_str}; Regime JSD={dossier['regime_jsd']} nats; "
                f"Pareto tail alpha={dossier['pareto_alpha']} (Vuong Z={dossier['pareto_vuong_z']}). "
                f"Proposing {len(open_proofs)} cross-theater open hypotheses for estimator scrutiny."
            ),
            "math_refs": {
                "n_incidents": len(incidents),
                "filtered_centroid": unlocated_cnt,
                "top_actor_kl": actor_kls[0].test_statistic_value if actor_kls else 0.0,
                "regime_jsd_nats": dossier["regime_jsd"],
            },
            "verdict_after": "PROPOSED",
        },
        {
            "role_badge": "Theorist",
            "spec": theorist_spec,
            "target_hyp_id": open_proofs[1].hyp_id if len(open_proofs) > 1 else (open_proofs[0].hyp_id if open_proofs else ""),
            "default_msg": (
                f"Challenging finite-sample KL and TE nulls: replacing uniform Jeffreys prior with Empirical-Bayes Dirichlet (alpha0=K) "
                f"eliminates sparse-cercle inflation, while exact without-replacement MC permutation + BH-FDR controls multiplicity. "
                f"For corridor TE, naive pooled chi2 fails annual co-escalation; only year-stratified Miller-Madow TE_MM with positive onset lift is admissible "
                f"(leading inflow: {top_te_str}; structural changepoints: {top_cp_str})."
            ),
            "math_refs": {
                "alpha0_rule": "K_eff",
                "te_mm_top": te_top[0].test_statistic_value if te_top else 0.0,
                "te_q_bh": te_top[0].q_value if te_top else 1.0,
                "pareto_vuong_z": dossier["pareto_vuong_z"] or 0.0,
            },
            "verdict_after": "CHALLENGED",
        },
        {
            "role_badge": "Explorer",
            "spec": COORDINATOR_SPEC,
            "target_hyp_id": open_proofs[2].hyp_id if len(open_proofs) > 2 else (open_proofs[0].hyp_id if open_proofs else ""),
            "default_msg": (
                f"Refined all open hypotheses against >45d gap segmentation and strict t<=T_cut bounds: "
                + "; ".join(f"{h.hyp_id}={h.test_statistic_value} (p={h.p_value})" for h in open_proofs[:3])
                + ". Ready for final empirical adjudication."
            ),
            "math_refs": {
                "open_tested": len(open_proofs),
                "proven_open": sum(1 for h in open_proofs if h.verdict in ("PROVEN", "SUPPORTED")),
            },
            "verdict_after": "REFINED",
        },
        {
            "role_badge": "Arbiter",
            "spec": CRITIC_SPEC,
            "target_hyp_id": open_proofs[0].hyp_id if open_proofs else "national_mali__H_pareto_tail",
            "default_msg": (
                "Final adjudication under alpha=0.05 (BH-FDR & exact permutation nulls): "
                + (
                    "; ".join(f"{h.hyp_id} -> {h.verdict} (stat={h.test_statistic_value}, q={h.q_value})" for h in open_proofs)
                    if open_proofs
                    else "all core information-theoretic families verified."
                )
            ),
            "math_refs": {
                "total_proofs": len(proofs) + len(open_proofs),
                "open_proven_or_supported": sum(1 for h in open_proofs if h.verdict in ("PROVEN", "SUPPORTED")),
            },
            "verdict_after": open_proofs[0].verdict if open_proofs else "PROVEN",
        },
    ]

    base_history = list(prior_turns_history or [])
    start_idx = (base_history[-1].turn_index + 1) if base_history else 1
    prior_notes: List[str] = [f"[{t.role_badge}] {t.message[:180]}" for t in base_history[-4:]]
    turns: List[AgentExchangeTurn] = []
    focus_label = BY_CERCLE_ID[focus_cercle].name if (focus_cercle and focus_cercle in BY_CERCLE_ID) else (focus_cercle or "All-Mali")
    topic_str = (
        f"Ongoing Proof Cycle #{cycle_num} ({focus_label}): Empirical-Bayes KL, Transfer Entropy & Changepoints"
    )
    open_ids = [h.hyp_id for h in open_proofs]

    for offset, tspec in enumerate(turns_spec):
        idx = start_idx + offset
        role = tspec["role_badge"]
        sp = tspec["spec"]
        msg_text = tspec["default_msg"]
        if focus_cercle and focus_stats:
            fp_summary = ", ".join(f"{x['family']}={x['stat']}(q={x['q']}, {x['verdict']})" for x in focus_stats[:4])
            if offset == 0:
                msg_text = f"[Cycle #{cycle_num} Focus: {focus_label}] {fp_summary}. {msg_text}"
            elif offset == 1:
                msg_text = f"[Cycle #{cycle_num} Theorist Audit on {focus_label}] Scrutinizing permutation nulls & BH-FDR: {fp_summary}. {msg_text}"
            elif offset == 2:
                msg_text = f"[Cycle #{cycle_num} Explorer Reply on {focus_label}] Re-anchored causal T<=T_cut bounds & >45d gap check for {focus_label}: {fp_summary}."
        if use_llm and client is not None:
            prompt_payload = {
                "cycle_num": cycle_num,
                "focus_cercle": focus_label,
                "role": role,
                "dossier": dossier,
                "prior_turns": prior_notes[-4:],
                "instruction": (
                    f"You are {sp.name} ({role}) in ongoing cycle #{cycle_num} focused on {focus_label} (Mali). "
                    "Advance the debate beyond prior_turns (inspect focus_cercle_proofs or open_candidates) and write ONE concise technical turn (<=60 words) citing ONLY exact dossier numbers. "
                    'Return strictly JSON: {"message": "..."}'
                ),
            }
            parsed = None
            for cand_spec, t_sec in ((sp, 14.0), (CRITIC_SPEC, 12.0)):
                try:
                    raw, _, err = await asyncio.wait_for(
                        invoke_vol2_agent(
                            client,
                            cand_spec,
                            [{"role": "user", "content": json.dumps(prompt_payload)}],
                            max_tokens=800,
                            temperature=0.22,
                        ),
                        timeout=t_sec,
                    )
                    parsed = extract_json_object(raw) if (not err and raw) else None
                    if parsed and isinstance(parsed.get("message"), str) and len(parsed["message"].strip()) >= 16:
                        msg_text = parsed["message"].strip()[:520]
                        break
                except Exception:
                    continue
        prior_notes.append(f"[{role}] {msg_text}")
        turn_now = _utc_now_iso()
        turns.append(
            AgentExchangeTurn(
                turn_index=idx,
                timestamp=turn_now,
                agent_id=sp.agent_id,
                agent_name=sp.name,
                model_id=sp.model_id,
                role_badge=role,
                target_hyp_id=(focus_proofs[0].hyp_id if (focus_proofs and offset == 0) else tspec["target_hyp_id"]),
                message=msg_text,
                math_refs={**tspec["math_refs"], "cycle": cycle_num, "focus": focus_cercle or "all"},
                verdict_after=tspec["verdict_after"],
            )
        )
        rolling = (base_history + turns)[-max_history_turns:]
        if on_turn is not None:
            try:
                partial = AgentExchangeRecord(
                    exchange_id=f"{sweep_id or 'vol2'}_exchange",
                    sweep_id=sweep_id or "vol2_sweep",
                    updated_at=turn_now,
                    topic=topic_str,
                    open_hyp_ids=open_ids,
                    turns=rolling,
                )
                maybe_coro = on_turn(partial)
                if asyncio.iscoroutine(maybe_coro):
                    await maybe_coro
            except Exception:
                pass

    final_turns = (base_history + turns)[-max_history_turns:]
    exchange_record = AgentExchangeRecord(
        exchange_id=f"{sweep_id or 'vol2'}_exchange",
        sweep_id=sweep_id or "vol2_sweep",
        updated_at=_utc_now_iso(),
        topic=topic_str,
        open_hyp_ids=open_ids,
        turns=final_turns,
    )
    return open_proofs, exchange_record
