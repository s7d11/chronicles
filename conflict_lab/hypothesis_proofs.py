"""Falsifiable mathematical hypothesis derivation and statistical proof engine for Volume II."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx

from . import AGENT_CAST, CRITIC_SPEC
from .gazetteer import BY_CERCLE_ID

AGENT_BY_ID = {s.agent_id: s for s in AGENT_CAST}
from .providers import extract_json_object, invoke_vol2_agent
from .schema import (
    AttackCluster,
    CercleMathProofState,
    HypothesisProof,
    VerdictStatus,
    VerifiedIncident,
)

MALI_DRY_MONTHS = {11, 12, 1, 2, 3, 4, 5}


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

    raw, _, err = await invoke_vol2_agent(client, spec, messages, max_tokens=1600, temperature=0.1)
    if err or not raw:
        # Fallback to cheap batched Gemini Flash-Lite critic if primary unavailable
        raw, _, _ = await invoke_vol2_agent(
            client, CRITIC_SPEC, messages, max_tokens=1600, temperature=0.1
        )
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
