"""Spatiotemporal cluster detection, corridor graph inference, and strict causal walk-forward forecasting."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .gazetteer import BY_CERCLE_ID, CERCLE_GAZETTEER, haversine_km
from .hypothesis_proofs import (
    MALI_DRY_MONTHS,
    _parse_iso_date,
    build_weekly_series,
    compute_beta_binomial_hazard,
    derive_information_theoretic_proofs,
    derive_proofs_for_cluster,
    eval_markov_transitions,
    eval_overdispersion,
    eval_seasonal_harmattan,
    fit_hawkes_process,
)
from .schema import (
    AttackCluster,
    CercleMathProofState,
    CerclePrediction,
    ClusterMemberRef,
    HypothesisProof,
    VerifiedIncident,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Spatiotemporal Clustering (Hotspot Identification)
# ---------------------------------------------------------------------------


def build_attack_clusters(
    incidents: Sequence[VerifiedIncident],
    *,
    spatial_eps_km: float = 45.0,
    temporal_eps_days: int = 45,
    min_cluster_size: int = 2,
    max_clusters: int = 18,
) -> Tuple[List[AttackCluster], Dict[str, List[VerifiedIncident]]]:
    """Group incidents into connected spatiotemporal clusters without crossing unconnected corridors."""
    if not incidents:
        return [], {}

    # Deduplicate/order ascending by date
    ordered = sorted(incidents, key=lambda x: (x.date_start, x.incident_id))
    dates = [_parse_iso_date(inc.date_start).toordinal() for inc in ordered]
    n = len(ordered)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        d_i = dates[i]
        inc_i = ordered[i]
        for j in range(i + 1, n):
            if dates[j] - d_i > temporal_eps_days:
                break
            inc_j = ordered[j]
            if inc_i.cercle_id == inc_j.cercle_id:
                union(i, j)
                continue
            dist = haversine_km(inc_i.lat, inc_i.lon, inc_j.lat, inc_j.lon)
            if dist <= spatial_eps_km:
                union(i, j)

    groups: Dict[int, List[VerifiedIncident]] = defaultdict(list)
    for i, inc in enumerate(ordered):
        groups[find(i)].append(inc)

    # Also ensure major cercles appear cleanly if fragmented across gaps
    selected_groups = [g for g in groups.values() if len(g) >= min_cluster_size]
    if not selected_groups and ordered:
        # Small/sparse dataset fallback: group directly by cercle_id
        by_c: Dict[str, List[VerifiedIncident]] = defaultdict(list)
        for inc in ordered:
            by_c[inc.cercle_id].append(inc)
        selected_groups = list(by_c.values())

    # Sort descending by event count, then recency
    selected_groups.sort(key=lambda g: (len(g), max(x.date_start for x in g)), reverse=True)
    selected_groups = selected_groups[:max_clusters]

    clusters: List[AttackCluster] = []
    cluster_members: Dict[str, List[VerifiedIncident]] = {}
    used_ids: Counter = Counter()

    for g in selected_groups:
        cercle_counts = Counter(inc.cercle_id for inc in g)
        dom_cercle = cercle_counts.most_common(1)[0][0]
        cercle_entry = BY_CERCLE_ID.get(dom_cercle)
        used_ids[dom_cercle] += 1
        suffix = f"_{used_ids[dom_cercle]}" if used_ids[dom_cercle] > 1 else ""
        cid = f"cluster_{dom_cercle}{suffix}"

        lat_c = sum(inc.lat for inc in g) / len(g)
        lon_c = sum(inc.lon for inc in g) / len(g)
        rad = max([5.0] + [haversine_km(lat_c, lon_c, inc.lat, inc.lon) for inc in g])
        actors = dict(Counter(inc.actor_primary for inc in g).most_common(8))
        tactics = dict(Counter(inc.tactic for inc in g).most_common(8))
        dry_share = sum(1 for inc in g if _parse_iso_date(inc.date_start).month in MALI_DRY_MONTHS) / len(g)
        corr_share = sum(1 for inc in g if inc.source_count >= 2 or len(inc.perspectives) >= 2) / len(g)

        recent_refs = [
            ClusterMemberRef(
                incident_id=inc.incident_id,
                date_start=inc.date_start,
                cercle_id=inc.cercle_id,
                locality=inc.locality or (cercle_entry.name if cercle_entry else inc.admin2),
                lat=inc.lat,
                lon=inc.lon,
                actor_primary=inc.actor_primary,
                tactic=inc.tactic,
                fatalities_best=inc.fatalities_best,
                source_count=inc.source_count,
            )
            for inc in sorted(g, key=lambda x: x.date_start, reverse=True)[:25]
        ]

        c_name = cercle_entry.name if cercle_entry else g[0].admin2
        label = f"{c_name} Corridor" if used_ids[dom_cercle] == 1 else f"{c_name} Wave {used_ids[dom_cercle]}"

        clusters.append(
            AttackCluster(
                cluster_id=cid,
                label=label,
                admin1=cercle_entry.admin1 if cercle_entry else g[0].admin1,
                dominant_cercle_id=dom_cercle,
                centroid_lat=round(lat_c, 5),
                centroid_lon=round(lon_c, 5),
                radius_km=round(rad, 1),
                event_count=len(g),
                fatalities_total=sum(inc.fatalities_best for inc in g),
                date_first=min(inc.date_start for inc in g),
                date_last=max(inc.date_start for inc in g),
                dominant_actors=actors,
                tactic_breakdown=tactics,
                dry_season_share=round(dry_share, 3),
                corroborated_share=round(corr_share, 3),
                sample_events=recent_refs,
            )
        )
        cluster_members[cid] = g

    return clusters, cluster_members


# ---------------------------------------------------------------------------
# Corridor Graph Inference (Sequential Actor Diffusion)
# ---------------------------------------------------------------------------


def build_corridor_edges(
    incidents: Sequence[VerifiedIncident],
    *,
    max_gap_days: int = 21,
    max_edges: int = 35,
) -> List[Dict[str, Any]]:
    """Detect directed spatial diffusion transitions by shared actor across distinct cercles."""
    by_actor: Dict[str, List[VerifiedIncident]] = defaultdict(list)
    for inc in sorted(incidents, key=lambda x: x.date_start):
        if inc.actor_primary and inc.actor_primary != "Unattributed":
            by_actor[inc.actor_primary].append(inc)

    edge_weights: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for actor, evs in by_actor.items():
        for a, b in zip(evs[:-1], evs[1:]):
            if a.cercle_id == b.cercle_id:
                continue
            gap = (_parse_iso_date(b.date_start) - _parse_iso_date(a.date_start)).days
            if 0 <= gap <= max_gap_days:
                dist = haversine_km(a.lat, a.lon, b.lat, b.lon)
                if dist <= 380.0:
                    key = (a.cercle_id, b.cercle_id)
                    if key not in edge_weights:
                        edge_weights[key] = {
                            "from_cercle": a.cercle_id,
                            "to_cercle": b.cercle_id,
                            "from_lat": round(a.lat, 4),
                            "from_lon": round(a.lon, 4),
                            "to_lat": round(b.lat, 4),
                            "to_lon": round(b.lon, 4),
                            "weight": 0,
                            "actors": Counter(),
                            "median_gap_days": [],
                        }
                    edge_weights[key]["weight"] += 1
                    edge_weights[key]["actors"][actor] += 1
                    edge_weights[key]["median_gap_days"].append(gap)

    out: List[Dict[str, Any]] = []
    for item in sorted(edge_weights.values(), key=lambda e: e["weight"], reverse=True)[:max_edges]:
        gaps = sorted(item["median_gap_days"])
        out.append(
            {
                "from_cercle": item["from_cercle"],
                "to_cercle": item["to_cercle"],
                "from_lat": item["from_lat"],
                "from_lon": item["from_lon"],
                "to_lat": item["to_lat"],
                "to_lon": item["to_lon"],
                "weight": item["weight"],
                "top_actor": item["actors"].most_common(1)[0][0],
                "median_gap_days": gaps[len(gaps) // 2] if gaps else 0,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Strict Causal Scoring Metrics (Pure Python)
# ---------------------------------------------------------------------------


def compute_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    total = len(pos) * len(neg)
    for sp in pos:
        for sn in neg:
            if sp > sn:
                wins += 1.0
            elif sp == sn:
                wins += 0.5
    return round(wins / total, 4)


def compute_pr_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    n_pos = sum(labels)
    if n_pos == 0:
        return 0.0
    paired = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    tp = 0
    fp = 0
    precisions: List[float] = []
    recalls: List[float] = []
    for _, y in paired:
        if y == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_pos)
    area = 0.0
    prev_r = 0.0
    for p, r in zip(precisions, recalls):
        area += p * (r - prev_r)
        prev_r = r
    return round(min(1.0, max(0.0, area)), 4)


def compute_brier_and_bss(labels: Sequence[int], probs: Sequence[float]) -> Tuple[float, float]:
    n = len(labels)
    if n == 0:
        return 0.25, 0.0
    brier = sum((p - y) ** 2 for y, p in zip(labels, probs)) / n
    base_rate = sum(labels) / n
    brier_clim = base_rate * (1.0 - base_rate)
    if brier_clim <= 1e-9:
        return round(brier, 5), 0.0
    bss = 1.0 - (brier / brier_clim)
    return round(brier, 5), round(bss, 4)


# ---------------------------------------------------------------------------
# Causal Feature Extractor (Strict t <= T_cutoff Guard)
# ---------------------------------------------------------------------------


def _score_cercle_at_cutoff(
    cercle_id: str,
    cercle_events: Sequence[VerifiedIncident],
    neighbor_probs: Dict[str, float],
    cutoff_ord: int,
    min_ord: int,
    *,
    beta_prior_a: float = 1.0,
    beta_prior_b: float = 12.0,
) -> Dict[str, float]:
    """Compute strictly causal 7d and 30d hazard probabilities using ONLY events <= cutoff_ord."""
    history = [e for e in cercle_events if _parse_iso_date(e.date_start).toordinal() <= cutoff_ord]
    span_days = max(28.0, float(cutoff_ord - min_ord + 1))

    if not history:
        base_7d = beta_prior_a / (beta_prior_a + beta_prior_b)
        entry = BY_CERCLE_ID.get(cercle_id)
        spill = 0.0
        if entry and entry.corridor_neighbors:
            spill = sum(neighbor_probs.get(nb, 0.0) for nb in entry.corridor_neighbors) / len(entry.corridor_neighbors)
        p7 = min(0.95, max(0.01, 0.7 * base_7d + 0.3 * spill))
        p30 = min(0.99, 1.0 - (1.0 - p7) ** 4.2)
        return {
            "p7": round(p7, 4),
            "p30": round(p30, 4),
            "expected_30d": round(-math.log(max(1e-6, 1.0 - p30)), 3),
            "lambda_now": round(base_7d / 7.0, 5),
            "mu": round(base_7d / 7.0, 5),
            "alpha": 0.0,
            "beta": 0.12,
            "branching": 0.0,
            "half_life": 5.78,
            "beta_mean": round(base_7d, 4),
            "ci_low": 0.002,
            "ci_high": round(min(0.45, base_7d * 3.0), 4),
            "active_weeks": 0,
            "total_weeks": max(4, int(span_days // 7)),
        }

    times = [float(_parse_iso_date(e.date_start).toordinal() - min_ord) for e in history]
    t_end = float(cutoff_ord - min_ord) + 0.5
    hw = fit_hawkes_process(times, t_end)

    cutoff_date = date.fromordinal(cutoff_ord)
    min_date = date.fromordinal(min_ord)
    _, weekly_bin = build_weekly_series([e.date_start for e in history], min_date, cutoff_date)
    bb = compute_beta_binomial_hazard(
        sum(weekly_bin), len(weekly_bin), alpha0=beta_prior_a, beta0=beta_prior_b
    )

    # Combine Hawkes 7d inhomogeneous intensity integral + Beta-Binomial stationary hazard + neighbor spillover
    mu, alpha, beta = hw["mu"], hw["alpha"], hw["beta"]
    decay_sum = sum(math.exp(-beta * max(0.0, t_end - t)) for t in times)
    integral_7d = mu * 7.0 + (alpha / beta if beta > 0 else 0.0) * decay_sum * (1.0 - math.exp(-7.0 * beta))
    integral_30d = mu * 30.0 + (alpha / beta if beta > 0 else 0.0) * decay_sum * (1.0 - math.exp(-30.0 * beta))

    hawkes_p7 = 1.0 - math.exp(-max(0.0, integral_7d))
    hawkes_p30 = 1.0 - math.exp(-max(0.0, integral_30d))

    # Recent 14d momentum & Harmattan month weighting
    recent_14d = sum(1 for t in times if (t_end - t) <= 14.0)
    momentum_bump = min(0.18, recent_14d * 0.04)

    entry = BY_CERCLE_ID.get(cercle_id)
    spill = 0.0
    if entry and entry.corridor_neighbors:
        spill = sum(neighbor_probs.get(nb, 0.0) for nb in entry.corridor_neighbors) / len(entry.corridor_neighbors)

    p7 = min(0.97, max(0.01, 0.45 * hawkes_p7 + 0.40 * bb["posterior_mean"] + 0.15 * spill + momentum_bump * 0.35))
    p30_from_p7 = 1.0 - (1.0 - p7) ** (30.0 / 7.0)
    p30 = min(0.99, max(p7, 0.55 * hawkes_p30 + 0.45 * p30_from_p7))

    return {
        "p7": round(p7, 4),
        "p30": round(p30, 4),
        "expected_30d": round(max(integral_30d, -math.log(max(1e-6, 1.0 - p30))), 3),
        "lambda_now": hw["lambda_now"],
        "mu": hw["mu"],
        "alpha": hw["alpha"],
        "beta": hw["beta"],
        "branching": hw["branching_ratio"],
        "half_life": hw["half_life_days"],
        "beta_mean": bb["posterior_mean"],
        "ci_low": bb["ci_low"],
        "ci_high": bb["ci_high"],
        "active_weeks": sum(weekly_bin),
        "total_weeks": len(weekly_bin),
    }


def run_walk_forward_and_predictions(
    incidents: Sequence[VerifiedIncident],
    clusters: Sequence[AttackCluster],
    cluster_members: Dict[str, List[VerifiedIncident]],
    *,
    reference_date: Optional[str] = None,
    max_folds: int = 16,
) -> Dict[str, Any]:
    """Run strictly causal walk-forward validation across 38 cercles + produce live proofs & forecasts."""
    now_iso = _utc_now_iso()
    ordered = sorted(incidents, key=lambda x: x.date_start)

    if ordered:
        min_d = _parse_iso_date(ordered[0].date_start)
        max_d = _parse_iso_date(reference_date) if reference_date else _parse_iso_date(ordered[-1].date_start)
    else:
        max_d = date.today()
        min_d = date.fromordinal(max_d.toordinal() - 180)

    min_ord = min_d.toordinal()
    max_ord = max_d.toordinal()
    total_span = max(28, max_ord - min_ord)
    cutoff_iso = max_d.isoformat()
    causal_ordered = [inc for inc in ordered if _parse_iso_date(inc.date_start) <= max_d]

    by_cercle: Dict[str, List[VerifiedIncident]] = defaultdict(list)
    for inc in causal_ordered:
        by_cercle[inc.cercle_id].append(inc)

    # Compute national weekly baseline active fraction
    _, nat_bin = build_weekly_series([inc.date_start for inc in causal_ordered], min_d, max_d)
    cercle_active_fracs = []
    for c in CERCLE_GAZETTEER:
        evs = by_cercle.get(c.cercle_id, [])
        _, cb = build_weekly_series([e.date_start for e in evs], min_d, max_d)
        cercle_active_fracs.append(sum(cb) / max(1, len(cb)))
    national_weekly_prob = max(0.02, min(0.5, sum(cercle_active_fracs) / max(1, len(cercle_active_fracs))))

    # Walk-forward evaluation cutoffs (strictly < eval_end - 7 so 7-day horizon outcome is observable!)
    # If there is an archival gap > 45d between historical GED coverage and live RSS headlines,
    # evaluate walk-forward folds at the tail of dense continuous monitoring so BSS compares honest active weeks.
    all_ords = sorted(_parse_iso_date(inc.date_start).toordinal() for inc in ordered)
    eval_end_ord = max_ord
    if not reference_date and len(all_ords) >= 20:
        for idx in range(len(all_ords) - 1, 0, -1):
            if (all_ords[idx] - all_ords[idx - 1]) <= 45:
                # Found dense coverage tail ending at all_ords[idx]
                if (max_ord - all_ords[idx]) > 45:
                    eval_end_ord = all_ords[idx]
                break

    eval_span = max(28, eval_end_ord - min_ord)
    step_days = 7 if eval_span >= 120 else max(3, eval_span // 10)
    cutoffs: List[int] = []
    cur = eval_end_ord - 7
    earliest_allowed = min_ord + max(14, int(eval_span * 0.25))
    while cur >= earliest_allowed and len(cutoffs) < max_folds:
        cutoffs.append(cur)
        cur -= step_days
    cutoffs.reverse()

    all_labels: List[int] = []
    all_probs: List[float] = []
    cercle_labels: Dict[str, List[int]] = defaultdict(list)
    cercle_probs: Dict[str, List[float]] = defaultdict(list)

    for t_cut in cutoffs:
        # First pass: local weekly empirical rates strictly <= t_cut for neighbor spillover
        local_emp: Dict[str, float] = {}
        for c in CERCLE_GAZETTEER:
            past = [e for e in by_cercle.get(c.cercle_id, []) if _parse_iso_date(e.date_start).toordinal() <= t_cut]
            weeks_so_far = max(2.0, (t_cut - min_ord) / 7.0)
            local_emp[c.cercle_id] = min(0.85, len(past) / weeks_so_far)

        for c in CERCLE_GAZETTEER:
            cid = c.cercle_id
            evs = by_cercle.get(cid, [])
            scored = _score_cercle_at_cutoff(cid, evs, local_emp, t_cut, min_ord)
            # Observable binary target in (t_cut, t_cut + 7]
            y = 1 if any(t_cut < _parse_iso_date(e.date_start).toordinal() <= t_cut + 7 for e in evs) else 0
            all_labels.append(y)
            all_probs.append(scored["p7"])
            cercle_labels[cid].append(y)
            cercle_probs[cid].append(scored["p7"])

    global_auroc = compute_auroc(all_labels, all_probs)
    global_prauc = compute_pr_auc(all_labels, all_probs)
    global_brier, global_bss = compute_brier_and_bss(all_labels, all_probs)

    # Derive cluster-level mathematical proofs (strictly bounded <= max_d)
    all_proofs: List[HypothesisProof] = []
    for cl in clusters:
        members = [m for m in cluster_members.get(cl.cluster_id, []) if _parse_iso_date(m.date_start) <= max_d]
        if members:
            all_proofs.extend(
                derive_proofs_for_cluster(cl, members, min_d, max_d, national_weekly_prob)
            )

    info_proofs, cercle_info_metrics = derive_information_theoretic_proofs(
        causal_ordered, clusters, cutoff_date=cutoff_iso
    )
    all_proofs.extend(info_proofs)

    # Final at-horizon (t <= max_ord) cercle states & predictions
    final_emp = {
        c.cercle_id: min(0.85, len(by_cercle.get(c.cercle_id, [])) / max(2.0, (max_ord - min_ord) / 7.0))
        for c in CERCLE_GAZETTEER
    }

    cercle_states: List[CercleMathProofState] = []
    cercle_preds: List[CerclePrediction] = []

    for c in CERCLE_GAZETTEER:
        cid = c.cercle_id
        evs = sorted(by_cercle.get(cid, []), key=lambda x: x.date_start)
        scored = _score_cercle_at_cutoff(cid, evs, final_emp, max_ord, min_ord)

        wc, _ = build_weekly_series([e.date_start for e in evs], min_d, max_d)
        disp = eval_overdispersion(wc)
        harm = eval_seasonal_harmattan([e.date_start for e in evs])
        mk = eval_markov_transitions([e.tactic for e in evs])

        actor_cnt = Counter(e.actor_primary for e in evs if e.actor_primary != "Unattributed")
        dom_actor = actor_cnt.most_common(1)[0][0] if actor_cnt else "Unattributed"
        next_tactic_probs = mk["next_probs"]
        pred_tactic = max(next_tactic_probs.items(), key=lambda kv: kv[1])[0] if next_tactic_probs else "other"

        c_lbls = cercle_labels.get(cid, [])
        c_prbs = cercle_probs.get(cid, [])
        c_auc = compute_auroc(c_lbls, c_prbs) if sum(c_lbls) > 0 and sum(c_lbls) < len(c_lbls) else global_auroc
        c_brier, c_bss = compute_brier_and_bss(c_lbls, c_prbs)

        t_days = [round(float(_parse_iso_date(e.date_start).toordinal() - min_ord), 2) for e in evs]
        info_st = cercle_info_metrics.get(cid, {})

        cercle_states.append(
            CercleMathProofState(
                cercle_id=cid,
                cercle_name=c.name,
                admin1=c.admin1,
                lat=c.lat,
                lon=c.lon,
                event_count=len(evs),
                event_timestamps_days=t_days[-180:],
                hawkes_mu=scored["mu"],
                hawkes_alpha=scored["alpha"],
                hawkes_beta=scored["beta"],
                hawkes_branching_ratio=scored["branching"],
                hawkes_half_life_days=scored["half_life"],
                hawkes_lambda_now=scored["lambda_now"],
                beta_alpha0=1.0,
                beta_beta0=12.0,
                beta_active_weeks=int(scored["active_weeks"]),
                beta_total_weeks=int(scored["total_weeks"]),
                beta_posterior_mean=scored["beta_mean"],
                beta_ci_low=scored["ci_low"],
                beta_ci_high=scored["ci_high"],
                dispersion_ratio=disp["dispersion_ratio"],
                dry_season_rate=harm["dry_rate_per_30d"],
                rainy_season_rate=harm["rainy_rate_per_30d"],
                dominant_actor=dom_actor,
                markov_next_tactics=next_tactic_probs,
                kl_actor_nats=float(info_st.get("kl_actor_nats", 0.0)),
                kl_actor_q_bh=float(info_st.get("kl_actor_q_bh", 1.0)),
                kl_tactic_nats=float(info_st.get("kl_tactic_nats", 0.0)),
                kl_tactic_q_bh=float(info_st.get("kl_tactic_q_bh", 1.0)),
                te_inflow_bits=float(info_st.get("te_inflow_bits", 0.0)),
                te_inflow_q_bh=float(info_st.get("te_inflow_q_bh", 1.0)),
                changepoint_date=str(info_st.get("changepoint_date", "")),
                kl_bins=list(info_st.get("kl_bins", [])),
                risk_7d_prob=scored["p7"],
                risk_30d_prob=scored["p30"],
                expected_30d_count=scored["expected_30d"],
                walk_forward_auroc=c_auc,
                walk_forward_brier=c_brier,
                brier_skill_score=c_bss,
                updated_at=now_iso,
            )
        )

        cercle_preds.append(
            CerclePrediction(
                cercle_id=cid,
                cercle_name=c.name,
                admin1=c.admin1,
                lat=c.lat,
                lon=c.lon,
                risk_7d_prob=scored["p7"],
                risk_30d_prob=scored["p30"],
                expected_30d_count=scored["expected_30d"],
                dominant_actor=dom_actor,
                predicted_next_tactic=pred_tactic,
                tactic_probs=next_tactic_probs,
                ci_low=scored["ci_low"],
                ci_high=scored["ci_high"],
                walk_forward_auroc=c_auc,
                walk_forward_brier=c_brier,
                brier_skill_score=c_bss,
                updated_at=now_iso,
            )
        )

    cercle_preds.sort(key=lambda p: p.risk_7d_prob, reverse=True)
    return {
        "proofs": all_proofs,
        "cercle_states": cercle_states,
        "predictions": cercle_preds,
        "walk_forward_folds": len(cutoffs),
        "global_auroc": global_auroc,
        "global_pr_auc": global_prauc,
        "global_brier": global_brier,
        "global_bss": global_bss,
    }
