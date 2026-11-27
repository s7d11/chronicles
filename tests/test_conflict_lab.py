"""Tests for Volume II Mali OSINT pipeline, mathematical proofs, walk-forward causality & isolation."""

from __future__ import annotations

import asyncio
import json
import random
from datetime import date, timedelta
from pathlib import Path

import pytest

from conflict_lab import AGENT_CAST
from conflict_lab.firestore_sync import (
    clamp_cluster_doc,
    clamp_exchange_doc,
    clamp_hypothesis_doc,
    clamp_incident_doc,
)
from conflict_lab.gazetteer import normalize_actor_name, resolve_mali_location
from conflict_lab.georesolve_dedup import deduplicate_and_georesolve
from conflict_lab.hf_sync import export_static_site_bundle
from conflict_lab.hypothesis_proofs import (
    benjamini_hochberg,
    betainc,
    beta_ppf,
    chi2_sf,
    compute_beta_binomial_hazard,
    derive_information_theoretic_proofs,
    eval_changepoint_poisson,
    eval_kl_divergence_loo,
    eval_markov_transitions,
    eval_overdispersion,
    eval_pareto_tail,
    eval_seasonal_harmattan,
    eval_transfer_entropy_binary,
    extract_dense_segment,
    fit_hawkes_process,
    run_autonomous_proof_debate,
)
from conflict_lab.pattern_model import (
    build_attack_clusters,
    build_corridor_edges,
    run_walk_forward_and_predictions,
)
from conflict_lab.providers import ALLOWED_OPENROUTER_FREE_MODELS, assert_openrouter_free_model
from conflict_lab.schema import AgentExchangeRecord, ObservatoryMeta, RawOsintRecord, VerifiedIncident


def test_model_isolation_and_openrouter_free_guardrail():
    for s in AGENT_CAST:
        # Strict Edition 4 isolation guarantees
        assert "nemotron-3-ultra-550b" not in s.model_id
        assert "gemini-3.5-flash" not in s.model_id
        assert s.provider in {"nvidia", "openrouter_free", "vertex", "deterministic"}
        if s.provider == "openrouter_free":
            assert s.model_id in ALLOWED_OPENROUTER_FREE_MODELS
            assert_openrouter_free_model(s.model_id)

    with pytest.raises(ValueError):
        assert_openrouter_free_model("openai/gpt-5")
    with pytest.raises(ValueError):
        assert_openrouter_free_model("anthropic/claude-sonnet-4")


def test_mali_gazetteer_resolution():
    r1, prec1 = resolve_mali_location(raw_location="Djenné", admin1_hint="Mopti")
    assert r1.cercle_id == "djenne"
    assert r1.admin1 == "Mopti"
    assert prec1 in {"exact_locality", "cercle_centroid"}

    r2, _ = resolve_mali_location(raw_location="Ansongo road near Labbezanga")
    assert r2.cercle_id in {"ansongo", "gao"}

    assert normalize_actor_name("JNIM (Macina Katiba)") == "JNIM / Katiba Coalition"
    assert normalize_actor_name("FAMa DIRPA / Forces Armees Maliennes") == "FAMa / Allied Security Forces"
    assert normalize_actor_name("ISGS militants") == "ISGS / Islamic State Sahel"


def test_spatiotemporal_deduplication_merges_multi_source():
    rec1 = RawOsintRecord(
        record_id="fama_1",
        source_name="FAMa DIRPA",
        source_perspective="official_fama_dirpa",
        published_date="2026-10-12",
        event_date="2026-10-11",
        title="Incident sector Bandiagara",
        raw_location="Bandiagara",
        admin2_hint="Bandiagara",
        actor_raw="FAMa",
        tactic_hint="ambush",
        fatalities_reported=2,
    )
    rec2 = RawOsintRecord(
        record_id="ucdp_2",
        source_name="UCDP GED",
        source_perspective="independent_event_db",
        published_date="2026-10-13",
        event_date="2026-10-12",
        title="JNIM ambush near Bandiagara road",
        raw_location="Bandiagara",
        admin2_hint="Bandiagara",
        actor_raw="JNIM",
        tactic_hint="ambush",
        fatalities_reported=6,
    )
    rec3 = RawOsintRecord(
        record_id="kidal_1",
        source_name="GDELT Doc 2.0",
        source_perspective="sahel_press",
        published_date="2026-10-12",
        event_date="2026-10-12",
        title="Explosion reported in Kidal",
        raw_location="Kidal",
        admin2_hint="Kidal",
        actor_raw="ISGS",
        tactic_hint="ied_mine",
        fatalities_reported=1,
    )

    verified = deduplicate_and_georesolve([rec1, rec2, rec3])
    assert len(verified) == 2
    bandiagara = next(v for v in verified if v.cercle_id == "bandiagara")
    assert bandiagara.source_count == 2
    assert set(bandiagara.perspectives) == {"official_fama_dirpa", "independent_event_db"}
    assert bandiagara.corroboration_tier == "high"
    assert bandiagara.fatalities_best >= 2


def test_math_proof_primitives_hawkes_dispersion_beta_markov():
    assert 0.04 < chi2_sf(3.8415, 1) < 0.055
    assert chi2_sf(25.0, 2) < 1e-4

    q95 = beta_ppf(0.95, 5.0, 15.0)
    assert abs(betainc(5.0, 15.0, q95) - 0.95) < 1e-4

    bursty_days = [5.0, 5.4, 6.0, 6.3, 7.0, 7.5, 40.0, 40.5, 41.0, 41.2, 42.0, 90.0, 90.3, 91.0]
    hw = fit_hawkes_process(bursty_days, T=120.0)
    assert hw["branching_ratio"] > 0.1
    assert hw["p_value"] < 0.05

    disp = eval_overdispersion([0, 0, 5, 6, 0, 0, 0, 7, 0, 0])
    assert disp["dispersion_ratio"] > 3.0
    assert disp["p_value"] < 0.01

    seq = ["ied_mine", "ambush"] * 8
    mk = eval_markov_transitions(seq)
    assert mk["g_stat"] > 5.0
    assert mk["p_value"] < 0.02

    dry_heavy = [
        "2025-12-01",
        "2025-12-10",
        "2026-01-05",
        "2026-02-11",
        "2026-03-04",
        "2026-04-02",
        "2026-07-01",
        "2026-08-01",
    ]
    seas = eval_seasonal_harmattan(dry_heavy)
    assert seas["dry_count"] == 6.0
    assert seas["rainy_count"] == 2.0

    bb = compute_beta_binomial_hazard(active_weeks=12, total_weeks=52)
    assert bb["ci_low"] < bb["posterior_mean"] < bb["ci_high"]


def test_walk_forward_causality_zero_future_leakage():
    start = date(2026, 1, 1)
    base_events: list[VerifiedIncident] = []
    for day_off in [4, 12, 20, 35, 42, 60, 75]:
        d = (start + timedelta(days=day_off)).isoformat()
        base_events.append(
            VerifiedIncident(
                incident_id=f"inc_{day_off}",
                date_start=d,
                date_end=d,
                admin1="Mopti",
                admin2="Bankass",
                cercle_id="bankass",
                lat=14.07,
                lon=-3.52,
                actor_primary="JNIM / Katiba Coalition",
                tactic="ied_mine",
                fatalities_best=3,
            )
        )

    future_events = list(base_events)
    for k in range(12):
        d = (start + timedelta(days=95 + k)).isoformat()
        future_events.append(
            VerifiedIncident(
                incident_id=f"fut_{k}",
                date_start=d,
                date_end=d,
                admin1="Kidal",
                admin2="Kidal",
                cercle_id="kidal",
                lat=18.44,
                lon=1.41,
                actor_primary="ISGS / Islamic State Sahel",
                tactic="ambush",
                fatalities_best=2,
            )
        )

    ref = (start + timedelta(days=90)).isoformat()
    cl1, mem1 = build_attack_clusters(base_events)
    cl2, mem2 = build_attack_clusters([e for e in future_events if e.date_start <= ref])
    res1 = run_walk_forward_and_predictions(base_events, cl1, mem1, reference_date=ref)
    res2 = run_walk_forward_and_predictions(future_events, cl2, mem2, reference_date=ref)

    p1_kidal = next(p for p in res1["predictions"] if p.cercle_id == "kidal")
    assert p1_kidal.risk_7d_prob < 0.35

    cl_full, _ = build_attack_clusters(future_events)
    assert len(cl_full) >= 1
    assert clamp_cluster_doc(cl_full[0])["cluster_id"] == cl_full[0].cluster_id
    assert clamp_incident_doc(future_events[0])["incident_id"] == future_events[0].incident_id
    if res2["proofs"]:
        assert clamp_hypothesis_doc(res2["proofs"][0])["hyp_id"] == res2["proofs"][0].hyp_id
    edges = build_corridor_edges(future_events)
    assert isinstance(edges, list)


def test_information_theoretic_proofs_and_data_hygiene():
    # 1. Empirical-Bayes LOO KL divergence + BH-FDR
    c_cnt = {"ISGS / Islamic State Sahel": 40, "JNIM / Katiba Coalition": 5, "FAMa / Allied Security Forces": 5}
    nat_cnt = {"ISGS / Islamic State Sahel": 60, "JNIM / Katiba Coalition": 300, "FAMa / Allied Security Forces": 140}
    kl = eval_kl_divergence_loo(
        c_cnt,
        nat_cnt,
        ["ISGS / Islamic State Sahel", "JNIM / Katiba Coalition", "FAMa / Allied Security Forces"],
        dimension="actor",
        n_perm=400,
    )
    assert kl["kl_eb"] > 0.4
    assert kl["p_value"] < 0.01
    assert len(kl["bins"]) == 3
    assert abs(sum(b["p_eb"] for b in kl["bins"]) - 1.0) < 1e-3
    assert benjamini_hochberg([0.001, 0.04, 0.5]) == [0.003, 0.06, 0.5]

    # 2. Year-stratified Neighborhood-Inflow Transfer Entropy (Y_{t+1} = X_t on irregular X)
    rng_te = random.Random(19)
    x_seq = [1 if rng_te.random() < 0.4 else 0 for _ in range(80)]
    y_seq = [0] + x_seq[:-1]
    yrs = [2023] * 40 + [2024] * 40
    te = eval_transfer_entropy_binary(x_seq, y_seq, yrs, n_perm=300)
    assert te["te_mm_bits"] > 0.2
    assert te["onset_lift"] > 0.5
    assert te["p_value"] < 0.01

    # 3. Exact Poisson Changepoint
    counts = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0] + [4, 5, 3, 6, 4, 5, 3, 4, 5, 6, 4, 5]
    wk_dates = [(date(2023, 1, 2) + timedelta(days=7 * i)).isoformat() for i in range(len(counts))]
    cp = eval_changepoint_poisson(counts, wk_dates, min_seg=10, n_perm=300)
    assert cp["sup_2dll"] > 15.0
    assert cp["p_value"] < 0.01
    assert cp["rate_after_wk"] > cp["rate_before_wk"]

    # 4. Pareto Heavy-Tail vs Truncated Lognormal
    rng = random.Random(7)
    pareto_vals = [int(4.5 * ((1.0 - rng.random()) ** (-1.0 / 1.1))) for _ in range(180)] + [1, 2, 3] * 60
    par = eval_pareto_tail(pareto_vals, x_min=5, x0=4.5)
    assert par["n_tail"] >= 150
    assert 1.5 < par["alpha_mle"] < 3.0
    assert isinstance(par["vuong_z"], float)

    # 5. Unlocated region_centroid filter + >45d gap isolation
    evs: list[VerifiedIncident] = []
    for k in range(20):
        d = (date(2024, 1, 1) + timedelta(days=k * 3)).isoformat()
        evs.append(
            VerifiedIncident(
                incident_id=f"loc_{k}",
                date_start=d,
                date_end=d,
                precision_level="exact_locality",
                admin1="Ménaka",
                admin2="Ménaka",
                cercle_id="menaka",
                lat=15.9,
                lon=2.4,
                actor_primary="ISGS / Islamic State Sahel",
                tactic="raid_assault",
                fatalities_best=6 if k % 2 == 0 else 2,
            )
        )
        evs.append(
            VerifiedIncident(
                incident_id=f"cen_{k}",
                date_start=d,
                date_end=d,
                precision_level="cercle_centroid",
                admin1="Bandiagara",
                admin2="Bandiagara",
                cercle_id="bandiagara",
                lat=14.35,
                lon=-3.61,
                actor_primary="JNIM / Katiba Coalition",
                tactic="ambush",
                fatalities_best=1,
            )
        )
    # 10 unlocated 2026 region_centroid events falling back to mopti after a >45d gap
    for k in range(10):
        d = (date(2026, 6, 1) + timedelta(days=k)).isoformat()
        evs.append(
            VerifiedIncident(
                incident_id=f"unloc_{k}",
                date_start=d,
                date_end=d,
                precision_level="region_centroid",
                admin1="Mopti",
                admin2="Mopti",
                cercle_id="mopti",
                lat=14.48,
                lon=-4.18,
                actor_primary="Unattributed Armed Group",
                tactic="other",
                fatalities_best=1,
            )
        )

    dense = extract_dense_segment(evs, gap_days=45)
    assert len(dense) == 40
    assert all(e.cercle_id != "mopti" for e in dense)

    clusters, _ = build_attack_clusters(evs)
    info_proofs, cercle_metrics = derive_information_theoretic_proofs(
        evs, clusters, n_perm_kl=200, n_perm_te=200, n_perm_cp=200
    )
    assert "mopti" not in cercle_metrics
    assert any(h.family == "H_kl_divergence" and h.cercle_id == "menaka" and h.q_value is not None for h in info_proofs)
    assert any(h.hyp_id == "national_mali__H_kl_regime_jsd" for h in info_proofs)


def test_autonomous_proof_debate_and_bundle_export(tmp_path: Path):
    evs: list[VerifiedIncident] = []
    for k in range(18):
        d = (date(2024, 1, 1) + timedelta(days=k * 4)).isoformat()
        evs.append(
            VerifiedIncident(
                incident_id=f"m_{k}",
                date_start=d,
                date_end=d,
                precision_level="exact_locality",
                admin1="Ménaka",
                admin2="Ménaka",
                cercle_id="menaka",
                lat=15.9,
                lon=2.4,
                actor_primary="ISGS / Islamic State Sahel",
                tactic="raid_assault",
                fatalities_best=7 if k % 3 == 0 else 2,
            )
        )
        evs.append(
            VerifiedIncident(
                incident_id=f"b_{k}",
                date_start=d,
                date_end=d,
                precision_level="cercle_centroid",
                admin1="Bandiagara",
                admin2="Bandiagara",
                cercle_id="bandiagara",
                lat=14.35,
                lon=-3.61,
                actor_primary="JNIM / Katiba Coalition",
                tactic="ambush",
                fatalities_best=1,
            )
        )

    clusters, members = build_attack_clusters(evs)
    model_out = run_walk_forward_and_predictions(evs, clusters, members)
    proofs = list(model_out["proofs"])
    open_proofs, ex_rec = asyncio.run(
        run_autonomous_proof_debate(None, evs, clusters, proofs, sweep_id="test_sweep", use_llm=False)
    )
    assert len(open_proofs) >= 1
    assert all(h.origin == "agent_open" and h.family == "H_open" for h in open_proofs)
    assert len(ex_rec.turns) >= 3
    assert {t.role_badge for t in ex_rec.turns} == {"Explorer", "Theorist", "Arbiter"}

    proofs.extend(open_proofs)
    clamped_h = clamp_hypothesis_doc(open_proofs[0])
    assert clamped_h["origin"] == "agent_open"
    assert isinstance(clamped_h["kl_bins"], list)
    clamped_ex = clamp_exchange_doc(ex_rec)
    assert len(clamped_ex["turns"]) == len(ex_rec.turns)

    meta = ObservatoryMeta(
        sweep_id="test_sweep",
        updated_at="2026-11-27T18:30:00Z",
        total_verified_incidents=len(evs),
        total_clusters=len(clusters),
        total_hypotheses=len(proofs),
        agent_exchanges=[t.model_dump() for t in ex_rec.turns],
    )
    exported = export_static_site_bundle(
        tmp_path,
        meta=meta,
        incidents=evs,
        clusters=clusters,
        hypotheses=proofs,
        proofs=model_out["cercle_states"],
        predictions=model_out["predictions"],
        exchanges=ex_rec,
    )
    ex_file = Path(exported["exchanges"])
    assert ex_file.is_file()
    loaded_ex = AgentExchangeRecord.model_validate(json.loads(ex_file.read_text(encoding="utf-8")))
    assert len(loaded_ex.turns) == len(ex_rec.turns)
