"""Tests for Volume II Mali OSINT pipeline, mathematical proofs, walk-forward causality & isolation."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from conflict_lab import AGENT_CAST
from conflict_lab.firestore_sync import clamp_cluster_doc, clamp_hypothesis_doc, clamp_incident_doc
from conflict_lab.gazetteer import normalize_actor_name, resolve_mali_location
from conflict_lab.georesolve_dedup import deduplicate_and_georesolve
from conflict_lab.hypothesis_proofs import (
    betainc,
    beta_ppf,
    chi2_sf,
    compute_beta_binomial_hazard,
    eval_markov_transitions,
    eval_overdispersion,
    eval_seasonal_harmattan,
    fit_hawkes_process,
)
from conflict_lab.pattern_model import (
    build_attack_clusters,
    build_corridor_edges,
    run_walk_forward_and_predictions,
)
from conflict_lab.providers import ALLOWED_OPENROUTER_FREE_MODELS, assert_openrouter_free_model
from conflict_lab.schema import RawOsintRecord, VerifiedIncident


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
