"""Tests for Edition No. 3 proof_lab metrics, 14-language corpus stress, and API routes."""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

from proof_lab import DATASET_LANGS, NIM_MODELS
from proof_lab.corpus import analyze_language_corpus
from proof_lab.metrics_core import (
    MorphTag,
    aligned_mcs,
    allow_promote_strict_ood,
    bounded_mer,
    decomposed_uwec,
    exact_levenshtein_distance,
    exact_mer,
    fsa_accepts,
    index_mcs,
    pfv_sandhi_licensed,
)
from proof_lab.runner import publish_edition3, run_language_proof
from swarm_lab.viz_server import LiveProofController, make_handler_class


def test_metrics_match_lean4_witnesses() -> None:
    # mcs_shift_collapse_witness & stageLcs_resists_shift_witness
    ref_s = [1, 2, 1, 2]
    pred_s = [-1, 1, 2, 1, 2]
    assert index_mcs(pred_s, ref_s) == 0.0
    assert aligned_mcs(pred_s, ref_s) == 4 / 5

    # mer_overgen_exceeds_one_witness & bounded_mer_always_unit_bounded
    ref_m = ["a", "b"]
    hyp_m = ["a", "b", "x", "y", "z"]
    assert exact_levenshtein_distance(ref_m, hyp_m) > len(ref_m)
    assert exact_mer(ref_m, hyp_m) == 1.5
    assert 0.0 <= bounded_mer(ref_m, hyp_m) <= 1.0

    # uwec_current_indicator_prefers_unknown
    u_orig_unk, ind_orig_unk, _ = decomposed_uwec([-1], [0.5], [0.5], invert_indicator=False)
    u_orig_ok, ind_orig_ok, _ = decomposed_uwec([0], [0.5], [0.5], invert_indicator=False)
    assert ind_orig_unk < ind_orig_ok

    u_inv_unk, ind_inv_unk, _ = decomposed_uwec([-1], [0.5], [0.5], invert_indicator=True)
    u_inv_ok, ind_inv_ok, _ = decomposed_uwec([0], [0.5], [0.5], invert_indicator=True)
    assert ind_inv_unk > ind_inv_ok and u_inv_unk > u_inv_ok

    # phi_inflation_without_fsa_validity & sandhi
    assert not fsa_accepts([MorphTag.INFL])
    assert fsa_accepts([MorphTag.STEM, MorphTag.DERIV, MorphTag.INFL])
    assert pfv_sandhi_licensed("nasal", "na")
    assert not pfv_sandhi_licensed("nasal", "ra")

    # strictOOD_blocks_mcs_regression
    assert not allow_promote_strict_ood(mcs=0.9, mcs_prime=0.7, phi=0.5, phi_prime=0.8, mer=0.1)


def test_corpus_stress_in_domain_and_bbo_ood() -> None:
    bam = analyze_language_corpus("bam", max_rows=8)
    bbo = analyze_language_corpus("bbo", max_rows=8)

    assert bam.n_samples == 8 and not bam.is_ood
    assert bbo.n_samples == 8 and bbo.is_ood
    assert len(bam.regimes) == 6
    assert len(bam.spotlights) >= 1

    # Over-generation saturates raw MER > 1 while bounded MER stays <= 1
    assert bam.regimes["overgeneration_drift"].mean_mer_raw > 1.0
    assert bam.regimes["overgeneration_drift"].mean_mer_bounded <= 1.0

    # Leading shift recovers MCS via aligned LCS
    assert bam.regimes["leading_shift"].mean_mcs_aligned > bam.regimes["leading_shift"].mean_mcs_index

    # Truncation corpus MCS exceeds sentence MCS
    assert bam.regimes["truncation_asymmetry"].mean_mcs_corpus > bam.regimes["truncation_asymmetry"].mean_mcs_index

    # In-domain bam has higher strict parse than OOD bbo
    assert bam.clean_strict_parse_pct > bbo.clean_strict_parse_pct


def test_edition3_publish_and_api_routes(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs" / "edition3"
    out_dir = tmp_path / "site" / "data"
    site_dir = tmp_path / "site"
    site_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(run_language_proof("bam", runs_dir=runs_dir, resume=False, max_rows=6, skip_llm=True))
    asyncio.run(run_language_proof("bbo", runs_dir=runs_dir, resume=False, max_rows=6, skip_llm=True))
    pub = publish_edition3(runs_dir=runs_dir, out_dir=out_dir)

    assert pub.langs == ["bam", "bbo"]
    assert pub.total_sentences == 12
    assert (out_dir / "edition3.json").is_file()
    assert (out_dir / "edition3_events.json").is_file()
    assert len(NIM_MODELS) == 4
    assert len(DATASET_LANGS) == 14

    ctrl = LiveProofController(runs_dir=runs_dir, site_dir=site_dir)
    handler = make_handler_class(runs_dir=site_dir, site_dir=site_dir, ed3_runs_dir=runs_dir, ed3_controller=ctrl)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        port = srv.server_address[1]
        with urlopen(f"http://127.0.0.1:{port}/api/ed3/health", timeout=5) as r:
            h = json.loads(r.read().decode("utf-8"))
            assert h["status"] == "ok" and h["edition"] == 3
            assert h["completed_langs"] == ["bam", "bbo"]
        with urlopen(f"http://127.0.0.1:{port}/api/ed3/runs", timeout=5) as r:
            runs = json.loads(r.read().decode("utf-8"))["runs"]
            assert len(runs) == 14
    finally:
        srv.shutdown()
        srv.server_close()
