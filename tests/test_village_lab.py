"""Unit tests for Edition No. 4 Living Lexicon Village (`village_lab`)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from village_lab import ALL_ADAPTION_MODELS, BASE_VILLAGERS, COUNCIL_SEATS, CRITIC_SPEC, ORCHESTRATOR_SPEC
from village_lab.corpus_miner import CorpusOOVMiner, _is_plausible_bam_line
from village_lab.hf_sync import build_ed4_hf_viewer_bundle, publish_ed4_site_data
from village_lab.orthography import check_standardization, classify_lemma_vs_rule, normalize_bambara_dnafla
from village_lab.schema import CriticVerdict, serialize_mdf_entry
from village_lab.shadow_verify import verify_shadow_candidate
from village_lab.world_engine import LanguageVillageEngine, compute_zone_path


def test_cast_matches_approved_models() -> None:
    assert ORCHESTRATOR_SPEC.name == "Mande"
    assert ORCHESTRATOR_SPEC.model_id == "nvidia/nemotron-3-ultra-550b-a55b"
    assert CRITIC_SPEC.model_id == "gemini-4-argon"
    assert len(BASE_VILLAGERS) == 5
    assert len(COUNCIL_SEATS) == 4
    assert len(set(ALL_ADAPTION_MODELS)) == 9


def test_dnafla_orthography_and_lemma_rule_gates() -> None:
    std_word, notes = normalize_bambara_dnafla("nyègèn")
    assert std_word == "ɲɛgɛn"
    assert any("ɲ" in n for n in notes)

    chk_noise = check_standardization("bonjour", "bam")
    assert not chk_noise.is_standard
    assert chk_noise.route_hint == "reject"

    chk_digit = check_standardization("bam2026", "bam")
    assert not chk_digit.is_standard
    assert chk_digit.route_hint == "reject"

    # Regular plural -w of attested noun should route to .gram rule
    route, reason = classify_lemma_vs_rule("musow", "n", "bam")
    assert route == "gram_rule", reason


def test_waypoint_pathfinding() -> None:
    path = compute_zone_path("millet_fields", "toguna_council")
    assert len(path) >= 2
    assert path[-1] == {"x": 0.0, "z": -3.5}


def test_shadow_dabax_candidate_verification(tmp_path: Path) -> None:
    eng = LanguageVillageEngine("bam", runs_dir=tmp_path / "runs", mock_mode=True)
    mdf = serialize_mdf_entry(
        lx="kɛlɛkɛlɛba",
        ps="n",
        ge="large cricket",
        gf="grand grillon",
        xv="An ye kɛlɛkɛlɛba ye kungo kɔnɔ.",
        xf="Nous avons vu un grand grillon en brousse.",
    )
    chk = verify_shadow_candidate(
        lang="bam",
        working_dict_path=eng.dict_path,
        working_gram_path=eng.gram_path,
        candidate_term="kɛlɛkɛlɛba",
        mdf_block=mdf,
    )
    assert chk.dict_compile_ok is True
    assert chk.gram_compile_ok is True
    assert chk.regression_pct <= 0.5


def test_mock_epoch_spawn_council_and_hf_viewer_bundle(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    out_dir = tmp_path / "site_data"
    hf_stage = tmp_path / "hf_stage"

    eng = LanguageVillageEngine("bam", runs_dir=runs_dir, mock_mode=True)
    base_count = eng.state.baseline_lx_count
    assert base_count >= 13000

    st = asyncio.run(eng.run_epoch(max_turns=2))
    assert st.total_turns == 2
    # Specialist dynamically spawned
    assert any(a.spawned_dynamically for a in st.active_agents)
    # Argon batched cleanly in 1 single call at epoch end
    assert eng.argon_batch_calls == 1
    assert st.total_candidates_reviewed >= 2
    assert st.added_lx_count >= 1
    # Events & council ledger persisted
    evs = [json.loads(ln) for ln in eng.events_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert any(e["kind"] == "agent_spawn" for e in evs)
    assert any(e["kind"] == "agent_move" and len(e["path"]) >= 1 for e in evs)
    assert any(e["kind"] == "critic_review" for e in evs)
    assert any(e["kind"] == "epoch_checkpoint" for e in evs)

    summary = publish_ed4_site_data(runs_dir, out_dir)
    assert (out_dir / "edition4.json").is_file()
    assert (out_dir / "edition4_events.json").is_file()
    assert summary["launch_lang"] == "bam"
    assert len(summary["langs"]) == 13

    counts = build_ed4_hf_viewer_bundle(runs_dir, hf_stage, summary)
    assert counts["language_progress"] == 13
    assert counts["exchanges"] >= 4
    assert counts["events"] >= 10
    card_txt = (hf_stage / "README.md").read_text(encoding="utf-8")
    assert "chronicles/editions/vol-01-no-04/world.html" in card_txt
    assert "Antigravity Team" in card_txt
    assert "huggingface datasets" not in card_txt.lower()
    assert "compute credits" not in card_txt.lower()

    # Every viewer JSONL file must have uniform primitive types per column
    for fname in ("council_ledger.jsonl", "exchanges.jsonl", "events.jsonl", "language_progress.jsonl"):
        rows = [json.loads(ln) for ln in (hf_stage / "viewer_data" / fname).read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not rows:
            continue
        keys = list(rows[0].keys())
        types = {k: type(rows[0][k]) for k in keys}
        for r in rows:
            assert list(r.keys()) == keys
            for k in keys:
                assert type(r[k]) is types[k]


def test_corpus_oov_miner_on_bam_txt(tmp_path: Path) -> None:
    assert not _is_plausible_bam_line("Le ministre a visité le marché central aujourd'hui")
    assert not _is_plausible_bam_line("Saya memohon maaf atas kelewatan ini")
    assert _is_plausible_bam_line("A bɛ taa sugu kɔnɔ ka malɔ san")

    eng = LanguageVillageEngine("bam", runs_dir=tmp_path / "runs", mock_mode=True)
    miner = CorpusOOVMiner(
        "bam",
        working_dict_index=eng.working_index,
        working_dict_path=eng.dict_path,
        working_gram_path=eng.gram_path,
        max_scan_lines=8000,
    )
    if miner.raw_path is None:
        return
    batch = miner.next_batch(5)
    assert len(batch) == 5
    for cand in batch:
        assert len(cand.term) >= 4
        assert cand.freq >= 2
        assert cand.source_file == "bam.txt"
        assert cand.example_sentence

