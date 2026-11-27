"""Unit tests for Edition No. 4 Living Lexicon Village (`village_lab`)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from village_lab import ALL_ADAPTION_MODELS, ALL_LANGS, BASE_VILLAGERS, COUNCIL_SEATS, CRITIC_SPEC, ORCHESTRATOR_SPEC
from village_lab.cli import _parse_job_langs, _seed_all_languages_to_firestore, _shard_langs_for_task
from village_lab.corpus_miner import CorpusOOVMiner, _is_plausible_bam_line
from village_lab.firestore_sync import (
    FirestoreLivePublisher,
    _serialize_bounded_json_list,
    _to_fs_fields,
    _to_fs_value,
)
from village_lab.hf_sync import build_ed4_hf_viewer_bundle, publish_ed4_site_data, push_ed4_state_to_hf
from village_lab.orthography import check_standardization, classify_lemma_vs_rule, normalize_bambara_dnafla
from village_lab.schema import (
    CouncilDecision,
    CriticVerdict,
    DabaXCheck,
    LanguageVillageState,
    OrthographyCheck,
    serialize_mdf_entry,
)
from village_lab.shadow_verify import verify_shadow_batch, verify_shadow_candidate
from village_lab.world_engine import (
    ARGON_BATCH_MAX,
    ARGON_BATCH_MIN,
    DEFAULT_CANDIDATES_PER_TURN,
    LanguageVillageEngine,
    compute_zone_path,
)


def test_cast_matches_approved_models() -> None:
    assert ORCHESTRATOR_SPEC.name == "Mande"
    assert ORCHESTRATOR_SPEC.model_id == "nvidia/nemotron-3-ultra-550b-a55b"
    assert CRITIC_SPEC.model_id == "gemini-3.5-flash"
    assert len(BASE_VILLAGERS) == 5
    assert len(COUNCIL_SEATS) == 4
    assert len(set(ALL_ADAPTION_MODELS)) == 9


def test_firestore_rest_encoding_and_env_guard(monkeypatch) -> None:
    monkeypatch.delenv("ED4_FIRESTORE_DB", raising=False)
    assert FirestoreLivePublisher.from_env() is None
    assert _to_fs_value(True) == {"booleanValue": True}
    assert _to_fs_value(42) == {"integerValue": "42"}
    assert _to_fs_value(98.75) == {"doubleValue": 98.75}
    fields = _to_fs_fields({"lang": "bam", "running": True, "seq": 7})
    assert fields["running"] == {"booleanValue": True}
    assert fields["seq"] == {"integerValue": "7"}
    assert fields["lang"] == {"stringValue": "bam"}


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


def test_argon_batch_bounds_and_firestore_bounded_json() -> None:
    assert ARGON_BATCH_MIN == 10
    assert ARGON_BATCH_MAX == 15

    # Build 20 oversized decisions (>60KB total) and confirm bounded JSON stays valid & keeps newest
    decisions = []
    for i in range(20):
        decisions.append(
            CouncilDecision(
                turn_id=i + 1,
                lang="bam",
                candidate_raw=f"kuma{i}",
                standardized_lx=f"kuma{i}",
                final_action="REJECT",
                orthography_check=OrthographyCheck(raw_form=f"kuma{i}", standardized_form=f"kuma{i}", is_standard=True),
                dabax_check=DabaXCheck(),
                summary_reason="x" * 3500,
            )
        )
    st = LanguageVillageState(lang="bam", lang_name="Bambara", recent_decisions=decisions)
    fields = FirestoreLivePublisher._build_state_fields(st, running=True, job_execution="exec-1")
    raw_json = fields["recent_decisions_json"]
    assert len(raw_json) <= 48000
    parsed = json.loads(raw_json)
    assert isinstance(parsed, list) and 1 <= len(parsed) < 20
    # Tail (newest turn_id=20) must be preserved
    assert parsed[-1]["turn_id"] == 20
    assert _serialize_bounded_json_list([{"big": "a" * 500}], 100) == "[]"


def test_firestore_seed_if_missing_and_13_lang_cli_sharding(tmp_path: Path, monkeypatch) -> None:
    assert _parse_job_langs("all") == list(ALL_LANGS)
    assert _parse_job_langs("uncovered") == [l for l in ALL_LANGS if l != "bam"]
    assert _shard_langs_for_task(list(ALL_LANGS), 0, 13) == ["bam"]
    assert _shard_langs_for_task(list(ALL_LANGS), 1, 13) == ["bmq"]
    assert _shard_langs_for_task(list(ALL_LANGS), 12, 13) == ["taq"]

    pub = FirestoreLivePublisher(project_id="mlgramv1", database_id="chronicles-live")
    monkeypatch.setattr(pub, "_get_token", lambda force_refresh=False: "fake-tok")

    calls = []

    class _FakeResp:
        def __init__(self, status_code: int, text: str = "{}") -> None:
            self.status_code = status_code
            self.text = text

    def _fake_patch(url, headers=None, params=None, json=None):
        calls.append((url, params, json))
        if url.endswith("/ed4_languages/bam"):
            return _FakeResp(409, "ALREADY_EXISTS")
        return _FakeResp(200, "{}")

    monkeypatch.setattr(pub._client, "patch", _fake_patch)
    seeded = _seed_all_languages_to_firestore(pub, tmp_path / "runs", "exec-seed")
    # 12 non-bam created (200) + 1 bam already exists (409)
    assert seeded == 12
    assert len(calls) == 13
    assert all(c[1] == {"currentDocument.exists": "false"} for c in calls)
    seeded_langs = {c[0].rsplit("/", 1)[-1] for c in calls}
    assert seeded_langs == set(ALL_LANGS)
    pub.close()


def test_push_ed4_state_to_hf_lang_only_avoids_shared_bundle(tmp_path: Path, monkeypatch) -> None:
    import huggingface_hub

    runs_dir = tmp_path / "runs"
    out_dir = tmp_path / "site_data"
    ldir = runs_dir / "bmq"
    ldir.mkdir(parents=True)
    for fname in ("state.json", "augmented.dict", "events.jsonl"):
        (ldir / fname).write_text("{}", encoding="utf-8")

    uploads = []

    class _FakeHfApi:
        def __init__(self, token: str = "") -> None:
            self.token = token

        def create_repo(self, **kwargs) -> None:
            pass

        def upload_folder(self, **kwargs) -> None:
            uploads.append(kwargs)

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)
    push_ed4_state_to_hf("s7d11/chronicles-edition4-state", runs_dir, "bmq", out_dir, "hf-tok", lang_only=True)
    assert len(uploads) == 1
    assert uploads[0]["path_in_repo"] == "edition4/bmq"
    assert not (out_dir / "edition4.json").exists()


def test_proper_noun_filter_and_batch_shadow_dabax(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "bam.txt").write_text(
        "\n".join(
            [
                "Mɔgɔw bɛ taa Dawuda yɔrɔ la bi.",
                "An bɛ Dawuda kɛlɛya fɔ sisan kɔnɔ.",
                "U bɛ furamuguba san furaw la sugu kɔnɔ.",
                "Furamuguba ka ɲi bana caman na fana.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    import village_lab.corpus_miner as cm

    monkeypatch.setattr(cm, "_resolve_raw_corpus_path", lambda lang: raw_dir / "bam.txt")
    eng = LanguageVillageEngine("bam", runs_dir=tmp_path / "runs", mock_mode=True)
    miner = CorpusOOVMiner(
        "bam",
        working_dict_index=eng.working_index,
        working_dict_path=eng.dict_path,
        working_gram_path=eng.gram_path,
    )
    terms = [c.term for c in miner._ranked]
    assert "dawuda" not in terms
    assert "furamuguba" in terms

    # Optimistic batch verify: valid item + broken .gram rule must isolate the failure
    good_mdf = serialize_mdf_entry(
        lx="furamuguba",
        ps="n",
        ge="herbal powder",
        gf="poudre médicinale",
        xv="U bɛ furamuguba san sugu kɔnɔ.",
    )
    res = verify_shadow_batch(
        lang="bam",
        working_dict_path=eng.dict_path,
        working_gram_path=eng.gram_path,
        items=[
            {"candidate_term": "furamuguba", "mdf_block": good_mdf, "va_addition": None, "gram_rule": ""},
            {"candidate_term": "brokenrule", "mdf_block": "", "va_addition": None, "gram_rule": "INVALID SYNTAX NO PATTERN"},
        ],
    )
    assert len(res) == 2
    assert res[0].dict_compile_ok is True
    assert res[1].gram_compile_ok is False
    assert DEFAULT_CANDIDATES_PER_TURN >= 16


