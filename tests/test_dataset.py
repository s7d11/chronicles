"""Tests for frozen MALI13 dev-mini splits and anti-leakage guarantees."""

from __future__ import annotations

import json
from pathlib import Path

from bench import LANGS
from bench.dataset import DEFAULT_OUT_DIR, audit_zero_overlap, load_split, verify_manifest


def test_dev_mini_manifest_and_counts():
    manifest = verify_manifest(DEFAULT_OUT_DIR)
    assert manifest.tier == "dev-mini"
    assert manifest.seed == 2026
    assert manifest.anti_leakage_overlap == 0
    assert manifest.langs == LANGS
    assert manifest.total_rows == 728
    assert manifest.files["morph.jsonl"]["rows"] == 130
    assert manifest.files["gen.jsonl"]["rows"] == 104
    assert manifest.files["lid.jsonl"]["rows"] == 130
    assert manifest.files["cloze.jsonl"]["rows"] == 104
    assert manifest.files["lm_efficiency.jsonl"]["rows"] == 260


def test_all_13_languages_and_no_bbo():
    expected = {
        "morph": 10,
        "gen": 8,
        "lid": 10,
        "cloze": 8,
        "efficiency": 20,
    }
    for task, per_lang in expected.items():
        rows = load_split(DEFAULT_OUT_DIR, task)
        counts = {}
        for r in rows:
            assert r.lang in LANGS
            assert r.lang != "bbo"
            counts[r.lang] = counts.get(r.lang, 0) + 1
        assert set(counts.keys()) == set(LANGS)
        for lang in LANGS:
            assert counts[lang] == per_lang, f"{task}:{lang} has {counts[lang]} != {per_lang}"


def test_anti_leakage_zero_overlap():
    assert audit_zero_overlap(DEFAULT_OUT_DIR) == 0


def test_cloze_choices_distinct():
    rows = load_split(DEFAULT_OUT_DIR, "cloze")
    for r in rows:
        assert "[MASK]" in r.masked_text
        assert r.answer in {"A", "B", "C", "D"}
        assert r.choices[r.answer] == r.target
        assert len({v.lower() for v in r.choices.values()}) == 4
