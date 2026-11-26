"""Tests for Hugging Face Dataset Viewer bundle & Dataset Card generation."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.publish_hf_viewer import build_hf_viewer_bundles

REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse_frontmatter(markdown_text: str) -> dict:
    assert markdown_text.startswith("---\n"), "Missing YAML frontmatter start"
    _, rest = markdown_text.split("---\n", 1)
    fm_raw, _body = rest.split("\n---\n", 1)
    return yaml.safe_load(fm_raw)


def _assert_uniform_jsonl(path: Path, expected_rows: int) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == expected_rows, f"{path.name}: expected {expected_rows} rows, got {len(lines)}"
    rows = [json.loads(line) for line in lines]
    keys = list(rows[0].keys())
    col_types = {k: type(rows[0][k]) for k in keys}
    for k, typ in col_types.items():
        assert typ in (str, int, float, bool), f"{path.name}.{k} has non-primitive type {typ}"
    for idx, row in enumerate(rows):
        assert list(row.keys()) == keys, f"{path.name} row {idx} key mismatch"
        for k in keys:
            assert type(row[k]) is col_types[k], (
                f"{path.name} row {idx} col {k} type {type(row[k])} != {col_types[k]}"
            )


def test_build_hf_viewer_bundles_schemas_and_cards(tmp_path: Path) -> None:
    bundles = build_hf_viewer_bundles(tmp_path)

    ed2 = bundles["edition2"]
    assert ed2["counts"] == {"sessions": 117, "leaderboard": 9, "events": 1831}
    ed3 = bundles["edition3"]
    assert ed3["counts"] == {
        "deductions": 56,
        "stress_regimes": 84,
        "events": 154,
        "lean_theorems": 10,
    }

    for key, info in bundles.items():
        card_path = info["dir"] / "README.md"
        assert card_path.is_file()
        card_text = card_path.read_text(encoding="utf-8")
        fm = _parse_frontmatter(card_text)

        assert isinstance(fm.get("configs"), list) and len(fm["configs"]) == len(info["counts"])
        assert any(c.get("default") is True for c in fm["configs"])
        for cfg in fm["configs"]:
            for df in cfg["data_files"]:
                rel_path = df["path"]
                target = info["dir"] / rel_path
                assert target.is_file(), f"{key}: missing config file {rel_path}"
                cfg_name = cfg["config_name"]
                _assert_uniform_jsonl(target, info["counts"][cfg_name])

        assert "https://s7d11.github.io/chronicles/" in card_text
        assert "@misc{" in card_text
        assert "author       = {s7d11}" in card_text
        assert "Antigravity Team" in card_text
        assert "compute credits" not in card_text.lower()
        assert "huggingface datasets" not in card_text.lower()


def test_repo_readme_citations_and_attributions() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for cite_key in (
        "s7d11_sebeni_chronicle_2026",
        "s7d11_sebeni_chronicle_ed1_2026",
        "s7d11_sebeni_chronicle_ed2_2026",
        "s7d11_sebeni_chronicle_ed3_2026",
    ):
        assert cite_key in readme
    assert "Antigravity Team" in readme
    assert "compute credits" not in readme.lower()
    assert "huggingface datasets" not in readme.lower()
