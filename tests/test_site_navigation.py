"""Verify bidirectional cross-edition navigation metadata and HTML hooks across site/."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE_DIR = ROOT / "site"


def _compute_reverse_edges(editions: list[dict]) -> dict[str, list[str]]:
    refs: dict[str, list[str]] = {item["id"]: [] for item in editions}
    for item in editions:
        src = item["id"]
        for b in item.get("builds_on", []):
            dst = b["id"]
            if src not in refs[dst]:
                refs[dst].append(src)
        parent = item.get("parent_edition")
        if parent and src not in refs[parent]:
            refs[parent].append(src)
    return refs


def test_editions_manifest_graph_integrity() -> None:
    manifest = json.loads((SITE_DIR / "data" / "editions.json").read_text(encoding="utf-8"))
    assert len(manifest) >= 5
    by_id = {}
    for item in manifest:
        pid = item.get("id")
        assert pid and pid not in by_id
        by_id[pid] = item
        assert item.get("short_title")
        assert item.get("vol_issue")
        assert item.get("title")
        target = SITE_DIR / item["url"]
        assert target.is_file(), f"Missing target HTML for {pid}: {target}"

    for item in manifest:
        pid = item["id"]
        parent = item.get("parent_edition")
        if parent is not None:
            assert parent in by_id and parent != pid
        for ref in item.get("builds_on", []):
            rid = ref.get("id")
            assert rid in by_id and rid != pid
            assert ref.get("note")

    # Bidirectional check: Edition No. 2 builds on Edition No. 1 & Supplement -> old pages reference No. 2
    ed2_refs = {r["id"] for r in by_id["vol-01-no-02"].get("builds_on", [])}
    assert "vol-01-no-01" in ed2_refs
    assert "morphotactics-supplement" in ed2_refs

    rev = _compute_reverse_edges(manifest)
    assert "vol-01-no-02" in rev["vol-01-no-01"]
    assert "vol-01-no-02" in rev["morphotactics-supplement"]
    assert "ed2-theater" in rev["vol-01-no-02"]


def test_all_html_pages_wire_cross_edition_nav() -> None:
    manifest = json.loads((SITE_DIR / "data" / "editions.json").read_text(encoding="utf-8"))
    known_ids = {item["id"] for item in manifest} | {"front"}
    html_files = sorted(SITE_DIR.rglob("*.html"))
    assert len(html_files) == 6

    for html_path in html_files:
        text = html_path.read_text(encoding="utf-8")
        m_root = re.search(r'<body[^>]*\bdata-root="([^"]+)"', text)
        m_id = re.search(r'<body[^>]*\bdata-page-id="([^"]+)"', text)
        assert m_root, f"{html_path} missing body data-root"
        assert m_id, f"{html_path} missing body data-page-id"
        data_root = m_root.group(1)
        page_id = m_id.group(1)
        assert page_id in known_ids, f"{html_path} unknown data-page-id={page_id}"

        expected_prefix = "" if data_root == "." else f"{data_root}/"
        assert (
            f'src="{expected_prefix}assets/chronicle.js"' in text
        ), f"{html_path} missing chronicle.js script at {expected_prefix}assets/chronicle.js"
        assert (
            'data-chronicle-nav="1"' in text
        ), f"{html_path} missing data-chronicle-nav=\"1\" on nav.ed-nav"

        editions_resolved = (html_path.parent / data_root / "data" / "editions.json").resolve()
        assert editions_resolved == (SITE_DIR / "data" / "editions.json").resolve()
