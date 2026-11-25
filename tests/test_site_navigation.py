"""Verify bidirectional cross-edition navigation, Chronicle validator, and agent hooks."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE_DIR = ROOT / "site"
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from check_chronicle import _compute_reverse_edges, validate_chronicle  # noqa: E402


def _make_html(
    *,
    data_root: str,
    page_id: str,
    theme: str = "newsprint",
    nav_attr: str = 'data-chronicle-nav="1"',
    extra_body: str = "",
) -> str:
    prefix = "" if data_root == "." else f"{data_root}/"
    return (
        f'<!DOCTYPE html><html lang="en" data-theme="{theme}"><head><meta charset="utf-8"/></head>'
        f'<body data-root="{data_root}" data-page-id="{page_id}">'
        f'<div class="sheet"><div class="folio-bar"><button id="themeToggle"></button></div>'
        f'<header class="masthead"></header>'
        f'<nav class="ed-nav" {nav_attr}></nav>'
        f"{extra_body}"
        f'<footer class="colophon"></footer></div>'
        f'<script src="{prefix}assets/chronicle.js"></script></body></html>'
    )


def _seed_minimal_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "site" / "data").mkdir(parents=True)
    (repo / "site" / "assets").mkdir(parents=True)
    (repo / "site" / "editions" / "vol-01-no-01").mkdir(parents=True)
    (repo / "site" / "editions" / "vol-01-no-02").mkdir(parents=True)
    (repo / "site" / "assets" / "chronicle.js").write_text("console.log('ok');\n", encoding="utf-8")

    (repo / "site" / "index.html").write_text(
        _make_html(
            data_root=".",
            page_id="front",
            extra_body='<a href="editions/vol-01-no-02/index.html">Edition No. 2</a>',
        ),
        encoding="utf-8",
    )
    (repo / "site" / "editions" / "vol-01-no-01" / "index.html").write_text(
        _make_html(data_root="../..", page_id="vol-01-no-01"),
        encoding="utf-8",
    )
    (repo / "site" / "editions" / "vol-01-no-02" / "index.html").write_text(
        _make_html(data_root="../..", page_id="vol-01-no-02"),
        encoding="utf-8",
    )
    manifest = [
        {
            "id": "vol-01-no-02",
            "short_title": "Edition No. 2",
            "issue_num": 2,
            "is_edition": True,
            "parent_edition": None,
            "vol_issue": "VOL. I · NO. 2",
            "date": "2026-11-25",
            "title": "Edition No. 2 Title",
            "summary": "Summary 2",
            "url": "editions/vol-01-no-02/index.html",
            "kind": "Lead Edition",
            "tagCls": "api",
            "builds_on": [{"id": "vol-01-no-01", "note": "Builds on Edition No. 1."}],
        },
        {
            "id": "vol-01-no-01",
            "short_title": "Edition No. 1",
            "issue_num": 1,
            "is_edition": True,
            "parent_edition": None,
            "vol_issue": "VOL. I · NO. 1",
            "date": "2026-11-24",
            "title": "Edition No. 1 Title",
            "summary": "Summary 1",
            "url": "editions/vol-01-no-01/index.html",
            "kind": "Lead Edition",
            "tagCls": "sebeni",
            "builds_on": [],
        },
    ]
    (repo / "site" / "data" / "editions.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return repo


def test_editions_manifest_graph_integrity() -> None:
    manifest = json.loads((SITE_DIR / "data" / "editions.json").read_text(encoding="utf-8"))
    assert len(manifest) >= 5
    by_id = {item["id"]: item for item in manifest}

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
        assert m_root and m_id
        assert m_id.group(1) in known_ids


def test_check_chronicle_passes_on_repo() -> None:
    errors = validate_chronicle(ROOT, check_git_staged=False, check_node=True)
    assert errors == [], f"Unexpected validator errors: {errors}"


def test_check_chronicle_catches_violations(tmp_path: Path) -> None:
    repo = _seed_minimal_repo(tmp_path)
    assert validate_chronicle(repo, check_git_staged=False, check_node=False) == []

    # 1. Unregistered orphan HTML page
    orphan = repo / "site" / "editions" / "orphan.html"
    orphan.write_text(_make_html(data_root="..", page_id="orphan"), encoding="utf-8")
    errs = validate_chronicle(repo, check_git_staged=False, check_node=False)
    assert any("Orphan HTML" in e for e in errs)
    orphan.unlink()

    # 2. Disconnected publication island + layout contract violation
    supp = repo / "site" / "supplements" / "island.html"
    supp.parent.mkdir(parents=True)
    supp.write_text(_make_html(data_root="..", page_id="island", nav_attr=""), encoding="utf-8")
    manifest = json.loads((repo / "site" / "data" / "editions.json").read_text(encoding="utf-8"))
    manifest.append(
        {
            "id": "island",
            "short_title": "Island",
            "issue_num": None,
            "is_edition": False,
            "parent_edition": None,
            "vol_issue": "ISLAND",
            "date": "2026-11-25",
            "title": "Island",
            "summary": "Disconnected",
            "url": "supplements/island.html",
            "kind": "Supplement",
            "tagCls": "slm",
            "builds_on": [],
        }
    )
    (repo / "site" / "data" / "editions.json").write_text(json.dumps(manifest), encoding="utf-8")
    errs = validate_chronicle(repo, check_git_staged=False, check_node=False)
    assert any("Disconnected publication island" in e for e in errs)
    assert any('data-chronicle-nav="1"' in e for e in errs)

    # 3. Stale Front Page when Edition No. 3 exists
    shutil.rmtree(supp.parent)
    ed3_dir = repo / "site" / "editions" / "vol-01-no-03"
    ed3_dir.mkdir(parents=True)
    (ed3_dir / "index.html").write_text(_make_html(data_root="../..", page_id="vol-01-no-03"), encoding="utf-8")
    manifest.pop()
    manifest.insert(
        0,
        {
            "id": "vol-01-no-03",
            "short_title": "Edition No. 3",
            "issue_num": 3,
            "is_edition": True,
            "parent_edition": None,
            "vol_issue": "VOL. I · NO. 3",
            "date": "2026-11-26",
            "title": "Edition No. 3 Title",
            "summary": "Summary 3",
            "url": "editions/vol-01-no-03/index.html",
            "kind": "Lead Edition",
            "tagCls": "api",
            "builds_on": [{"id": "vol-01-no-02", "note": "Continues Edition No. 2."}],
        },
    )
    (repo / "site" / "data" / "editions.json").write_text(json.dumps(manifest), encoding="utf-8")
    errs = validate_chronicle(repo, check_git_staged=False, check_node=False)
    assert any("latest numbered edition 'vol-01-no-03'" in e for e in errs)


def test_antigravity_pre_commit_hook_protocol() -> None:
    hook = ROOT / ".agents" / "hooks" / "pre_commit_gate.py"
    assert hook.is_file()
    for cmd in ("git status --short", 'git commit -m "Dry check"'):
        payload = json.dumps({"toolCall": {"name": "run_command", "args": {"CommandLine": cmd}}})
        res = subprocess.run([sys.executable, str(hook)], input=payload, capture_output=True, text=True, check=True)
        out = json.loads(res.stdout.strip())
        assert out["decision"] == "allow", f"Unexpected deny for {cmd}: {out}"
