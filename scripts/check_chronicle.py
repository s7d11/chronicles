#!/usr/bin/env python3
"""Deterministic validator for The Sebeni Chronicle commit, layout, and connectivity rules."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

FORBIDDEN_TERM = "gravity" + "x"


def _compute_reverse_edges(editions: list[dict]) -> dict[str, list[str]]:
    refs: dict[str, list[str]] = {item["id"]: [] for item in editions}
    for item in editions:
        src = item["id"]
        for ref in item.get("builds_on", []):
            dst = ref["id"]
            if src not in refs[dst]:
                refs[dst].append(src)
        parent = item.get("parent_edition")
        if parent and src not in refs[parent]:
            refs[parent].append(src)
    return refs


def validate_chronicle(
    repo_root: Path,
    *,
    check_git_staged: bool = True,
    check_node: bool = True,
) -> list[str]:
    """Validate layout contracts, publication connectivity, Front Page promotion, and hygiene."""
    errors: list[str] = []
    site_dir = repo_root / "site"
    editions_path = site_dir / "data" / "editions.json"

    if not editions_path.is_file():
        return [f"Missing publication manifest: {editions_path}"]

    try:
        manifest = json.loads(editions_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"Invalid JSON in {editions_path}: {exc}"]

    if not isinstance(manifest, list) or not manifest:
        return [f"{editions_path} must be a non-empty JSON list."]

    # 1. Schema & target file integrity
    by_id: dict[str, dict] = {}
    urls_to_id: dict[str, str] = {}
    required_str_fields = ("id", "short_title", "vol_issue", "date", "title", "summary", "url", "kind", "tagCls")
    for idx, item in enumerate(manifest):
        if not isinstance(item, dict):
            errors.append(f"editions.json[{idx}] must be an object.")
            continue
        for field in required_str_fields:
            val = item.get(field)
            if not isinstance(val, str) or not val.strip():
                errors.append(f"editions.json[{idx}] missing non-empty string field '{field}'.")
        pid = item.get("id")
        if isinstance(pid, str) and pid:
            if pid in by_id:
                errors.append(f"Duplicate publication id in editions.json: '{pid}'.")
            else:
                by_id[pid] = item
        if not isinstance(item.get("is_edition"), bool):
            errors.append(f"editions.json['{pid or idx}'] missing boolean 'is_edition'.")
        elif item["is_edition"] and not isinstance(item.get("issue_num"), int):
            errors.append(f"editions.json['{pid or idx}'] is_edition=true must have integer 'issue_num'.")
        if not isinstance(item.get("builds_on"), list):
            errors.append(f"editions.json['{pid or idx}'] missing list field 'builds_on'.")

        rel_url = str(item.get("url") or "").split("#", 1)[0]
        target = site_dir / rel_url
        if not rel_url or not target.is_file():
            errors.append(f"editions.json['{pid or idx}'] url '{rel_url}' does not exist under site/.")
        else:
            if rel_url in urls_to_id:
                errors.append(f"Duplicate url '{rel_url}' in editions.json.")
            urls_to_id[rel_url] = str(pid or idx)

    # Validate directional edges & per-type lineage rules
    for pid, item in by_id.items():
        parent = item.get("parent_edition")
        if parent is not None:
            if parent == pid:
                errors.append(f"Publication '{pid}' cannot set parent_edition to itself.")
            elif parent not in by_id:
                errors.append(f"Publication '{pid}' references unknown parent_edition '{parent}'.")

        builds = item.get("builds_on") or []
        for b_idx, ref in enumerate(builds):
            if not isinstance(ref, dict):
                errors.append(f"Publication '{pid}' builds_on[{b_idx}] must be an object {{id, note}}.")
                continue
            rid = ref.get("id")
            note = ref.get("note")
            if rid == pid:
                errors.append(f"Publication '{pid}' builds_on[{b_idx}] cannot reference itself.")
            elif rid not in by_id:
                errors.append(f"Publication '{pid}' builds_on references unknown id '{rid}'.")
            if not isinstance(note, str) or not note.strip():
                errors.append(f"Publication '{pid}' builds_on['{rid}'] must include a non-empty 'note'.")

        if item.get("is_edition") and isinstance(item.get("issue_num"), int) and item["issue_num"] > 1:
            if len(builds) < 1:
                errors.append(
                    f"Numbered edition '{pid}' (issue_num={item['issue_num']}) must declare at least one 'builds_on' reference to earlier work."
                )
        elif item.get("is_edition") is False:
            if parent is None and len(builds) < 1:
                errors.append(
                    f"Companion/supplement '{pid}' must declare 'parent_edition' or at least one 'builds_on' edge (cannot be disconnected)."
                )

    # 2. Zero orphan HTML files
    html_files = sorted(site_dir.rglob("*.html")) if site_dir.is_dir() else []
    if not (site_dir / "index.html").is_file():
        errors.append("Missing Front Page site/index.html.")
    for hp in html_files:
        rel_posix = hp.relative_to(site_dir).as_posix()
        if rel_posix == "index.html":
            continue
        if rel_posix not in urls_to_id:
            errors.append(f"Orphan HTML file not registered in site/data/editions.json: 'site/{rel_posix}'.")

    # 3. Single connected component check across all publications
    if by_id and not any("unknown" in e for e in errors):
        adj: dict[str, set[str]] = {pid: set() for pid in by_id}
        for pid, item in by_id.items():
            parent = item.get("parent_edition")
            if isinstance(parent, str) and parent in by_id:
                adj[pid].add(parent)
                adj[parent].add(pid)
            for ref in item.get("builds_on") or []:
                rid = ref.get("id") if isinstance(ref, dict) else None
                if isinstance(rid, str) and rid in by_id:
                    adj[pid].add(rid)
                    adj[rid].add(pid)
        numbered = sorted(
            [item for item in by_id.values() if item.get("is_edition") and isinstance(item.get("issue_num"), int)],
            key=lambda x: x["issue_num"],
        )
        for a, b in zip(numbered, numbered[1:]):
            adj[a["id"]].add(b["id"])
            adj[b["id"]].add(a["id"])

        start_id = next(iter(by_id))
        visited: set[str] = {start_id}
        q = deque([start_id])
        while q:
            cur = q.popleft()
            for nxt in adj[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)
        disconnected = sorted(set(by_id) - visited)
        if disconnected:
            errors.append(
                f"Disconnected publication island(s) in site/data/editions.json: {', '.join(disconnected)}."
            )

    # 4. Layout & cross-edition wiring contract on every HTML page
    for hp in html_files:
        rel_posix = hp.relative_to(site_dir).as_posix()
        text = hp.read_text(encoding="utf-8", errors="ignore")

        if not re.search(r'<html\b[^>]*\bdata-theme="newsprint"', text):
            errors.append(f"site/{rel_posix}: <html> must default to data-theme=\"newsprint\".")

        m_root = re.search(r'<body\b[^>]*\bdata-root="([^"]+)"', text)
        m_id = re.search(r'<body\b[^>]*\bdata-page-id="([^"]+)"', text)
        if not m_root or not m_id:
            errors.append(f"site/{rel_posix}: <body> must declare data-root and data-page-id.")
        else:
            data_root = m_root.group(1)
            page_id = m_id.group(1)
            expected_id = "front" if rel_posix == "index.html" else urls_to_id.get(rel_posix)
            if expected_id and page_id != expected_id:
                errors.append(f"site/{rel_posix}: data-page-id='{page_id}' does not match expected '{expected_id}'.")

            if (hp.parent / data_root / "data" / "editions.json").resolve() != editions_path.resolve():
                errors.append(f"site/{rel_posix}: data-root='{data_root}' does not resolve to site/data/editions.json.")

            prefix = "" if data_root == "." else f"{data_root.rstrip('/')}/"
            if f'src="{prefix}assets/chronicle.js"' not in text:
                errors.append(f"site/{rel_posix}: missing <script src=\"{prefix}assets/chronicle.js\">.")

        for marker, label in (
            ('class="sheet"', 'class="sheet"'),
            ('class="folio-bar"', 'class="folio-bar"'),
            ('id="themeToggle"', 'id="themeToggle"'),
            ('data-chronicle-nav="1"', 'data-chronicle-nav="1"'),
            ('<footer class="colophon"', '<footer class="colophon">'),
        ):
            if marker not in text:
                errors.append(f"site/{rel_posix}: missing required Newsprint layout element {label}.")

        is_fullscreen = 'data-fullscreen="1"' in text
        if not is_fullscreen and '<header class="masthead"' not in text:
            errors.append(f"site/{rel_posix}: non-fullscreen page missing <header class=\"masthead\">.")

    # 5. Front Page promotion check
    front_html_path = site_dir / "index.html"
    numbered = [item for item in by_id.values() if item.get("is_edition") and isinstance(item.get("issue_num"), int)]
    if numbered and front_html_path.is_file():
        latest = max(numbered, key=lambda x: x["issue_num"])
        front_text = front_html_path.read_text(encoding="utf-8", errors="ignore")
        if latest["url"] not in front_text:
            errors.append(
                f"site/index.html must link to the latest numbered edition '{latest['id']}' ({latest['url']})."
            )
        if latest["short_title"].lower() not in front_text.lower() and latest["vol_issue"].lower() not in front_text.lower():
            errors.append(
                f"site/index.html must reference the latest numbered edition '{latest['short_title']}'."
            )

    # 6. Forbidden term & hygiene checks
    for scope_name in ("site", "swarm_lab", "deploy", "README.md", "AGENTS.md"):
        scope = repo_root / scope_name
        if scope.is_file():
            targets = [scope]
        elif scope.is_dir():
            targets = [
                p
                for p in scope.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".png", ".jpg"}
            ]
        else:
            targets = []
        for p in targets:
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if FORBIDDEN_TERM in content.lower():
                errors.append(f"Forbidden legacy project reference found in {p.relative_to(repo_root)}.")

    if check_git_staged and (repo_root / ".git").exists():
        try:
            staged = subprocess.check_output(
                ["git", "diff", "--cached", "--name-only"],
                cwd=str(repo_root),
                text=True,
                stderr=subprocess.DEVNULL,
            ).splitlines()
            for fname in staged:
                base = Path(fname).name
                if base == ".env" or base.endswith(".env"):
                    errors.append(f"Secret file '{fname}' must never be staged or committed.")
        except Exception:
            pass

    if check_node and site_dir.is_dir() and shutil.which("node"):
        for js_file in sorted((site_dir / "assets").glob("*.js")):
            proc = subprocess.run(
                ["node", "--check", str(js_file)],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                errors.append(f"JS syntax check failed for {js_file.relative_to(repo_root)}: {proc.stderr.strip()}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Chronicle commit, layout & connectivity rules.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--skip-git-staged", action="store_true")
    parser.add_argument("--skip-node", action="store_true")
    args = parser.parse_args()

    errs = validate_chronicle(
        Path(args.repo_root).resolve(),
        check_git_staged=not args.skip_git_staged,
        check_node=not args.skip_node,
    )
    if errs:
        print("Chronicle pre-commit validation FAILED (see AGENTS.md):", file=sys.stderr)
        for err in errs:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print("Chronicle validation passed: connected publication graph, Newsprint layout & hygiene OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
