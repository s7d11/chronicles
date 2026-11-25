# The Sebeni Chronicle — Mandatory Agent, Commit & Publication Rules

> **STOP AND READ BEFORE EVERY `git commit`.**
> Every agent operating in this repository MUST read and satisfy these rules before committing or publishing.
> Both `.agents/hooks.json` (Antigravity `PreToolUse` hook) and `.githooks/pre-commit` (Git hook) run `python3 scripts/check_chronicle.py` automatically and block any non-compliant commit.

---

## 1. Mandatory Pre-Commit Gate

Before running `git commit`:

1. Ensure every changed or added page obeys Sections 2–5 below.
2. Run the deterministic site validator:
   ```bash
   python3 scripts/check_chronicle.py
   ```
3. Run the automated test suite whenever code, site assets, navigation, or schemas changed:
   ```bash
   /home/seb/sebenx/bin/python3 -m pytest -q tests/
   ```
4. Never bypass `.githooks/pre-commit` (`git commit --no-verify` is forbidden).
5. On a fresh clone, ensure versioned git hooks are active:
   ```bash
   git config core.hooksPath .githooks
   ```

---

## 2. Single Connected Publication System (Zero Disconnected Pages)

All publications in **The Sebeni Chronicle** form **one unified, bidirectionally linked system**. Every new edition, supplement, or interactive observatory MUST integrate with the rest of the repository so:
- older publications automatically surface links to newer ones, and
- newer publications visibly reference and navigate back to older foundations.

Required manifest contract for `site/data/editions.json`:

1. **Register every HTML page**: Every `.html` file under `site/` (other than `site/index.html`, which uses `data-page-id="front"`) MUST have an entry in `site/data/editions.json` with:
   - `id` (unique slug, e.g. `"vol-01-no-03"`),
   - `short_title` (compact nav label, e.g. `"Edition No. 3"`),
   - `vol_issue`, `date`, `title`, `summary`, `url` (POSIX path relative to `site/`), `kind`, `tagCls`,
   - `is_edition` (`true` for numbered editions, `false` for companions/supplements),
   - `issue_num` (`1`, `2`, `3`, ... for numbered editions; `null` for companions),
   - `parent_edition` (`null` for numbered editions, or parent edition `id` for companions/supplements),
   - `builds_on` (list of `{ "id": "<prior-publication-id>", "note": "<editorial rationale>" }` objects).
2. **Zero orphan HTML files & zero isolated islands**:
   - No unregistered `.html` files may exist anywhere under `site/`.
   - Over undirected edges (`builds_on`, `parent_edition`, and consecutive numbered `issue_num` links), **100% of publications in `site/data/editions.json` MUST form a single connected component**.
3. **Mandatory directional lineage**:
   - Every new numbered edition (`issue_num > 1`) MUST declare at least one `builds_on` edge referencing an earlier publication with a clear `note`.
   - Every companion/supplement (`is_edition: false`) MUST declare `parent_edition` or at least one `builds_on` edge.
4. **Automatic reverse linking**: `site/assets/chronicle.js` reads `site/data/editions.json`, renders the shared top navigation, injects the top editorial lineage banner (**`Newer Edition Published →`** on older issues / **`← Builds On Earlier Dispatches`** on the latest issue), and injects the bottom **Chronicle Continuity & Cross-Edition Dispatch Map** (`Previous Edition | Front Page & Archive | Newer Edition`). Never replace or fork that engine with ad-hoc hardcoded-only navs.

---

## 3. Front Page Promotion Rule (`site/index.html`)

Whenever a new numbered edition (`is_edition: true`) is added:
1. Update `site/index.html` so the **highest `issue_num` edition leads the Front Page** (lead dispatch headline, summary deck, and telemetry), referencing its `url` and `short_title` / `vol_issue`.
2. Keep prior editions reachable in the Archive ledger (`#archive` hydrated from `site/data/editions.json`) and secondary sections (such as the ongoing MALI13 leaderboard).

---

## 4. Newsprint Layout & Visual Consistency Contract

Every HTML page under `site/` MUST preserve the Chronicle broadsheet visual language and shared DOM contract:

1. **Light Newsprint default**: `<html lang="en" data-theme="newsprint">` with `#themeToggle` for optional Evening Edition toggling. Never ship dark-by-default visualizations or pages.
2. **Required DOM shell on every `site/**/*.html` page**:
   - `<body data-root="..." data-page-id="...">` where `data-root` resolves cleanly back to `site/` (`"."`, `".."`, `"../.."`) and `data-page-id` is `"front"` on `site/index.html` or matches `id` in `site/data/editions.json`.
   - `<div class="sheet">` wrapping the broadsheet container.
   - `<div class="folio-bar">` including `id="themeToggle"`.
   - `<nav class="ed-nav" data-chronicle-nav="1">` (containing static fallback links including `Front Page` and the latest edition; external trailing links inside it that must persist across hydration may use `data-keep-nav="1"`).
   - `<header class="masthead">` on every non-fullscreen editorial page (only fullscreen interactive stages tagged `data-fullscreen="1"` may omit `.masthead`).
   - `<footer class="colophon">` at the bottom of `.sheet`.
   - `<script src="<root>/assets/chronicle.js"></script>` matching `data-root`.
3. **Sub-navigation separation**: In-page section anchor links (e.g., `#sec-pareto` on `leaderboard.html`) MUST live in `<nav class="ed-subnav">`, leaving `<nav class="ed-nav" data-chronicle-nav="1">` dedicated to global Chronicle cross-edition navigation.
4. **Static-First GitHub Pages**: All live views on GitHub Pages MUST default to static JSON replay under `site/data/` (`0` automatic Cloud Run requests on page load); any remote backend connection must remain opt-in (e.g. **Connect Live** or `?live=1`).

---

## 5. Git Commit, Security & Attribution Rules

1. **Author identity**: All commits MUST be authored by `s7d11 <sebastcode@outlook.com>`.
2. **No destructive git commands**: Never run `git push --force` / `-f`, `git reset --hard`, `git clean -f`, `git branch -D`, `git checkout .`, or `git restore .` without explicit approval in the active conversation.
3. **Secrets**: Never stage, commit, or print contents of `.env` (`ADOPTION_KEY`, `HF_KEY`, `NVIDIA_KEY`, `SWARM_ADMIN_TOKEN`).
4. **Scope boundary**: Never modify `/home/seb/sebenx/sebeni/`.
5. **Accuracy & Attribution**:
   - Credit **Antigravity Team (`Antigravity for Science`)**, **Adaption Labs**, **NVIDIA Build**, **Hugging Face Datasets**, and **`Gemini 4 Neon-High-Orchestrator`**.
   - Never claim Google Cloud compute credits were provided (only Antigravity for Science access was provided; cloud runs are self-funded), and never reference unrelated GCP project IDs in committed files.
