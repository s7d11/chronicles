# The Sebeni Chronicle

> *Out-of-reach investigations, What-If dispatches, and ~98% AI-generated computational science built with Google Antigravity for Science and Google Cloud — read by humans or their agents.* See [intro.md](intro.md).

**Live Publication:** `https://s7d11.github.io/chronicles/`

## Local Preview

```bash
python3 -m http.server -d site 8080
```

Then open `http://localhost:8080/`.

## Running the MALI13 Benchmark (`dev-mini`)

The `bench/` harness imports the installed `sebeni` (`beni` / `DabaX`) package to build anti-leakage frozen evaluation splits and score models across all 13 national languages of Mali (`MULTI13`).

```bash
# 1. Build or verify frozen dev-mini splits (728 samples + SHA-256 manifest)
python3 -m bench.cli build
python3 -m bench.cli build --verify-only

# 2. Run models with crash-safe resumable checkpointing into runs/mali13/
python3 -m bench.cli run --models all --resume

# 3. Re-score cached runs offline using DabaX(strict_parse=True)
python3 -m bench.cli score

# 4. Publish site/data/leaderboard.json & site/data/samples.json
python3 -m bench.cli publish --out-dir site/data
```

## Adding a New Newspaper Edition

1. Create `site/editions/vol-01-no-XX/index.html` reusing `site/assets/chronicle.css` and `site/assets/chronicle.js`.
2. Append an entry to `site/data/editions.json` (which automatically updates the Archive Ledger on the Front Page).
3. Commit and push to `main`; `.github/workflows/pages.yml` deploys `site/` to GitHub Pages automatically.
