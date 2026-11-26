#!/usr/bin/env python3
"""Build and optionally upload Hugging Face Dataset Viewer splits and Cards.

Reads the static Chronicle snapshots under ``site/data/edition2*.json`` and
``site/data/edition3*.json``, exports strictly-typed flat JSONL tables under
``viewer_data/`` plus a YAML-configured ``README.md`` Dataset Card for both
public repositories:

- ``s7d11/chronicles-edition2-state``
- ``s7d11/chronicles-edition3-state``

Existing checkpoint directories (``edition2/``, ``edition3/``, ``site_data/``)
on Hugging Face are left untouched so ``sync-from-hf`` and Cloud Run restore
remain fully compatible.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
SITE_DATA_DIR = REPO_ROOT / "site" / "data"

CHRONICLE_URL = "https://s7d11.github.io/chronicles/"
GITHUB_URL = "https://github.com/s7d11/chronicles"

ED2_REPO = "s7d11/chronicles-edition2-state"
ED3_REPO = "s7d11/chronicles-edition3-state"
PRIVATE_RUNTIME_REPO = "s7d11/sebeni-runtime-private"


def _s(value: Any) -> str:
    """Coerce value cleanly to a string (empty string for None)."""
    if value is None:
        return ""
    return str(value)


def _i(value: Any, default: int = 0) -> int:
    """Coerce value cleanly to an int."""
    if value is None or isinstance(value, bool):
        return int(value) if isinstance(value, bool) else default
    return int(value)


def _f(value: Any, default: float = 0.0) -> float:
    """Coerce value cleanly to a float."""
    if value is None or isinstance(value, bool):
        return float(int(value)) if isinstance(value, bool) else default
    return float(value)


def _b(value: Any, default: bool = False) -> bool:
    """Coerce value cleanly to a bool."""
    if value is None:
        return default
    return bool(value)


def _j(value: Any, default: Any = None) -> str:
    """Serialize nested metadata cleanly as compact UTF-8 JSON text."""
    return json.dumps(default if value is None else value, ensure_ascii=False, separators=(",", ":"))


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    """Write rows as UTF-8 JSONL and return row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _load_json(path: Path) -> Any:
    """Load JSON file from disk."""
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _extract_ed2_rows(
    ed2: Dict[str, Any], ed2_events: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    """Flatten Edition No. 2 JSON payloads into uniform viewer tables."""
    sessions: List[Dict[str, Any]] = []
    for lang_key, lang_obj in ed2.get("languages", {}).items():
        lang_code = _s(lang_obj.get("lang") or lang_key)
        lang_name = _s(lang_obj.get("lang_name"))
        for cluster in lang_obj.get("clusters", []):
            plan = cluster.get("plan") or {}
            shadow = cluster.get("shadow") or {}
            native_gen = cluster.get("native_gen") or {}
            sessions.append(
                {
                    "lang": lang_code,
                    "lang_name": lang_name,
                    "cluster_slug": _s(cluster.get("cluster_slug")),
                    "model_name": _s(cluster.get("model_name")),
                    "n_agents": _i(plan.get("n_agents")),
                    "topology": _s(plan.get("topology")),
                    "plan_rationale": _s(plan.get("rationale")),
                    "rounds_used": _i(cluster.get("rounds_used")),
                    "unanimous": _b(cluster.get("unanimous")),
                    "agreement_pct": _f(cluster.get("agreement_pct")),
                    "n_exchanges": _i(cluster.get("n_exchanges")),
                    "dict_entries_added": _i(shadow.get("dict_entries_added")),
                    "gram_rules_proposed": _i(shadow.get("gram_rules_proposed")),
                    "dict_compile_ok": _b(shadow.get("dict_compile_ok")),
                    "gram_compile_ok": _b(shadow.get("gram_compile_ok")),
                    "baseline_test_parse_pct": _f(shadow.get("baseline_test_parse_pct")),
                    "shadow_test_parse_pct": _f(shadow.get("shadow_test_parse_pct")),
                    "test_delta_pct": _f(shadow.get("test_delta_pct")),
                    "baseline_devmini_parse_pct": _f(shadow.get("baseline_devmini_parse_pct")),
                    "shadow_devmini_parse_pct": _f(shadow.get("shadow_devmini_parse_pct")),
                    "devmini_delta_pct": _f(shadow.get("devmini_delta_pct")),
                    "phantom_rate_pct": _f(shadow.get("phantom_rate_pct")),
                    "native_baseline_parse_pct": _f(native_gen.get("baseline_parse_pct")),
                    "native_shadow_parse_pct": _f(native_gen.get("shadow_parse_pct")),
                    "native_leakage_pct": _f(native_gen.get("leakage_pct")),
                    "native_header_compliance_pct": _f(native_gen.get("header_compliance")),
                    "native_verified_by_peer": _b(native_gen.get("verified_by_peer")),
                    "coordinator_score": _f(cluster.get("coordinator_score")),
                    "coordinator_note": _s(cluster.get("coordinator_note")),
                    "native_command": _s(native_gen.get("command_template")),
                    "native_output": _s(native_gen.get("output_text")),
                    "dict_deltas_json": _j(cluster.get("dict_deltas"), []),
                    "gram_rules_json": _j(cluster.get("gram_rules"), []),
                }
            )

    leaderboard: List[Dict[str, Any]] = []
    for row in ed2.get("leaderboard", []):
        leaderboard.append(
            {
                "cluster_slug": _s(row.get("cluster_slug")),
                "model_name": _s(row.get("model_name")),
                "langs_completed": _i(row.get("langs_completed")),
                "avg_agents": _f(row.get("avg_agents")),
                "dominant_topology": _s(row.get("dominant_topology")),
                "topology_counts_json": _j(row.get("topology_counts"), {}),
                "unanimous_rate_pct": _f(row.get("unanimous_rate_pct")),
                "avg_agreement_pct": _f(row.get("avg_agreement_pct")),
                "avg_rounds": _f(row.get("avg_rounds")),
                "total_dict_added": _i(row.get("total_dict_added")),
                "compile_ok_rate_pct": _f(row.get("compile_ok_rate_pct")),
                "avg_test_delta_pct": _f(row.get("avg_test_delta_pct")),
                "avg_devmini_delta_pct": _f(row.get("avg_devmini_delta_pct")),
                "avg_native_parse_pct": _f(row.get("avg_native_parse_pct")),
                "avg_header_compliance_pct": _f(row.get("avg_header_compliance_pct")),
                "avg_coordinator_score": _f(row.get("avg_coordinator_score")),
            }
        )

    events: List[Dict[str, Any]] = []
    for lang_key, ev_list in ed2_events.get("events_by_lang", {}).items():
        for ev in ev_list:
            events.append(
                {
                    "lang": _s(ev.get("lang") or lang_key),
                    "seq": _i(ev.get("seq")),
                    "ts": _f(ev.get("ts")),
                    "cluster": _s(ev.get("cluster")),
                    "kind": _s(ev.get("kind")),
                    "src": _s(ev.get("src")),
                    "dst": _s(ev.get("dst")),
                    "round_num": _i(ev.get("round_num")),
                    "label": _s(ev.get("label")),
                    "summary": _s(ev.get("summary")),
                    "meta_json": _j(ev.get("meta"), {}),
                }
            )

    return {"sessions": sessions, "leaderboard": leaderboard, "events": events}


def _extract_ed3_rows(
    ed3: Dict[str, Any], ed3_events: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    """Flatten Edition No. 3 JSON payloads into uniform viewer tables."""
    deductions: List[Dict[str, Any]] = []
    stress_regimes: List[Dict[str, Any]] = []

    for lang_key, lang_obj in ed3.get("languages", {}).items():
        corpus = lang_obj.get("corpus") or {}
        lang_code = _s(lang_obj.get("lang") or corpus.get("lang") or lang_key)
        lang_name = _s(corpus.get("lang_name"))
        family = _s(corpus.get("family"))
        is_ood = _b(corpus.get("is_ood"))
        dabax_baseline = _s(corpus.get("dabax_baseline"))

        models_raw = lang_obj.get("models", [])
        models_iter = models_raw.values() if isinstance(models_raw, dict) else models_raw
        for model in models_iter:
            cited = model.get("lean_theorems_cited") or []
            cited_str = ", ".join(_s(x) for x in cited) if isinstance(cited, list) else _s(cited)
            deductions.append(
                {
                    "lang": lang_code,
                    "lang_name": lang_name,
                    "family": family,
                    "is_ood": is_ood,
                    "dabax_baseline": dabax_baseline,
                    "model_slug": _s(model.get("model_slug")),
                    "model_name": _s(model.get("model_name")),
                    "model_id": _s(model.get("model_id")),
                    "params_label": _s(model.get("params_label")),
                    "verdict": _s(model.get("verdict")),
                    "effectiveness_score": _f(model.get("effectiveness_score")),
                    "oracle_verified_count": _i(model.get("oracle_verified_count")),
                    "oracle_total_count": _i(model.get("oracle_total_count")),
                    "round1_latency_s": _f(model.get("round1_latency_s")),
                    "round2_latency_s": _f(model.get("round2_latency_s")),
                    "round1_error": _s(model.get("round1_error")),
                    "round2_error": _s(model.get("round2_error")),
                    "round1_summary": _s(model.get("round1_summary")),
                    "mer_assessment": _s(model.get("mer_assessment")),
                    "mcs_assessment": _s(model.get("mcs_assessment")),
                    "uwec_assessment": _s(model.get("uwec_assessment")),
                    "lean_theorems_cited": cited_str,
                    "key_evidence": _s(model.get("key_evidence")),
                    "recommended_hardening": _s(model.get("recommended_hardening")),
                    "round1_reasoning": _s(model.get("round1_reasoning")),
                    "round2_reasoning": _s(model.get("round2_reasoning")),
                    "probes_json": _j(model.get("probes"), []),
                }
            )

        regimes_raw = corpus.get("regimes") or {}
        regimes_iter = regimes_raw.values() if isinstance(regimes_raw, dict) else regimes_raw
        for regime in regimes_iter:
            stress_regimes.append(
                {
                    "lang": lang_code,
                    "lang_name": lang_name,
                    "family": family,
                    "is_ood": is_ood,
                    "dabax_baseline": dabax_baseline,
                    "regime_id": _s(regime.get("regime_id")),
                    "label": _s(regime.get("label")),
                    "theorem_ref": _s(regime.get("theorem_ref")),
                    "n_sentences": _i(regime.get("n_sentences")),
                    "mean_mer_raw": _f(regime.get("mean_mer_raw")),
                    "mean_mer_bounded": _f(regime.get("mean_mer_bounded")),
                    "mer_gt_one_pct": _f(regime.get("mer_gt_one_pct")),
                    "mean_mcs_index": _f(regime.get("mean_mcs_index")),
                    "mean_mcs_aligned": _f(regime.get("mean_mcs_aligned")),
                    "mean_mcs_corpus": _f(regime.get("mean_mcs_corpus")),
                    "mean_uwec_orig": _f(regime.get("mean_uwec_orig")),
                    "mean_uwec_inv": _f(regime.get("mean_uwec_inv")),
                    "mean_phi": _f(regime.get("mean_phi")),
                    "mean_strict_parse": _f(regime.get("mean_strict_parse")),
                    "strict_ood_reject_pct": _f(regime.get("strict_ood_reject_pct")),
                }
            )

    events: List[Dict[str, Any]] = []
    for lang_key, ev_list in ed3_events.get("events_by_lang", {}).items():
        for ev in ev_list:
            events.append(
                {
                    "lang": _s(ev.get("lang") or lang_key),
                    "seq": _i(ev.get("seq")),
                    "ts": _s(ev.get("ts")),
                    "kind": _s(ev.get("kind")),
                    "model_slug": _s(ev.get("model_slug")),
                    "model_name": _s(ev.get("model_name")),
                    "label": _s(ev.get("label")),
                    "summary": _s(ev.get("summary")),
                    "reasoning_excerpt": _s(ev.get("reasoning_excerpt")),
                    "metrics_json": _j(ev.get("metrics"), {}),
                }
            )

    lean_theorems: List[Dict[str, Any]] = []
    for thm in ed3.get("lean_theorems", []):
        lean_theorems.append(
            {
                "file": _s(thm.get("file")),
                "name": _s(thm.get("name")),
                "statement": _s(thm.get("statement")),
                "meaning": _s(thm.get("meaning")),
            }
        )

    return {
        "deductions": deductions,
        "stress_regimes": stress_regimes,
        "events": events,
        "lean_theorems": lean_theorems,
    }


def _render_ed2_card(counts: Dict[str, int]) -> str:
    """Build Dataset Card README.md for s7d11/chronicles-edition2-state."""
    return f"""---
pretty_name: "The Sebeni Chronicle — Edition No. 2: Swarm Linguistics Lab"
license: mit
language:
  - multilingual
  - en
  - bm
  - ff
task_categories:
  - text-generation
  - other
tags:
  - mali
  - african-languages
  - computational-linguistics
  - morphological-analysis
  - multi-agent
  - sebeni-chronicle
configs:
  - config_name: sessions
    default: true
    data_files:
      - split: train
        path: viewer_data/sessions.jsonl
  - config_name: leaderboard
    data_files:
      - split: train
        path: viewer_data/leaderboard.jsonl
  - config_name: events
    data_files:
      - split: train
        path: viewer_data/events.jsonl
---

# The Sebeni Chronicle — Edition No. 2: Swarm Linguistics Lab

> **Read the full interactive broadsheet dispatch:**
> - **Chronicle Front Page:** [{CHRONICLE_URL}]({CHRONICLE_URL})
> - **Edition No. 2 Dispatch:** [{CHRONICLE_URL}editions/vol-01-no-02/]({CHRONICLE_URL}editions/vol-01-no-02/)
> - **Live Swarm Theater:** [{CHRONICLE_URL}editions/vol-01-no-02/theater.html]({CHRONICLE_URL}editions/vol-01-no-02/theater.html)
> - **Source Repository:** [{GITHUB_URL}]({GITHUB_URL})

## About This Dataset

This dataset archives the complete outputs of **Edition No. 2** of *The Sebeni Chronicle*. Nine multi-agent LLM clusters collaborated across all **13 national languages of Mali** (`bam`, `bmq`, `boz`, `dtm`, `ful`, `kao`, `mey`, `mku`, `myk`, `ses`, `snk`, `spp`, `taq`) to diagnose parser gaps, propose verified Shadow `DabaX` dictionary additions (**+96** compiled headwords with strict non-regression checks), and compose original native-register dispatches verified by peer review.

Visit [Edition No. 2 on The Sebeni Chronicle]({CHRONICLE_URL}editions/vol-01-no-02/) and the [Interactive Swarm Theater]({CHRONICLE_URL}editions/vol-01-no-02/theater.html) for the full narrative analysis, parse-delta charts, and agent-to-agent replay.

## Dataset Viewer Configurations

| Config | Rows | Description |
| --- | ---: | --- |
| `sessions` *(default)* | {counts['sessions']} | Per-language, per-cluster sessions (`13` languages × `9` clusters): topology, Shadow `DabaX` deltas, strict parse rates, native dispatch text, and coordinator scores. |
| `leaderboard` | {counts['leaderboard']} | Aggregate cluster scorecard across all 13 Malian languages. |
| `events` | {counts['events']} | Chronological inter-agent handoff, debate, Shadow `DabaX` compilation, and coordinator verdict events. |

*(Raw checkpoint artifacts under `edition2/` and `site_data/` are preserved for runtime state sync.)*

## Quick Usage

```python
from datasets import load_dataset

sessions = load_dataset("{ED2_REPO}", "sessions", split="train")
leaderboard = load_dataset("{ED2_REPO}", "leaderboard", split="train")
events = load_dataset("{ED2_REPO}", "events", split="train")
```

## Credits

- **Antigravity Team** ([Google Antigravity for Science](https://antigravity.google/use-cases/science/))
- **Adaption Labs** ([docs.adaptionlabs.ai](https://docs.adaptionlabs.ai/))
- **NVIDIA Build** ([build.nvidia.com](https://build.nvidia.com/))
- **`Gemini 4 Neon-High-Orchestrator`** (code, pipeline orchestration, and broadsheet prose)

## Citation (BibTeX)

```bibtex
@misc{{s7d11_sebeni_chronicle_ed2_2026,
  author       = {{s7d11}},
  title        = {{The Sebeni Chronicle, Edition No. 2: Swarm Linguistics Lab across 13 National Languages of Mali}},
  year         = {{2026}},
  howpublished = {{\\url{{{CHRONICLE_URL}editions/vol-01-no-02/}}}},
  note         = {{Dataset: \\url{{https://huggingface.co/datasets/{ED2_REPO}}}}}
}}
```
"""


def _render_ed3_card(counts: Dict[str, int]) -> str:
    """Build Dataset Card README.md for s7d11/chronicles-edition3-state."""
    return f"""---
pretty_name: "The Sebeni Chronicle — Edition No. 3: SebeniProofs Metric Stress Test"
license: mit
language:
  - multilingual
  - en
  - bm
  - ff
task_categories:
  - text-generation
  - other
tags:
  - mali
  - african-languages
  - lean4
  - formal-verification
  - morphological-analysis
  - evaluation-metrics
  - sebeni-chronicle
configs:
  - config_name: deductions
    default: true
    data_files:
      - split: train
        path: viewer_data/deductions.jsonl
  - config_name: stress_regimes
    data_files:
      - split: train
        path: viewer_data/stress_regimes.jsonl
  - config_name: events
    data_files:
      - split: train
        path: viewer_data/events.jsonl
  - config_name: lean_theorems
    data_files:
      - split: train
        path: viewer_data/lean_theorems.jsonl
---

# The Sebeni Chronicle — Edition No. 3: SebeniProofs Metric Stress Test

> **Read the full interactive broadsheet dispatch:**
> - **Chronicle Front Page:** [{CHRONICLE_URL}]({CHRONICLE_URL})
> - **Edition No. 3 Dispatch:** [{CHRONICLE_URL}editions/vol-01-no-03/]({CHRONICLE_URL}editions/vol-01-no-03/)
> - **Metric Stress Observatory:** [{CHRONICLE_URL}editions/vol-01-no-03/observatory.html]({CHRONICLE_URL}editions/vol-01-no-03/observatory.html)
> - **Source Repository:** [{GITHUB_URL}]({GITHUB_URL})

## About This Dataset

This dataset archives the empirical stress test and live model deduction audit for **Edition No. 3** of *The Sebeni Chronicle*. Three formally verified `SebeniProofs` (Lean 4.36.0) evaluation metrics — **MER** (Morphological Error Rate), **MCS** (Morphological Coverage Score), and **UWEC** (Unknown Word Error Coverage) — are evaluated across **4,200 sentences** (`67,623` tokens) from `dataset_300_samples.jsonl` spanning all **13 `MALI13` languages** (`bam`, `bmq`, `boz`, `dtm`, `ful`, `hsy`, `mku`, `mlq`, `myk`, `seq`, `ses`, `snk`, `taq`) plus an out-of-distribution control (`bbo`) across 6 formal stress regimes.

Four live NVIDIA Build models (`Mistral Large V4`, `Nemotron-3-Ultra-550B`, `Nemotron-3.5-Super-120B`, and `Minimax M3.1 Flash`; `meta/muse-glimmer-30b` reached NVIDIA Build EOL on `2026-11-24`) inspected the telemetry, ran independent deterministic oracle probes against `DabaX`, and delivered round-by-round deductions. Explore the charts, Lean 4 theorem map, and replay in the [Edition No. 3 Dispatch]({CHRONICLE_URL}editions/vol-01-no-03/) and [Metric Stress Observatory]({CHRONICLE_URL}editions/vol-01-no-03/observatory.html).

## Dataset Viewer Configurations

| Config | Rows | Description |
| --- | ---: | --- |
| `deductions` *(default)* | {counts['deductions']} | Live two-round model deductions & deterministic `DabaX` oracle probe checks (`14` languages × `4` models). |
| `stress_regimes` | {counts['stress_regimes']} | Formal `SebeniProofs` metric values across `14` languages × `6` adversarial & shift regimes (`4,200` sentences). |
| `events` | {counts['events']} | Step-by-step live observatory stream (`corpus_init`, `round1_deduction`, `oracle_probe`, `round2_deduction`, `lang_consensus`). |
| `lean_theorems` | {counts['lean_theorems']} | Formal Lean 4.36.0 theorem statements and plain-English interpretations from `SebeniProofs`. |

*(Raw checkpoint artifacts under `edition3/` and `site_data/` are preserved for runtime state sync.)*

## Quick Usage

```python
from datasets import load_dataset

deductions = load_dataset("{ED3_REPO}", "deductions", split="train")
regimes = load_dataset("{ED3_REPO}", "stress_regimes", split="train")
events = load_dataset("{ED3_REPO}", "events", split="train")
theorems = load_dataset("{ED3_REPO}", "lean_theorems", split="train")
```

## Credits

- **Antigravity Team** ([Google Antigravity for Science](https://antigravity.google/use-cases/science/))
- **Adaption Labs** ([docs.adaptionlabs.ai](https://docs.adaptionlabs.ai/))
- **NVIDIA Build** ([build.nvidia.com](https://build.nvidia.com/))
- **`Gemini 4 Neon-High-Orchestrator`** (code, pipeline orchestration, and broadsheet prose)

## Citation (BibTeX)

```bibtex
@misc{{s7d11_sebeni_chronicle_ed3_2026,
  author       = {{s7d11}},
  title        = {{The Sebeni Chronicle, Edition No. 3: Stress-Testing Formally Verified SebeniProofs Metrics (MER, MCS, UWEC) on MALI13}},
  year         = {{2026}},
  howpublished = {{\\url{{{CHRONICLE_URL}editions/vol-01-no-03/}}}},
  note         = {{Dataset: \\url{{https://huggingface.co/datasets/{ED3_REPO}}}}}
}}
```
"""


def build_hf_viewer_bundles(
    out_dir: Path, site_data_dir: Path = SITE_DATA_DIR
) -> Dict[str, Any]:
    """Generate viewer JSONL splits and Dataset Cards for Edition 2 and Edition 3."""
    ed2 = _load_json(site_data_dir / "edition2.json")
    ed2_events = _load_json(site_data_dir / "edition2_events.json")
    ed3 = _load_json(site_data_dir / "edition3.json")
    ed3_events = _load_json(site_data_dir / "edition3_events.json")

    ed2_rows = _extract_ed2_rows(ed2, ed2_events)
    ed3_rows = _extract_ed3_rows(ed3, ed3_events)

    ed2_dir = Path(out_dir) / "chronicles-edition2-state"
    ed3_dir = Path(out_dir) / "chronicles-edition3-state"

    ed2_counts = {
        name: _write_jsonl(ed2_dir / "viewer_data" / f"{name}.jsonl", rows)
        for name, rows in ed2_rows.items()
    }
    ed3_counts = {
        name: _write_jsonl(ed3_dir / "viewer_data" / f"{name}.jsonl", rows)
        for name, rows in ed3_rows.items()
    }

    (ed2_dir / "README.md").write_text(_render_ed2_card(ed2_counts), encoding="utf-8")
    (ed3_dir / "README.md").write_text(_render_ed3_card(ed3_counts), encoding="utf-8")

    return {
        "edition2": {"dir": ed2_dir, "repo_id": ED2_REPO, "counts": ed2_counts},
        "edition3": {"dir": ed3_dir, "repo_id": ED3_REPO, "counts": ed3_counts},
    }


def _load_hf_token(env_path: Optional[Path] = None) -> str:
    """Resolve HF token from environment or .env without ever logging it."""
    tok = os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN") or ""
    if tok:
        return tok.strip()
    candidate = env_path or (REPO_ROOT / ".env")
    if candidate.is_file():
        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            if key in {"HF_KEY", "HF_TOKEN"}:
                cleaned = val.strip().strip('"').strip("'")
                if cleaned:
                    return cleaned
    return ""


def upload_bundles(bundles: Dict[str, Any], token: str) -> None:
    """Upload README.md and viewer_data/*.jsonl to both public HF datasets."""
    if not token:
        raise RuntimeError("HF_KEY / HF_TOKEN is required for --upload")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    for key in ("edition2", "edition3"):
        info = bundles[key]
        repo_id = info["repo_id"]
        assert repo_id != PRIVATE_RUNTIME_REPO, "Refusing to touch private runtime repo"
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(info["dir"]),
            commit_message="Add Dataset Card (Chronicle links + BibTeX) and typed viewer_data splits for HF Dataset Viewer",
        )
        print(f"[hf-viewer] Uploaded {repo_id}: {info['counts']}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory for generated bundles (uses a temp dir if omitted).",
    )
    parser.add_argument(
        "--site-data-dir",
        type=Path,
        default=SITE_DATA_DIR,
        help="Path to site/data JSON directory.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload generated README.md and viewer_data/*.jsonl to public HF datasets.",
    )
    args = parser.parse_args()

    if args.out_dir is not None:
        bundles = build_hf_viewer_bundles(args.out_dir, site_data_dir=args.site_data_dir)
        for key, info in bundles.items():
            print(f"[hf-viewer] Built {key} -> {info['dir']} ({info['counts']})", flush=True)
        if args.upload:
            upload_bundles(bundles, _load_hf_token())
    else:
        with tempfile.TemporaryDirectory() as tmp:
            bundles = build_hf_viewer_bundles(Path(tmp), site_data_dir=args.site_data_dir)
            for key, info in bundles.items():
                print(f"[hf-viewer] Built {key} ({info['counts']})", flush=True)
            if args.upload:
                upload_bundles(bundles, _load_hf_token())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
