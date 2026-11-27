"""HF Dataset checkpointing + Dataset Viewer export + GitHub Pages static publishing for Edition No. 4."""

from __future__ import annotations

import json
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from swarm_lab import ALL_LANGS, normalize_lang_code
from village_lab import (
    DEFAULT_LAUNCH_LANG,
    ED4_STATE_REPO,
    LANGUAGE_NATIVE_NAMES,
    TARGET_DICT_ENTRIES,
)
from village_lab.schema import LanguageVillageState


def _render_ed4_hf_card(counts: Dict[str, int]) -> str:
    chronicle_url = "https://s7d11.github.io/chronicles/"
    github_url = "https://github.com/s7d11/chronicles"
    return f"""---
pretty_name: "The Sebeni Chronicle — Edition No. 4: The Living Lexicon Village (50k MALI13)"
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
  - lexicography
  - morphotactics
  - multi-agent
  - sebeni-chronicle
configs:
  - config_name: council_ledger
    default: true
    data_files:
      - split: train
        path: viewer_data/council_ledger.jsonl
  - config_name: exchanges
    data_files:
      - split: train
        path: viewer_data/exchanges.jsonl
  - config_name: events
    data_files:
      - split: train
        path: viewer_data/events.jsonl
  - config_name: language_progress
    data_files:
      - split: train
        path: viewer_data/language_progress.jsonl
---

# The Sebeni Chronicle — Edition No. 4: The Living Lexicon Village (50k MALI13)

> **Explore the Live 3D Sudano-Sahelian Village & Broadsheet:**
> - **Chronicle Front Page:** [{chronicle_url}]({chronicle_url})
> - **Edition No. 4 Dispatch:** [{chronicle_url}editions/vol-01-no-04/]({chronicle_url}editions/vol-01-no-04/)
> - **3D Living Lexicon Village (Three.js):** [{chronicle_url}editions/vol-01-no-04/world.html]({chronicle_url}editions/vol-01-no-04/world.html)
> - **Source Repository:** [{github_url}]({github_url})

## About This Dataset

**Edition No. 4** documents a cloud-executed multi-agent linguistic society growing standardized Toolbox **MDF** dictionaries (`\\lx` / `\\va`) and morphotactic `.gram` grammars toward **50,000 entries** across the 13 national languages of Mali (`MALI13`, launching first on Bambara `bam`).

Every candidate entry passes a **5-Gate Standardization Pipeline**:
1. **DNAFLA/AMALAN Orthography Hygiene** & NFC/homoglyph normalization.
2. **Lemma-vs-Rule Boundary** (inflections routed to `.gram`; dialectal variants to `\\va`).
3. **Isolated Shadow `DabaX(strict_parse=True)`** compile & zero `dev-mini` regression check.
4. **4-Agent Council of Field Linguists** (`qwen3.5-397b`, `glm-5.2`, `qwen3.8-27b`, `gemma-4-26b-a4b-it`).
5. **Primary Critic Audit** by **`gemini-4-argon`**, orchestrated by **Mande** (`nvidia/nemotron-3-ultra-550b-a55b`) and 5+ dynamically spawned Adaption Labs villagers.

## Dataset Viewer Configurations

| Config | Rows | Description |
| --- | ---: | --- |
| `council_ledger` *(default)* | {counts['council_ledger']} | Standardized MDF proposals, Gate 1-3 diagnostics, 4-seat Council votes, and Gemini 4 Argon critiques. |
| `exchanges` | {counts['exchanges']} | Natural MALI13 village dialogues with EN/FR glosses and dynamic specialist spawns. |
| `events` | {counts['events']} | Spatial waypoint movements, courier scrolls, deliberations, and commits for 3D replay. |
| `language_progress` | {counts['language_progress']} | Progress ledger across all 13 MALI13 languages toward 50,000 standardized entries. |

## Credits

- **Antigravity Team** ([Google Antigravity for Science](https://antigravity.google/use-cases/science/))
- **Adaption Labs** ([docs.adaptionlabs.ai](https://docs.adaptionlabs.ai/))
- **NVIDIA Build** ([build.nvidia.com](https://build.nvidia.com/))
- **`Gemini 4 Neon-High-Orchestrator`** (code, pipeline orchestration, and broadsheet prose)

## Citation (BibTeX)

```bibtex
@misc{{s7d11_sebeni_chronicle_ed4_2026,
  author       = {{s7d11}},
  title        = {{The Sebeni Chronicle, Edition No. 4: The Living Lexicon Village — Building a 50k Standardized MALI13 Dictionary}},
  year         = {{2026}},
  howpublished = {{\\url{{{chronicle_url}editions/vol-01-no-04/}}}},
  note         = {{Dataset: \\url{{https://huggingface.co/datasets/{ED4_STATE_REPO}}}}}
}}
```
"""


def publish_ed4_site_data(
    runs_dir: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    """Export compact static-first site/data/edition4.json & edition4_events.json."""
    from swarm_lab.resources import load_baseline_resources
    from village_lab.world_engine import _default_agents

    runs_dir = Path(runs_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    languages_payload: Dict[str, Any] = {}
    events_by_lang: Dict[str, List[Dict[str, Any]]] = {}

    for lang in ALL_LANGS:
        canon = normalize_lang_code(lang)
        ldir = runs_dir / canon
        state_p = ldir / "state.json"
        if state_p.is_file():
            st = LanguageVillageState.model_validate_json(state_p.read_text(encoding="utf-8"))
            languages_payload[canon] = st.model_dump()
        else:
            base = load_baseline_resources(canon)
            gram_lines = len(Path(base["gram_path"]).read_text(encoding="utf-8", errors="replace").splitlines()) if Path(base["gram_path"]).is_file() else 0
            languages_payload[canon] = {
                "lang": canon,
                "lang_name": LANGUAGE_NATIVE_NAMES.get(canon, canon),
                "target_entries": TARGET_DICT_ENTRIES,
                "baseline_lx_count": int(base.get("dict_entries") or 0),
                "added_lx_count": 0,
                "added_va_count": 0,
                "baseline_gram_lines": gram_lines,
                "added_gram_rules_count": 0,
                "total_turns": 0,
                "total_candidates_reviewed": 0,
                "total_rejected": 0,
                "baseline_devmini_pct": 0.0,
                "latest_devmini_pct": 0.0,
                "updated_at": "",
                "orchestrator_model": "nvidia/nemotron-3-ultra-550b-a55b",
                "critic_model": "gemini-4-argon",
                "active_agents": [a.model_dump() for a in _default_agents()],
                "domain_counts": {},
                "recent_exchanges": [],
                "recent_decisions": [],
            }

        ev_p = ldir / "events.jsonl"
        evs: List[Dict[str, Any]] = []
        if ev_p.is_file():
            for ln in ev_p.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    evs.append(json.loads(ln))
        events_by_lang[canon] = evs[-1500:]

    ed4_summary = {
        "edition": "Vol. I · No. 4",
        "launch_lang": DEFAULT_LAUNCH_LANG,
        "target_entries": TARGET_DICT_ENTRIES,
        "langs": list(ALL_LANGS),
        "total_events": sum(len(v) for v in events_by_lang.values()),
        "total_added_lx": sum(int(v["added_lx_count"]) for v in languages_payload.values()),
        "total_added_va": sum(int(v["added_va_count"]) for v in languages_payload.values()),
        "total_added_gram": sum(int(v["added_gram_rules_count"]) for v in languages_payload.values()),
        "languages": languages_payload,
    }
    ed4_events = {
        "edition": "Vol. I · No. 4",
        "launch_lang": DEFAULT_LAUNCH_LANG,
        "langs": list(ALL_LANGS),
        "events_by_lang": events_by_lang,
    }
    (out_dir / "edition4.json").write_text(json.dumps(ed4_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "edition4_events.json").write_text(json.dumps(ed4_events, ensure_ascii=False), encoding="utf-8")
    return ed4_summary


def build_ed4_hf_viewer_bundle(runs_dir: Path, stage_dir: Path, summary: Dict[str, Any]) -> Dict[str, int]:
    """Create strictly-typed flat JSONL viewer_data splits + HF README.md under stage_dir."""
    vdir = Path(stage_dir) / "viewer_data"
    vdir.mkdir(parents=True, exist_ok=True)

    ledger_rows: List[Dict[str, Any]] = []
    exchange_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    prog_rows: List[Dict[str, Any]] = []

    for lang in ALL_LANGS:
        canon = normalize_lang_code(lang)
        ldir = Path(runs_dir) / canon
        linfo = summary["languages"][canon]
        prog_rows.append(
            {
                "lang": str(canon),
                "lang_name": str(linfo.get("lang_name") or canon),
                "target_entries": int(linfo.get("target_entries") or TARGET_DICT_ENTRIES),
                "baseline_lx_count": int(linfo.get("baseline_lx_count") or 0),
                "added_lx_count": int(linfo.get("added_lx_count") or 0),
                "added_va_count": int(linfo.get("added_va_count") or 0),
                "baseline_gram_lines": int(linfo.get("baseline_gram_lines") or 0),
                "added_gram_rules_count": int(linfo.get("added_gram_rules_count") or 0),
                "total_turns": int(linfo.get("total_turns") or 0),
                "total_candidates_reviewed": int(linfo.get("total_candidates_reviewed") or 0),
                "total_rejected": int(linfo.get("total_rejected") or 0),
                "completion_pct": round(
                    100.0 * (int(linfo.get("baseline_lx_count") or 0) + int(linfo.get("added_lx_count") or 0)) / TARGET_DICT_ENTRIES,
                    3,
                ),
                "updated_at": str(linfo.get("updated_at") or ""),
            }
        )

        led_p = ldir / "council_ledger.jsonl"
        if led_p.is_file():
            for ln in led_p.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                d = json.loads(ln)
                ortho = d.get("orthography_check") or {}
                dx = d.get("dabax_check") or {}
                cr = d.get("critic") or {}
                ledger_rows.append(
                    {
                        "lang": str(d.get("lang") or canon),
                        "turn_id": int(d.get("turn_id") or 0),
                        "candidate_raw": str(d.get("candidate_raw") or ""),
                        "standardized_lx": str(d.get("standardized_lx") or ""),
                        "variant_va": str(d.get("variant_va") or ""),
                        "pos": str(d.get("pos") or ""),
                        "gloss_en": str(d.get("gloss_en") or ""),
                        "gloss_fr": str(d.get("gloss_fr") or ""),
                        "semantic_domain": str(d.get("semantic_domain") or ""),
                        "final_action": str(d.get("final_action") or ""),
                        "dnafla_standard": bool(ortho.get("is_standard")),
                        "dabax_compile_ok": bool(dx.get("dict_compile_ok")) and bool(dx.get("gram_compile_ok")),
                        "devmini_regression_pct": float(dx.get("regression_pct") or 0.0),
                        "critic_verdict": str(cr.get("verdict") or ""),
                        "critic_rationale": str(cr.get("rationale") or ""),
                        "mdf_block": str(d.get("mdf_block") or ""),
                        "gram_rule": str(d.get("gram_rule") or ""),
                        "summary_reason": str(d.get("summary_reason") or ""),
                    }
                )

        exc_p = ldir / "exchanges.jsonl"
        if exc_p.is_file():
            for ln in exc_p.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                ex = json.loads(ln)
                for utt in ex.get("utterances") or []:
                    exchange_rows.append(
                        {
                            "lang": str(ex.get("lang") or canon),
                            "turn_id": int(ex.get("turn_id") or 0),
                            "zone": str(ex.get("zone") or ""),
                            "semantic_domain": str(ex.get("semantic_domain") or ""),
                            "topic": str(ex.get("topic") or ""),
                            "spawned_agent_id": str(ex.get("spawned_agent_id") or ""),
                            "speaker_id": str(utt.get("agent_id") or ""),
                            "speaker_name": str(utt.get("speaker_name") or ""),
                            "model_id": str(utt.get("model_id") or ""),
                            "text_native": str(utt.get("text_native") or ""),
                            "gloss_en": str(utt.get("gloss_en") or ""),
                            "gloss_fr": str(utt.get("gloss_fr") or ""),
                            "candidate_terms": ", ".join(str(x) for x in (utt.get("candidate_terms") or [])),
                        }
                    )

        ev_p = ldir / "events.jsonl"
        if ev_p.is_file():
            for ln in ev_p.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                ev = json.loads(ln)
                event_rows.append(
                    {
                        "lang": str(ev.get("lang") or canon),
                        "seq": int(ev.get("seq") or 0),
                        "ts": str(ev.get("ts") or ""),
                        "turn_id": int(ev.get("turn_id") or 0),
                        "kind": str(ev.get("kind") or ""),
                        "agent_id": str(ev.get("agent_id") or ""),
                        "zone": str(ev.get("zone") or ""),
                        "target_zone": str(ev.get("target_zone") or ""),
                        "label": str(ev.get("label") or ""),
                        "summary": str(ev.get("summary") or ""),
                    }
                )

    def _dump_jsonl(path: Path, rows: List[Dict[str, Any]]) -> int:
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        return len(rows)

    counts = {
        "council_ledger": _dump_jsonl(vdir / "council_ledger.jsonl", ledger_rows),
        "exchanges": _dump_jsonl(vdir / "exchanges.jsonl", exchange_rows),
        "events": _dump_jsonl(vdir / "events.jsonl", event_rows),
        "language_progress": _dump_jsonl(vdir / "language_progress.jsonl", prog_rows),
    }
    (Path(stage_dir) / "README.md").write_text(_render_ed4_hf_card(counts), encoding="utf-8")
    return counts


def restore_ed4_state_from_hf(state_repo: str, runs_dir: Path, token: Optional[str] = None) -> List[str]:
    if not state_repo:
        return []
    from huggingface_hub import snapshot_download

    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            snap = Path(
                snapshot_download(
                    repo_id=state_repo,
                    repo_type="dataset",
                    local_dir=tmp,
                    token=token or None,
                    allow_patterns=["edition4/**"],
                )
            )
            src_root = snap / "edition4"
            if src_root.is_dir():
                for src in src_root.rglob("*"):
                    if not src.is_file():
                        continue
                    rel = src.relative_to(src_root)
                    dst = runs_dir / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
    except Exception as exc:
        print(f"[village-hf] Starting fresh Edition 4 state ({type(exc).__name__}: {exc})", flush=True)
        return []

    return [l for l in ALL_LANGS if (runs_dir / normalize_lang_code(l) / "state.json").is_file()]


def _upload_folder_with_retry(
    api: Any,
    *,
    repo_id: str,
    folder_path: str,
    commit_message: str,
    path_in_repo: Optional[str] = None,
    attempts: int = 4,
) -> None:
    """Upload a staged folder to HF Hub with jittered retry on concurrent commit conflicts."""
    last_exc: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            kwargs: Dict[str, Any] = {
                "repo_id": repo_id,
                "repo_type": "dataset",
                "folder_path": folder_path,
                "commit_message": commit_message,
            }
            if path_in_repo:
                kwargs["path_in_repo"] = path_in_repo
            api.upload_folder(**kwargs)
            return
        except Exception as exc:
            last_exc = exc
            if attempt + 1 >= attempts:
                break
            time.sleep(min(15.0, (2.0 ** attempt) + random.uniform(0.3, 1.8)))
    assert last_exc is not None
    raise last_exc


def push_ed4_state_to_hf(
    state_repo: str,
    runs_dir: Path,
    lang: str,
    site_data_dir: Path,
    token: str,
    *,
    lang_only: bool = False,
) -> None:
    if not state_repo or not token:
        return
    from huggingface_hub import HfApi

    canon = normalize_lang_code(lang)
    api = HfApi(token=token)
    api.create_repo(repo_id=state_repo, repo_type="dataset", private=False, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        ed4_lang_dst = stage / "edition4" / canon
        ed4_lang_dst.mkdir(parents=True, exist_ok=True)
        ldir = Path(runs_dir) / canon
        for fname in ("state.json", "augmented.dict", "augmented.gram", "events.jsonl", "council_ledger.jsonl", "exchanges.jsonl"):
            fp = ldir / fname
            if fp.is_file():
                shutil.copy2(fp, ed4_lang_dst / fname)
        if lang_only:
            _upload_folder_with_retry(
                api,
                repo_id=state_repo,
                folder_path=str(ed4_lang_dst),
                path_in_repo=f"edition4/{canon}",
                commit_message=f"Checkpoint Edition 4 language ({canon})",
            )
            return

        summary = publish_ed4_site_data(runs_dir, site_data_dir)
        for sfname in ("edition4.json", "edition4_events.json"):
            sf = Path(site_data_dir) / sfname
            if sf.is_file():
                (stage / "site_data").mkdir(parents=True, exist_ok=True)
                shutil.copy2(sf, stage / "site_data" / sfname)
        build_ed4_hf_viewer_bundle(runs_dir, stage, summary)
        _upload_folder_with_retry(
            api,
            repo_id=state_repo,
            folder_path=str(stage),
            commit_message=f"Update Edition 4 Living Lexicon Village ({canon})",
        )


def push_ed4_bundle_to_hf(
    state_repo: str,
    runs_dir: Path,
    site_data_dir: Path,
    token: str,
) -> Dict[str, Any]:
    """Publish unified site_data/*, viewer_data/*, and README.md from a complete runs_dir."""
    summary = publish_ed4_site_data(runs_dir, site_data_dir)
    if not state_repo or not token:
        return summary
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=state_repo, repo_type="dataset", private=False, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for sfname in ("edition4.json", "edition4_events.json"):
            sf = Path(site_data_dir) / sfname
            if sf.is_file():
                (stage / "site_data").mkdir(parents=True, exist_ok=True)
                shutil.copy2(sf, stage / "site_data" / sfname)
        build_ed4_hf_viewer_bundle(runs_dir, stage, summary)
        _upload_folder_with_retry(
            api,
            repo_id=state_repo,
            folder_path=str(stage),
            commit_message="Update Edition 4 unified site_data and viewer_data bundle",
        )
    return summary
