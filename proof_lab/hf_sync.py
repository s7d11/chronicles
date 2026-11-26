"""Hugging Face Dataset state persistence & restore for Edition No. 3 (`chronicles-edition3-state`)."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from proof_lab import DATASET_LANGS


def restore_ed3_state_from_hf(state_repo: str, runs_dir: Path, token: Optional[str] = None) -> List[str]:
    if not state_repo:
        return []
    from huggingface_hub import snapshot_download

    runs_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            snap = Path(
                snapshot_download(
                    repo_id=state_repo,
                    repo_type="dataset",
                    local_dir=tmp,
                    token=token or None,
                    allow_patterns=["edition3/**"],
                )
            )
            src_root = snap / "edition3"
            if src_root.is_dir():
                for src in src_root.rglob("*"):
                    if not src.is_file():
                        continue
                    rel = src.relative_to(src_root)
                    dst = runs_dir / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
    except Exception as exc:
        print(f"[proof-hf] Starting fresh Edition 3 state ({type(exc).__name__}: {exc})", flush=True)
        return []

    return [l for l in DATASET_LANGS if (runs_dir / l / "result.json").is_file()]


def push_ed3_language_to_hf(
    state_repo: str,
    runs_dir: Path,
    lang: str,
    site_data_dir: Path,
    token: str,
) -> None:
    if not state_repo or not token:
        return
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=state_repo, repo_type="dataset", private=False, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        ed3 = stage / "edition3"
        res_fp = runs_dir / lang / "result.json"
        if res_fp.is_file():
            (ed3 / lang).mkdir(parents=True, exist_ok=True)
            shutil.copy2(res_fp, ed3 / lang / "result.json")
        live_fp = runs_dir / "live" / f"{lang}.jsonl"
        if live_fp.is_file():
            (ed3 / "live").mkdir(parents=True, exist_ok=True)
            shutil.copy2(live_fp, ed3 / "live" / f"{lang}.jsonl")
        for fname in ("edition3.json", "edition3_events.json"):
            sf = site_data_dir / fname
            if sf.is_file():
                (stage / "site_data").mkdir(parents=True, exist_ok=True)
                shutil.copy2(sf, stage / "site_data" / fname)
        api.upload_folder(
            repo_id=state_repo,
            repo_type="dataset",
            folder_path=str(stage),
            commit_message=f"Update Edition 3 proof state: {lang}",
        )
