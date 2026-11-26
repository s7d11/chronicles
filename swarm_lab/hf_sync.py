"""Private runtime packer and Hugging Face Dataset checkpoint sync for live online deployment."""

from __future__ import annotations

import io
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional

from swarm_lab import ALL_LANGS


def pack_sebeni_runtime(out_tar: Path) -> Path:
    """Bundle minimal beni runtime, 13 baselines, test.json, and unpacked daba into tar.gz."""
    import beni
    import daba

    beni_root = Path(beni.__file__).resolve().parent
    daba_loc = Path(daba.__file__).resolve()
    out_tar = Path(out_tar)
    out_tar.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(out_tar, "w:gz") as tf:
        # 1. Minimal beni modules (no samples/training dumps/caches)
        for rel in ("__init__.py", "core", "utils"):
            src = beni_root / rel
            if not src.exists():
                continue
            if src.is_file():
                tf.add(src, arcname=f"beni/{rel}")
                continue
            for fp in sorted(src.rglob("*")):
                if fp.is_dir() or "__pycache__" in fp.parts or fp.suffix in {".pyc", ".pyo"}:
                    continue
                rel_p = fp.relative_to(beni_root)
                tf.add(fp, arcname=f"beni/{rel_p.as_posix()}")

        # 2. test.json, raw/dataset_300_samples.jsonl, raw/<lang>.txt for MALI13 + bbo, and 13 baseline.{gram,dict} files
        test_json = beni_root / "data" / "test.json"
        if test_json.is_file():
            tf.add(test_json, arcname="beni/data/test.json")
        raw_dir = beni_root / "data" / "raw"
        ds300 = raw_dir / "dataset_300_samples.jsonl"
        if ds300.is_file():
            tf.add(ds300, arcname="beni/data/raw/dataset_300_samples.jsonl")
        for lang_txt in list(ALL_LANGS) + ["mlq", "seq", "bbo"]:
            rfp = raw_dir / f"{lang_txt}.txt"
            if rfp.is_file():
                tf.add(rfp, arcname=f"beni/data/raw/{lang_txt}.txt")
        baselines_dir = beni_root / "data" / "baselines"
        for lang in ALL_LANGS:
            ldir = baselines_dir / lang
            if not ldir.is_dir():
                continue
            for fname in ("baseline.gram", "baseline.dict"):
                fp = ldir / fname
                if fp.is_file():
                    tf.add(fp, arcname=f"beni/data/baselines/{lang}/{fname}")

        # 3. Unpack pure-Python daba package (handles zip .egg cleanly)
        if ".egg" in daba_loc.as_posix() and not daba_loc.parent.is_dir():
            egg_path = next(p for p in [daba_loc] + list(daba_loc.parents) if p.suffix == ".egg" and p.is_file())
            with zipfile.ZipFile(egg_path) as zf:
                for info in zf.infolist():
                    if not info.filename.startswith("daba/") or info.is_dir():
                        continue
                    if "__pycache__" in info.filename or info.filename.endswith((".pyc", ".pyo")):
                        continue
                    data = zf.read(info.filename)
                    ti = tarfile.TarInfo(name=info.filename)
                    ti.size = len(data)
                    ti.mode = 0o644
                    tf.addfile(ti, io.BytesIO(data))
        else:
            daba_root = daba_loc.parent
            for fp in sorted(daba_root.rglob("*")):
                if fp.is_dir() or "__pycache__" in fp.parts or fp.suffix in {".pyc", ".pyo"}:
                    continue
                rel_p = fp.relative_to(daba_root)
                tf.add(fp, arcname=f"daba/{rel_p.as_posix()}")

    size_mb = out_tar.stat().st_size / (1024 * 1024)
    if size_mb > 25.0:
        raise RuntimeError(f"Runtime archive unexpectedly large ({size_mb:.2f} MB > 25 MB)")
    return out_tar


def download_and_extract_runtime(repo_id: str, dest_dir: Path, token: str) -> Path:
    """Fetch private sebeni_runtime.tar.gz from HF Dataset and extract to dest_dir."""
    from huggingface_hub import hf_hub_download

    if not token:
        raise RuntimeError("HF_KEY / HF_TOKEN is required to download private runtime")
    tar_path = hf_hub_download(
        repo_id=repo_id,
        filename="sebeni_runtime.tar.gz",
        repo_type="dataset",
        token=token,
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(dest_dir, filter="data")
    return dest_dir


def restore_state_from_hf(state_repo: str, runs_dir: Path, token: Optional[str] = None) -> List[str]:
    """Download persisted checkpoints & live JSONL from HF Dataset into runs_dir."""
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
                    allow_patterns=["edition2/**"],
                )
            )
            src_root = snap / "edition2"
            if src_root.is_dir():
                for src in src_root.rglob("*"):
                    if not src.is_file():
                        continue
                    rel = src.relative_to(src_root)
                    dst = runs_dir / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
    except Exception as exc:
        print(f"[hf_sync] Note: starting fresh state ({type(exc).__name__}: {exc})", flush=True)
        return []

    completed = []
    for lang in ALL_LANGS:
        if (runs_dir / lang / "language_summary.json").is_file():
            completed.append(lang)
    return completed


def push_language_state_to_hf(
    state_repo: str,
    runs_dir: Path,
    lang: str,
    site_data_dir: Path,
    token: str,
) -> None:
    """Upload one completed language checkpoint + live JSONL + edition2*.json to HF Dataset."""
    if not state_repo or not token:
        return
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        ed2 = stage / "edition2"
        lang_src = runs_dir / lang
        if lang_src.is_dir():
            for name in ("language_summary.json",):
                fp = lang_src / name
                if fp.is_file():
                    (ed2 / lang).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(fp, ed2 / lang / name)
            for sub in lang_src.iterdir():
                if sub.is_dir() and (sub / "result.json").is_file():
                    (ed2 / lang / sub.name).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(sub / "result.json", ed2 / lang / sub.name / "result.json")
        live_fp = runs_dir / "live" / f"{lang}.jsonl"
        if live_fp.is_file():
            (ed2 / "live").mkdir(parents=True, exist_ok=True)
            shutil.copy2(live_fp, ed2 / "live" / f"{lang}.jsonl")
        for fname in ("edition2.json", "edition2_events.json"):
            sf = site_data_dir / fname
            if sf.is_file():
                (stage / "site_data").mkdir(parents=True, exist_ok=True)
                shutil.copy2(sf, stage / "site_data" / fname)
        api.upload_folder(
            repo_id=state_repo,
            repo_type="dataset",
            folder_path=str(stage),
            commit_message=f"Update Edition 2 state: {lang}",
        )


def seed_state_to_hf(state_repo: str, runs_dir: Path, site_data_dir: Path, token: str) -> List[str]:
    """Push all locally finished languages + site_data to initialize the remote state dataset."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=state_repo, repo_type="dataset", private=False, exist_ok=True)
    seeded: List[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        ed2 = stage / "edition2"
        for lang_dir in sorted(runs_dir.iterdir()):
            if not lang_dir.is_dir() or lang_dir.name in {"live", "_rt_base"}:
                continue
            ls_fp = lang_dir / "language_summary.json"
            if not ls_fp.is_file():
                continue
            seeded.append(lang_dir.name)
            (ed2 / lang_dir.name).mkdir(parents=True, exist_ok=True)
            shutil.copy2(ls_fp, ed2 / lang_dir.name / "language_summary.json")
            for sub in lang_dir.iterdir():
                if sub.is_dir() and (sub / "result.json").is_file():
                    (ed2 / lang_dir.name / sub.name).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(sub / "result.json", ed2 / lang_dir.name / sub.name / "result.json")
            live_fp = runs_dir / "live" / f"{lang_dir.name}.jsonl"
            if live_fp.is_file():
                (ed2 / "live").mkdir(parents=True, exist_ok=True)
                shutil.copy2(live_fp, ed2 / "live" / f"{lang_dir.name}.jsonl")
        for fname in ("edition2.json", "edition2_events.json"):
            sf = site_data_dir / fname
            if sf.is_file():
                (stage / "site_data").mkdir(parents=True, exist_ok=True)
                shutil.copy2(sf, stage / "site_data" / fname)
        api.upload_folder(
            repo_id=state_repo,
            repo_type="dataset",
            folder_path=str(stage),
            commit_message=f"Seed Edition 2 checkpoints ({', '.join(seeded)})",
        )
    return seeded
