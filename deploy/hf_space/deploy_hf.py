"""One-command Hugging Face Docker Space & persistent Dataset deployer for Edition No. 2."""

from __future__ import annotations

import argparse
import os
import secrets
import shutil
import sys
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.env import load_env
from swarm_lab.hf_sync import pack_sebeni_runtime, seed_state_to_hf
from swarm_lab.runner import DEFAULT_ED2_RUNS_DIR, DEFAULT_SITE_DATA_DIR

SPACE_DEPLOY_DIR = Path(__file__).resolve().parent


def _ensure_admin_token(env_path: Path) -> str:
    tok = os.environ.get("SWARM_ADMIN_TOKEN") or ""
    if tok:
        return tok
    tok = secrets.token_urlsafe(24)
    os.environ["SWARM_ADMIN_TOKEN"] = tok
    if env_path.is_file():
        existing = env_path.read_text(encoding="utf-8")
        if "SWARM_ADMIN_TOKEN=" not in existing:
            sep = "" if existing.endswith("\n") else "\n"
            env_path.write_text(existing + f"{sep}SWARM_ADMIN_TOKEN={tok}\n", encoding="utf-8")
    return tok


def _copy_clean_tree(src: Path, dst: Path) -> None:
    for fp in src.rglob("*"):
        if "__pycache__" in fp.parts or fp.suffix in {".pyc", ".pyo"}:
            continue
        rel = fp.relative_to(src)
        target = dst / rel
        if fp.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fp, target)


def deploy(
    *,
    owner: str = "s7d11",
    runtime_repo_name: str = "sebeni-runtime-private",
    state_repo_name: str = "chronicles-edition2-state",
    space_name: str = "sebeni-swarm-live",
    wait_seconds: int = 420,
) -> None:
    load_env()
    hf_tok = os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN") or ""
    adoption_key = os.environ.get("ADOPTION_KEY") or ""
    nvidia_key = os.environ.get("NVIDIA_KEY") or ""
    if not hf_tok or not adoption_key or not nvidia_key:
        raise RuntimeError("Missing HF_KEY, ADOPTION_KEY, or NVIDIA_KEY in environment/.env")

    admin_tok = _ensure_admin_token(REPO_ROOT / ".env")
    api = HfApi(token=hf_tok)

    runtime_repo = f"{owner}/{runtime_repo_name}"
    state_repo = f"{owner}/{state_repo_name}"
    space_repo = f"{owner}/{space_name}"

    # 1. Create PRIVATE dataset repo and verify privacy before uploading runtime tarball
    api.create_repo(repo_id=runtime_repo, repo_type="dataset", private=True, exist_ok=True)
    ds_info = api.dataset_info(repo_id=runtime_repo, token=hf_tok)
    if not getattr(ds_info, "private", False):
        raise RuntimeError(f"Refusing upload: {runtime_repo} is NOT private!")

    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "sebeni_runtime.tar.gz"
        pack_sebeni_runtime(tar_path)
        size_mb = tar_path.stat().st_size / (1024 * 1024)
        print(f"[deploy_hf] Uploading private runtime ({size_mb:.2f} MB) -> {runtime_repo}", flush=True)
        api.upload_file(
            path_or_fileobj=tar_path,
            path_in_repo="sebeni_runtime.tar.gz",
            repo_id=runtime_repo,
            repo_type="dataset",
            commit_message="Upload minimal private Sebeni + DabaX runtime",
        )

    # 2. Seed Edition 2 persistent state dataset (taq, ful, dtm, bam)
    seeded = seed_state_to_hf(state_repo, DEFAULT_ED2_RUNS_DIR, DEFAULT_SITE_DATA_DIR, hf_tok)
    print(f"[deploy_hf] Seeded {state_repo} with {len(seeded)} languages: {seeded}", flush=True)

    # 3. Create / update public Docker Space
    api.create_repo(
        repo_id=space_repo,
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for fname in ("Dockerfile", "README.md", "entrypoint.sh"):
            shutil.copy2(SPACE_DEPLOY_DIR / fname, stage / fname)
        for root_file in ("pyproject.toml", "prompt_native.csv"):
            shutil.copy2(REPO_ROOT / root_file, stage / root_file)
        _copy_clean_tree(REPO_ROOT / "bench", stage / "bench")
        _copy_clean_tree(REPO_ROOT / "swarm_lab", stage / "swarm_lab")
        _copy_clean_tree(REPO_ROOT / "site", stage / "site")

        print(f"[deploy_hf] Uploading Space code -> {space_repo}", flush=True)
        api.upload_folder(
            repo_id=space_repo,
            repo_type="space",
            folder_path=str(stage),
            commit_message="Deploy live 13-language Edition No. 2 Swarm Theater",
        )

    # 4. Configure Space Secrets & Variables
    for k, v in [
        ("HF_KEY", hf_tok),
        ("ADOPTION_KEY", adoption_key),
        ("NVIDIA_KEY", nvidia_key),
        ("SWARM_ADMIN_TOKEN", admin_tok),
    ]:
        api.add_space_secret(repo_id=space_repo, key=k, value=v)
    for k, v in [
        ("AUTO_RUN_ALL_13", "1"),
        ("SEBENI_RUNTIME_REPO", runtime_repo),
        ("SWARM_STATE_REPO", state_repo),
    ]:
        api.add_space_variable(repo_id=space_repo, key=k, value=v)

    print(f"[deploy_hf] Space URL: https://huggingface.co/spaces/{space_repo}", flush=True)
    print(f"[deploy_hf] Direct App URL: https://{owner}-{space_name}.hf.space", flush=True)

    # 5. Poll Space runtime stage cleanly
    deadline = time.time() + wait_seconds
    last_stage = None
    while time.time() < deadline:
        rt = api.get_space_runtime(repo_id=space_repo)
        stage_str = str(getattr(rt, "stage", ""))
        if stage_str != last_stage:
            print(f"[deploy_hf] Space runtime stage: {stage_str}", flush=True)
            last_stage = stage_str
        if "RUNNING" in stage_str and "BUILDING" not in stage_str:
            print("[deploy_hf] Space is RUNNING!", flush=True)
            return
        if "ERROR" in stage_str:
            raise RuntimeError(f"HF Space failed with stage={stage_str}: {rt}")
        time.sleep(10)
    print(f"[deploy_hf] Timed out waiting for RUNNING (last stage={last_stage})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy Sebeni Swarm Theater to HF Spaces")
    parser.add_argument("--owner", default="s7d11")
    parser.add_argument("--wait-seconds", type=int, default=420)
    args = parser.parse_args()
    deploy(owner=args.owner, wait_seconds=args.wait_seconds)


if __name__ == "__main__":
    main()
