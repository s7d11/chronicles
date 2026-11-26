"""Deploy Edition No. 2 live 13-language Swarm Theater & runner to Google Cloud Run (mlgramv1)."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.env import load_env  # noqa: E402

DOCKER_SRC_DIR = REPO_ROOT / "deploy" / "hf_space"


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


def deploy_cloud_run(
    *,
    project: str = "mlgramv1",
    region: str = "us-east4",
    service: str = "sebeni-swarm-live",
    runtime_repo: str = "s7d11/sebeni-runtime-private",
    state_repo: str = "s7d11/chronicles-edition2-state",
    ed3_state_repo: str = "s7d11/chronicles-edition3-state",
    keep_warm: bool = False,
    auto_run_ed2: bool = False,
    auto_run_ed3: bool = False,
) -> str:
    load_env()
    hf_tok = os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN") or ""
    adoption_key = os.environ.get("ADOPTION_KEY") or ""
    nvidia_key = os.environ.get("NVIDIA_KEY") or ""
    if not hf_tok or not adoption_key or not nvidia_key:
        raise RuntimeError("Missing HF_KEY, ADOPTION_KEY, or NVIDIA_KEY in environment/.env")
    admin_tok = _ensure_admin_token(REPO_ROOT / ".env")

    gcloud_env = {k: v for k, v in os.environ.items() if k != "GOOGLE_APPLICATION_CREDENTIALS"}
    gcloud_env["CLOUDSDK_CORE_PROJECT"] = project
    gcloud_env["CLOUDSDK_BILLING_QUOTA_PROJECT"] = project

    print(f"[cloud-run] Enabling Run/CloudBuild/ArtifactRegistry APIs on {project}...", flush=True)
    subprocess.run(
        [
            "gcloud",
            "services",
            "enable",
            "run.googleapis.com",
            "cloudbuild.googleapis.com",
            "artifactregistry.googleapis.com",
            "--project",
            project,
            "--quiet",
        ],
        env=gcloud_env,
        check=True,
    )

    proj_num = subprocess.check_output(
        ["gcloud", "projects", "describe", project, "--format", "value(projectNumber)"],
        env=gcloud_env,
        text=True,
    ).strip()
    build_sa = f"{proj_num}-compute@developer.gserviceaccount.com"
    print(f"[cloud-run] Ensuring documented Cloud Build IAM roles on {build_sa}...", flush=True)
    for role in (
        "roles/cloudbuild.builds.builder",
        "roles/serviceusage.serviceUsageConsumer",
        "roles/artifactregistry.writer",
        "roles/logging.logWriter",
    ):
        subprocess.run(
            [
                "gcloud",
                "projects",
                "add-iam-policy-binding",
                project,
                f"--member=serviceAccount:{build_sa}",
                f"--role={role}",
                "--quiet",
            ],
            env=gcloud_env,
            stdout=subprocess.DEVNULL,
            check=True,
        )
    time.sleep(15)

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "ctx"
        stage.mkdir(parents=True, exist_ok=True)
        for fname in ("Dockerfile", "entrypoint.sh"):
            shutil.copy2(DOCKER_SRC_DIR / fname, stage / fname)
        for root_file in ("pyproject.toml", "prompt_native.csv"):
            shutil.copy2(REPO_ROOT / root_file, stage / root_file)
        _copy_clean_tree(REPO_ROOT / "bench", stage / "bench")
        _copy_clean_tree(REPO_ROOT / "swarm_lab", stage / "swarm_lab")
        _copy_clean_tree(REPO_ROOT / "proof_lab", stage / "proof_lab")
        _copy_clean_tree(REPO_ROOT / "site", stage / "site")

        env_vars = {
            "HF_KEY": hf_tok,
            "ADOPTION_KEY": adoption_key,
            "NVIDIA_KEY": nvidia_key,
            "SWARM_ADMIN_TOKEN": admin_tok,
            "AUTO_RUN_ALL_13": "1" if auto_run_ed2 else "0",
            "AUTO_RUN_ED3": "1" if auto_run_ed3 else "0",
            "SEBENI_RUNTIME_REPO": runtime_repo,
            "SWARM_STATE_REPO": state_repo,
            "ED3_STATE_REPO": ed3_state_repo,
        }
        env_yaml = Path(tmp) / "env.yaml"
        env_yaml.write_text(
            "\n".join(f"{k}: {json.dumps(str(v))}" for k, v in env_vars.items()) + "\n",
            encoding="utf-8",
        )
        os.chmod(env_yaml, 0o600)

        print(f"[cloud-run] Deploying {service} to {project} ({region})...", flush=True)
        subprocess.run(
            [
                "gcloud",
                "run",
                "deploy",
                service,
                "--project",
                project,
                "--region",
                region,
                "--source",
                str(stage),
                "--allow-unauthenticated",
                "--port",
                "8080",
                "--cpu",
                "1",
                "--memory",
                "1Gi",
                "--min-instances",
                "1" if keep_warm else "0",
                "--max-instances",
                "1",
                "--no-cpu-throttling" if keep_warm else "--cpu-throttling",
                "--timeout",
                "3600",
                "--env-vars-file",
                str(env_yaml),
                "--quiet",
            ],
            env=gcloud_env,
            check=True,
        )

    out = subprocess.check_output(
        [
            "gcloud",
            "run",
            "services",
            "describe",
            service,
            "--project",
            project,
            "--region",
            region,
            "--format",
            "value(status.url)",
        ],
        env=gcloud_env,
        text=True,
    ).strip()
    print(f"[cloud-run] Live Service URL: {out}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy Sebeni Swarm Theater to Cloud Run")
    parser.add_argument("--project", default="mlgramv1")
    parser.add_argument("--region", default="us-east4")
    parser.add_argument("--service", default="sebeni-swarm-live")
    parser.add_argument(
        "--keep-warm",
        action="store_true",
        help="Pin min-instances=1 with no CPU throttling for active batch runs (incurs billing).",
    )
    parser.add_argument("--auto-run-ed2", action="store_true", help="Auto-start pending Edition 2 runs on boot.")
    parser.add_argument("--auto-run-ed3", action="store_true", help="Auto-start pending Edition 3 runs on boot.")
    args = parser.parse_args()
    deploy_cloud_run(
        project=args.project,
        region=args.region,
        service=args.service,
        keep_warm=args.keep_warm,
        auto_run_ed2=args.auto_run_ed2,
        auto_run_ed3=args.auto_run_ed3,
    )


if __name__ == "__main__":
    main()
