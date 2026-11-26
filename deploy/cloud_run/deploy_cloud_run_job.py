"""One-command Cloud Run Job deployer for unattended 24h Edition No. 4 execution.

Deploys the Edition No. 4 Living Lexicon Village worker as a Google Cloud Run Job
(`sebeni-ed4-job`, up to 24h timeout per execution, $0 when idle) streaming live
telemetry and events to Cloud Firestore while checkpointing artifacts to Hugging Face.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.env import load_env
from deploy.cloud_run.deploy_cloud_run import DOCKER_SRC_DIR, _copy_clean_tree, _ensure_admin_token


def deploy_job(
    *,
    project: str = "mlgramv1",
    region: str = "us-east4",
    job_name: str = "sebeni-ed4-job",
    langs: str = "bam",
    turns_per_lang: int = 40,
    checkpoint_every: int = 5,
    critic_model: str = "gemini-3.5-flash",
    firestore_db: str = "chronicles-live",
    execute: bool = False,
) -> None:
    load_env()
    hf_tok = os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN") or ""
    adoption_key = os.environ.get("ADOPTION_KEY") or ""
    nvidia_key = os.environ.get("NVIDIA_KEY") or ""
    gemini_key = os.environ.get("GEMINI_API_KEY") or ""
    if not hf_tok or not adoption_key or not nvidia_key:
        raise RuntimeError("Missing HF_KEY, ADOPTION_KEY, or NVIDIA_KEY in environment/.env")

    admin_tok = _ensure_admin_token(REPO_ROOT / ".env")
    gcloud_env = {k: v for k, v in os.environ.items() if k != "GOOGLE_APPLICATION_CREDENTIALS"}
    gcloud_env["CLOUDSDK_CORE_PROJECT"] = project
    gcloud_env["CLOUDSDK_BILLING_QUOTA_PROJECT"] = project

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "ctx"
        stage.mkdir(parents=True, exist_ok=True)
        for fname in ("Dockerfile", "entrypoint.sh"):
            import shutil

            shutil.copy2(DOCKER_SRC_DIR / fname, stage / fname)
        for root_file in ("pyproject.toml", "prompt_native.csv"):
            import shutil

            shutil.copy2(REPO_ROOT / root_file, stage / root_file)
        for pkg in ("bench", "swarm_lab", "proof_lab", "village_lab", "site"):
            _copy_clean_tree(REPO_ROOT / pkg, stage / pkg)

        env_vars = {
            "JOB_MODE": "ed4",
            "HF_KEY": hf_tok,
            "ADOPTION_KEY": adoption_key,
            "NVIDIA_KEY": nvidia_key,
            "SWARM_ADMIN_TOKEN": admin_tok,
            "SEBENI_RUNTIME_REPO": "s7d11/sebeni-runtime-private",
            "SWARM_STATE_REPO": "s7d11/chronicles-edition2-state",
            "ED3_STATE_REPO": "s7d11/chronicles-edition3-state",
            "ED4_STATE_REPO": "s7d11/chronicles-edition4-state",
            "GCP_VERTEX_PROJECT": project,
            "ED4_FIRESTORE_PROJECT": project,
            "ED4_FIRESTORE_DB": firestore_db,
            "ED4_CRITIC_MODEL": critic_model,
            "ED4_JOB_LANGS": langs,
            "ED4_JOB_TURNS": str(int(turns_per_lang)),
            "ED4_CHECKPOINT_EVERY": str(int(checkpoint_every)),
        }
        if gemini_key:
            env_vars["GEMINI_API_KEY"] = gemini_key

        env_yaml = Path(tmp) / "env.yaml"
        env_yaml.write_text(
            "\n".join(f"{k}: {json.dumps(str(v))}" for k, v in env_vars.items()) + "\n",
            encoding="utf-8",
        )
        os.chmod(env_yaml, 0o600)

        print(
            f"[cloud-run-job] Deploying {job_name} to {project} ({region}) | "
            f"critic={critic_model}, db={firestore_db}, langs={langs}, turns={turns_per_lang}",
            flush=True,
        )
        subprocess.run(
            [
                "gcloud",
                "run",
                "jobs",
                "deploy",
                job_name,
                "--project",
                project,
                "--region",
                region,
                "--source",
                str(stage),
                "--cpu",
                "2",
                "--memory",
                "2Gi",
                "--task-timeout",
                "24h",
                "--max-retries",
                "0",
                "--env-vars-file",
                str(env_yaml),
                "--quiet",
            ],
            env=gcloud_env,
            check=True,
        )

    if execute:
        print(f"[cloud-run-job] Triggering async execution of {job_name}...", flush=True)
        subprocess.run(
            [
                "gcloud",
                "run",
                "jobs",
                "execute",
                job_name,
                "--project",
                project,
                "--region",
                region,
                "--async",
                "--quiet",
            ],
            env=gcloud_env,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy Edition 4 Cloud Run Job (sebeni-ed4-job)")
    parser.add_argument("--project", default="mlgramv1")
    parser.add_argument("--region", default="us-east4")
    parser.add_argument("--job", default="sebeni-ed4-job")
    parser.add_argument("--langs", default="bam", help="Comma-separated MALI13 codes or 'all'")
    parser.add_argument("--turns", type=int, default=40, help="Turns per language")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--critic-model", default="gemini-3.5-flash")
    parser.add_argument("--firestore-db", default="chronicles-live")
    parser.add_argument("--execute", action="store_true", help="Immediately launch an async execution")
    args = parser.parse_args()
    deploy_job(
        project=args.project,
        region=args.region,
        job_name=args.job,
        langs=args.langs,
        turns_per_lang=args.turns,
        checkpoint_every=args.checkpoint_every,
        critic_model=args.critic_model,
        firestore_db=args.firestore_db,
        execute=args.execute,
    )


if __name__ == "__main__":
    main()
