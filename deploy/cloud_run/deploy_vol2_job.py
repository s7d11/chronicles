"""Deployer for Volume II Mali OSINT & Mathematical Proof Cloud Run Job (`sebeni-vol2-job`).

Runs isolated from Edition No. 4 (`ADOPTION_KEY` & `nemotron-3-ultra-550b` excluded),
using NVIDIA Build dedicated models, OpenRouter `:free` models, and Vertex
`gemini-3.1-flash-lite` on an optional Cloud Scheduler cadence (`0 */2 * * *`).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conflict_lab.cli import load_repo_dotenv
from deploy.cloud_run.deploy_cloud_run import DOCKER_SRC_DIR, _copy_clean_tree


def _resolve_gcloud() -> str:
    found = shutil.which("gcloud")
    if found:
        return found
    candidate = Path.home() / "google-cloud-sdk" / "bin" / "gcloud"
    if candidate.is_file():
        return str(candidate)
    return "gcloud"


def deploy_vol2_job(
    *,
    project: str = "mlgramv1",
    region: str = "us-east4",
    job_name: str = "sebeni-vol2-job",
    firestore_db: str = "chronicles-live",
    max_ucdp_rows: int = 1200,
    execute: bool = False,
    schedule_cron: str = "",
) -> None:
    load_repo_dotenv()
    hf_tok = (os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN") or "").strip()
    nvidia_key = (os.environ.get("NVIDIA_KEY") or "").strip()
    openrouter_key = (os.environ.get("OPENROUTER") or os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not hf_tok or not nvidia_key:
        raise RuntimeError("Missing HF_KEY or NVIDIA_KEY in environment/.env")

    gcloud_bin = _resolve_gcloud()
    gcloud_env = {k: v for k, v in os.environ.items() if k != "GOOGLE_APPLICATION_CREDENTIALS"}
    gcloud_env["CLOUDSDK_CORE_PROJECT"] = project
    gcloud_env["CLOUDSDK_BILLING_QUOTA_PROJECT"] = project

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "ctx"
        stage.mkdir(parents=True, exist_ok=True)
        for fname in ("Dockerfile", "entrypoint.sh"):
            shutil.copy2(DOCKER_SRC_DIR / fname, stage / fname)
        for root_file in ("pyproject.toml", "prompt_native.csv"):
            shutil.copy2(REPO_ROOT / root_file, stage / root_file)
        for pkg in ("bench", "swarm_lab", "proof_lab", "village_lab", "conflict_lab", "site"):
            _copy_clean_tree(REPO_ROOT / pkg, stage / pkg)

        # Strict isolation: NEVER include ADOPTION_KEY or Edition 4 models
        env_vars = {
            "JOB_MODE": "vol2",
            "VOL2_CONTINUOUS": "1",
            "VOL2_MAX_HOURS": "23.2",
            "VOL2_CYCLE_SLEEP_SEC": "45",
            "VOL2_OSINT_EVERY": "6",
            "VOL2_CHECKPOINT_EVERY": "6",
            "HF_KEY": hf_tok,
            "NVIDIA_KEY": nvidia_key,
            "GCP_VERTEX_PROJECT": project,
            "VOL2_FIRESTORE_PROJECT": project,
            "VOL2_FIRESTORE_DB": firestore_db,
            "VOL2_STATE_REPO": "s7d11/chronicles-vol2-mali",
            "VOL2_CRITIC_MODEL": "gemini-3.1-flash-lite",
            "VOL2_MAX_UCDP_ROWS": str(int(max_ucdp_rows)),
        }
        if openrouter_key:
            env_vars["OPENROUTER"] = openrouter_key

        env_yaml = Path(tmp) / "env.yaml"
        env_yaml.write_text(
            "\n".join(f"{k}: {json.dumps(str(v))}" for k, v in env_vars.items()) + "\n",
            encoding="utf-8",
        )
        os.chmod(env_yaml, 0o600)

        print(f"[vol2-deploy] Deploying {job_name} to {project} ({region}) | db={firestore_db}", flush=True)
        subprocess.run(
            [
                gcloud_bin,
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
                "1",
                "--memory",
                "1Gi",
                "--tasks",
                "1",
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

    if schedule_cron:
        subprocess.run(
            [gcloud_bin, "services", "enable", "cloudscheduler.googleapis.com", "--project", project, "--quiet"],
            env=gcloud_env,
            check=True,
        )
        sa_email = subprocess.check_output(
            [
                gcloud_bin,
                "run",
                "jobs",
                "describe",
                job_name,
                "--project",
                project,
                "--region",
                region,
                "--format",
                "value(template.template.serviceAccount)",
            ],
            env=gcloud_env,
            text=True,
        ).strip()
        if not sa_email:
            proj_num = subprocess.check_output(
                [gcloud_bin, "projects", "describe", project, "--format", "value(projectNumber)"],
                env=gcloud_env,
                text=True,
            ).strip()
            sa_email = f"{proj_num}-compute@developer.gserviceaccount.com"

        sched_name = f"{job_name}-sched"
        run_uri = f"https://{region}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/{project}/jobs/{job_name}:run"
        action = "update"
        check_proc = subprocess.run(
            [gcloud_bin, "scheduler", "jobs", "describe", sched_name, "--project", project, "--location", region],
            env=gcloud_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if check_proc.returncode != 0:
            action = "create"
        print(f"[vol2-deploy] Configuring Cloud Scheduler ({action} {sched_name} @ '{schedule_cron}')...", flush=True)
        subprocess.run(
            [
                gcloud_bin,
                "scheduler",
                "jobs",
                action,
                "http",
                sched_name,
                "--project",
                project,
                "--location",
                region,
                "--schedule",
                schedule_cron,
                "--uri",
                run_uri,
                "--http-method",
                "POST",
                "--oauth-service-account-email",
                sa_email,
                "--quiet",
            ],
            env=gcloud_env,
            check=True,
        )

    if execute:
        print(f"[vol2-deploy] Triggering async execution of {job_name}...", flush=True)
        subprocess.run(
            [
                gcloud_bin,
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
    parser = argparse.ArgumentParser(description="Deploy Volume II Mali OSINT Cloud Run Job")
    parser.add_argument("--project", default="mlgramv1")
    parser.add_argument("--region", default="us-east4")
    parser.add_argument("--job", default="sebeni-vol2-job")
    parser.add_argument("--firestore-db", default="chronicles-live")
    parser.add_argument("--max-ucdp-rows", type=int, default=1200)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--schedule", default="", help="Optional cron expression, e.g. '0 */2 * * *'")
    args = parser.parse_args()
    deploy_vol2_job(
        project=args.project,
        region=args.region,
        job_name=args.job,
        firestore_db=args.firestore_db,
        max_ucdp_rows=args.max_ucdp_rows,
        execute=args.execute,
        schedule_cron=args.schedule,
    )


if __name__ == "__main__":
    main()
