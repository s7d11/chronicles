#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/home/user/app"
RUNTIME_DIR="${APP_DIR}/runtime_vendor"
RUNS_DIR="${APP_DIR}/runs/edition2"
SITE_DATA_DIR="${APP_DIR}/site/data"
RUNTIME_REPO="${SEBENI_RUNTIME_REPO:-s7d11/sebeni-runtime-private}"
STATE_REPO="${SWARM_STATE_REPO:-s7d11/chronicles-edition2-state}"

echo "[entrypoint] Fetching private runtime from ${RUNTIME_REPO}..."
python3 - <<'PY'
import os
from pathlib import Path
from swarm_lab.hf_sync import download_and_extract_runtime, restore_state_from_hf

tok = os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN") or ""
runtime_repo = os.environ.get("SEBENI_RUNTIME_REPO", "s7d11/sebeni-runtime-private")
state_repo = os.environ.get("SWARM_STATE_REPO", "s7d11/chronicles-edition2-state")

download_and_extract_runtime(runtime_repo, Path("/home/user/app/runtime_vendor"), tok)
import beni, daba
print(f"[entrypoint] Runtime ready: beni={beni.__file__}, daba={daba.__file__}")

restored = restore_state_from_hf(state_repo, Path("/home/user/app/runs/edition2"), token=tok)
print(f"[entrypoint] Restored {len(restored)} languages from {state_repo}: {restored}")
PY

python3 -m swarm_lab.cli publish --runs-dir "${RUNS_DIR}" --out-dir "${SITE_DATA_DIR}"

echo "[entrypoint] Launching live 13-language Swarm Theater on port ${PORT:-7860}..."
exec python3 -m swarm_lab.cli viz \
    --host 0.0.0.0 \
    --port "${PORT:-7860}" \
    --runs-dir "${RUNS_DIR}" \
    --auto-run-all-13
