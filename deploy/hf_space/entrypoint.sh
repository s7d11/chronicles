#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/home/user/app"
RUNTIME_DIR="${APP_DIR}/runtime_vendor"
RUNS_DIR="${APP_DIR}/runs/edition2"
ED3_RUNS_DIR="${APP_DIR}/runs/edition3"
SITE_DATA_DIR="${APP_DIR}/site/data"
RUNTIME_REPO="${SEBENI_RUNTIME_REPO:-s7d11/sebeni-runtime-private}"
STATE_REPO="${SWARM_STATE_REPO:-s7d11/chronicles-edition2-state}"
ED3_STATE_REPO="${ED3_STATE_REPO:-s7d11/chronicles-edition3-state}"

echo "[entrypoint] Fetching private runtime from ${RUNTIME_REPO}..."
python3 - <<'PY'
import os
from pathlib import Path
from swarm_lab.hf_sync import download_and_extract_runtime, restore_state_from_hf
from proof_lab.hf_sync import restore_ed3_state_from_hf

tok = os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN") or ""
runtime_repo = os.environ.get("SEBENI_RUNTIME_REPO", "s7d11/sebeni-runtime-private")
state_repo = os.environ.get("SWARM_STATE_REPO", "s7d11/chronicles-edition2-state")
ed3_state_repo = os.environ.get("ED3_STATE_REPO", "s7d11/chronicles-edition3-state")

download_and_extract_runtime(runtime_repo, Path("/home/user/app/runtime_vendor"), tok)
import beni, daba
print(f"[entrypoint] Runtime ready: beni={beni.__file__}, daba={daba.__file__}")

restored = restore_state_from_hf(state_repo, Path("/home/user/app/runs/edition2"), token=tok)
print(f"[entrypoint] Edition 2 restored {len(restored)} languages: {restored}")

restored_ed3 = restore_ed3_state_from_hf(ed3_state_repo, Path("/home/user/app/runs/edition3"), token=tok)
print(f"[entrypoint] Edition 3 restored {len(restored_ed3)} languages: {restored_ed3}")
PY

python3 -m swarm_lab.cli publish --runs-dir "${RUNS_DIR}" --out-dir "${SITE_DATA_DIR}"
python3 -m proof_lab.cli publish --runs-dir "${ED3_RUNS_DIR}" --out-dir "${SITE_DATA_DIR}"

ARGS=(--host 0.0.0.0 --port "${PORT:-7860}" --runs-dir "${RUNS_DIR}" --ed3-runs-dir "${ED3_RUNS_DIR}")
if [[ "${AUTO_RUN_ALL_13:-0}" == "1" ]]; then
  ARGS+=(--auto-run-all-13)
fi
if [[ "${AUTO_RUN_ED3:-0}" == "1" ]]; then
  ARGS+=(--auto-run-ed3)
fi

echo "[entrypoint] Launching Sebeni Chronicle Live Server on port ${PORT:-7860}..."
exec python3 -m swarm_lab.cli viz "${ARGS[@]}"
