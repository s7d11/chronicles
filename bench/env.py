"""Environment variable loader for local .env without logging secrets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def load_env(env_path: Optional[Path] = None) -> None:
    """Populate os.environ from a local .env file if keys are not already set."""
    candidate = env_path or (Path(__file__).resolve().parent.parent / ".env")
    if not candidate.is_file():
        return
    for raw_line in candidate.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val
