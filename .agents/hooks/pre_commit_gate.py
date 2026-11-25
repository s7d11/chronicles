#!/usr/bin/env python3
"""Antigravity PreToolUse hook: enforces Chronicle commit/layout/connectivity rules."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

COMMIT_CMD_RE = re.compile(
    r"(^|[;&|\n]\s*)(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*git(?:\s+-\S+(?:\s+\S+)?)*\s+commit\b"
)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:
        print(json.dumps({"decision": "deny", "reason": f"Invalid hook JSON payload: {exc}"}))
        return 0

    tool_call = payload.get("toolCall") or {}
    args = tool_call.get("args") or {}
    cmd = str(args.get("CommandLine") or "")

    if not COMMIT_CMD_RE.search(cmd):
        print(json.dumps({"decision": "allow"}))
        return 0

    repo_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(repo_root / "scripts"))
    from check_chronicle import validate_chronicle  # noqa: E402

    errors = validate_chronicle(repo_root, check_git_staged=True, check_node=True)
    if errors:
        reason = (
            "Chronicle pre-commit rules failed (read AGENTS.md before committing):\n- "
            + "\n- ".join(errors)
        )
        print(json.dumps({"decision": "deny", "reason": reason}))
        return 0

    print(json.dumps({"decision": "allow"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
