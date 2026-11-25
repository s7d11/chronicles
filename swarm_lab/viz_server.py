"""Stdlib HTTP + Server-Sent Events (SSE) live visualization server for Edition No. 2."""

from __future__ import annotations

import json
import mimetypes
import os
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
SITE_DIR = REPO_ROOT / "site"
DEFAULT_LIVE_DIR = REPO_ROOT / "runs" / "edition2" / "live"


def list_live_runs(live_dir: Path = DEFAULT_LIVE_DIR) -> List[Dict[str, object]]:
    """Return all live/recorded language streams ordered by most recent modification."""
    if not live_dir.is_dir():
        return []
    items: List[Dict[str, object]] = []
    for p in live_dir.glob("*.jsonl"):
        try:
            lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            done = False
            if lines:
                last = json.loads(lines[-1])
                done = last.get("kind") == "run_complete"
            items.append(
                {
                    "lang": p.stem,
                    "events": len(lines),
                    "mtime": p.stat().st_mtime,
                    "done": done,
                }
            )
        except Exception:
            continue
    items.sort(key=lambda r: -float(r["mtime"]))
    return items


class SwarmVizHandler(SimpleHTTPRequestHandler):
    """Serves static site/ assets alongside live SSE /api/stream?lang= feeds."""

    live_dir: Path = DEFAULT_LIVE_DIR

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SITE_DIR), **kwargs)

    def log_message(self, format: str, *args) -> None:
        return

    def _send_json(self, obj: object, status: int = 200) -> None:
        body = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/runs":
            self._send_json({"runs": list_live_runs(self.live_dir)})
            return
        if parsed.path == "/api/latest":
            runs = list_live_runs(self.live_dir)
            self._send_json({"lang": runs[0]["lang"] if runs else None, "runs": runs})
            return
        if parsed.path == "/api/stream":
            qs = parse_qs(parsed.query)
            lang = (qs.get("lang") or [""])[0].strip().lower()
            if not lang:
                runs = list_live_runs(self.live_dir)
                lang = str(runs[0]["lang"]) if runs else ""
            if not lang:
                self._send_json({"error": "No live run found"}, status=404)
                return
            self._stream_sse(lang)
            return

        # Serve static files cleanly
        super().do_GET()

    def _stream_sse(self, lang: str) -> None:
        jsonl_path = self.live_dir / f"{lang}.jsonl"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        pos = 0
        idle_ticks = 0
        try:
            while idle_ticks < 3600:
                if jsonl_path.is_file():
                    with jsonl_path.open("r", encoding="utf-8") as f:
                        f.seek(pos)
                        chunk = f.readlines()
                        pos = f.tell()
                    if chunk:
                        idle_ticks = 0
                        for line in chunk:
                            s = line.strip()
                            if not s:
                                continue
                            self.wfile.write(f"data: {s}\n\n".encode("utf-8"))
                        self.wfile.flush()
                        try:
                            if json.loads(chunk[-1]).get("kind") == "run_complete":
                                self.wfile.write(b"event: done\ndata: {}\n\n")
                                self.wfile.flush()
                                return
                        except Exception:
                            pass
                        continue
                idle_ticks += 1
                if idle_ticks % 20 == 0:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                time.sleep(0.25)
        except (BrokenPipeError, ConnectionResetError):
            return


def serve_viz(*, port: int = 8765, host: str = "127.0.0.1", live_dir: Path = DEFAULT_LIVE_DIR) -> None:
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/css", ".css")

    class BoundHandler(SwarmVizHandler):
        pass

    BoundHandler.live_dir = live_dir
    server = ThreadingHTTPServer((host, port), BoundHandler)
    print(f"[swarm-viz] Listening on http://{host}:{port}/editions/vol-01-no-02/theater.html", flush=True)
    server.serve_forever()
