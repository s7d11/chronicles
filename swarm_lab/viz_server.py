"""Stdlib HTTP + Server-Sent Events (SSE) live visualization & 13-language orchestrator."""

from __future__ import annotations

import asyncio
import hmac
import json
import mimetypes
import os
import threading
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence
from urllib.parse import parse_qs, urlparse

from swarm_lab import ALL_LANGS, DEFAULT_LANGS
from swarm_lab.hf_sync import push_language_state_to_hf

REPO_ROOT = Path(__file__).resolve().parent.parent
SITE_DIR = REPO_ROOT / "site"
DEFAULT_RUNS_DIR = REPO_ROOT / "runs" / "edition2"


def ordered_all_langs() -> List[str]:
    """Return the 4 Edition 2 launch languages first, followed by the remaining 9."""
    return list(DEFAULT_LANGS) + [l for l in ALL_LANGS if l not in DEFAULT_LANGS]


def list_live_runs(
    runs_dir: Path = DEFAULT_RUNS_DIR,
    active_lang: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Return status across all 13 MULTI13 languages ordered with active/done first."""
    live_dir = runs_dir / "live"
    items: List[Dict[str, object]] = []
    for lang in ordered_all_langs():
        live_p = live_dir / f"{lang}.jsonl"
        summary_ok = (runs_dir / lang / "language_summary.json").is_file()
        n_ev = 0
        mtime = 0.0
        done = summary_ok
        if live_p.is_file():
            try:
                lines = [ln for ln in live_p.read_text(encoding="utf-8").splitlines() if ln.strip()]
                n_ev = len(lines)
                mtime = live_p.stat().st_mtime
                if lines and not done:
                    done = json.loads(lines[-1]).get("kind") == "run_complete"
            except Exception:
                pass
        elif summary_ok:
            mtime = (runs_dir / lang / "language_summary.json").stat().st_mtime

        items.append(
            {
                "lang": lang,
                "events": n_ev,
                "mtime": mtime,
                "done": done,
                "active": (lang == active_lang),
            }
        )
    return items


class LiveSwarmController:
    """Single-flight background runner executing Edition No. 2 sequentially across languages."""

    def __init__(
        self,
        *,
        runs_dir: Path = DEFAULT_RUNS_DIR,
        site_dir: Path = SITE_DIR,
        state_repo: Optional[str] = None,
        run_fn: Optional[Callable[[str, Path, bool], object]] = None,
    ):
        self.runs_dir = runs_dir
        self.site_dir = site_dir
        self.state_repo = state_repo or os.environ.get("SWARM_STATE_REPO", "")
        self._run_fn = run_fn
        self._lock = threading.Lock()
        self.running = False
        self.active_lang: Optional[str] = None
        self.stop_requested = False
        self.last_error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None

    def completed_langs(self) -> List[str]:
        return [l for l in ordered_all_langs() if (self.runs_dir / l / "language_summary.json").is_file()]

    def health_snapshot(self) -> Dict[str, object]:
        done = self.completed_langs()
        pending = [l for l in ordered_all_langs() if l not in done]
        return {
            "status": "ok",
            "running": self.running,
            "active_lang": self.active_lang,
            "completed_langs": done,
            "pending_langs": pending,
            "total_langs": len(ALL_LANGS),
            "last_error": self.last_error,
            "state_repo": self.state_repo or None,
        }

    def start(self, langs: Sequence[str], *, resume: bool = True) -> bool:
        clean_langs = [l for l in langs if l in ALL_LANGS]
        if not clean_langs:
            return False
        with self._lock:
            if self.running:
                return False
            self.running = True
            self.stop_requested = False
            self.last_error = None
            t = threading.Thread(
                target=self._worker,
                args=(clean_langs, resume),
                daemon=True,
                name="edition2-live-runner",
            )
            self._thread = t
            t.start()
            return True

    def stop(self) -> None:
        with self._lock:
            self.stop_requested = True

    def _worker(self, langs: List[str], resume: bool) -> None:
        from swarm_lab.runner import publish_edition2, run_language_swarm

        site_data_dir = self.site_dir / "data"
        try:
            for lang in langs:
                if self.stop_requested:
                    break
                if resume and (self.runs_dir / lang / "language_summary.json").is_file():
                    continue
                self.active_lang = lang
                try:
                    if self._run_fn is not None:
                        self._run_fn(lang, self.runs_dir, resume)
                    else:
                        asyncio.run(run_language_swarm(lang, runs_dir=self.runs_dir, resume=resume))
                        publish_edition2(runs_dir=self.runs_dir, out_dir=site_data_dir)
                        hf_tok = os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN") or ""
                        if self.state_repo and hf_tok:
                            push_language_state_to_hf(
                                self.state_repo,
                                self.runs_dir,
                                lang,
                                site_data_dir,
                                hf_tok,
                            )
                except Exception as exc:
                    self.last_error = f"{lang}: {type(exc).__name__}: {str(exc)[:220]}"
                    print(f"[swarm-runner] error on {lang}: {self.last_error}", flush=True)
        finally:
            self.active_lang = None
            self.running = False


class SwarmVizHandler(SimpleHTTPRequestHandler):
    """Serves static site/ assets alongside CORS SSE streams and admin triggers."""

    runs_dir: Path = DEFAULT_RUNS_DIR
    site_dir: Path = SITE_DIR
    controller: Optional[LiveSwarmController] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.site_dir), **kwargs)

    def log_message(self, format: str, *args) -> None:
        return

    def _cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")

    def _send_json(self, obj: object, status: int = 200) -> None:
        body = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file_json(self, path: Path) -> None:
        if not path.is_file():
            self._send_json({"error": f"{path.name} not generated yet"}, status=404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self._cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _check_admin_auth(self) -> Optional[tuple]:
        expected = os.environ.get("SWARM_ADMIN_TOKEN") or ""
        if not expected:
            return (503, {"error": "SWARM_ADMIN_TOKEN is not configured"})
        auth = self.headers.get("Authorization") or ""
        if not auth.startswith("Bearer "):
            return (401, {"error": "Missing Bearer token"})
        token = auth[len("Bearer ") :].strip()
        if not hmac.compare_digest(token, expected):
            return (401, {"error": "Invalid admin token"})
        return None

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        ctrl = self.controller
        active_lang = ctrl.active_lang if ctrl else None

        if parsed.path == "/api/health":
            snap = ctrl.health_snapshot() if ctrl else {"status": "ok", "running": False}
            self._send_json(snap)
            return
        if parsed.path == "/api/runs":
            self._send_json({"runs": list_live_runs(self.runs_dir, active_lang=active_lang)})
            return
        if parsed.path == "/api/latest":
            runs = list_live_runs(self.runs_dir, active_lang=active_lang)
            chosen = active_lang or next((r["lang"] for r in runs if int(r["events"]) > 0), runs[0]["lang"])
            self._send_json({"lang": chosen, "runs": runs})
            return
        if parsed.path == "/api/edition2.json":
            self._send_file_json(self.site_dir / "data" / "edition2.json")
            return
        if parsed.path == "/api/edition2_events.json":
            self._send_file_json(self.site_dir / "data" / "edition2_events.json")
            return
        if parsed.path == "/api/stream":
            qs = parse_qs(parsed.query)
            lang = (qs.get("lang") or [""])[0].strip().lower()
            if not lang:
                lang = str(active_lang or ordered_all_langs()[0])
            if lang not in ALL_LANGS:
                self._send_json({"error": f"Unknown language '{lang}'"}, status=404)
                return
            self._stream_sse(lang)
            return

        if parsed.path == "/" and (os.environ.get("SPACE_ID") or os.environ.get("K_SERVICE")):
            self.send_response(302)
            self.send_header("Location", "/editions/vol-01-no-02/theater.html")
            self.end_headers()
            return

        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/admin/trigger", "/api/admin/stop"}:
            self._send_json({"error": "Not found"}, status=404)
            return
        auth_err = self._check_admin_auth()
        if auth_err is not None:
            self._send_json(auth_err[1], status=auth_err[0])
            return
        ctrl = self.controller
        if ctrl is None:
            self._send_json({"error": "Controller unavailable"}, status=503)
            return

        if parsed.path == "/api/admin/stop":
            ctrl.stop()
            self._send_json({"stopping": True})
            return

        length = int(self.headers.get("Content-Length") or "0")
        raw_body = self.rfile.read(min(length, 65536)).decode("utf-8", errors="replace") if length > 0 else "{}"
        try:
            body = json.loads(raw_body or "{}")
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body"}, status=400)
            return

        req_langs = body.get("langs", "all")
        resume = bool(body.get("resume", True))
        if req_langs == "all":
            langs = ordered_all_langs()
        elif isinstance(req_langs, list):
            langs = [str(x).strip().lower() for x in req_langs if str(x).strip().lower() in ALL_LANGS]
        else:
            langs = [x.strip().lower() for x in str(req_langs).split(",") if x.strip().lower() in ALL_LANGS]

        if not langs:
            self._send_json({"error": "No valid MULTI13 language codes provided"}, status=400)
            return
        if not ctrl.start(langs, resume=resume):
            self._send_json({"error": "A swarm run is already active", "active_lang": ctrl.active_lang}, status=409)
            return
        self._send_json({"accepted": True, "langs": langs, "resume": resume}, status=202)

    def _stream_sse(self, lang: str) -> None:
        jsonl_path = self.runs_dir / "live" / f"{lang}.jsonl"
        self.send_response(HTTPStatus.OK)
        self._cors_headers()
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        pos = 0
        idle_ticks = 0
        try:
            while idle_ticks < 14400:
                if jsonl_path.is_file():
                    with jsonl_path.open("r", encoding="utf-8") as f:
                        f.seek(pos)
                        chunk = f.readlines()
                        pos = f.tell()
                    if chunk:
                        idle_ticks = 0
                        last_kind = ""
                        for line in chunk:
                            s = line.strip()
                            if not s:
                                continue
                            self.wfile.write(f"data: {s}\n\n".encode("utf-8"))
                            try:
                                last_kind = str(json.loads(s).get("kind", ""))
                            except Exception:
                                pass
                        self.wfile.flush()
                        if last_kind == "run_complete":
                            self.wfile.write(b"event: done\ndata: {}\n\n")
                            self.wfile.flush()
                            return
                        continue
                idle_ticks += 1
                if idle_ticks % 40 == 0:  # every 10s
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                time.sleep(0.25)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return


def make_handler_class(
    *,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    site_dir: Path = SITE_DIR,
    controller: Optional[LiveSwarmController] = None,
) -> type:
    """Create a bound SwarmVizHandler class (useful for tests & multi-instance servers)."""

    class BoundHandler(SwarmVizHandler):
        pass

    BoundHandler.runs_dir = runs_dir
    BoundHandler.site_dir = site_dir
    BoundHandler.controller = controller
    return BoundHandler


def serve_viz(
    *,
    port: int = 8765,
    host: str = "127.0.0.1",
    runs_dir: Path = DEFAULT_RUNS_DIR,
    site_dir: Path = SITE_DIR,
    auto_run_all_13: bool = False,
    state_repo: Optional[str] = None,
) -> None:
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/css", ".css")

    controller = LiveSwarmController(
        runs_dir=runs_dir,
        site_dir=site_dir,
        state_repo=state_repo,
    )
    if auto_run_all_13:
        pending = [l for l in ordered_all_langs() if not (runs_dir / l / "language_summary.json").is_file()]
        if pending:
            print(f"[swarm-viz] Auto-starting remaining {len(pending)}/13 languages: {', '.join(pending)}", flush=True)
            controller.start(pending, resume=True)
        else:
            print("[swarm-viz] All 13 languages already completed.", flush=True)

    handler_cls = make_handler_class(runs_dir=runs_dir, site_dir=site_dir, controller=controller)
    server = ThreadingHTTPServer((host, port), handler_cls)
    print(f"[swarm-viz] Listening on http://{host}:{port}/editions/vol-01-no-02/theater.html", flush=True)
    server.serve_forever()
