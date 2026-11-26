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
from proof_lab import DATASET_LANGS
from proof_lab.hf_sync import push_ed3_language_to_hf
from village_lab import DEFAULT_LAUNCH_LANG, ED4_STATE_REPO, ORCHESTRATOR_SPEC, CRITIC_SPEC
from village_lab.hf_sync import publish_ed4_site_data, push_ed4_state_to_hf

REPO_ROOT = Path(__file__).resolve().parent.parent
SITE_DIR = REPO_ROOT / "site"
DEFAULT_RUNS_DIR = REPO_ROOT / "runs" / "edition2"
DEFAULT_ED3_RUNS_DIR = REPO_ROOT / "runs" / "edition3"
DEFAULT_ED4_RUNS_DIR = REPO_ROOT / "runs" / "edition4"


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

    def reset(self, lang: str = DEFAULT_LAUNCH_LANG) -> bool:
        import shutil

        clean_lang = lang.strip().lower()
        if clean_lang not in ALL_LANGS:
            return False
        with self._lock:
            if self.running:
                return False
            ldir = self.runs_dir / clean_lang
            if ldir.is_dir():
                shutil.rmtree(ldir, ignore_errors=True)
            publish_ed4_site_data(self.runs_dir, self.site_dir / "data")
            return True

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


class LiveProofController:
    """Single-flight background runner executing Edition No. 3 sequentially across 14 dataset_300 langs."""

    def __init__(
        self,
        *,
        runs_dir: Path = DEFAULT_ED3_RUNS_DIR,
        site_dir: Path = SITE_DIR,
        state_repo: Optional[str] = None,
        run_fn: Optional[Callable[[str, Path, bool], object]] = None,
    ):
        self.runs_dir = runs_dir
        self.site_dir = site_dir
        self.state_repo = state_repo or os.environ.get("ED3_STATE_REPO", "")
        self._run_fn = run_fn
        self._lock = threading.Lock()
        self.running = False
        self.active_lang: Optional[str] = None
        self.stop_requested = False
        self.last_error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None

    def completed_langs(self) -> List[str]:
        return [l for l in DATASET_LANGS if (self.runs_dir / l / "result.json").is_file()]

    def list_runs(self) -> List[Dict[str, object]]:
        live_dir = self.runs_dir / "live"
        items: List[Dict[str, object]] = []
        for lang in DATASET_LANGS:
            live_p = live_dir / f"{lang}.jsonl"
            res_p = self.runs_dir / lang / "result.json"
            done = res_p.is_file()
            n_ev = 0
            mtime = 0.0
            if live_p.is_file():
                try:
                    lines = [ln for ln in live_p.read_text(encoding="utf-8").splitlines() if ln.strip()]
                    n_ev = len(lines)
                    mtime = live_p.stat().st_mtime
                except Exception:
                    pass
            elif done:
                mtime = res_p.stat().st_mtime
            items.append({"lang": lang, "events": n_ev, "mtime": mtime, "done": done, "active": (lang == self.active_lang)})
        return items

    def health_snapshot(self) -> Dict[str, object]:
        done = self.completed_langs()
        pending = [l for l in DATASET_LANGS if l not in done]
        return {
            "status": "ok",
            "edition": 3,
            "running": self.running,
            "active_lang": self.active_lang,
            "completed_langs": done,
            "pending_langs": pending,
            "total_langs": len(DATASET_LANGS),
            "last_error": self.last_error,
            "state_repo": self.state_repo or None,
        }

    def start(self, langs: Sequence[str], *, resume: bool = True) -> bool:
        clean_langs = [l for l in langs if l in DATASET_LANGS]
        if not clean_langs:
            return False
        with self._lock:
            if self.running:
                return False
            self.running = True
            self.stop_requested = False
            self.last_error = None
            t = threading.Thread(target=self._worker, args=(clean_langs, resume), daemon=True, name="edition3-live-runner")
            self._thread = t
            t.start()
            return True

    def stop(self) -> None:
        with self._lock:
            self.stop_requested = True

    def reset(self, lang: str = DEFAULT_LAUNCH_LANG) -> bool:
        import shutil

        clean_lang = lang.strip().lower()
        if clean_lang not in ALL_LANGS:
            return False
        with self._lock:
            if self.running:
                return False
            ldir = self.runs_dir / clean_lang
            if ldir.is_dir():
                shutil.rmtree(ldir, ignore_errors=True)
            publish_ed4_site_data(self.runs_dir, self.site_dir / "data")
            return True

    def _worker(self, langs: List[str], resume: bool) -> None:
        from proof_lab.runner import publish_edition3, run_language_proof

        site_data_dir = self.site_dir / "data"
        try:
            for lang in langs:
                if self.stop_requested:
                    break
                if resume and (self.runs_dir / lang / "result.json").is_file():
                    continue
                self.active_lang = lang
                try:
                    if self._run_fn is not None:
                        self._run_fn(lang, self.runs_dir, resume)
                    else:
                        asyncio.run(run_language_proof(lang, runs_dir=self.runs_dir, resume=resume))
                        publish_edition3(runs_dir=self.runs_dir, out_dir=site_data_dir)
                        hf_tok = os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN") or ""
                        if self.state_repo and hf_tok:
                            push_ed3_language_to_hf(self.state_repo, self.runs_dir, lang, site_data_dir, hf_tok)
                except Exception as exc:
                    self.last_error = f"{lang}: {type(exc).__name__}: {str(exc)[:220]}"
                    print(f"[proof-runner] error on {lang}: {self.last_error}", flush=True)
        finally:
            self.active_lang = None
            self.running = False


class LiveVillageController:
    """Single-flight background runner executing Edition No. 4 Living Lexicon Village epochs."""

    def __init__(
        self,
        *,
        runs_dir: Path = DEFAULT_ED4_RUNS_DIR,
        site_dir: Path = SITE_DIR,
        state_repo: Optional[str] = None,
    ):
        self.runs_dir = runs_dir
        self.site_dir = site_dir
        self.state_repo = state_repo or os.environ.get("ED4_STATE_REPO", ED4_STATE_REPO)
        self._lock = threading.Lock()
        self.running = False
        self.active_lang: Optional[str] = None
        self.max_turns: int = 3
        self.stop_requested = False
        self.last_error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None

    def status_snapshot(self) -> Dict[str, object]:
        lang_stats: Dict[str, object] = {}
        for l in ALL_LANGS:
            st_p = self.runs_dir / l / "state.json"
            ev_p = self.runs_dir / l / "events.jsonl"
            n_ev = 0
            if ev_p.is_file():
                try:
                    n_ev = sum(1 for ln in ev_p.read_text(encoding="utf-8").splitlines() if ln.strip())
                except Exception:
                    pass
            if st_p.is_file():
                try:
                    st = json.loads(st_p.read_text(encoding="utf-8"))
                    lang_stats[l] = {
                        "baseline_lx": int(st.get("baseline_lx_count") or 0),
                        "added_lx": int(st.get("added_lx_count") or 0),
                        "added_va": int(st.get("added_va_count") or 0),
                        "added_gram": int(st.get("added_gram_rules_count") or 0),
                        "total_turns": int(st.get("total_turns") or 0),
                        "events": n_ev,
                    }
                except Exception:
                    lang_stats[l] = {"events": n_ev}
            else:
                lang_stats[l] = {"events": n_ev}
        return {
            "status": "ok",
            "edition": 4,
            "running": self.running,
            "active_lang": self.active_lang,
            "launch_lang": DEFAULT_LAUNCH_LANG,
            "orchestrator_model": ORCHESTRATOR_SPEC.model_id,
            "critic_model": CRITIC_SPEC.model_id,
            "state_repo": self.state_repo or None,
            "last_error": self.last_error,
            "languages": lang_stats,
        }

    def start(self, lang: str = DEFAULT_LAUNCH_LANG, *, max_turns: int = 3) -> bool:
        clean_lang = lang.strip().lower()
        if clean_lang not in ALL_LANGS:
            return False
        with self._lock:
            if self.running:
                return False
            self.running = True
            self.active_lang = clean_lang
            self.max_turns = max(1, min(int(max_turns), 25))
            self.stop_requested = False
            self.last_error = None
            t = threading.Thread(
                target=self._worker,
                args=(clean_lang, self.max_turns),
                daemon=True,
                name="edition4-village-runner",
            )
            self._thread = t
            t.start()
            return True

    def stop(self) -> None:
        with self._lock:
            self.stop_requested = True

    def reset(self, lang: str = DEFAULT_LAUNCH_LANG) -> bool:
        import shutil

        clean_lang = lang.strip().lower()
        if clean_lang not in ALL_LANGS:
            return False
        with self._lock:
            if self.running:
                return False
            ldir = self.runs_dir / clean_lang
            if ldir.is_dir():
                shutil.rmtree(ldir, ignore_errors=True)
            publish_ed4_site_data(self.runs_dir, self.site_dir / "data")
            return True

    def _worker(self, lang: str, max_turns: int) -> None:
        from village_lab.world_engine import LanguageVillageEngine

        site_data_dir = self.site_dir / "data"
        try:
            eng = LanguageVillageEngine(lang, runs_dir=self.runs_dir)

            def _on_turn(_turn_num: int) -> None:
                publish_ed4_site_data(self.runs_dir, site_data_dir)
                hf_tok = os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN") or ""
                if self.state_repo and hf_tok:
                    push_ed4_state_to_hf(self.state_repo, self.runs_dir, lang, site_data_dir, hf_tok)

            asyncio.run(eng.run_epoch(max_turns=max_turns, on_turn_complete=_on_turn))
            publish_ed4_site_data(self.runs_dir, site_data_dir)
            hf_tok = os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN") or ""
            if self.state_repo and hf_tok:
                push_ed4_state_to_hf(self.state_repo, self.runs_dir, lang, site_data_dir, hf_tok)
        except Exception as exc:
            self.last_error = f"{lang}: {type(exc).__name__}: {str(exc)[:220]}"
            print(f"[village-runner] error on {lang}: {self.last_error}", flush=True)
        finally:
            self.active_lang = None
            self.running = False


class SwarmVizHandler(SimpleHTTPRequestHandler):
    """Serves static site/ assets alongside CORS SSE streams and admin triggers."""

    runs_dir: Path = DEFAULT_RUNS_DIR
    ed3_runs_dir: Path = DEFAULT_ED3_RUNS_DIR
    ed4_runs_dir: Path = DEFAULT_ED4_RUNS_DIR
    site_dir: Path = SITE_DIR
    controller: Optional[LiveSwarmController] = None
    ed3_controller: Optional[LiveProofController] = None
    ed4_controller: Optional[LiveVillageController] = None

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
            self._stream_jsonl_sse(self.runs_dir / "live" / f"{lang}.jsonl")
            return

        # Edition No. 3 live routes (SebeniProofs 14-language stress tests & direct deductions)
        ed3 = self.ed3_controller
        ed3_active = ed3.active_lang if ed3 else None
        if parsed.path == "/api/ed3/health":
            snap = ed3.health_snapshot() if ed3 else {"status": "ok", "edition": 3, "running": False}
            self._send_json(snap)
            return
        if parsed.path == "/api/ed3/runs":
            runs = ed3.list_runs() if ed3 else [{"lang": l, "events": 0, "mtime": 0.0, "done": False, "active": False} for l in DATASET_LANGS]
            self._send_json({"runs": runs})
            return
        if parsed.path == "/api/ed3/edition3.json":
            self._send_file_json(self.site_dir / "data" / "edition3.json")
            return
        if parsed.path == "/api/ed3/edition3_events.json":
            self._send_file_json(self.site_dir / "data" / "edition3_events.json")
            return
        if parsed.path == "/api/ed3/stream":
            qs = parse_qs(parsed.query)
            lang = (qs.get("lang") or [""])[0].strip().lower() or str(ed3_active or DATASET_LANGS[0])
            if lang not in DATASET_LANGS:
                self._send_json({"error": f"Unknown Edition 3 language '{lang}'"}, status=404)
                return
            ed3_root = ed3.runs_dir if ed3 else self.ed3_runs_dir
            self._stream_jsonl_sse(ed3_root / "live" / f"{lang}.jsonl")
            return

        # Edition No. 4 live routes (50k Living Lexicon Village & 4-Agent Council + Argon Critic)
        ed4 = self.ed4_controller
        ed4_active = ed4.active_lang if ed4 else None
        ed4_root = ed4.runs_dir if ed4 else self.ed4_runs_dir
        if parsed.path == "/api/ed4/status":
            snap = ed4.status_snapshot() if ed4 else {"status": "ok", "edition": 4, "running": False}
            self._send_json(snap)
            return
        if parsed.path == "/api/ed4/state":
            qs = parse_qs(parsed.query)
            lang = (qs.get("lang") or [""])[0].strip().lower() or str(ed4_active or DEFAULT_LAUNCH_LANG)
            if lang not in ALL_LANGS:
                self._send_json({"error": f"Unknown Edition 4 language '{lang}'"}, status=404)
                return
            st_path = ed4_root / lang / "state.json"
            if st_path.is_file():
                self._send_file_json(st_path)
            else:
                self._send_file_json(self.site_dir / "data" / "edition4.json")
            return
        if parsed.path == "/api/ed4/edition4.json":
            self._send_file_json(self.site_dir / "data" / "edition4.json")
            return
        if parsed.path == "/api/ed4/edition4_events.json":
            self._send_file_json(self.site_dir / "data" / "edition4_events.json")
            return
        if parsed.path == "/api/ed4/stream":
            qs = parse_qs(parsed.query)
            lang = (qs.get("lang") or [""])[0].strip().lower() or str(ed4_active or DEFAULT_LAUNCH_LANG)
            if lang not in ALL_LANGS:
                self._send_json({"error": f"Unknown Edition 4 language '{lang}'"}, status=404)
                return
            self._stream_jsonl_sse(ed4_root / lang / "events.jsonl", done_kind="epoch_checkpoint")
            return

        if parsed.path == "/" and (os.environ.get("SPACE_ID") or os.environ.get("K_SERVICE")):
            self.send_response(302)
            self.send_header("Location", "/editions/vol-01-no-04/world.html")
            self.end_headers()
            return

        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        allowed = {
            "/api/admin/trigger",
            "/api/admin/stop",
            "/api/ed3/admin/trigger",
            "/api/ed3/admin/stop",
            "/api/ed4/start",
            "/api/ed4/stop",
            "/api/ed4/reset",
        }
        if parsed.path not in allowed:
            self._send_json({"error": "Not found"}, status=404)
            return
        auth_err = self._check_admin_auth()
        if auth_err is not None:
            self._send_json(auth_err[1], status=auth_err[0])
            return

        if parsed.path.startswith("/api/ed4/"):
            ed4_ctrl = self.ed4_controller
            if ed4_ctrl is None:
                self._send_json({"error": "Edition 4 controller unavailable"}, status=503)
                return
            if parsed.path.endswith("/stop"):
                ed4_ctrl.stop()
                self._send_json({"stopping": True})
                return
            length = int(self.headers.get("Content-Length") or "0")
            raw_body = self.rfile.read(min(length, 65536)).decode("utf-8", errors="replace") if length > 0 else "{}"
            try:
                body = json.loads(raw_body or "{}")
            except json.JSONDecodeError:
                self._send_json({"error": "Invalid JSON body"}, status=400)
                return
            req_lang = str(body.get("lang") or DEFAULT_LAUNCH_LANG).strip().lower()
            if parsed.path.endswith("/reset"):
                if req_lang not in ALL_LANGS:
                    self._send_json({"error": f"Invalid language code '{req_lang}'"}, status=400)
                    return
                if not ed4_ctrl.reset(req_lang):
                    self._send_json({"error": "Cannot reset while epoch is running"}, status=409)
                    return
                self._send_json({"reset": True, "lang": req_lang}, status=200)
                return
            max_turns = int(body.get("max_turns") or 3)
            if req_lang not in ALL_LANGS:
                self._send_json({"error": f"Invalid language code '{req_lang}'"}, status=400)
                return
            if not ed4_ctrl.start(req_lang, max_turns=max_turns):
                self._send_json({"error": "An Edition 4 epoch is already active", "active_lang": ed4_ctrl.active_lang}, status=409)
                return
            self._send_json({"accepted": True, "lang": req_lang, "max_turns": max_turns}, status=202)
            return

        is_ed3 = parsed.path.startswith("/api/ed3/")
        ctrl = self.ed3_controller if is_ed3 else self.controller
        if ctrl is None:
            self._send_json({"error": "Controller unavailable"}, status=503)
            return

        if parsed.path.endswith("/stop"):
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

        valid_set = list(DATASET_LANGS) if is_ed3 else ordered_all_langs()
        req_langs = body.get("langs", "all")
        resume = bool(body.get("resume", True))
        if req_langs == "all":
            langs = valid_set
        elif isinstance(req_langs, list):
            langs = [str(x).strip().lower() for x in req_langs if str(x).strip().lower() in valid_set]
        else:
            langs = [x.strip().lower() for x in str(req_langs).split(",") if x.strip().lower() in valid_set]

        if not langs:
            self._send_json({"error": "No valid language codes provided"}, status=400)
            return
        if not ctrl.start(langs, resume=resume):
            self._send_json({"error": "A run is already active", "active_lang": ctrl.active_lang}, status=409)
            return
        self._send_json({"accepted": True, "langs": langs, "resume": resume}, status=202)

    def _stream_jsonl_sse(self, jsonl_path: Path, *, done_kind: str = "run_complete") -> None:
        self.send_response(HTTPStatus.OK)
        self._cors_headers()
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        pos = 0
        idle_ticks = 0
        initial_drain = True
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
                        is_ed4_epoch = done_kind == "epoch_checkpoint" and self.ed4_controller and self.ed4_controller.running
                        if last_kind == done_kind and not (initial_drain and is_ed4_epoch):
                            self.wfile.write(b"event: done\ndata: {}\n\n")
                            self.wfile.flush()
                            return
                        initial_drain = False
                        continue
                initial_drain = False
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
    ed3_runs_dir: Path = DEFAULT_ED3_RUNS_DIR,
    ed3_controller: Optional[LiveProofController] = None,
    ed4_runs_dir: Path = DEFAULT_ED4_RUNS_DIR,
    ed4_controller: Optional[LiveVillageController] = None,
) -> type:
    """Create a bound SwarmVizHandler class (useful for tests & multi-instance servers)."""

    class BoundHandler(SwarmVizHandler):
        pass

    BoundHandler.runs_dir = runs_dir
    BoundHandler.ed3_runs_dir = ed3_runs_dir
    BoundHandler.ed4_runs_dir = ed4_runs_dir
    BoundHandler.site_dir = site_dir
    BoundHandler.controller = controller
    BoundHandler.ed3_controller = ed3_controller
    BoundHandler.ed4_controller = ed4_controller
    return BoundHandler


def serve_viz(
    *,
    port: int = 8765,
    host: str = "127.0.0.1",
    runs_dir: Path = DEFAULT_RUNS_DIR,
    site_dir: Path = SITE_DIR,
    auto_run_all_13: bool = False,
    state_repo: Optional[str] = None,
    ed3_runs_dir: Optional[Path] = None,
    auto_run_ed3: bool = False,
    ed3_state_repo: Optional[str] = None,
    ed4_runs_dir: Optional[Path] = None,
    auto_run_ed4: bool = False,
    ed4_state_repo: Optional[str] = None,
) -> None:
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/css", ".css")

    ed3_dir = ed3_runs_dir or DEFAULT_ED3_RUNS_DIR
    ed4_dir = ed4_runs_dir or DEFAULT_ED4_RUNS_DIR
    controller = LiveSwarmController(runs_dir=runs_dir, site_dir=site_dir, state_repo=state_repo)
    ed3_ctrl = LiveProofController(runs_dir=ed3_dir, site_dir=site_dir, state_repo=ed3_state_repo)
    ed4_ctrl = LiveVillageController(runs_dir=ed4_dir, site_dir=site_dir, state_repo=ed4_state_repo)

    if auto_run_all_13:
        pending = [l for l in ordered_all_langs() if not (runs_dir / l / "language_summary.json").is_file()]
        if pending:
            print(f"[swarm-viz] Auto-starting remaining {len(pending)}/13 Ed.2 languages: {', '.join(pending)}", flush=True)
            controller.start(pending, resume=True)
    if auto_run_ed3:
        pending3 = [l for l in DATASET_LANGS if not (ed3_dir / l / "result.json").is_file()]
        if pending3:
            print(f"[proof-viz] Auto-starting remaining {len(pending3)}/14 Ed.3 languages: {', '.join(pending3)}", flush=True)
            ed3_ctrl.start(pending3, resume=True)
    if auto_run_ed4:
        print(f"[village-viz] Auto-starting Edition 4 epoch on {DEFAULT_LAUNCH_LANG}...", flush=True)
        ed4_ctrl.start(DEFAULT_LAUNCH_LANG, max_turns=3)

    handler_cls = make_handler_class(
        runs_dir=runs_dir,
        site_dir=site_dir,
        controller=controller,
        ed3_runs_dir=ed3_dir,
        ed3_controller=ed3_ctrl,
        ed4_runs_dir=ed4_dir,
        ed4_controller=ed4_ctrl,
    )
    server = ThreadingHTTPServer((host, port), handler_cls)
    print(f"[chronicle-viz] Listening on http://{host}:{port}/editions/vol-01-no-04/world.html", flush=True)
    server.serve_forever()
