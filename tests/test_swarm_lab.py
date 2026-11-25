"""Offline unit tests for swarm_lab topologies, resources, Shadow DabaX, and publishing."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, List

from bench.adapters.base import AdapterCapabilities, GenerationResult, ModelAdapter
from swarm_lab import ALL_LANGS, TOPOLOGIES
from swarm_lab.coordinator import CoordinatorAgent
from swarm_lab.resources import (
    load_baseline_resources,
    load_prompt_native,
    load_test_texts,
    run_shadow_dabax_test,
)
from swarm_lab.runner import publish_edition2, run_edition2
from swarm_lab.schema import DictDelta, Edition2Payload
from swarm_lab.topology import canonical_edges, normalize_cluster_plan


class _MockSwarmAdapter(ModelAdapter):
    """Deterministic multi-topology mock adapter for offline swarm testing."""

    capabilities = AdapterCapabilities(is_api=True)

    def __init__(self, n_agents: int = 3, topology: str = "ring", native_lang: str = "taq"):
        self.n_agents = n_agents
        self.topology = topology
        self.native_lang = native_lang
        self.calls = 0

    async def count_tokens_batch(self, texts: List[str]):
        return [len(t.split()) for t in texts], "mock"

    async def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> GenerationResult:
        self.calls += 1
        sys_txt = messages[0]["content"]
        if "self-managed research cluster" in sys_txt:
            payload = {
                "n_agents": self.n_agents,
                "topology": self.topology,
                "agents": [{"role": f"Role{i+1}", "mandate": "Inspect stems"} for i in range(max(1, min(5, self.n_agents)))],
                "rationale": "Test cluster plan",
            }
            return GenerationResult(text=json.dumps(payload), raw_text=json.dumps(payload), latency_s=0.01)
        if "deliberation Round" in sys_txt:
            payload = {
                "agree_with_cluster": True,
                "dissent": "",
                "words": [
                    {"surface": "BROŽE", "morphemes": ["brože"], "pos": "n", "gloss": "project"},
                ],
                "dict_deltas": [
                    {"lx": "brože", "ps": "n", "ge": "project", "gf": "projet"},
                ],
                "gram_rules": [],
                "summary": "Confirmed project loanword gloss.",
            }
            return GenerationResult(text=json.dumps(payload), raw_text=json.dumps(payload), latency_s=0.01)
        if "Assess the target words" in sys_txt:
            payload = {
                "words": [
                    {"surface": "BROŽE", "morphemes": ["brože"], "pos": "n", "gloss": "project"},
                ],
                "dict_deltas": [{"lx": "brože", "ps": "n", "ge": "project", "gf": "projet"}],
                "gram_rules": [],
                "summary": "Initial split of BROŽE completed.",
            }
            return GenerationResult(text=json.dumps(payload), raw_text=json.dumps(payload), latency_s=0.01)
        if "Master Swarm Coordinator" in sys_txt:
            return GenerationResult(text="Coordinator brief for mock test.", raw_text="Coordinator brief.", latency_s=0.01)
        if "auditing all model clusters" in sys_txt:
            payload = {
                "verdict": "3-agent ring won on clean Shadow DabaX compile.",
                "scores": {
                    "glm-5-3-flash": {"score": 88.0, "note": "Unanimous in Round 2 with clean dict patch."},
                },
            }
            return GenerationResult(text=json.dumps(payload), raw_text=json.dumps(payload), latency_s=0.01)
        # Native generation or peer verification
        txt = "Eyǎf: ĂDDĂWLĂT N-MALI\nTiyər: Tamətte iyyăt.\nƏmmək: Konstitisyon.\nǍnnǎxwi: dǎɣ-ăddunya-ta-dǎɣ."
        return GenerationResult(text=txt, raw_text=txt, latency_s=0.01)


def test_iso_alias_loading() -> None:
    tests = load_test_texts()
    prompts = load_prompt_native()
    for code in ALL_LANGS:
        assert code in tests, f"Missing {code} in test.json"
        assert code in prompts, f"Missing {code} in prompt_native.csv"
    assert prompts["spp"]["Title"] == "Juntahamɛgi"


def test_cluster_plan_bounds_and_topologies() -> None:
    p_low = normalize_cluster_plan({"n_agents": 1, "topology": "mesh"}, cluster_slug="m1", model_name="M1")
    p_high = normalize_cluster_plan({"n_agents": 9, "topology": "unknown"}, cluster_slug="m2", model_name="M2")
    assert p_low.n_agents == 2
    assert p_high.n_agents == 5
    assert p_high.topology == "star"
    for topo in TOPOLOGIES:
        plan = normalize_cluster_plan({"n_agents": 4, "topology": topo}, cluster_slug="m", model_name="M")
        edges = canonical_edges(plan.agents, topo)
        assert len(edges) >= 3


def test_shadow_dabax_isolation_and_zero_phantom(tmp_path: Path) -> None:
    res = load_baseline_resources("taq")
    assert res["dict_entries"] > 100
    shadow_res = run_shadow_dabax_test(
        "taq",
        "ĂDDĂWLĂT N-MALI xyzqaw",
        [DictDelta(lx="xyzqaw", ps="n", ge="novel", gf="nouveau")],
        [],
        tmp_path / "shadow_taq",
    )
    assert shadow_res.dict_compile_ok is True
    assert shadow_res.dict_entries_added == 1
    assert shadow_res.shadow_test_parse_pct >= shadow_res.baseline_test_parse_pct
    assert shadow_res.phantom_rate_pct == 0.0


def test_end_to_end_mock_swarm_and_publish(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    out_dir = tmp_path / "site_data"
    coord = CoordinatorAgent(adapter_override=_MockSwarmAdapter())

    def _mk(spec):
        return _MockSwarmAdapter(n_agents=3, topology="ring")

    summaries = asyncio.run(
        run_edition2(
            langs=["taq"],
            cluster_slugs=["glm-5-3-flash"],
            runs_dir=runs_dir,
            resume=False,
            coordinator=coord,
            adapter_factory=_mk,
        )
    )
    assert len(summaries) == 1
    ls = summaries[0]
    assert ls.clusters[0].unanimous is True
    assert ls.clusters[0].plan.n_agents == 3

    payload = publish_edition2(runs_dir=runs_dir, out_dir=out_dir)
    validated = Edition2Payload.model_validate_json((out_dir / "edition2.json").read_text(encoding="utf-8"))
    assert validated.langs == ["taq"]
    events_doc = json.loads((out_dir / "edition2_events.json").read_text(encoding="utf-8"))
    kinds = {e["kind"] for e in events_doc["events_by_lang"]["taq"]}
    assert {"coordinator_brief", "cluster_spawn", "agent_exchange", "consensus_reached", "shadow_patch", "native_generation", "run_complete"}.issubset(kinds)


def test_pack_sebeni_runtime_minimal(tmp_path: Path) -> None:
    import tarfile
    from swarm_lab.hf_sync import pack_sebeni_runtime

    tar_path = pack_sebeni_runtime(tmp_path / "sebeni_runtime.tar.gz")
    assert tar_path.is_file()
    assert tar_path.stat().st_size < 25 * 1024 * 1024
    with tarfile.open(tar_path, "r:gz") as tf:
        names = set(tf.getnames())
    assert "beni/__init__.py" in names
    assert "beni/data/test.json" in names
    assert "beni/data/baselines/taq/baseline.dict" in names
    assert any(n.startswith("daba/") and n.endswith(".py") for n in names)
    assert not any(".env" in n or "dataset_300_samples" in n for n in names)


def test_viz_server_health_cors_and_admin_auth(tmp_path: Path, monkeypatch) -> None:
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer
    from swarm_lab.viz_server import LiveSwarmController, make_handler_class

    monkeypatch.setenv("SWARM_ADMIN_TOKEN", "unit-test-secret-token")
    runs_dir = tmp_path / "runs"
    site_dir = tmp_path / "site"
    (site_dir / "data").mkdir(parents=True)
    runs_dir.mkdir(parents=True)

    called_langs = []

    def _fake_run(lang: str, rd: Path, resume: bool) -> None:
        called_langs.append(lang)

    ctrl = LiveSwarmController(runs_dir=runs_dir, site_dir=site_dir, run_fn=_fake_run)
    cls = make_handler_class(runs_dir=runs_dir, site_dir=site_dir, controller=ctrl)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), cls)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=5) as r:
            assert r.status == 200
            assert r.headers.get("Access-Control-Allow-Origin") == "*"
            h = json.loads(r.read().decode("utf-8"))
            assert h["total_langs"] == 13

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs", timeout=5) as r:
            assert r.status == 200
            runs = json.loads(r.read().decode("utf-8"))["runs"]
            assert len(runs) == 13

        req_no_auth = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/admin/trigger",
            data=b'{"langs": ["bmq"]}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req_no_auth, timeout=5)
            raise AssertionError("Expected 401 without Bearer token")
        except urllib.error.HTTPError as e:
            assert e.code == 401

        req_ok = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/admin/trigger",
            data=b'{"langs": ["bmq"]}',
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer unit-test-secret-token",
            },
            method="POST",
        )
        with urllib.request.urlopen(req_ok, timeout=5) as r:
            assert r.status == 202
    finally:
        srv.shutdown()
        srv.server_close()
