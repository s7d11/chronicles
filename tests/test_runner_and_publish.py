"""End-to-end integration test using MockAdapter, --resume, and publish_site_data."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from bench.adapters.mock import MockAdapter
from bench.dataset import DEFAULT_OUT_DIR
from bench.publish import publish_site_data
from bench.runner import run_model
from bench.schema import LeaderboardPayload, ModelSpec


def test_runner_resume_and_publish(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    site_data = tmp_path / "site_data"
    spec = ModelSpec(
        slug="mock-sebeni",
        display_name="Mock-Sebeni",
        provider="mock",
        model_id="mock-0",
        track="Sebeni",
        active_params_b=0.6,
        compute_label="Mock GPU",
    )

    adapter1 = MockAdapter()
    asyncio.run(
        run_model(
            spec,
            data_dir=DEFAULT_OUT_DIR,
            runs_dir=runs_dir,
            tasks=("morph", "gen", "lid", "cloze", "efficiency"),
            resume=True,
            limit_per_task=13,
            adapter_override=adapter1,
        )
    )
    assert adapter1.calls == 52  # 13 * 4 tasks

    # Second run with --resume must make 0 additional generation calls
    adapter2 = MockAdapter()
    asyncio.run(
        run_model(
            spec,
            data_dir=DEFAULT_OUT_DIR,
            runs_dir=runs_dir,
            tasks=("morph", "gen", "lid", "cloze", "efficiency"),
            resume=True,
            limit_per_task=13,
            adapter_override=adapter2,
        )
    )
    assert adapter2.calls == 0

    payload = publish_site_data(
        data_dir=DEFAULT_OUT_DIR,
        runs_dir=runs_dir,
        out_dir=site_data,
        registry={"mock-sebeni": spec},
    )
    validated = LeaderboardPayload.model_validate(
        json.loads((site_data / "leaderboard.json").read_text(encoding="utf-8"))
    )
    assert len(validated.models) == 1
    assert validated.models[0].id == "Mock-Sebeni"
    assert validated.models[0].bpb == 1.25
    assert (site_data / "samples.json").is_file()
    assert payload.total_samples == 728


def test_stratified_per_lang_runner(tmp_path: Path):
    runs_dir = tmp_path / "runs_strat"
    spec = ModelSpec(
        slug="mock-strat",
        display_name="Mock-Strat",
        provider="mock",
        model_id="mock-0",
        track="Sebeni",
        active_params_b=0.6,
    )
    adapter = MockAdapter()
    asyncio.run(
        run_model(
            spec,
            data_dir=DEFAULT_OUT_DIR,
            runs_dir=runs_dir,
            tasks=("morph", "lid"),
            resume=False,
            per_lang=1,
            adapter_override=adapter,
        )
    )
    # 1 per lang * 13 langs * 2 tasks = 26 calls
    assert adapter.calls == 26
