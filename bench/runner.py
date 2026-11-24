"""Resumable MALI13 benchmark runner with JSONL completion caching."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from bench import LANGS
from bench.adapters.base import ModelAdapter
from bench.adapters.local_sebeni import LocalSebeniAdapter
from bench.adapters.mock import MockAdapter
from bench.adapters.openai_compat import OpenAICompatAdapter
from bench.adapters.vertex import VertexGeminiAdapter
from bench.dataset import DEFAULT_OUT_DIR, load_split
from bench.env import load_env
from bench.prompts import (
    build_cloze_messages,
    build_gen_messages,
    build_lid_messages,
    build_morph_messages,
)
from bench.schema import CompletionRecord, EfficiencyStats, ModelSpec

DEFAULT_RUNS_DIR = Path(__file__).resolve().parent.parent / "runs" / "mali13"


def create_adapter(spec: ModelSpec) -> ModelAdapter:
    load_env()
    if spec.provider == "mock":
        return MockAdapter()
    if spec.provider == "local_sebeni":
        return LocalSebeniAdapter(
            base_model_id=spec.model_id,
            lora_repo=spec.lora_repo or "mlsftwrs/sebeni-gemma-4-e2b-grpo-50",
            active_params_b=spec.active_params_b or 2.0,
        )
    if spec.provider == "adaption":
        key = os.environ.get("ADOPTION_KEY") or os.environ.get("ADAPTION_API_KEY") or ""
        if not key:
            raise RuntimeError("ADOPTION_KEY missing in environment/.env for Adaption Labs")
        return OpenAICompatAdapter(
            provider="adaption",
            model_id=spec.model_id,
            api_key=key,
            base_url="https://api.prod.adaptionlabs.ai/api/v1",
            hf_tokenizer_id=spec.hf_tokenizer_id,
            active_params_b=spec.active_params_b,
            price_in_per_1m=spec.price_in_per_1m,
            price_out_per_1m=spec.price_out_per_1m,
        )
    if spec.provider == "nim":
        key = os.environ.get("NVIDIA_KEY") or ""
        if not key:
            raise RuntimeError("NVIDIA_KEY missing in environment/.env for NVIDIA NIM")
        return OpenAICompatAdapter(
            provider="nim",
            model_id=spec.model_id,
            api_key=key,
            base_url="https://integrate.api.nvidia.com/v1",
            hf_tokenizer_id=spec.hf_tokenizer_id,
            active_params_b=spec.active_params_b,
            price_in_per_1m=spec.price_in_per_1m,
            price_out_per_1m=spec.price_out_per_1m,
        )
    if spec.provider == "vertex":
        return VertexGeminiAdapter(
            model_id=spec.model_id,
            project_id=os.environ.get("VERTEX_PROJECT", "gravityx-509611"),
            location=os.environ.get("VERTEX_LOCATION", "global"),
            price_in_per_1m=spec.price_in_per_1m,
            price_out_per_1m=spec.price_out_per_1m,
        )
    raise ValueError(f"Unsupported provider: {spec.provider}")


def load_cached_records(cache_path: Path) -> Dict[str, CompletionRecord]:
    out: Dict[str, CompletionRecord] = {}
    if not cache_path.is_file():
        return out
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = CompletionRecord.model_validate_json(line)
            out[rec.sample_id] = rec
        except Exception:
            continue
    return out


def _append_record(cache_path: Path, rec: CompletionRecord) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(rec.model_dump_json() + "\n")


async def run_efficiency(
    spec: ModelSpec,
    adapter: ModelAdapter,
    data_dir: Path,
    run_dir: Path,
    *,
    resume: bool = True,
) -> EfficiencyStats:
    eff_path = run_dir / "efficiency.json"
    if resume and eff_path.is_file():
        try:
            return EfficiencyStats.model_validate_json(eff_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    samples = load_split(data_dir, "efficiency")
    texts = [s.text for s in samples]
    counts, tok_mode = await adapter.count_tokens_batch(texts)

    # Also measure ASCII vs Diacritic token fertility directly on unique attested words
    ascii_words: List[str] = []
    diac_words: List[str] = []
    for s in samples:
        ascii_words.extend(s.ascii_tokens)
        diac_words.extend(s.diacritic_tokens)

    def _dedup_weighted(words_list: Sequence[str], max_unique: int = 200) -> Tuple[List[str], Dict[str, int]]:
        freq: Dict[str, int] = {}
        order: List[str] = []
        for w in words_list:
            if w not in freq:
                if len(order) >= max_unique:
                    continue
                order.append(w)
                freq[w] = 1
            else:
                freq[w] += 1
        return order, freq

    u_ascii, f_ascii = _dedup_weighted(ascii_words, max_unique=200 if spec.provider == "vertex" else 800)
    u_diac, f_diac = _dedup_weighted(diac_words, max_unique=200 if spec.provider == "vertex" else 800)

    ascii_counts, _ = await adapter.count_tokens_batch(u_ascii) if u_ascii else ([], tok_mode)
    diac_counts, _ = await adapter.count_tokens_batch(u_diac) if u_diac else ([], tok_mode)

    total_subwords = sum(counts)
    total_words = sum(s.n_words for s in samples)
    total_morphemes = sum(s.n_dabax_morphemes for s in samples)
    total_bytes = sum(s.utf8_bytes for s in samples)

    per_lang_sub: Dict[str, int] = {l: 0 for l in LANGS}
    per_lang_w: Dict[str, int] = {l: 0 for l in LANGS}
    for s, c in zip(samples, counts):
        per_lang_sub[s.lang] += c
        per_lang_w[s.lang] += s.n_words

    per_lang_fert = {
        l: round(per_lang_sub[l] / max(1, per_lang_w[l]), 3) for l in LANGS
    }
    a_tot = sum(c * f_ascii[w] for w, c in zip(u_ascii, ascii_counts))
    a_den = sum(f_ascii[w] for w in u_ascii)
    d_tot = sum(c * f_diac[w] for w, c in zip(u_diac, diac_counts))
    d_den = sum(f_diac[w] for w in u_diac)
    ascii_fert = (a_tot / max(1, a_den)) if u_ascii else 1.0
    diac_fert = (d_tot / max(1, d_den)) if u_diac else ascii_fert
    diac_pen = diac_fert / max(0.1, ascii_fert)

    bpb = await adapter.compute_bpb(texts[:26]) if adapter.capabilities.supports_bpb else None

    stats = EfficiencyStats(
        word_fertility=round(total_subwords / max(1, total_words), 3),
        morpheme_fertility=round(total_subwords / max(1, total_morphemes), 3),
        bytes_per_token=round(total_bytes / max(1, total_subwords), 3),
        diacritic_penalty=round(diac_pen, 3),
        ascii_fertility=round(ascii_fert, 3),
        diacritic_fertility=round(diac_fert, 3),
        bpb=bpb,
        tokenizer_mode=tok_mode,
        per_lang_fertility=per_lang_fert,
    )
    eff_path.parent.mkdir(parents=True, exist_ok=True)
    eff_path.write_text(stats.model_dump_json(indent=2) + "\n", encoding="utf-8")

    # Save shatter samples if adapter exposes piece tokenization
    shatter_words = {
        "taq": "dăɣ-ăddunya-ta-dăɣ",
        "ful": "suɓaaɓe",
        "spp": "SARIYAƝUNGI",
        "mey": "ƏLWAAJIBAAT",
        "dtm": "PÀŊÁBE",
    }
    pieces_out = {}
    for lang, w in shatter_words.items():
        p = adapter.tokenize_pieces(w)
        if p:
            pieces_out[lang] = p
    if pieces_out:
        (run_dir / "shatter.json").write_text(
            json.dumps(pieces_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if adapter.vram_mb() is not None:
        (run_dir / "telemetry.json").write_text(
            json.dumps({"vram_mb": adapter.vram_mb()}) + "\n", encoding="utf-8"
        )
    return stats


def _select_stratified(samples: Sequence, per_lang: Optional[int], limit: Optional[int]) -> List:
    """Select up to per_lang samples per language deterministically across all 13 languages."""
    out = list(samples)
    if per_lang is not None and per_lang > 0:
        counts: Dict[str, int] = {}
        picked = []
        for s in out:
            c = counts.get(s.lang, 0)
            if c < per_lang:
                picked.append(s)
                counts[s.lang] = c + 1
        out = picked
    if limit is not None:
        out = out[:limit]
    return out


async def run_model(
    spec: ModelSpec,
    *,
    data_dir: Path = DEFAULT_OUT_DIR,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    tasks: Sequence[str] = ("morph", "gen", "lid", "cloze", "efficiency"),
    resume: bool = True,
    concurrency: int = 4,
    per_lang: Optional[int] = None,
    limit_per_task: Optional[int] = None,
    adapter_override: Optional[ModelAdapter] = None,
) -> Path:
    """Execute selected tasks for one model with crash-safe JSONL checkpointing."""
    model_run_dir = runs_dir / spec.slug
    model_run_dir.mkdir(parents=True, exist_ok=True)
    (model_run_dir / "spec.json").write_text(spec.model_dump_json(indent=2) + "\n", encoding="utf-8")

    adapter = adapter_override or create_adapter(spec)
    eff_concurrent = 1 if (not adapter.capabilities.is_api or spec.provider in {"adaption", "nim"}) else max(1, concurrency)
    sem = asyncio.Semaphore(eff_concurrent)
    file_lock = asyncio.Lock()

    try:
        is_local = not adapter.capabilities.is_api
        is_nim = spec.provider == "nim"
        max_toks_map = {
            "morph": 384 if is_local else (1024 if is_nim else 2048),
            "gen": 96 if is_local else (256 if is_nim else 384),
            "lid": 32 if is_local else (256 if is_nim else 512),
            "cloze": 32 if is_local else (256 if is_nim else 512),
        }
        aborted_reason: Optional[str] = None

        for task in tasks:
            if aborted_reason:
                break
            if task == "efficiency":
                print(f"[{spec.slug}] running efficiency...", flush=True)
                await run_efficiency(spec, adapter, data_dir, model_run_dir, resume=resume)
                continue

            samples = _select_stratified(load_split(data_dir, task), per_lang, limit_per_task)
            cache_path = model_run_dir / f"{task}.completions.jsonl"
            existing = load_cached_records(cache_path) if resume else {}
            todo = [
                s for s in samples
                if s.id not in existing or existing[s.id].error is not None
            ]
            print(
                f"[{spec.slug}] {task}: {len(samples) - len(todo)}/{len(samples)} cached, running {len(todo)}",
                flush=True,
            )
            if not todo:
                continue

            consec_errors = 0
            circuit_tripped = False

            async def _run_one(sample) -> None:
                nonlocal consec_errors, circuit_tripped, aborted_reason
                async with sem:
                    if circuit_tripped:
                        return
                    if task == "morph":
                        msgs = build_morph_messages(sample.lang, sample.text)
                        res = await adapter.generate(msgs, max_tokens=max_toks_map["morph"], temperature=0.0, json_mode=True)
                    elif task == "gen":
                        msgs = build_gen_messages(sample.lang, sample.prompt_prefix)
                        res = await adapter.generate(msgs, max_tokens=max_toks_map["gen"], temperature=0.2, json_mode=False)
                    elif task == "lid":
                        msgs = build_lid_messages(sample.text)
                        res = await adapter.generate(msgs, max_tokens=max_toks_map["lid"], temperature=0.0, json_mode=True)
                    elif task == "cloze":
                        msgs = build_cloze_messages(sample.lang, sample.masked_text, sample.choices)
                        res = await adapter.score_cloze(msgs, sample.choices)
                    else:
                        raise ValueError(task)

                    rec = CompletionRecord(
                        sample_id=sample.id,
                        task=task,
                        lang=sample.lang,
                        completion=res.text,
                        raw_text=res.raw_text,
                        latency_s=round(res.latency_s, 4),
                        prompt_tokens=res.prompt_tokens,
                        completion_tokens=res.completion_tokens,
                        error=res.error,
                        cloze_mode=res.cloze_mode,
                        logprob_scores=res.logprob_scores,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                    async with file_lock:
                        _append_record(cache_path, rec)
                        if res.error:
                            consec_errors += 1
                            if consec_errors >= 8 and not circuit_tripped:
                                circuit_tripped = True
                                aborted_reason = f"Circuit breaker tripped on {task} after 8 consecutive errors: {res.error}"
                                print(f"[{spec.slug}] CIRCUIT BREAKER: {aborted_reason}", flush=True)
                        else:
                            consec_errors = 0

            await asyncio.gather(*[_run_one(s) for s in todo])
    finally:
        await adapter.close()

    return model_run_dir


async def run_many_models(
    specs: Sequence[ModelSpec],
    *,
    data_dir: Path = DEFAULT_OUT_DIR,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    tasks: Sequence[str] = ("morph", "gen", "lid", "cloze", "efficiency"),
    resume: bool = True,
    concurrency_per_model: int = 4,
    per_lang: Optional[int] = None,
    limit_per_task: Optional[int] = None,
) -> List[Tuple[str, Optional[str]]]:
    """Run API providers and local Sebeni concurrently with per-provider semaphores."""
    provider_sems: Dict[str, asyncio.Semaphore] = {
        "adaption": asyncio.Semaphore(1),
        "nim": asyncio.Semaphore(1),
        "vertex": asyncio.Semaphore(2),
        "local_sebeni": asyncio.Semaphore(1),
        "mock": asyncio.Semaphore(4),
    }

    async def _run_guarded(spec: ModelSpec) -> Tuple[str, Optional[str]]:
        psem = provider_sems.get(spec.provider, asyncio.Semaphore(2))
        async with psem:
            try:
                await run_model(
                    spec,
                    data_dir=data_dir,
                    runs_dir=runs_dir,
                    tasks=tasks,
                    resume=resume,
                    concurrency=1 if spec.provider in {"local_sebeni", "adaption", "nim"} else concurrency_per_model,
                    per_lang=per_lang,
                    limit_per_task=limit_per_task,
                )
                return spec.slug, None
            except Exception as exc:
                return spec.slug, f"{type(exc).__name__}: {exc}"

    return list(await asyncio.gather(*[_run_guarded(s) for s in specs]))
