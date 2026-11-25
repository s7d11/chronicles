"""Nemotron-3-Ultra-550B coordinator on NVIDIA Build & random prompt_native.csv directives."""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bench import LANG_NAMES
from bench.adapters.base import ModelAdapter
from bench.adapters.registry import get_model_spec
from bench.metrics import _extract_json_flexible
from bench.runner import create_adapter
from swarm_lab import ALL_LANGS, COORDINATOR_SLUG
from swarm_lab.schema import ClusterResult
from swarm_lab.topology import EventLogger


def build_random_native_command(
    lang: str,
    native_labels: Dict[str, str],
    *,
    seed: int = 2026,
) -> Dict[str, Any]:
    """Deterministically draw a random command archetype from prompt_native.csv fields."""
    idx = ALL_LANGS.index(lang) if lang in ALL_LANGS else 0
    rng = random.Random(seed + idx * 37)
    lbl_text = native_labels.get("Text", "Text")
    lbl_title = native_labels.get("Title", "Title")
    lbl_content = native_labels.get("Content", "Content")
    lbl_context = native_labels.get("Context", "Context")
    lbl_grammar = native_labels.get("Grammar", "Grammar")

    archetypes = [
        (
            "Civic Charter",
            f"Compose a short civic proclamation strictly in {LANG_NAMES.get(lang, lang)} ({lang}) "
            f"using the native headers '{lbl_title}:', '{lbl_content}:', '{lbl_context}:', and '{lbl_grammar}:'.",
        ),
        (
            "Linguistic Dispatch",
            f"Write an original brief dispatch strictly in {LANG_NAMES.get(lang, lang)} ({lang}) structured with "
            f"'{lbl_title}:', '{lbl_text}:', '{lbl_content}:', '{lbl_context}:', and '{lbl_grammar}:' demonstrating your reinforced stems.",
        ),
        (
            "Grammar-Anchored Commentary",
            f"Produce a concise commentary strictly in {LANG_NAMES.get(lang, lang)} ({lang}) organized under "
            f"'{lbl_title}:', '{lbl_context}:', '{lbl_content}:', and '{lbl_grammar}:', applying only verified native vocabulary.",
        ),
    ]
    archetype, command = rng.choice(archetypes)
    return {
        "archetype": archetype,
        "command": command,
        "labels": native_labels,
    }


def _heuristic_cluster_score(c: ClusterResult) -> float:
    base = (
        0.35 * c.agreement_pct
        + 0.25 * (100.0 if c.shadow.dict_compile_ok else 20.0)
        + 0.25 * c.native_gen.shadow_parse_pct
        + 0.15 * c.native_gen.header_compliance
        + min(10.0, max(-5.0, c.shadow.test_delta_pct * 2.0))
        - 0.3 * c.native_gen.leakage_pct
    )
    return round(max(0.0, min(100.0, base)), 1)


class CoordinatorAgent:
    """Orchestrates language briefings and comparative audits via Nemotron-3-Ultra-550B."""

    def __init__(self, adapter_override: Optional[ModelAdapter] = None):
        spec = get_model_spec(COORDINATOR_SLUG)
        self.spec = spec
        self._adapter = adapter_override or create_adapter(spec)

    async def brief_language(
        self,
        lang: str,
        diag_packet: Dict[str, Any],
        native_cmd: Dict[str, Any],
        event_logger: EventLogger,
    ) -> str:
        lang_name = LANG_NAMES.get(lang, lang)
        fallback = (
            f"Analyze {diag_packet.get('n_tokens', 0)} tokens in {lang_name} ({lang}; baseline DabaX parse "
            f"{diag_packet.get('baseline_parse_pct', 0.0)}%, {diag_packet.get('unparsed_count', 0)} OOV tokens). "
            f"Focus clusters on reconciling stems with baseline.dict ({diag_packet.get('dict_entries_total', 0)} entries), "
            f"validating isolated Shadow DabaX deltas, and executing directive '{native_cmd['archetype']}'."
        )
        msgs = [
            {
                "role": "system",
                "content": (
                    "You are the Master Swarm Coordinator (Nemotron-3-Ultra-550B) overseeing 9 autonomous "
                    f"multi-agent model clusters studying {lang_name} ({lang}). Write a concise 3-sentence technical brief "
                    "highlighting key morphotactic risks, unparsed tokens, and consensus goals."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Baseline parse: {diag_packet.get('baseline_parse_pct', 0.0)}% "
                    f"({diag_packet.get('unparsed_count', 0)} unparsed tokens, "
                    f"{diag_packet.get('dict_entries_total', 0)} dict entries).\n"
                    f"Sample target words: {', '.join(t['surface'] for t in diag_packet.get('target_words', [])[:10])}\n"
                    f"Native task: {native_cmd['command']}"
                ),
            },
        ]
        res = await self._adapter.generate(msgs, max_tokens=1024, temperature=0.2)
        brief = res.text.strip() if (res.text and not res.error) else fallback
        await event_logger.emit(
            lang=lang,
            cluster="coordinator",
            kind="coordinator_brief",
            src="coordinator",
            dst="all_clusters",
            round_num=0,
            label=f"Brief · {lang}",
            summary=brief,
            meta={"baseline_parse_pct": diag_packet.get("baseline_parse_pct", 0.0), "archetype": native_cmd["archetype"]},
        )
        return brief

    async def audit_language_clusters(
        self,
        lang: str,
        clusters: List[ClusterResult],
        event_logger: EventLogger,
    ) -> Tuple[List[ClusterResult], str]:
        if not clusters:
            return clusters, "No clusters evaluated."

        lang_name = LANG_NAMES.get(lang, lang)
        compact = [
            {
                "slug": c.cluster_slug,
                "n_agents": c.plan.n_agents,
                "topology": c.plan.topology,
                "unanimous": c.unanimous,
                "agreement_pct": c.agreement_pct,
                "rounds": c.rounds_used,
                "dict_added": c.shadow.dict_entries_added,
                "shadow_compile_ok": c.shadow.dict_compile_ok,
                "test_delta_pct": c.shadow.test_delta_pct,
                "native_parse_pct": c.native_gen.shadow_parse_pct,
                "header_compliance": c.native_gen.header_compliance,
                "leakage_pct": c.native_gen.leakage_pct,
                "sample": c.native_gen.output_text[:220],
            }
            for c in clusters
        ]
        msgs = [
            {
                "role": "system",
                "content": (
                    "You are Nemotron-3-Ultra-550B auditing all model clusters on a single Malian language text. "
                    "Score each cluster 0-100 based on consensus quality, Shadow DabaX compile/delta, and native output purity, "
                    "and summarize which topology won. Respond ONLY with JSON:\n"
                    '{"verdict": "2-sentence overview", "scores": {"slug": {"score": 82.0, "note": "1-sentence critique"}}}'
                ),
            },
            {
                "role": "user",
                "content": f"Language: {lang_name} ({lang})\nClusters:\n{json.dumps(compact, ensure_ascii=False)[:3800]}",
            },
        ]
        res = await self._adapter.generate(msgs, max_tokens=2048, temperature=0.1, json_mode=True)
        parsed = _extract_json_flexible(res.text) if not res.error else {}
        scores_map = parsed.get("scores") if isinstance(parsed.get("scores"), dict) else {}

        for c in clusters:
            h_score = _heuristic_cluster_score(c)
            entry = scores_map.get(c.cluster_slug)
            if isinstance(entry, dict) and entry.get("score") is not None:
                try:
                    llm_score = max(0.0, min(100.0, float(entry["score"])))
                    c.coordinator_score = round(0.5 * llm_score + 0.5 * h_score, 1)
                except (TypeError, ValueError):
                    c.coordinator_score = h_score
                c.coordinator_note = str(entry.get("note") or "").strip()[:220]
            else:
                c.coordinator_score = h_score
                c.coordinator_note = (
                    f"{c.plan.n_agents}x {c.plan.topology} ({c.agreement_pct:.0f}% agree, "
                    f"shadow {c.shadow.test_delta_pct:+.1f}%, native {c.native_gen.shadow_parse_pct:.1f}% parsed)."
                )
            await event_logger.emit(
                lang=lang,
                cluster=c.cluster_slug,
                kind="coordinator_audit",
                src="coordinator",
                dst=f"{c.cluster_slug}:a1",
                round_num=c.rounds_used + 2,
                label=f"Score {c.coordinator_score:.1f}",
                summary=c.coordinator_note,
                meta={"score": c.coordinator_score},
            )

        best = max(clusters, key=lambda x: x.coordinator_score)
        verdict = str(parsed.get("verdict") or "").strip()
        if not verdict:
            verdict = (
                f"{best.model_name} ({best.plan.n_agents}-agent {best.plan.topology}) led {lang_name} ({lang}) "
                f"with {best.coordinator_score:.1f} Coordinator score, {best.agreement_pct:.0f}% consensus, "
                f"and {best.native_gen.shadow_parse_pct:.1f}% DabaX native-command parse coverage."
            )
        await event_logger.emit(
            lang=lang,
            cluster="coordinator",
            kind="run_complete",
            src="coordinator",
            dst="all_clusters",
            round_num=99,
            label=f"Done · {lang}",
            summary=verdict,
        )
        return clusters, verdict

    async def close(self) -> None:
        await self._adapter.close()
