"""Autonomous 2-5 agent cluster topologies, consensus loop, and SSE event emitter."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bench import LANG_NAMES
from bench.adapters.base import ModelAdapter
from bench.metrics import _extract_json_flexible
from swarm_lab import MAX_CLUSTER_AGENTS, MIN_CLUSTER_AGENTS, TOPOLOGIES
from swarm_lab.resources import (
    evaluate_native_generation,
    lookup_dict_candidates,
    run_shadow_dabax_test,
)
from swarm_lab.schema import (
    AgentSpec,
    ClusterPlan,
    ClusterResult,
    DictDelta,
    ExchangeSnippet,
    SwarmEvent,
    TopologyType,
    WordAssessment,
)


class EventLogger:
    """Append-only JSONL event stream writer with monotonic sequence numbers."""

    def __init__(self, jsonl_path: Path, *, reset: bool = False):
        self.path = jsonl_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._seq = 0
        if reset and self.path.is_file():
            self.path.unlink()
        elif self.path.is_file():
            lines = [ln for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if lines:
                try:
                    self._seq = int(json.loads(lines[-1]).get("seq", len(lines)))
                except Exception:
                    self._seq = len(lines)

    async def emit(
        self,
        *,
        lang: str,
        cluster: str,
        kind: str,
        src: str,
        dst: str,
        round_num: int = 0,
        label: str = "",
        summary: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> SwarmEvent:
        async with self._lock:
            self._seq += 1
            ev = SwarmEvent(
                seq=self._seq,
                ts=round(time.time(), 3),
                lang=lang,
                cluster=cluster,
                kind=kind,  # type: ignore[arg-type]
                src=src,
                dst=dst,
                round_num=round_num,
                label=label,
                summary=summary[:320],
                meta=meta or {},
            )
            with self.path.open("a", encoding="utf-8") as f:
                f.write(ev.model_dump_json() + "\n")
                f.flush()
            return ev


def canonical_edges(agents: Sequence[AgentSpec], topology: TopologyType) -> List[Tuple[str, str]]:
    """Construct clean directed communication edges for the selected topology."""
    ids = [a.id for a in agents]
    n = len(ids)
    if n < 2:
        return []
    edges: List[Tuple[str, str]] = []
    if topology == "star":
        hub = ids[0]
        for w in ids[1:]:
            edges.append((hub, w))
            edges.append((w, hub))
    elif topology == "ring":
        for i in range(n):
            edges.append((ids[i], ids[(i + 1) % n]))
    elif topology == "pipeline":
        for i in range(n - 1):
            edges.append((ids[i], ids[i + 1]))
    elif topology == "hierarchical":
        lead = ids[0]
        if n >= 4:
            sub = ids[1]
            edges.append((lead, sub))
            edges.append((sub, lead))
            for w in ids[2:]:
                edges.append((sub, w))
                edges.append((w, sub))
        else:
            for w in ids[1:]:
                edges.append((lead, w))
                edges.append((w, lead))
    else:  # mesh
        for i in range(n):
            for j in range(n):
                if i != j:
                    edges.append((ids[i], ids[j]))
    return edges


def normalize_cluster_plan(
    raw: Dict[str, Any],
    *,
    cluster_slug: str,
    model_name: str,
) -> ClusterPlan:
    """Validate and normalize LLM cluster plan, strictly clamping n_agents to [2, 5]."""
    raw_agents = raw.get("agents") if isinstance(raw.get("agents"), list) else []
    n_req = raw.get("n_agents") or len(raw_agents) or 3
    try:
        n_int = int(n_req)
    except (TypeError, ValueError):
        n_int = 3
    n_clamped = max(MIN_CLUSTER_AGENTS, min(MAX_CLUSTER_AGENTS, n_int))

    topo_raw = str(raw.get("topology") or "star").strip().lower()
    topo: TopologyType = topo_raw if topo_raw in TOPOLOGIES else "star"  # type: ignore[assignment]

    default_roles = [
        ("Morphologist", "Analyze affixation, clitics, and stems against baseline.gram."),
        ("Lexicographer", "Cross-check stems, tone/orthography, and glosses in baseline.dict."),
        ("Arbiter", "Challenge uncertain splits and verify agreement across all peers."),
        ("Syntactician", "Align word boundaries, POS categories, and clause context."),
        ("Generator", "Draft and proofread native-language output for prompt_native.csv."),
    ]
    agents: List[AgentSpec] = []
    for i in range(n_clamped):
        item = raw_agents[i] if i < len(raw_agents) and isinstance(raw_agents[i], dict) else {}
        d_role, d_mandate = default_roles[i]
        aid = f"a{i + 1}"
        agents.append(
            AgentSpec(
                id=aid,
                role=str(item.get("role") or d_role).strip()[:40],
                mandate=str(item.get("mandate") or d_mandate).strip()[:160],
            )
        )

    valid_ids = {a.id for a in agents}
    edges: List[Tuple[str, str]] = []
    if isinstance(raw.get("edges"), list):
        for pair in raw["edges"]:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                u, v = str(pair[0]), str(pair[1])
                if u in valid_ids and v in valid_ids and u != v and (u, v) not in edges:
                    edges.append((u, v))
    if not edges:
        edges = canonical_edges(agents, topo)

    rationale = str(raw.get("rationale") or f"Self-selected {n_clamped}-agent {topo} cluster.").strip()[:280]
    return ClusterPlan(
        cluster_slug=cluster_slug,
        model_name=model_name,
        n_agents=n_clamped,
        topology=topo,
        agents=agents,
        edges=edges,
        rationale=rationale,
    )


def _format_target_dossier(targets: Sequence[Dict[str, Any]]) -> str:
    lines = []
    for i, t in enumerate(targets, 1):
        hits = ", ".join(
            f"{h['lx']} ({h['ps']}: {h['ge'] or h['gf']})"
            for h in (t.get("dict_hits") or [])
        ) or "NO_DICT_HIT"
        lines.append(
            f"{i}. '{t['surface']}' [stage={t['stage']}, base_morph={t.get('baseline_morphemes', [])}, dict={hits}]"
        )
    return "\n".join(lines)


def _parse_assessments(raw: Dict[str, Any], target_words: Sequence[Dict[str, Any]]) -> List[WordAssessment]:
    items = raw.get("words") if isinstance(raw.get("words"), list) else []
    by_surface: Dict[str, WordAssessment] = {}
    for w in items:
        if not isinstance(w, dict):
            continue
        surf = str(w.get("surface") or "").strip()
        if not surf:
            continue
        morphs = [str(m).strip() for m in (w.get("morphemes") or []) if str(m).strip()]
        if not morphs:
            morphs = [surf]
        by_surface[surf] = WordAssessment(
            surface=surf,
            morphemes=morphs,
            pos=str(w.get("pos") or "").strip()[:24],
            gloss=str(w.get("gloss") or "").strip()[:120],
            notes=str(w.get("notes") or "").strip()[:120],
        )
    if not by_surface:
        for t in target_words[:6]:
            surf = t["surface"]
            by_surface[surf] = WordAssessment(
                surface=surf,
                morphemes=t.get("baseline_morphemes") or [surf],
                pos=",".join(t.get("baseline_ps") or [])[:24] or "n",
                gloss=str(t.get("baseline_gloss") or "")[:120],
            )
    return list(by_surface.values())


def _parse_dict_deltas(raw: Dict[str, Any]) -> List[DictDelta]:
    out: List[DictDelta] = []
    items = raw.get("dict_deltas") if isinstance(raw.get("dict_deltas"), list) else []
    for d in items[:15]:
        if not isinstance(d, dict):
            continue
        lx = str(d.get("lx") or "").strip()
        ge = str(d.get("ge") or "").strip()
        if lx and ge:
            out.append(
                DictDelta(
                    lx=lx,
                    ps=str(d.get("ps") or "n").strip()[:16] or "n",
                    ge=ge,
                    gf=str(d.get("gf") or "").strip(),
                )
            )
    return out


async def run_cluster_session(
    *,
    cluster_slug: str,
    model_name: str,
    adapter: ModelAdapter,
    lang: str,
    test_text: str,
    diag_packet: Dict[str, Any],
    coordinator_brief: str,
    native_cmd: Dict[str, Any],
    event_logger: EventLogger,
    cluster_run_dir: Path,
) -> ClusterResult:
    """Execute one model's self-organized 2-5 agent session on one language text."""
    lang_name = LANG_NAMES.get(lang, lang)
    dossier = _format_target_dossier(diag_packet.get("target_words", []))

    # Phase 1: Autonomous Cluster Spawn & Topology Choice (2-5 agents)
    spawn_msgs = [
        {
            "role": "system",
            "content": (
                f"You are organizing a self-managed research cluster for {lang_name} ({lang}). "
                "Choose 2 to 5 specialized agent instances (2 <= n_agents <= 5) and one topology "
                "from [star, ring, mesh, hierarchical, pipeline]. "
                "Respond ONLY with JSON:\n"
                '{"n_agents": 3, "topology": "star|ring|mesh|hierarchical|pipeline", '
                '"agents": [{"role": "...", "mandate": "..."}], "rationale": "..."}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Coordinator Brief:\n{coordinator_brief}\n\n"
                f"Target text tokens: {diag_packet.get('n_tokens', 0)}, "
                f"baseline DabaX parse: {diag_packet.get('baseline_parse_pct', 0.0)}%.\n"
                f"Target words needing reinforcement:\n{dossier}"
            ),
        },
    ]
    spawn_res = await adapter.generate(spawn_msgs, max_tokens=2048, temperature=0.2, json_mode=True)
    plan = normalize_cluster_plan(
        _extract_json_flexible(spawn_res.text),
        cluster_slug=cluster_slug,
        model_name=model_name,
    )
    await event_logger.emit(
        lang=lang,
        cluster=cluster_slug,
        kind="cluster_spawn",
        src="coordinator",
        dst=f"{cluster_slug}:{plan.agents[0].id}",
        round_num=0,
        label=f"{plan.n_agents}x {plan.topology}",
        summary=f"Spawned {plan.n_agents} agents ({plan.topology}): " + ", ".join(a.role for a in plan.agents),
        meta={"n_agents": plan.n_agents, "topology": plan.topology, "agents": [a.model_dump() for a in plan.agents], "edges": plan.edges},
    )

    # Emit grounded baseline lookup links for the top unparsed tokens
    sample_hits = []
    for tw in (diag_packet.get("target_words") or [])[:5]:
        hits = lookup_dict_candidates(lang, tw["surface"], limit=1)
        if hits:
            sample_hits.append(f"{tw['surface']}->{hits[0]['lx']}({hits[0]['ps']})")
    await event_logger.emit(
        lang=lang,
        cluster=cluster_slug,
        kind="resource_lookup",
        src=f"{cluster_slug}:{plan.agents[0].id}",
        dst="res:baseline",
        round_num=1,
        label=f"{diag_packet.get('dict_entries_total', 0)} dict",
        summary="Consulted baseline.gram & baseline.dict: " + (", ".join(sample_hits) or "excerpt loaded"),
    )

    # Phase 2: Multi-Round Word Assessment & Consensus Loop
    incoming: Dict[str, List[str]] = {a.id: [] for a in plan.agents}
    outgoing: Dict[str, List[str]] = {a.id: [] for a in plan.agents}
    for u, v in plan.edges:
        outgoing.setdefault(u, []).append(v)
        incoming.setdefault(v, []).append(u)

    agent_states: Dict[str, Dict[str, Any]] = {}
    exchanges: List[ExchangeSnippet] = []
    n_exchanges = 0
    unanimous = False
    rounds_used = 1
    agreement_pct = 0.0
    dissent_notes: List[str] = []

    def _next_hop(aid: str) -> str:
        outs = outgoing.get(aid) or []
        if outs:
            return outs[0]
        return plan.agents[0].id if aid != plan.agents[0].id else plan.agents[-1].id

    # Round 1: Initial assessments along topology order
    for idx, ag in enumerate(plan.agents):
        peer_ctx = ""
        if plan.topology in ("ring", "pipeline") and idx > 0:
            prev_id = plan.agents[idx - 1].id
            if prev_id in agent_states:
                peer_ctx = f"\nUpstream peer ({prev_id}) notes: {json.dumps(agent_states[prev_id].get('raw', {}), ensure_ascii=False)[:1100]}"
        r1_msgs = [
            {
                "role": "system",
                "content": (
                    f"You are agent {ag.id} ({ag.role}) inside a {plan.n_agents}-agent {plan.topology} cluster analyzing {lang_name} ({lang}). "
                    f"Mandate: {ag.mandate}\n"
                    "Assess the target words using baseline.gram & baseline.dict evidence. "
                    "Respond ONLY with JSON:\n"
                    '{"words": [{"surface": "...", "morphemes": ["..."], "pos": "n|v|...", "gloss": "...", "notes": "..."}], '
                    '"dict_deltas": [{"lx": "...", "ps": "...", "ge": "...", "gf": "..."}], '
                    '"gram_rules": [], "summary": "1-sentence takeaway"}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Text ({lang}):\n{test_text[:1000]}\n\n"
                    f"baseline.gram excerpt:\n{diag_packet.get('gram_excerpt', '')[:1400]}\n\n"
                    f"Target Words:\n{dossier}{peer_ctx}"
                ),
            },
        ]
        res = await adapter.generate(r1_msgs, max_tokens=2400, temperature=0.1, json_mode=True)
        parsed = _extract_json_flexible(res.text)
        summary = str(parsed.get("summary") or f"{ag.role} proposed {len(parsed.get('words') or [])} word analyses.").strip()[:200]
        agent_states[ag.id] = {
            "raw": parsed,
            "words": _parse_assessments(parsed, diag_packet.get("target_words", [])),
            "deltas": _parse_dict_deltas(parsed),
            "rules": [str(r) for r in (parsed.get("gram_rules") or []) if isinstance(r, str)],
            "agree": True,
            "summary": summary,
        }
        dst_id = _next_hop(ag.id)
        n_exchanges += 1
        exchanges.append(ExchangeSnippet(round_num=1, src=ag.id, dst=dst_id, summary=summary))
        await event_logger.emit(
            lang=lang,
            cluster=cluster_slug,
            kind="agent_exchange",
            src=f"{cluster_slug}:{ag.id}",
            dst=f"{cluster_slug}:{dst_id}",
            round_num=1,
            label=f"R1 · {ag.role}",
            summary=summary,
        )

    # Rounds 2..3: Deliberation & Unanimity Verification
    for rnd in (2, 3):
        rounds_used = rnd
        votes: List[bool] = []
        dissent_notes = []
        for ag in plan.agents:
            peers = [p for p in incoming.get(ag.id, []) if p in agent_states] or [
                a.id for a in plan.agents if a.id != ag.id
            ]
            peer_bundle = {
                pid: {
                    "summary": agent_states[pid]["summary"],
                    "words": [w.model_dump() for w in agent_states[pid]["words"][:10]],
                    "dict_deltas": [d.model_dump() for d in agent_states[pid]["deltas"][:6]],
                }
                for pid in peers[:3]
            }
            delib_msgs = [
                {
                    "role": "system",
                    "content": (
                        f"You are {ag.id} ({ag.role}) in deliberation Round {rnd} for {lang_name} ({lang}). "
                        "Inspect peer analyses, resolve disagreements on morphemes/glosses/POS, and vote whether "
                        "the cluster has reached unified agreement on the word assessments. "
                        "Respond ONLY with JSON:\n"
                        '{"agree_with_cluster": true, "dissent": "", "words": [{"surface": "...", "morphemes": ["..."], "pos": "...", "gloss": "..."}], '
                        '"dict_deltas": [{"lx": "...", "ps": "...", "ge": "...", "gf": "..."}], "gram_rules": [], "summary": "..."}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Your prior summary: {agent_states[ag.id]['summary']}\n"
                        f"Peer proposals along topology edges:\n{json.dumps(peer_bundle, ensure_ascii=False)[:2600]}"
                    ),
                },
            ]
            res = await adapter.generate(delib_msgs, max_tokens=2400, temperature=0.0, json_mode=True)
            parsed = _extract_json_flexible(res.text)
            agree = bool(parsed.get("agree_with_cluster", False))
            dissent_txt = str(parsed.get("dissent") or "").strip()
            summary = str(parsed.get("summary") or ("Concurred with peer analyses." if agree else dissent_txt or "Raised revision")).strip()[:200]

            if parsed.get("words"):
                agent_states[ag.id]["words"] = _parse_assessments(parsed, diag_packet.get("target_words", []))
            if parsed.get("dict_deltas"):
                agent_states[ag.id]["deltas"] = _parse_dict_deltas(parsed)
            if parsed.get("gram_rules"):
                agent_states[ag.id]["rules"] = [str(r) for r in parsed["gram_rules"] if isinstance(r, str)]
            agent_states[ag.id]["agree"] = agree
            agent_states[ag.id]["summary"] = summary
            votes.append(agree)
            if not agree and dissent_txt:
                dissent_notes.append(f"{ag.id}: {dissent_txt[:140]}")

            dst_id = _next_hop(ag.id)
            n_exchanges += 1
            exchanges.append(ExchangeSnippet(round_num=rnd, src=ag.id, dst=dst_id, summary=summary, agree=agree))
            await event_logger.emit(
                lang=lang,
                cluster=cluster_slug,
                kind="consensus_vote",
                src=f"{cluster_slug}:{ag.id}",
                dst=f"{cluster_slug}:{dst_id}",
                round_num=rnd,
                label="AGREE" if agree else "REVISE",
                summary=summary,
                meta={"agree": agree},
            )

        agree_count = sum(1 for v in votes if v)
        agreement_pct = round(100.0 * agree_count / max(1, len(votes)), 1)
        if agree_count == len(votes):
            unanimous = True
            await event_logger.emit(
                lang=lang,
                cluster=cluster_slug,
                kind="consensus_reached",
                src=f"{cluster_slug}:{plan.agents[0].id}",
                dst=f"{cluster_slug}:all",
                round_num=rnd,
                label="UNANIMOUS",
                summary=f"All {plan.n_agents} agents reached 100% consensus in Round {rnd}.",
            )
            break

    # Aggregate Reinforced Linguistic Notebook (prefer arbiter/lead then merge unique deltas)
    lead_state = agent_states[plan.agents[-1].id]
    consensus_words: List[WordAssessment] = lead_state["words"]
    merged_deltas: List[DictDelta] = []
    seen_lx = set()
    for ag in plan.agents:
        for d in agent_states[ag.id]["deltas"]:
            k = d.lx.strip().lower()
            if k and k not in seen_lx:
                seen_lx.add(k)
                merged_deltas.append(d)
    merged_rules: List[str] = []
    for ag in plan.agents:
        for r in agent_states[ag.id]["rules"]:
            if r not in merged_rules:
                merged_rules.append(r)

    # Phase 3: Isolated Shadow DabaX Verification
    shadow_dir = cluster_run_dir / "shadow"
    shadow_res = await asyncio.to_thread(
        run_shadow_dabax_test,
        lang,
        test_text,
        merged_deltas,
        merged_rules,
        shadow_dir,
    )
    await event_logger.emit(
        lang=lang,
        cluster=cluster_slug,
        kind="shadow_patch",
        src=f"{cluster_slug}:{plan.agents[0].id}",
        dst="res:shadow_dabax",
        round_num=rounds_used,
        label=f"{shadow_res.test_delta_pct:+.1f}% test",
        summary=(
            f"Shadow DabaX (+{shadow_res.dict_entries_added} dict, {shadow_res.gram_rules_proposed} gram): "
            f"compile={'OK' if shadow_res.dict_compile_ok else 'ERR'}, "
            f"test {shadow_res.baseline_test_parse_pct}%->{shadow_res.shadow_test_parse_pct}% "
            f"(dev-mini {shadow_res.devmini_delta_pct:+.2f}%)"
        ),
        meta={"compile_ok": shadow_res.dict_compile_ok, "delta": shadow_res.test_delta_pct},
    )

    # Phase 4: Native-Command Generation from prompt_native.csv
    await event_logger.emit(
        lang=lang,
        cluster=cluster_slug,
        kind="native_command",
        src="res:prompt_native",
        dst=f"{cluster_slug}:{plan.agents[0].id}",
        round_num=rounds_used + 1,
        label=native_cmd["archetype"],
        summary=native_cmd["command"],
    )

    notebook_summary = json.dumps(
        {
            "confirmed_words": [w.model_dump() for w in consensus_words[:12]],
            "proposed_lexicon": [d.model_dump() for d in merged_deltas[:10]],
        },
        ensure_ascii=False,
    )
    gen_msgs = [
        {
            "role": "system",
            "content": (
                f"You are {plan.agents[0].id} ({plan.agents[0].role}) writing strictly in {lang_name} ({lang}). "
                "Use your cluster's Reinforced Linguistic Notebook and the exact native section labels from prompt_native.csv. "
                "Respond ONLY in pure target language (no English/French explanations)."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Directive:\n{native_cmd['command']}\n\n"
                f"Reference excerpt ({lang}):\n{test_text[:650]}\n\n"
                f"Reinforced Notebook:\n{notebook_summary[:1800]}"
            ),
        },
    ]
    draft_res = await adapter.generate(gen_msgs, max_tokens=2048, temperature=0.35, json_mode=False)

    verifier = plan.agents[-1]
    verify_msgs = [
        {
            "role": "system",
            "content": (
                f"You are {verifier.id} ({verifier.role}) reviewing a draft in {lang_name} ({lang}). "
                "Preserve required native labels from prompt_native.csv, remove any French/English words, "
                "and output ONLY the clean final text in " + lang_name + "."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Required directive:\n{native_cmd['command']}\n\n"
                f"Draft:\n{draft_res.text[:1600]}"
            ),
        },
    ]
    final_res = await adapter.generate(verify_msgs, max_tokens=2048, temperature=0.1, json_mode=False)
    cand_final = final_res.text.strip()
    if not cand_final or "Role:" in cand_final[:80] or "Task:" in cand_final[:120]:
        cand_final = draft_res.text.strip()
    final_text = cand_final or draft_res.text.strip()

    native_eval = await asyncio.to_thread(
        evaluate_native_generation,
        lang,
        native_cmd["command"],
        native_cmd["labels"],
        final_text,
        shadow_dir,
        bool(final_res.text.strip()),
    )
    n_exchanges += 1
    exchanges.append(
        ExchangeSnippet(
            round_num=rounds_used + 1,
            src=plan.agents[0].id,
            dst=verifier.id,
            summary=f"Peer-verified native draft ({native_eval.shadow_parse_pct}% parsed, {native_eval.header_compliance}% headers).",
            agree=True,
        )
    )
    await event_logger.emit(
        lang=lang,
        cluster=cluster_slug,
        kind="native_generation",
        src=f"{cluster_slug}:{plan.agents[0].id}",
        dst=f"{cluster_slug}:{verifier.id}",
        round_num=rounds_used + 1,
        label=f"{native_eval.shadow_parse_pct:.1f}% DabaX",
        summary=f"Native {native_cmd['archetype']}: parse={native_eval.shadow_parse_pct}%, leakage={native_eval.leakage_pct}%, headers={native_eval.header_compliance}%",
        meta={"parse_pct": native_eval.shadow_parse_pct, "headers": native_eval.header_compliance},
    )

    return ClusterResult(
        lang=lang,
        cluster_slug=cluster_slug,
        model_name=model_name,
        plan=plan,
        rounds_used=rounds_used,
        unanimous=unanimous,
        agreement_pct=agreement_pct,
        n_exchanges=n_exchanges,
        consensus_words=consensus_words,
        dict_deltas=merged_deltas,
        gram_rules=merged_rules,
        dissent_notes=dissent_notes,
        exchanges=exchanges,
        shadow=shadow_res,
        native_gen=native_eval,
    )
