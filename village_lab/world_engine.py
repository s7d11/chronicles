"""Stateful Edition No. 4 Living Lexicon Village Engine (Mande + 9 Adaption + Gemini 4 Argon)."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import httpx

from swarm_lab import ALL_LANGS, normalize_lang_code
from swarm_lab.resources import load_baseline_resources
from village_lab import (
    BASE_VILLAGERS,
    COUNCIL_SEATS,
    CRITIC_SPEC,
    LANGUAGE_NATIVE_NAMES,
    ORCHESTRATOR_SPEC,
    SEMANTIC_DOMAINS,
    SPECIALIST_TEMPLATES,
    TARGET_DICT_ENTRIES,
    VILLAGE_ZONES,
)
from village_lab.orthography import (
    check_standardization,
    classify_lemma_vs_rule,
    parse_working_dict_index,
)
from village_lab.providers import (
    call_adaption,
    call_gemini_argon,
    call_nim_mande,
    extract_json_object,
)
from village_lab.schema import (
    CouncilDecision,
    CouncilSeatVote,
    CriticVerdict,
    DabaXCheck,
    DialogueUtterance,
    LanguageVillageState,
    VillageAgent,
    VillageEvent,
    VillageExchange,
    serialize_mdf_entry,
)
from village_lab.shadow_verify import _append_va_to_dict_file, _sanitize_gram_rule, verify_shadow_candidate


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def compute_zone_path(src_zone: str, dst_zone: str) -> List[Dict[str, float]]:
    """Compute shortest footpath waypoint coordinates between two village zones."""
    if src_zone not in VILLAGE_ZONES or dst_zone not in VILLAGE_ZONES:
        return [{"x": 0.0, "z": 0.0}]
    if src_zone == dst_zone:
        zinfo = VILLAGE_ZONES[dst_zone]
        return [{"x": float(zinfo["x"]), "z": float(zinfo["z"])}]

    q = deque([[src_zone]])
    seen = {src_zone}
    route = [src_zone, dst_zone]
    while q:
        cur = q.popleft()
        last = cur[-1]
        if last == dst_zone:
            route = cur
            break
        for nxt in VILLAGE_ZONES[last].get("neighbors", []):  # type: ignore[union-attr]
            if nxt not in seen:
                seen.add(nxt)
                q.append(cur + [nxt])

    return [
        {"x": float(VILLAGE_ZONES[z]["x"]), "z": float(VILLAGE_ZONES[z]["z"])}
        for z in route
    ]


def _default_agents() -> List[VillageAgent]:
    agents: List[VillageAgent] = []
    specs = [ORCHESTRATOR_SPEC] + list(BASE_VILLAGERS) + list(COUNCIL_SEATS) + [CRITIC_SPEC]
    for sp in specs:
        z = VILLAGE_ZONES[sp.home_zone]
        agents.append(
            VillageAgent(
                agent_id=sp.agent_id,
                name=sp.name,
                role=sp.role,  # type: ignore[arg-type]
                title=sp.title,
                model_id=sp.model_id,
                provider=sp.provider,
                home_zone=sp.home_zone,
                zone=sp.home_zone,
                x=float(z["x"]),
                z=float(z["z"]),
                domain=sp.domain,
                seat=sp.seat,
                color=sp.color,
            )
        )
    return agents


ARGON_BATCH_MIN = 10
ARGON_BATCH_MAX = 25


class LanguageVillageEngine:
    """Runs turns for one MALI13 language toward 50,000 standardized MDF entries."""

    def __init__(
        self,
        lang: str,
        *,
        runs_dir: Path,
        event_callback: Optional[Callable[[VillageEvent], None]] = None,
        mock_mode: bool = False,
        argon_batch_min: int = ARGON_BATCH_MIN,
        argon_batch_max: int = ARGON_BATCH_MAX,
    ):
        self.lang = normalize_lang_code(lang)
        self.runs_dir = Path(runs_dir)
        self.lang_dir = self.runs_dir / self.lang
        self.lang_dir.mkdir(parents=True, exist_ok=True)
        self.event_callback = event_callback
        self.mock_mode = mock_mode
        self.argon_batch_min = max(1, int(argon_batch_min))
        self.argon_batch_max = max(self.argon_batch_min, int(argon_batch_max))
        self._pending_argon: List[Dict[str, Any]] = []
        self.argon_batch_calls: int = 0

        self.dict_path = self.lang_dir / "augmented.dict"
        self.gram_path = self.lang_dir / "augmented.gram"
        self.state_path = self.lang_dir / "state.json"
        self.events_path = self.lang_dir / "events.jsonl"
        self.ledger_path = self.lang_dir / "council_ledger.jsonl"
        self.exchanges_path = self.lang_dir / "exchanges.jsonl"

        self.base_res = load_baseline_resources(self.lang)
        self._init_or_restore()

    def _init_or_restore(self) -> None:
        if not self.dict_path.is_file():
            shutil.copyfile(self.base_res["dict_path"], self.dict_path)
        if not self.gram_path.is_file():
            shutil.copyfile(self.base_res["gram_path"], self.gram_path)

        self.working_index, total_lx, total_va = parse_working_dict_index(self.dict_path)
        base_lx = int(self.base_res.get("dict_entries") or total_lx)
        base_gram_lines = len(Path(self.base_res["gram_path"]).read_text(encoding="utf-8", errors="replace").splitlines())

        if self.state_path.is_file():
            self.state = LanguageVillageState.model_validate_json(self.state_path.read_text(encoding="utf-8"))
            self.state.baseline_lx_count = base_lx
            self.state.added_lx_count = max(self.state.added_lx_count, max(0, total_lx - base_lx))
        else:
            self.state = LanguageVillageState(
                lang=self.lang,
                lang_name=LANGUAGE_NATIVE_NAMES.get(self.lang, self.lang),
                target_entries=TARGET_DICT_ENTRIES,
                baseline_lx_count=base_lx,
                added_lx_count=0,
                added_va_count=0,
                baseline_gram_lines=base_gram_lines,
                added_gram_rules_count=0,
                updated_at=_utc_now_iso(),
                active_agents=_default_agents(),
                domain_counts={d: 0 for d in SEMANTIC_DOMAINS},
            )
            self._save_state()

        self._next_seq = self._count_existing_events() + 1
        if self._next_seq == 1:
            self._emit(
                turn_id=0,
                kind="village_init",
                agent_id=ORCHESTRATOR_SPEC.agent_id,
                zone="baobab_plaza",
                label=f"{self.state.lang_name} Village Awakened",
                summary=(
                    f"Baseline loaded: {base_lx:,} MDF \\lx entries & {base_gram_lines} .gram lines. "
                    f"Cast: Mande (nemotron-3-ultra-550b-a55b), 9 Adaption Labs agents, Gemini 4 Argon Critic."
                ),
                meta={
                    "baseline_lx_count": base_lx,
                    "target_entries": TARGET_DICT_ENTRIES,
                    "n_agents": len(self.state.active_agents),
                },
            )

    def _count_existing_events(self) -> int:
        if not self.events_path.is_file():
            return 0
        return sum(1 for ln in self.events_path.read_text(encoding="utf-8").splitlines() if ln.strip())

    def _save_state(self) -> None:
        self.state.updated_at = _utc_now_iso()
        self.state.recent_exchanges = self.state.recent_exchanges[-18:]
        self.state.recent_decisions = self.state.recent_decisions[-30:]
        self.state_path.write_text(self.state.model_dump_json(indent=2), encoding="utf-8")

    def _emit(
        self,
        *,
        turn_id: int,
        kind: str,
        agent_id: str = "",
        target_agent_id: str = "",
        zone: str = "",
        target_zone: str = "",
        path: Optional[List[Dict[str, float]]] = None,
        label: str = "",
        summary: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> VillageEvent:
        ev = VillageEvent(
            seq=self._next_seq,
            ts=_utc_now_iso(),
            lang=self.lang,
            turn_id=turn_id,
            kind=kind,  # type: ignore[arg-type]
            agent_id=agent_id,
            target_agent_id=target_agent_id,
            zone=zone,
            target_zone=target_zone,
            path=path or [],
            label=label,
            summary=summary,
            meta=meta or {},
        )
        self._next_seq += 1
        with self.events_path.open("a", encoding="utf-8") as fh:
            fh.write(ev.model_dump_json() + "\n")
        if self.event_callback is not None:
            try:
                self.event_callback(ev)
            except Exception:
                pass
        return ev

    def _get_agent(self, agent_id: str) -> Optional[VillageAgent]:
        for ag in self.state.active_agents:
            if ag.agent_id == agent_id:
                return ag
        return None

    def _move_agent(self, agent: VillageAgent, dst_zone: str, turn_id: int, reason: str) -> None:
        if dst_zone not in VILLAGE_ZONES:
            return
        src_zone = agent.zone
        path = compute_zone_path(src_zone, dst_zone)
        agent.zone = dst_zone
        agent.x = path[-1]["x"]
        agent.z = path[-1]["z"]
        self._emit(
            turn_id=turn_id,
            kind="agent_move",
            agent_id=agent.agent_id,
            zone=src_zone,
            target_zone=dst_zone,
            path=path,
            label=f"{agent.name} -> {VILLAGE_ZONES[dst_zone]['label']}",
            summary=reason,
            meta={"role": agent.role, "model_id": agent.model_id},
        )

    def _maybe_spawn_specialist(self, requested_slug: str, reason: str, turn_id: int) -> Optional[VillageAgent]:
        existing_ids = {a.agent_id for a in self.state.active_agents}
        chosen = None
        for tpl in SPECIALIST_TEMPLATES:
            if requested_slug and requested_slug.lower() in tpl.agent_id.lower() and tpl.agent_id not in existing_ids:
                chosen = tpl
                break
        if chosen is None:
            for tpl in SPECIALIST_TEMPLATES:
                if tpl.agent_id not in existing_ids:
                    chosen = tpl
                    break
        if chosen is None:
            return None

        zinfo = VILLAGE_ZONES[chosen.home_zone]
        entry_path = compute_zone_path("millet_fields", chosen.home_zone)
        ag = VillageAgent(
            agent_id=chosen.agent_id,
            name=chosen.name,
            role="specialist",
            title=chosen.title,
            model_id=chosen.model_id,
            provider=chosen.provider,
            home_zone=chosen.home_zone,
            zone=chosen.home_zone,
            x=float(zinfo["x"]),
            z=float(zinfo["z"]),
            domain=chosen.domain,
            color=chosen.color,
            spawned_dynamically=True,
            spawn_reason=reason,
        )
        self.state.active_agents.append(ag)
        self._emit(
            turn_id=turn_id,
            kind="agent_spawn",
            agent_id=ag.agent_id,
            zone="millet_fields",
            target_zone=ag.home_zone,
            path=entry_path,
            label=f"Mande summoned specialist: {ag.name} ({ag.title})",
            summary=f"{reason} | Model: {ag.model_id}",
            meta=ag.model_dump(),
        )
        return ag

    async def _mande_plan_turn(self, client: httpx.AsyncClient, turn_id: int) -> Dict[str, Any]:
        domain_idx = (turn_id - 1) % len(SEMANTIC_DOMAINS)
        fallback_domain = SEMANTIC_DOMAINS[domain_idx]
        villager_pool = [a for a in self.state.active_agents if a.role in {"villager", "specialist"}]
        fallback_pair = [villager_pool[(turn_id - 1) % len(villager_pool)].agent_id, villager_pool[turn_id % len(villager_pool)].agent_id]
        zones_play = ["market_square", "baobab_plaza", "millet_fields", "banco_compounds", "artisan_forge", "riverbank_well"]
        fallback_zone = zones_play[(turn_id - 1) % len(zones_play)]
        should_spawn_fallback = (turn_id % 2 == 1) and len([a for a in self.state.active_agents if a.role == "specialist"]) < len(SPECIALIST_TEMPLATES)

        if self.mock_mode:
            return {
                "semantic_domain": fallback_domain,
                "zone": fallback_zone,
                "topic": f"Traditional {fallback_domain} work & tools in {self.state.lang_name}",
                "participants": fallback_pair,
                "spawn_specialist": "sp-numu" if should_spawn_fallback else "",
                "spawn_reason": f"Need specialized {fallback_domain} terminology" if should_spawn_fallback else "",
                "directive": f"Gather at {fallback_zone} and discuss specialized rural terms not yet common in urban speech.",
            }

        available_sp = [t.agent_id for t in SPECIALIST_TEMPLATES if t.agent_id not in {a.agent_id for a in self.state.active_agents}]
        prompt = (
            f"You are Mande, Chief Orchestrator guiding a traditional Sudano-Sahelian village building a 50,000-entry "
            f"standardized MDF lexicon for {self.state.lang_name} ({self.lang}).\n"
            f"Current baseline: {self.state.baseline_lx_count} \\lx entries; added so far: +{self.state.added_lx_count} \\lx, "
            f"+{self.state.added_va_count} \\va, +{self.state.added_gram_rules_count} .gram rules.\n"
            f"Domain coverage counts: {json.dumps(self.state.domain_counts)}\n"
            f"Available villagers: {[(a.agent_id, a.name, a.domain) for a in villager_pool]}\n"
            f"Available zones: {zones_play}\n"
            f"Unspawned specialist IDs you may summon on necessity: {available_sp}\n\n"
            "Pick an underrepresented semantic domain, gathering zone, 2 villager agent_ids, a concrete everyday conversation "
            "topic eliciting authentic rural/traditional roots (verbs, concrete nouns, ideophones), and optionally summon 1 "
            "specialist ID if domain depth requires it.\n"
            "Respond strictly as compact JSON:\n"
            '{"semantic_domain":"...","zone":"...","topic":"...","participants":["v1-...","v2-..."],"spawn_specialist":"","spawn_reason":"","directive":"..."}'
        )
        raw, _, err = await call_nim_mande(
            client,
            [{"role": "user", "content": prompt}],
            model_id=ORCHESTRATOR_SPEC.model_id,
        )
        parsed = extract_json_object(raw) if not err else None
        if not isinstance(parsed, dict):
            return {
                "semantic_domain": fallback_domain,
                "zone": fallback_zone,
                "topic": f"{fallback_domain} fieldwork exchange",
                "participants": fallback_pair,
                "spawn_specialist": "sp-numu" if should_spawn_fallback else "",
                "spawn_reason": f"Expand specialized vocabulary ({err or 'fallback'})" if should_spawn_fallback else "",
                "directive": f"Convene in {fallback_zone} to surface unattested stems for {fallback_domain}.",
            }
        zone = str(parsed.get("zone") or fallback_zone)
        if zone not in zones_play:
            zone = fallback_zone
        valid_ids = {a.agent_id for a in villager_pool}
        parts = [str(p) for p in (parsed.get("participants") or []) if str(p) in valid_ids]
        if len(parts) < 2:
            parts = fallback_pair
        return {
            "semantic_domain": str(parsed.get("semantic_domain") or fallback_domain),
            "zone": zone,
            "topic": str(parsed.get("topic") or fallback_domain),
            "participants": parts[:3],
            "spawn_specialist": str(parsed.get("spawn_specialist") or ""),
            "spawn_reason": str(parsed.get("spawn_reason") or "Expand specialized cultural lexicon"),
            "directive": str(parsed.get("directive") or f"Discuss {fallback_domain} in natural {self.lang}."),
        }

    async def _run_villager_dialogue(
        self,
        client: httpx.AsyncClient,
        turn_id: int,
        plan: Dict[str, Any],
        participants: List[VillageAgent],
    ) -> VillageExchange:
        domain = plan["semantic_domain"]
        zone = plan["zone"]
        topic = plan["topic"]
        utterances: List[DialogueUtterance] = []

        if self.mock_mode:
            mock_samples = [
                ("nɛgɛso-dilanni", "kɛlɛkɛlɛ", "bicycle repair / seasonal insect", "Unattested trade/nature items"),
                ("kɔrɔlenba", "nyɛgɛn", "very ancient / decorated", "Contains ny->ɲ standardization test + derived -len inflection test"),
            ]
            for idx, ag in enumerate(participants):
                w1, w2, gloss, _ = mock_samples[idx % len(mock_samples)]
                utterances.append(
                    DialogueUtterance(
                        agent_id=ag.agent_id,
                        speaker_name=ag.name,
                        model_id=ag.model_id,
                        text_native=f"An bɛ {w1} ani {w2} fɔ bi {zone} kɔnɔ.",
                        gloss_en=f"We speak about {gloss} in the village today.",
                        gloss_fr=f"Nous parlons de {gloss} aujourd'hui au village.",
                        candidate_terms=[w1, w2],
                    )
                )
            return VillageExchange(
                turn_id=turn_id,
                lang=self.lang,
                zone=zone,
                topic=topic,
                semantic_domain=domain,
                participants=[a.agent_id for a in participants],
                orchestrator_directive=plan["directive"],
                utterances=utterances,
            )

        sys_msg = (
            "Respond quickly in under 150 tokens total thinking + answer. "
            "Output ONLY valid single-line JSON with NO markdown fences. "
            "All `candidate_terms` must be single words or hyphenated compounds (no spaces)."
        )
        prior_lines = []
        for idx, ag in enumerate(participants):
            history_txt = "\n".join(prior_lines) if prior_lines else "(opening turn)"
            prompt = (
                f"You are {ag.name} ({ag.title}, domain: {ag.domain}) speaking naturally in authentic {self.state.lang_name} ({self.lang}) "
                f"at {VILLAGE_ZONES[zone]['label']}.\n"
                f"Mande's directive: {plan['directive']}\n"
                f"Topic: {topic} ({domain}).\n"
                f"Prior dialogue:\n{history_txt}\n\n"
                f"Write ONE natural 1-2 sentence utterance in standardized DNAFLA {self.lang} orthography "
                f"(no French/English words in `text_native`), concise EN/FR translations, "
                f"and 2 authentic unattested or specialized single-word/hyphenated cultural stems (`candidate_terms`) with 1-4 word glosses.\n"
                '{"text_native":"...","gloss_en":"...","gloss_fr":"...","candidate_terms":[{"term":"word1","pos":"n","ge":"short gloss","gf":"glose courte"}]}'
            )
            raw, _, err = await call_adaption(
                client,
                ag.model_id,
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0.3,
                json_mode=True,
            )
            parsed = extract_json_object(raw) if not err else None
            if isinstance(parsed, dict) and parsed.get("text_native"):
                txt = str(parsed["text_native"]).strip()
                g_en = str(parsed.get("gloss_en") or "").strip()
                g_fr = str(parsed.get("gloss_fr") or g_en).strip()
                raw_cands = parsed.get("candidate_terms") or []
                cands = []
                cand_glosses: Dict[str, Dict[str, str]] = {}
                for item in raw_cands:
                    if isinstance(item, dict) and item.get("term"):
                        t_str = str(item["term"]).strip()
                        cands.append(t_str)
                        cand_glosses[t_str] = {
                            "pos": str(item.get("pos") or "n").strip(),
                            "ge": str(item.get("ge") or "").strip(),
                            "gf": str(item.get("gf") or item.get("ge") or "").strip(),
                        }
                    elif isinstance(item, str) and item.strip():
                        cands.append(item.strip())
                cands = cands[:3]
            else:
                txt = f"I ni ce! {ag.name} bɛ baara kɛ {topic[:40]} la."
                g_en = f"Greetings! {ag.name} is working on {domain}."
                g_fr = f"Salutations! {ag.name} travaille dans le domaine {domain}."
                cands = []
                cand_glosses = {}
            utt = DialogueUtterance(
                agent_id=ag.agent_id,
                speaker_name=ag.name,
                model_id=ag.model_id,
                text_native=txt,
                gloss_en=g_en,
                gloss_fr=g_fr,
                candidate_terms=cands,
                candidate_glosses=cand_glosses,
            )
            utterances.append(utt)
            prior_lines.append(f"{ag.name}: {txt} ({g_en})")

        return VillageExchange(
            turn_id=turn_id,
            lang=self.lang,
            zone=zone,
            topic=topic,
            semantic_domain=domain,
            participants=[a.agent_id for a in participants],
            orchestrator_directive=plan["directive"],
            utterances=utterances,
        )

    async def _precheck_and_council(
        self,
        client: httpx.AsyncClient,
        turn_id: int,
        cand_raw: str,
        utt: DialogueUtterance,
        domain: str,
    ) -> Optional[Dict[str, Any]]:
        """Run Gates 1-4; return immediate REJECT decision or queue dict for batched Gemini 4 Argon review."""
        ortho = check_standardization(cand_raw, self.lang, working_dict_keys=self.working_index)
        route_hint, route_rationale = classify_lemma_vs_rule(
            ortho.standardized_form or cand_raw,
            "n",
            self.lang,
            working_dict_keys=self.working_index,
        )
        if not ortho.is_standard:
            ortho.route_hint = "reject"
        elif ortho.route_hint != "mdf_va" and route_hint == "gram_rule":
            ortho.route_hint = "gram_rule"

        if not ortho.is_standard or ortho.already_in_dict:
            self._commit_decision(
                CouncilDecision(
                    turn_id=turn_id,
                    lang=self.lang,
                    candidate_raw=cand_raw,
                    standardized_lx=ortho.standardized_form,
                    semantic_domain=domain,
                    example_native=utt.text_native,
                    example_fr=utt.gloss_fr or utt.gloss_en,
                    final_action="REJECT",
                    orthography_check=ortho,
                    dabax_check=DabaXCheck(error="Skipped DabaX compile after Gate 1 rejection"),
                    summary_reason=ortho.rationale,
                )
            )
            return None

        # Gate 4 (Council of 4 Field Linguists deliberation — Adaption Labs)
        seat_votes: List[CouncilSeatVote] = []
        proposed_lx = ortho.existing_headword if ortho.route_hint == "mdf_va" else ortho.standardized_form
        proposed_va = ortho.standardized_form if ortho.route_hint == "mdf_va" else ""
        cg = (utt.candidate_glosses or {}).get(cand_raw) or {}
        proposed_ps = str(cg.get("pos") or "n")[:16]
        proposed_ge = str(cg.get("ge") or cand_raw)[:80]
        proposed_gf = str(cg.get("gf") or proposed_ge)[:80]
        proposed_gram = ""

        if self.mock_mode:
            for sp in COUNCIL_SEATS:
                v = "REVISE_TO_VA" if ortho.route_hint == "mdf_va" else ("ROUTE_TO_GRAM" if ortho.route_hint == "gram_rule" else "APPROVE_LX")
                seat_votes.append(
                    CouncilSeatVote(
                        seat=sp.seat,
                        agent_id=sp.agent_id,
                        name=sp.name,
                        model_id=sp.model_id,
                        role_title=sp.title,
                        vote=v,  # type: ignore[arg-type]
                        rationale=f"{sp.name} checked standardized form '{proposed_lx}' via {sp.domain}.",
                    )
                )
        else:
            brief = (
                f"Language: {self.state.lang_name} ({self.lang})\n"
                f"Candidate token: '{cand_raw}'\n"
                f"Gate-1 Standardized form: '{ortho.standardized_form}' | Existing match: '{ortho.existing_headword or 'none'}'\n"
                f"Gate-1 Notes: {ortho.rationale}\n"
                f"Gate-2 Lemma-vs-Rule hint: {route_hint} ({route_rationale})\n"
                f"Context utterance: \"{utt.text_native}\"\n"
                f"EN gloss: \"{utt.gloss_en}\" | FR gloss: \"{utt.gloss_fr}\"\n"
                f"Domain: {domain}"
            )
            for sp in COUNCIL_SEATS:
                c_prompt = (
                    f"You are {sp.name}, {sp.title} on the 4-Agent Council of Field Linguists.\n"
                    "Rule: NEVER rubber-stamp raw dialectal, reduplicated nonsense, or Frenchified spellings; enforce DNAFLA/AMALAN standard form, "
                    "attach tonal/dialectal variants under \\va, route transparent inflections to .gram patterns, or REJECT invalid forms.\n"
                    "CRITICAL: Keep reasoning under 120 words and output compact single-line JSON. `gloss_en` and `gloss_fr` MUST be 1-5 word lexical glosses for the headword only.\n\n"
                    f"{brief}\n\n"
                    '{"vote":"APPROVE_LX|REVISE_TO_VA|ROUTE_TO_GRAM|REJECT","standardized_lx":"...","variant_va":"","pos":"n|v|adj|adv|ideo","gloss_en":"concise headword gloss","gloss_fr":"glose lexicale courte","gram_rule":"","rationale":"..."}'
                )
                raw, _, err = await call_adaption(
                    client,
                    sp.model_id,
                    [
                        {"role": "system", "content": "Output valid single-line JSON only after brief <120 word reasoning."},
                        {"role": "user", "content": c_prompt},
                    ],
                    max_tokens=4096,
                    temperature=0.1,
                    json_mode=True,
                )
                obj = extract_json_object(raw) if not err else None
                vote_str = str((obj or {}).get("vote") or "").upper()
                if vote_str not in {"APPROVE_LX", "REVISE_TO_VA", "ROUTE_TO_GRAM", "REJECT"}:
                    vote_str = (
                        "REVISE_TO_VA" if ortho.route_hint == "mdf_va"
                        else ("ROUTE_TO_GRAM" if ortho.route_hint == "gram_rule" else "APPROVE_LX")
                    )
                if obj:
                    if sp.seat == 1 and obj.get("standardized_lx") and vote_str != "REVISE_TO_VA":
                        cand_std = str(obj["standardized_lx"]).strip()
                        recheck = check_standardization(cand_std, self.lang, working_dict_keys=self.working_index)
                        if recheck.is_standard and not recheck.already_in_dict:
                            proposed_lx = recheck.standardized_form
                    if sp.seat == 2 and obj.get("gram_rule"):
                        proposed_gram = str(obj["gram_rule"]).strip()
                    if sp.seat == 3:
                        proposed_ps = str(obj.get("pos") or proposed_ps).strip()[:16]
                        proposed_ge = str(obj.get("gloss_en") or proposed_ge).strip()[:120]
                        proposed_gf = str(obj.get("gloss_fr") or proposed_gf).strip()[:120]
                seat_votes.append(
                    CouncilSeatVote(
                        seat=sp.seat,
                        agent_id=sp.agent_id,
                        name=sp.name,
                        model_id=sp.model_id,
                        role_title=sp.title,
                        vote=vote_str,  # type: ignore[arg-type]
                        rationale=str((obj or {}).get("rationale") or err or "Validated standard form."),
                    )
                )

        # Tally Council supermajority (>= 3/4 non-reject)
        non_reject = [v for v in seat_votes if v.vote != "REJECT"]
        if len(non_reject) < 3:
            self._commit_decision(
                CouncilDecision(
                    turn_id=turn_id,
                    lang=self.lang,
                    candidate_raw=cand_raw,
                    standardized_lx=proposed_lx,
                    variant_va=proposed_va,
                    pos=proposed_ps,
                    gloss_en=proposed_ge,
                    gloss_fr=proposed_gf,
                    semantic_domain=domain,
                    example_native=utt.text_native,
                    example_fr=utt.gloss_fr or utt.gloss_en,
                    final_action="REJECT",
                    orthography_check=ortho,
                    dabax_check=DabaXCheck(error="Rejected by Council supermajority (<3/4 votes)"),
                    seat_votes=seat_votes,
                    summary_reason=f"Rejected by Council ({4 - len(non_reject)}/4 reject votes).",
                )
            )
            return None

        # Pick dominant Council action
        vote_counts: Dict[str, int] = {}
        for v in non_reject:
            vote_counts[v.vote] = vote_counts.get(v.vote, 0) + 1
        top_vote = max(vote_counts.items(), key=lambda kv: kv[1])[0]
        if ortho.route_hint == "mdf_va" and ortho.existing_headword:
            council_action = "COMMIT_VA"
            proposed_lx = ortho.existing_headword
            proposed_va = ortho.standardized_form
        elif top_vote == "ROUTE_TO_GRAM" and proposed_gram:
            council_action = "COMMIT_GRAM"
        else:
            council_action = "COMMIT_LX"

        mdf_preview = serialize_mdf_entry(
            lx=proposed_lx,
            va=proposed_va,
            ps=proposed_ps,
            ge=proposed_ge,
            gf=proposed_gf,
            domain=domain,
            xv=utt.text_native,
            xf=utt.gloss_fr or utt.gloss_en,
        )

        # Gate 3: Deterministic Shadow DabaX(strict_parse=True) Verification
        dabax_chk = verify_shadow_candidate(
            lang=self.lang,
            working_dict_path=self.dict_path,
            working_gram_path=self.gram_path,
            candidate_term=proposed_va or proposed_lx,
            mdf_block=mdf_preview if council_action == "COMMIT_LX" else "",
            va_addition={"target_lx": proposed_lx, "va": proposed_va} if council_action == "COMMIT_VA" else None,
            gram_rule=proposed_gram if council_action == "COMMIT_GRAM" else "",
        )
        self._emit(
            turn_id=turn_id,
            kind="dabax_gate",
            agent_id="c4-dabax-verifier",
            zone="toguna_council",
            label=f"Shadow DabaX Gate: {proposed_va or proposed_lx}",
            summary=(
                f"dict_ok={dabax_chk.dict_compile_ok}, gram_ok={dabax_chk.gram_compile_ok}, "
                f"dev-mini={dabax_chk.shadow_devmini_pct:.2f}% (regression={dabax_chk.regression_pct:.2f}%)"
            ),
            meta=dabax_chk.model_dump(),
        )
        if not dabax_chk.dict_compile_ok or not dabax_chk.gram_compile_ok or dabax_chk.error:
            self._commit_decision(
                CouncilDecision(
                    turn_id=turn_id,
                    lang=self.lang,
                    candidate_raw=cand_raw,
                    standardized_lx=proposed_lx,
                    variant_va=proposed_va,
                    pos=proposed_ps,
                    gloss_en=proposed_ge,
                    gloss_fr=proposed_gf,
                    semantic_domain=domain,
                    example_native=utt.text_native,
                    example_fr=utt.gloss_fr or utt.gloss_en,
                    gram_rule=proposed_gram,
                    final_action="REJECT",
                    orthography_check=ortho,
                    dabax_check=dabax_chk,
                    seat_votes=seat_votes,
                    summary_reason=dabax_chk.error or "Failed strict DabaX compilation gate.",
                )
            )
            return None

        return {
            "turn_id": turn_id,
            "cand_raw": cand_raw,
            "proposed_lx": proposed_lx,
            "proposed_va": proposed_va,
            "proposed_ps": proposed_ps,
            "proposed_ge": proposed_ge,
            "proposed_gf": proposed_gf,
            "proposed_gram": proposed_gram,
            "domain": domain,
            "utt": utt,
            "ortho": ortho,
            "route_rationale": route_rationale,
            "dabax_chk": dabax_chk,
            "seat_votes": seat_votes,
            "council_action": council_action,
            "mdf_preview": mdf_preview,
        }

    async def _flush_argon_batch(self, client: httpx.AsyncClient, turn_id: int) -> List[CouncilDecision]:
        """Audit up to ARGON_BATCH_MAX queued candidates in ONE batched Gemini 4 Argon call."""
        if not self._pending_argon:
            return []

        decisions: List[CouncilDecision] = []
        while self._pending_argon:
            batch = self._pending_argon[: self.argon_batch_max]
            self._pending_argon = self._pending_argon[self.argon_batch_max :]
            self.argon_batch_calls += 1

            reviews_by_idx: Dict[int, CriticVerdict] = {}
            if self.mock_mode:
                for i, item in enumerate(batch):
                    reviews_by_idx[i] = CriticVerdict(
                        verdict="CRITIC_APPROVE",
                        revised_action=item["council_action"],
                        standardized_lx=item["proposed_lx"],
                        rationale=f"Batched Argon confirmed '{item['proposed_lx']}' complies with DNAFLA standard (batch size={len(batch)}).",
                        latency_s=0.1,
                    )
            else:
                items_txt = []
                for i, item in enumerate(batch):
                    items_txt.append(
                        f"[{i}] Cand='{item['cand_raw']}' -> Std='{item['proposed_lx']}' (va='{item['proposed_va']}'), "
                        f"Action={item['council_action']}, Pos={item['proposed_ps']}, GE='{item['proposed_ge']}', GF='{item['proposed_gf']}', "
                        f"Domain='{item['domain']}', Reg={item['dabax_chk'].regression_pct:.2f}%, Ex='{item['utt'].text_native[:110]}'"
                        + (f", Gram='{item['proposed_gram']}'" if item["proposed_gram"] else "")
                    )
                argon_prompt = (
                    f"You are Gemini 4 Argon, Primary Standardization Critic overseeing the Council of Field Linguists for {self.state.lang_name} ({self.lang}).\n"
                    f"Review this BATCH of {len(batch)} Council-approved & DabaX-compiled candidates together in ONE pass. "
                    "Reject calques, French/English intrusions, hallucinated stems, sentence-length glosses in \\ge/\\gf, or unsafe .gram rules.\n\n"
                    + "\n".join(items_txt)
                    + '\n\nRespond strictly as JSON with one entry per idx (0..N-1):\n'
                    '{"reviews":[{"idx":0,"verdict":"CRITIC_APPROVE|CRITIC_REVISE|CRITIC_VETO","revised_action":"COMMIT_LX|COMMIT_VA|COMMIT_GRAM|REJECT","standardized_lx":"...","rationale":"..."}]}'
                )
                raw_a, lat_a, err_a = await call_gemini_argon(
                    client,
                    [{"role": "user", "content": argon_prompt}],
                    max_tokens=3200,
                )
                parsed_a = extract_json_object(raw_a) if not err_a else None
                raw_reviews = (parsed_a or {}).get("reviews") if isinstance(parsed_a, dict) else None
                if isinstance(raw_reviews, list):
                    for r in raw_reviews:
                        if not isinstance(r, dict) or r.get("idx") is None:
                            continue
                        try:
                            idx_val = int(r["idx"])
                        except (TypeError, ValueError):
                            continue
                        if not (0 <= idx_val < len(batch)):
                            continue
                        item = batch[idx_val]
                        v_str = str(r.get("verdict") or "CRITIC_VETO").upper()
                        if v_str not in {"CRITIC_APPROVE", "CRITIC_REVISE", "CRITIC_VETO"}:
                            v_str = "CRITIC_VETO"
                        rev_act = str(r.get("revised_action") or item["council_action"]).upper()
                        if rev_act not in {"COMMIT_LX", "COMMIT_VA", "COMMIT_GRAM", "REJECT"}:
                            rev_act = item["council_action"]
                        if v_str == "CRITIC_VETO":
                            rev_act = "REJECT"
                        reviews_by_idx[idx_val] = CriticVerdict(
                            verdict=v_str,  # type: ignore[arg-type]
                            revised_action=rev_act,  # type: ignore[arg-type]
                            standardized_lx=str(r.get("standardized_lx") or item["proposed_lx"]).strip() or item["proposed_lx"],
                            rationale=str(r.get("rationale") or "Batched Argon review."),
                            latency_s=round(lat_a / max(1, len(batch)), 3),
                        )

                for i, item in enumerate(batch):
                    if i not in reviews_by_idx:
                        reviews_by_idx[i] = CriticVerdict(
                            verdict="CRITIC_VETO",
                            revised_action="REJECT",
                            standardized_lx=item["proposed_lx"],
                            rationale=f"Argon batch item {i} omitted or failed ({err_a or 'parse error'}); strict fail-closed veto.",
                            latency_s=0.0,
                            error=err_a or "omitted_from_batch",
                        )

            for i, item in enumerate(batch):
                critic_verdict = reviews_by_idx[i]
                item_turn = int(item["turn_id"])
                proposed_lx = str(item["proposed_lx"])
                proposed_va = str(item["proposed_va"])
                utt: DialogueUtterance = item["utt"]

                self._emit(
                    turn_id=turn_id,
                    kind="critic_review",
                    agent_id=CRITIC_SPEC.agent_id,
                    zone="critic_bench",
                    target_zone="toguna_council",
                    label=f"Gemini 4 Argon (Batch {i+1}/{len(batch)}): {critic_verdict.verdict} ({proposed_va or proposed_lx})",
                    summary=critic_verdict.rationale[:280],
                    meta=critic_verdict.model_dump(),
                )

                final_action = critic_verdict.revised_action or item["council_action"]
                if critic_verdict.verdict == "CRITIC_VETO" or final_action == "REJECT":
                    dec = CouncilDecision(
                        turn_id=item_turn,
                        lang=self.lang,
                        candidate_raw=item["cand_raw"],
                        standardized_lx=proposed_lx,
                        variant_va=proposed_va,
                        pos=item["proposed_ps"],
                        gloss_en=item["proposed_ge"],
                        gloss_fr=item["proposed_gf"],
                        semantic_domain=item["domain"],
                        example_native=utt.text_native,
                        example_fr=utt.gloss_fr or utt.gloss_en,
                        gram_rule=item["proposed_gram"],
                        final_action="REJECT",
                        orthography_check=item["ortho"],
                        dabax_check=item["dabax_chk"],
                        seat_votes=item["seat_votes"],
                        critic=critic_verdict,
                        summary_reason=critic_verdict.rationale,
                    )
                else:
                    if critic_verdict.standardized_lx and final_action == "COMMIT_LX":
                        re_std = check_standardization(
                            critic_verdict.standardized_lx, self.lang, working_dict_keys=self.working_index
                        )
                        if re_std.is_standard and not re_std.already_in_dict:
                            proposed_lx = re_std.standardized_form

                    # Re-check deduplication against working dict before commit
                    dup_chk = check_standardization(proposed_va or proposed_lx, self.lang, working_dict_keys=self.working_index)
                    if final_action == "COMMIT_LX" and dup_chk.already_in_dict:
                        dec = CouncilDecision(
                            turn_id=item_turn,
                            lang=self.lang,
                            candidate_raw=item["cand_raw"],
                            standardized_lx=proposed_lx,
                            variant_va=proposed_va,
                            pos=item["proposed_ps"],
                            gloss_en=item["proposed_ge"],
                            gloss_fr=item["proposed_gf"],
                            semantic_domain=item["domain"],
                            example_native=utt.text_native,
                            example_fr=utt.gloss_fr or utt.gloss_en,
                            final_action="REJECT",
                            orthography_check=item["ortho"],
                            dabax_check=item["dabax_chk"],
                            seat_votes=item["seat_votes"],
                            critic=critic_verdict,
                            summary_reason=f"Already committed earlier in batch under \\lx {dup_chk.existing_headword}.",
                        )
                    else:
                        final_mdf = serialize_mdf_entry(
                            lx=proposed_lx,
                            va=proposed_va,
                            ps=item["proposed_ps"],
                            ge=item["proposed_ge"],
                            gf=item["proposed_gf"],
                            domain=item["domain"],
                            xv=utt.text_native,
                            xf=utt.gloss_fr or utt.gloss_en,
                        )
                        dec = CouncilDecision(
                            turn_id=item_turn,
                            lang=self.lang,
                            candidate_raw=item["cand_raw"],
                            standardized_lx=proposed_lx,
                            variant_va=proposed_va,
                            pos=item["proposed_ps"],
                            gloss_en=item["proposed_ge"],
                            gloss_fr=item["proposed_gf"],
                            semantic_domain=item["domain"],
                            example_native=utt.text_native,
                            example_fr=utt.gloss_fr or utt.gloss_en,
                            gram_rule=item["proposed_gram"],
                            final_action=final_action,  # type: ignore[arg-type]
                            orthography_check=item["ortho"],
                            dabax_check=item["dabax_chk"],
                            seat_votes=item["seat_votes"],
                            critic=critic_verdict,
                            mdf_block=final_mdf,
                            summary_reason=critic_verdict.rationale,
                        )

                self._commit_decision(dec)
                decisions.append(dec)
        return decisions

    def _commit_decision(self, decision: CouncilDecision) -> None:
        self.state.total_candidates_reviewed += 1
        with self.ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(decision.model_dump_json() + "\n")

        if decision.final_action == "COMMIT_LX":
            with self.dict_path.open("a", encoding="utf-8") as fh:
                fh.write("\n" + decision.mdf_block.strip() + "\n")
            self.working_index, total_lx, _ = parse_working_dict_index(self.dict_path)
            self.state.added_lx_count = max(self.state.added_lx_count + 1, total_lx - self.state.baseline_lx_count)
            self.state.latest_devmini_pct = decision.dabax_check.shadow_devmini_pct or self.state.latest_devmini_pct
            if self.state.baseline_devmini_pct <= 0.0:
                self.state.baseline_devmini_pct = decision.dabax_check.baseline_devmini_pct
            self._emit(
                turn_id=decision.turn_id,
                kind="lexicon_commit",
                agent_id="c3-mdf-scribe",
                zone="toguna_council",
                label=f"Committed MDF \\lx {decision.standardized_lx}",
                summary=f"+1 standardized headword ({decision.pos}: {decision.gloss_en}) | Progress: {self.state.baseline_lx_count + self.state.added_lx_count:,} / {TARGET_DICT_ENTRIES:,}",
                meta=decision.model_dump(),
            )
        elif decision.final_action == "COMMIT_VA" and decision.variant_va:
            if _append_va_to_dict_file(self.dict_path, target_lx=decision.standardized_lx, va_form=decision.variant_va):
                self.working_index, _, _ = parse_working_dict_index(self.dict_path)
                self.state.added_va_count += 1
            self._emit(
                turn_id=decision.turn_id,
                kind="lexicon_commit",
                agent_id="c3-mdf-scribe",
                zone="toguna_council",
                label=f"Attached MDF \\va {decision.variant_va} -> \\lx {decision.standardized_lx}",
                summary=decision.summary_reason[:220],
                meta=decision.model_dump(),
            )
        elif decision.final_action == "COMMIT_GRAM" and decision.gram_rule:
            clean_g = _sanitize_gram_rule(decision.gram_rule)
            if clean_g:
                with self.gram_path.open("a", encoding="utf-8") as fh:
                    fh.write("\n# SEBENI_ED4_COUNCIL_ARGON\n" + clean_g + "\n")
                self.state.added_gram_rules_count += 1
            self._emit(
                turn_id=decision.turn_id,
                kind="gram_commit",
                agent_id="c2-morphotactics",
                zone="toguna_council",
                label=f"Integrated Morphotactic Rule ({decision.standardized_lx})",
                summary=decision.gram_rule[:220],
                meta=decision.model_dump(),
            )
        else:
            self.state.total_rejected += 1
            self._emit(
                turn_id=decision.turn_id,
                kind="council_reject",
                agent_id=CRITIC_SPEC.agent_id if decision.critic else "c1-orthography",
                zone="toguna_council",
                label=f"Standardization Rejected: {decision.candidate_raw}",
                summary=decision.summary_reason[:240],
                meta=decision.model_dump(),
            )

        self.state.recent_decisions.append(decision)
        self._save_state()

    async def run_turn(self, client: httpx.AsyncClient) -> Tuple[VillageExchange, List[CouncilDecision]]:
        """Execute one full village turn: Mande plan -> walking -> dialogue -> Council -> DabaX -> Argon."""
        turn_id = self.state.total_turns + 1
        plan = await self._mande_plan_turn(client, turn_id)

        mande = self._get_agent(ORCHESTRATOR_SPEC.agent_id)
        if mande is not None and plan["zone"] != mande.zone:
            self._move_agent(mande, plan["zone"], turn_id, f"Mande walks to oversee {plan['semantic_domain']}")

        spawned = None
        if plan.get("spawn_specialist"):
            spawned = self._maybe_spawn_specialist(
                plan["spawn_specialist"],
                plan.get("spawn_reason") or f"Needed for {plan['semantic_domain']}",
                turn_id,
            )

        self._emit(
            turn_id=turn_id,
            kind="orchestrator_directive",
            agent_id=ORCHESTRATOR_SPEC.agent_id,
            zone=plan["zone"],
            label=f"Mande Turn #{turn_id}: {plan['semantic_domain']}",
            summary=plan["directive"],
            meta=plan,
        )

        participants: List[VillageAgent] = []
        for aid in plan["participants"]:
            ag = self._get_agent(aid)
            if ag is not None:
                participants.append(ag)
        if spawned is not None and spawned.agent_id not in {p.agent_id for p in participants}:
            participants.append(spawned)
        if not participants:
            participants = [a for a in self.state.active_agents if a.role == "villager"][:2]

        for ag in participants:
            if ag.zone != plan["zone"]:
                self._move_agent(ag, plan["zone"], turn_id, f"Joining discussion on {plan['topic']}")

        exchange = await self._run_villager_dialogue(client, turn_id, plan, participants)
        if spawned is not None:
            exchange.spawned_agent_id = spawned.agent_id

        for utt in exchange.utterances:
            self._emit(
                turn_id=turn_id,
                kind="dialogue_turn",
                agent_id=utt.agent_id,
                zone=plan["zone"],
                label=f"{utt.speaker_name}: {utt.text_native[:72]}",
                summary=f"{utt.gloss_en} | Candidates: {', '.join(utt.candidate_terms) or 'none'}",
                meta=utt.model_dump(),
            )

        # Courier scroll sends unique candidates from dialogue zone to Toguna Council
        seen_cands = set()
        cand_pairs: List[Tuple[str, DialogueUtterance]] = []
        for utt in exchange.utterances:
            for c in utt.candidate_terms:
                ck = c.strip().lower()
                if ck and ck not in seen_cands:
                    seen_cands.add(ck)
                    cand_pairs.append((c.strip(), utt))
        cand_pairs = cand_pairs[:5]

        if cand_pairs:
            self._emit(
                turn_id=turn_id,
                kind="courier_scroll",
                agent_id=participants[0].agent_id,
                target_agent_id="c1-orthography",
                zone=plan["zone"],
                target_zone="toguna_council",
                path=compute_zone_path(plan["zone"], "toguna_council"),
                label=f"Field Scroll -> Toguna Council ({len(cand_pairs)} terms)",
                summary=f"Submitting {', '.join(c for c, _ in cand_pairs)} for DNAFLA & DabaX review.",
                meta={"candidates": [c for c, _ in cand_pairs]},
            )

        decisions: List[CouncilDecision] = []
        for cand_raw, utt in cand_pairs:
            self._emit(
                turn_id=turn_id,
                kind="council_deliberation",
                agent_id="c1-orthography",
                zone="toguna_council",
                label=f"Toguna Reviewing '{cand_raw}'",
                summary="Examining orthography, lemma-vs-.gram boundary, and strict DabaX compliance.",
                meta={"candidate": cand_raw},
            )
            queued = await self._precheck_and_council(client, turn_id, cand_raw, utt, plan["semantic_domain"])
            if queued is not None:
                self._pending_argon.append(queued)
                self._emit(
                    turn_id=turn_id,
                    kind="council_deliberation",
                    agent_id=CRITIC_SPEC.agent_id,
                    zone="toguna_council",
                    label=f"Queued for Gemini 4 Argon Batch ({len(self._pending_argon)}/{self.argon_batch_min})",
                    summary=f"'{queued['proposed_lx']}' passed Gates 1-4; awaiting batched Argon review.",
                    meta={"queue_len": len(self._pending_argon), "batch_min": self.argon_batch_min},
                )

        if len(self._pending_argon) >= self.argon_batch_min:
            decisions.extend(await self._flush_argon_batch(client, turn_id))

        self.state.total_turns = turn_id
        self.state.domain_counts[plan["semantic_domain"]] = self.state.domain_counts.get(plan["semantic_domain"], 0) + 1
        with self.exchanges_path.open("a", encoding="utf-8") as fh:
            fh.write(exchange.model_dump_json() + "\n")
        self.state.recent_exchanges.append(exchange)
        self._save_state()
        return exchange, decisions

    async def run_epoch(
        self,
        *,
        max_turns: int = 3,
        on_turn_complete: Optional[Callable[[int], None]] = None,
    ) -> LanguageVillageState:
        """Execute up to `max_turns` turns in sequence and emit epoch_checkpoint."""
        async with httpx.AsyncClient() as client:
            for idx in range(max_turns):
                await self.run_turn(client)
                if on_turn_complete is not None:
                    on_turn_complete(idx + 1)
            if self._pending_argon:
                await self._flush_argon_batch(client, self.state.total_turns)
                if on_turn_complete is not None:
                    on_turn_complete(max_turns)
        self._emit(
            turn_id=self.state.total_turns,
            kind="epoch_checkpoint",
            agent_id=ORCHESTRATOR_SPEC.agent_id,
            zone="baobab_plaza",
            label=f"Cloud Epoch Checkpointed ({self.state.baseline_lx_count + self.state.added_lx_count:,} / {TARGET_DICT_ENTRIES:,})",
            summary=(
                f"Turns={self.state.total_turns}, Added +\\lx={self.state.added_lx_count}, "
                f"+\\va={self.state.added_va_count}, +.gram={self.state.added_gram_rules_count}."
            ),
            meta={"added_lx": self.state.added_lx_count, "added_va": self.state.added_va_count},
        )
        return self.state
