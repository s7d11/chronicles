"""Typed Pydantic schemas & Toolbox MDF formatters for Edition No. 4."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class VillageAgent(BaseModel):
    agent_id: str
    name: str
    role: Literal["orchestrator", "villager", "specialist", "council", "critic"]
    title: str
    model_id: str
    provider: str
    home_zone: str
    zone: str
    x: float
    z: float
    domain: str
    seat: int = 0
    color: str = "#8a5a18"
    spawned_dynamically: bool = False
    spawn_reason: str = ""


class DialogueUtterance(BaseModel):
    agent_id: str
    speaker_name: str
    model_id: str
    text_native: str
    gloss_en: str = ""
    gloss_fr: str = ""
    candidate_terms: List[str] = Field(default_factory=list)
    candidate_glosses: Dict[str, Dict[str, str]] = Field(default_factory=dict)


class VillageExchange(BaseModel):
    turn_id: int
    lang: str
    zone: str
    topic: str
    semantic_domain: str
    participants: List[str] = Field(default_factory=list)
    spawned_agent_id: Optional[str] = None
    orchestrator_directive: str = ""
    utterances: List[DialogueUtterance] = Field(default_factory=list)


class OrthographyCheck(BaseModel):
    raw_form: str
    standardized_form: str
    is_standard: bool
    normalized_changed: bool = False
    already_in_dict: bool = False
    existing_headword: str = ""
    route_hint: Literal["mdf_lx", "mdf_va", "gram_rule", "reject"] = "mdf_lx"
    issues: List[str] = Field(default_factory=list)
    rationale: str = ""


class DabaXCheck(BaseModel):
    dict_compile_ok: bool = False
    gram_compile_ok: bool = True
    candidate_parsed: bool = False
    baseline_devmini_pct: float = 0.0
    shadow_devmini_pct: float = 0.0
    regression_pct: float = 0.0
    error: str = ""


class CouncilSeatVote(BaseModel):
    seat: int
    agent_id: str
    name: str
    model_id: str
    role_title: str
    vote: Literal["APPROVE_LX", "REVISE_TO_VA", "ROUTE_TO_GRAM", "REJECT"]
    rationale: str


class CriticVerdict(BaseModel):
    critic_name: str = "Gemini 4 Argon"
    model_id: str = "gemini-4-argon"
    verdict: Literal["CRITIC_APPROVE", "CRITIC_REVISE", "CRITIC_VETO"]
    revised_action: Optional[Literal["COMMIT_LX", "COMMIT_VA", "COMMIT_GRAM", "REJECT"]] = None
    standardized_lx: str = ""
    rationale: str = ""
    latency_s: float = 0.0
    error: str = ""


class CouncilDecision(BaseModel):
    turn_id: int
    lang: str
    candidate_raw: str
    standardized_lx: str
    variant_va: str = ""
    pos: str = "n"
    gloss_en: str = ""
    gloss_fr: str = ""
    semantic_domain: str = ""
    example_native: str = ""
    example_fr: str = ""
    gram_rule: str = ""
    final_action: Literal["COMMIT_LX", "COMMIT_VA", "COMMIT_GRAM", "REJECT"]
    orthography_check: OrthographyCheck
    dabax_check: DabaXCheck
    seat_votes: List[CouncilSeatVote] = Field(default_factory=list)
    critic: Optional[CriticVerdict] = None
    mdf_block: str = ""
    summary_reason: str = ""


class VillageEvent(BaseModel):
    seq: int
    ts: str
    lang: str
    turn_id: int
    kind: Literal[
        "village_init",
        "orchestrator_directive",
        "agent_spawn",
        "agent_move",
        "dialogue_turn",
        "courier_scroll",
        "council_deliberation",
        "dabax_gate",
        "critic_review",
        "lexicon_commit",
        "gram_commit",
        "council_reject",
        "epoch_checkpoint",
    ]
    agent_id: str = ""
    target_agent_id: str = ""
    zone: str = ""
    target_zone: str = ""
    path: List[Dict[str, float]] = Field(default_factory=list)
    label: str = ""
    summary: str = ""
    meta: Dict[str, Any] = Field(default_factory=dict)


class LanguageVillageState(BaseModel):
    lang: str
    lang_name: str
    target_entries: int = 50_000
    baseline_lx_count: int = 0
    added_lx_count: int = 0
    added_va_count: int = 0
    baseline_gram_lines: int = 0
    added_gram_rules_count: int = 0
    total_turns: int = 0
    total_candidates_reviewed: int = 0
    total_rejected: int = 0
    baseline_devmini_pct: float = 0.0
    latest_devmini_pct: float = 0.0
    updated_at: str = ""
    orchestrator_model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    critic_model: str = "gemini-4-argon"
    active_agents: List[VillageAgent] = Field(default_factory=list)
    domain_counts: Dict[str, int] = Field(default_factory=dict)
    recent_exchanges: List[VillageExchange] = Field(default_factory=list)
    recent_decisions: List[CouncilDecision] = Field(default_factory=list)


def _clean_mdf_field(value: str, *, single_token: bool = False, max_len: int = 160) -> str:
    txt = re.sub(r"[\r\n\\]+", " ", value or "").strip()
    if single_token:
        txt = txt.split()[0] if txt else ""
    return txt[:max_len].strip()


def serialize_mdf_entry(
    *,
    lx: str,
    ps: str,
    ge: str,
    gf: str,
    va: str = "",
    xv: str = "",
    xf: str = "",
    domain: str = "",
    date_str: Optional[str] = None,
) -> str:
    """Format a standardized Toolbox MDF 4.0 lexical entry."""
    clean_lx = _clean_mdf_field(lx, single_token=True, max_len=64)
    clean_ps = re.sub(r"[^A-Za-z0-9_./-]", "", ps or "n")[:16] or "n"
    clean_ge = _clean_mdf_field(ge or "gloss", max_len=140) or "gloss"
    clean_gf = _clean_mdf_field(gf or clean_ge, max_len=140) or clean_ge
    clean_va = _clean_mdf_field(va, single_token=True, max_len=64)
    clean_xv = _clean_mdf_field(xv, max_len=240)
    clean_xf = _clean_mdf_field(xf, max_len=240)
    clean_sd = _clean_mdf_field(domain, max_len=80)
    dt = date_str or datetime.now(timezone.utc).strftime("%d/%b/%Y")

    lines = [f"\\lx {clean_lx}"]
    if clean_va and clean_va.lower() != clean_lx.lower():
        lines.append(f"\\va {clean_va}")
    lines.extend([f"\\ps {clean_ps}", f"\\ge {clean_ge}", f"\\gf {clean_gf}"])
    if clean_sd:
        lines.append(f"\\sd {clean_sd}")
    if clean_xv:
        lines.append(f"\\xv {clean_xv}")
        if clean_xf:
            lines.append(f"\\xf {clean_xf}")
    lines.append(f"\\dt {dt}")
    lines.append("\\rem SEBENI_ED4_COUNCIL_ARGON")
    lines.append("")
    return "\n".join(lines)
