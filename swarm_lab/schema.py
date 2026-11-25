"""Pydantic v2 schemas for Edition No. 2 self-organizing swarm experiments and Three.js replay."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

TopologyType = Literal["star", "ring", "mesh", "hierarchical", "pipeline"]
EventKind = Literal[
    "coordinator_brief",
    "cluster_spawn",
    "resource_lookup",
    "agent_exchange",
    "consensus_vote",
    "consensus_reached",
    "shadow_patch",
    "native_command",
    "native_generation",
    "coordinator_audit",
    "run_complete",
]


class AgentSpec(BaseModel):
    """Single spawned agent instance inside a model's 2-5 agent cluster."""

    id: str
    role: str
    mandate: str


class ClusterPlan(BaseModel):
    """Self-selected topology and agent roster (strictly 2-5 agents per cluster)."""

    cluster_slug: str
    model_name: str
    n_agents: int = Field(ge=2, le=5)
    topology: TopologyType
    agents: List[AgentSpec]
    edges: List[Tuple[str, str]]
    rationale: str = ""


class WordAssessment(BaseModel):
    """Consensus linguistic analysis of a single surface token."""

    surface: str
    morphemes: List[str] = Field(default_factory=list)
    pos: str = ""
    gloss: str = ""
    notes: str = ""


class DictDelta(BaseModel):
    """Candidate Toolbox/MDF lexicon entry proposed by cluster consensus."""

    lx: str
    ps: str = "n"
    ge: str = ""
    gf: str = ""


class SwarmEvent(BaseModel):
    """Real-time / replayable event streamed to the Three.js Swarm Theater."""

    seq: int
    ts: float
    lang: str
    cluster: str
    kind: EventKind
    src: str
    dst: str
    round_num: int = 0
    label: str = ""
    summary: str = ""
    meta: Dict[str, Any] = Field(default_factory=dict)


class ShadowPatchResult(BaseModel):
    """Isolated Shadow DabaX(strict_parse=True) compilation & coverage delta."""

    dict_entries_added: int = 0
    gram_rules_proposed: int = 0
    dict_compile_ok: bool = False
    gram_compile_ok: bool = True
    baseline_test_parse_pct: float = 0.0
    shadow_test_parse_pct: float = 0.0
    test_delta_pct: float = 0.0
    baseline_devmini_parse_pct: float = 0.0
    shadow_devmini_parse_pct: float = 0.0
    devmini_delta_pct: float = 0.0
    phantom_rate_pct: float = 0.0
    error: Optional[str] = None


class NativeGenResult(BaseModel):
    """Generated text following a randomized directive from prompt_native.csv."""

    command_template: str
    native_fields: Dict[str, str] = Field(default_factory=dict)
    output_text: str = ""
    baseline_parse_pct: float = 0.0
    shadow_parse_pct: float = 0.0
    leakage_pct: float = 0.0
    header_compliance: float = 0.0
    verified_by_peer: bool = False


class ExchangeSnippet(BaseModel):
    """Compact human-readable excerpt of an inter-agent exchange for publication."""

    round_num: int
    src: str
    dst: str
    summary: str
    agree: Optional[bool] = None


class ClusterResult(BaseModel):
    """Complete result bundle for one model cluster on one language text."""

    lang: str
    cluster_slug: str
    model_name: str
    plan: ClusterPlan
    rounds_used: int = 1
    unanimous: bool = False
    agreement_pct: float = 0.0
    n_exchanges: int = 0
    consensus_words: List[WordAssessment] = Field(default_factory=list)
    dict_deltas: List[DictDelta] = Field(default_factory=list)
    gram_rules: List[str] = Field(default_factory=list)
    dissent_notes: List[str] = Field(default_factory=list)
    exchanges: List[ExchangeSnippet] = Field(default_factory=list)
    shadow: ShadowPatchResult
    native_gen: NativeGenResult
    coordinator_score: float = 0.0
    coordinator_note: str = ""


class LanguageSummary(BaseModel):
    """All clusters & coordinator telemetry for one test.json language text."""

    lang: str
    lang_name: str
    test_text: str
    native_labels: Dict[str, str]
    native_command: str
    coordinator_brief: str
    coordinator_verdict: str
    target_words: List[Dict[str, Any]]
    clusters: List[ClusterResult] = Field(default_factory=list)


class Edition2Payload(BaseModel):
    """Published top-level payload in site/data/edition2.json."""

    generated_at: str
    coordinator_model: str
    generation_model: str
    langs: List[str]
    languages: Dict[str, LanguageSummary]
    leaderboard: List[Dict[str, Any]]
