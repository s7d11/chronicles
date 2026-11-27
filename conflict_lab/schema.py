"""Pydantic schemas for Volume II Mali OSINT incidents, clusters, mathematical proofs & forecasts."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

SourcePerspective = Literal[
    "independent_event_db",
    "multilateral_un",
    "official_fama_dirpa",
    "public_x_post",
    "sahel_press",
]

TacticType = Literal[
    "ied_mine",
    "ambush",
    "raid_assault",
    "indirect_fire",
    "kidnapping_intimidation",
    "airstrike_clash",
    "blockade_siege",
    "other",
]

VerdictStatus = Literal["PROVEN", "SUPPORTED", "INCONCLUSIVE", "FALSIFIED"]


class RawOsintRecord(BaseModel):
    record_id: str
    source_name: str
    source_perspective: SourcePerspective
    source_url: str = ""
    published_date: str
    event_date: str
    title: str
    summary: str = ""
    raw_location: str = ""
    admin1_hint: str = ""
    admin2_hint: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None
    actor_raw: str = ""
    tactic_hint: str = "other"
    fatalities_reported: int = 0
    confidence: float = 0.7


class VerifiedIncident(BaseModel):
    incident_id: str
    date_start: str
    date_end: str
    precision_level: Literal["exact_locality", "cercle_centroid", "region_centroid"] = "cercle_centroid"
    admin1: str
    admin2: str
    cercle_id: str
    locality: str = ""
    lat: float
    lon: float
    actor_primary: str
    actor_secondary: str = ""
    tactic: TacticType = "other"
    fatalities_best: int = 0
    source_count: int = 1
    sources: List[str] = Field(default_factory=list)
    perspectives: List[SourcePerspective] = Field(default_factory=list)
    corroboration_tier: Literal["high", "medium", "single_source"] = "single_source"
    headline: str = ""
    summary: str = ""
    adjudicator_note: str = ""


class ClusterMemberRef(BaseModel):
    incident_id: str
    date_start: str
    cercle_id: str
    locality: str
    lat: float
    lon: float
    actor_primary: str
    tactic: str
    fatalities_best: int = 0
    source_count: int = 1


class AttackCluster(BaseModel):
    cluster_id: str
    label: str
    admin1: str
    dominant_cercle_id: str
    centroid_lat: float
    centroid_lon: float
    radius_km: float
    event_count: int
    fatalities_total: int = 0
    date_first: str
    date_last: str
    dominant_actors: Dict[str, int] = Field(default_factory=dict)
    tactic_breakdown: Dict[str, int] = Field(default_factory=dict)
    dry_season_share: float = 0.5
    corroborated_share: float = 0.0
    sample_events: List[ClusterMemberRef] = Field(default_factory=list)


class HypothesisProof(BaseModel):
    hyp_id: str
    cluster_id: str
    cercle_id: str
    family: str
    title: str
    symbolic_formula: str
    null_hypothesis: str
    alternative_hypothesis: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    test_statistic_name: str
    test_statistic_value: float
    p_value: float
    q_value: Optional[float] = None
    log_likelihood_ratio: float = 0.0
    verdict: VerdictStatus
    alpha: float = 0.05
    n_observations: int = 0
    derivation_steps: List[str] = Field(default_factory=list)
    kl_bins: List[Dict[str, Any]] = Field(default_factory=list)
    origin: Literal["deterministic", "agent_open"] = "deterministic"
    proposed_by: str = ""
    challenged_by: str = ""
    adjudicated_by: str = ""
    analyst_narrative: str = ""
    updated_at: str = ""


class AgentExchangeTurn(BaseModel):
    turn_index: int
    timestamp: str
    agent_id: str
    agent_name: str
    model_id: str
    role_badge: Literal["Explorer", "Theorist", "Arbiter"]
    target_hyp_id: str = ""
    message: str
    math_refs: Dict[str, Any] = Field(default_factory=dict)
    verdict_after: str = ""


class AgentExchangeRecord(BaseModel):
    exchange_id: str
    sweep_id: str = ""
    updated_at: str = ""
    topic: str = ""
    open_hyp_ids: List[str] = Field(default_factory=list)
    turns: List[AgentExchangeTurn] = Field(default_factory=list)


class CercleMathProofState(BaseModel):
    cercle_id: str
    cercle_name: str
    admin1: str
    lat: float
    lon: float
    event_count: int
    event_timestamps_days: List[float] = Field(default_factory=list)
    hawkes_mu: float
    hawkes_alpha: float
    hawkes_beta: float
    hawkes_branching_ratio: float
    hawkes_half_life_days: float
    hawkes_lambda_now: float
    beta_alpha0: float = 1.0
    beta_beta0: float = 12.0
    beta_active_weeks: int = 0
    beta_total_weeks: int = 52
    beta_posterior_mean: float = 0.0
    beta_ci_low: float = 0.0
    beta_ci_high: float = 1.0
    dispersion_ratio: float = 1.0
    dry_season_rate: float = 0.0
    rainy_season_rate: float = 0.0
    dominant_actor: str = "Unattributed"
    markov_next_tactics: Dict[str, float] = Field(default_factory=dict)
    kl_actor_nats: float = 0.0
    kl_actor_q_bh: float = 1.0
    kl_tactic_nats: float = 0.0
    kl_tactic_q_bh: float = 1.0
    te_inflow_bits: float = 0.0
    te_inflow_q_bh: float = 1.0
    changepoint_date: str = ""
    kl_bins: List[Dict[str, Any]] = Field(default_factory=list)
    risk_7d_prob: float = 0.0
    risk_30d_prob: float = 0.0
    expected_30d_count: float = 0.0
    walk_forward_auroc: float = 0.5
    walk_forward_brier: float = 0.25
    brier_skill_score: float = 0.0
    updated_at: str = ""


class CerclePrediction(BaseModel):
    cercle_id: str
    cercle_name: str
    admin1: str
    lat: float
    lon: float
    risk_7d_prob: float
    risk_30d_prob: float
    expected_30d_count: float
    dominant_actor: str
    predicted_next_tactic: str
    tactic_probs: Dict[str, float] = Field(default_factory=dict)
    ci_low: float
    ci_high: float
    walk_forward_auroc: float
    walk_forward_brier: float
    brier_skill_score: float
    updated_at: str = ""


class ObservatoryMeta(BaseModel):
    sweep_id: str
    updated_at: str
    total_raw_records: int = 0
    total_verified_incidents: int = 0
    total_clusters: int = 0
    total_hypotheses: int = 0
    proven_or_supported_count: int = 0
    walk_forward_folds: int = 0
    global_auroc: float = 0.5
    global_pr_auc: float = 0.0
    global_brier: float = 0.25
    global_bss: float = 0.0
    source_counts: Dict[str, int] = Field(default_factory=dict)
    perspective_counts: Dict[str, int] = Field(default_factory=dict)
    actor_counts: Dict[str, int] = Field(default_factory=dict)
    tactic_counts: Dict[str, int] = Field(default_factory=dict)
    corridor_edges: List[Dict[str, Any]] = Field(default_factory=list)
    active_agents: List[Dict[str, Any]] = Field(default_factory=list)
    agent_exchanges: List[Dict[str, Any]] = Field(default_factory=list)
    running: bool = False
