"""Pydantic schemas for Edition No. 3 stress tests, direct deductions & publishing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RegimeStats(BaseModel):
    regime_id: str
    label: str
    theorem_ref: str
    n_sentences: int
    mean_mer_raw: float
    mean_mer_bounded: float
    mer_gt_one_pct: float
    mean_mcs_index: float
    mean_mcs_aligned: float
    mean_mcs_corpus: float
    mean_uwec_orig: float
    mean_uwec_inv: float
    mean_phi: float
    mean_strict_parse: float
    strict_ood_reject_pct: float = 0.0


class SpotlightExample(BaseModel):
    idx: int
    text: str
    tokens_preview: List[Dict[str, Any]]
    ref_morphemes: List[str]
    ref_stages: List[int]
    fsa_tags: List[str]
    fsa_valid: bool
    ood_cross_stages: List[int]
    ood_cross_morphemes: List[str]
    clean_phi: float
    clean_strict_parse: float
    ood_strict_parse: float


class CorpusStressResult(BaseModel):
    lang: str
    lang_name: str
    family: str
    dabax_baseline: str
    ood_cross_baseline: str
    is_ood: bool
    alias_note: str
    n_samples: int
    n_tokens_total: int
    clean_phi_pct: float
    clean_strict_parse_pct: float
    empr_token_pct: float
    diacritic_word_pct: float
    cohens_d_mer_corrupt: float
    cohens_d_mcs_shift: float
    cohens_d_uwec_ood: float
    ood_strict_gap_pct: float
    shift_mcs_recovery_pct: float
    mer_saturation_prevented_pct: float
    truncation_denom_bias_pct: float
    regimes: Dict[str, RegimeStats]
    spotlights: List[SpotlightExample]


class OracleProbeResult(BaseModel):
    probe_name: str
    perturbed_text: str
    expected_regime: str
    model_predicted_direction: str
    oracle_baseline: str
    oracle_mer_raw: float
    oracle_mer_bounded: float
    oracle_mcs_index: float
    oracle_mcs_aligned: float
    oracle_uwec_inv: float
    oracle_strict_parse: float
    direction_verified: bool


class ModelDeductionResult(BaseModel):
    model_slug: str
    model_name: str
    model_id: str
    params_label: str
    round1_latency_s: float
    round2_latency_s: float
    round1_error: Optional[str] = None
    round2_error: Optional[str] = None
    round1_reasoning: str = ""
    round1_summary: str = ""
    probes: List[OracleProbeResult] = Field(default_factory=list)
    oracle_verified_count: int = 0
    oracle_total_count: int = 0
    round2_reasoning: str = ""
    verdict: str = "PENDING"
    effectiveness_score: float = 0.0
    mer_assessment: str = ""
    mcs_assessment: str = ""
    uwec_assessment: str = ""
    lean_theorems_cited: List[str] = Field(default_factory=list)
    key_evidence: str = ""
    recommended_hardening: str = ""


class LanguageProofResult(BaseModel):
    lang: str
    generated_at: str
    corpus: CorpusStressResult
    models: List[ModelDeductionResult]
    consensus_effectiveness: float
    consensus_verdict: str
    oracle_pass_rate_pct: float
    n_events: int


class ProofEvent(BaseModel):
    seq: int
    ts: str
    lang: str
    kind: str
    model_slug: str
    model_name: str
    label: str
    summary: str
    reasoning_excerpt: str = ""
    metrics: Dict[str, Any] = Field(default_factory=dict)


class Edition3Summary(BaseModel):
    edition: str = "Vol. I · No. 3"
    generated_at: str
    dataset_name: str = "dataset_300_samples.jsonl"
    langs: List[str]
    total_sentences: int
    total_events: int
    muse_eol_note: str
    lean_theorems: List[Dict[str, str]]
    model_leaderboard: List[Dict[str, Any]]
    global_metrics: Dict[str, Any]
    languages: Dict[str, LanguageProofResult]
