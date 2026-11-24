"""Pydantic v2 schemas for MALI13 splits, checkpoints, scorecards, and website JSON."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class RefToken(BaseModel):
    surface: str
    stage: str
    morphemes: List[str] = Field(default_factory=list)


class MorphSample(BaseModel):
    id: str
    lang: str
    text: str
    source: str = "raw_heldout"
    ref_tokens: List[RefToken] = Field(default_factory=list)
    ref_morphemes: List[str] = Field(default_factory=list)
    ref_stages: List[str] = Field(default_factory=list)
    ref_strict_rate: float
    n_inflected: int


class GenSample(BaseModel):
    id: str
    lang: str
    prompt_prefix: str
    ref_continuation: str
    source: str = "raw_heldout"


class LidSample(BaseModel):
    id: str
    lang: str
    text: str
    source: str = "raw_heldout"


class ClozeSample(BaseModel):
    id: str
    lang: str
    masked_text: str
    target: str
    choices: Dict[str, str]
    answer: str
    distractor_types: List[str] = Field(default_factory=list)


class EfficiencySample(BaseModel):
    id: str
    lang: str
    text: str
    utf8_bytes: int
    n_words: int
    n_dabax_morphemes: int
    ascii_tokens: List[str] = Field(default_factory=list)
    diacritic_tokens: List[str] = Field(default_factory=list)


class SplitManifest(BaseModel):
    tier: str
    seed: int
    anti_leakage_overlap: int
    excluded_hash_count: int
    total_rows: int
    langs: List[str]
    files: Dict[str, Dict[str, Any]]


class ModelSpec(BaseModel):
    slug: str
    display_name: str
    provider: Literal["local_sebeni", "vertex", "adaption", "nim", "mock"]
    model_id: str
    track: Literal["Sebeni", "Google", "Open Frontier"]
    active_params_b: Optional[float] = None
    total_params_b: Optional[float] = None
    price_in_per_1m: float = 0.0
    price_out_per_1m: float = 0.0
    lora_repo: Optional[str] = None
    hf_tokenizer_id: Optional[str] = None
    compute_label: str = ""
    notes: str = ""


class CompletionRecord(BaseModel):
    sample_id: str
    task: Literal["morph", "gen", "lid", "cloze"]
    lang: str
    completion: str = ""
    raw_text: str = ""
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None
    cloze_mode: str = "generate"
    logprob_scores: Optional[Dict[str, float]] = None
    timestamp: str = ""


class EfficiencyStats(BaseModel):
    word_fertility: float
    morpheme_fertility: float
    bytes_per_token: float
    diacritic_penalty: float
    ascii_fertility: float
    diacritic_fertility: float
    bpb: Optional[float] = None
    tokenizer_mode: str = "exact"
    per_lang_fertility: Dict[str, float] = Field(default_factory=dict)


class ModelScorecard(BaseModel):
    slug: str
    id: str
    provider: str
    model_id: str
    track: Literal["Sebeni", "Google", "Open Frontier"]
    trackCls: str
    q: float
    q_min3: float
    q_gap: float
    morph: float
    gen: float
    lid: float
    cloze: float
    fert: float
    morph_fert: float
    bytes_per_tok: float
    diac: float
    ascii_fert: float
    ext_fert: float
    bpb: Optional[float] = None
    lat: float
    tok_s: float
    params: Optional[float] = None
    vram_mb: Optional[float] = None
    cost: float
    compute: str = ""
    tokenizer_mode: str = "exact"
    completion_rate: float = 1.0
    n_errors: int = 0


class LeaderboardPayload(BaseModel):
    generated_at: str
    tier: str
    manifest_sha256: str
    total_samples: int
    langs: List[str]
    lang_names: Dict[str, str]
    models: List[ModelScorecard]
    per_lang: Dict[str, Dict[str, List[float]]]
    lid_confusion: Dict[str, List[List[int]]]
    shatter_samples: Dict[str, Any]
    pending_models: List[Dict[str, str]] = Field(default_factory=list)
