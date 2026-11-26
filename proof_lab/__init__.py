"""Edition No. 3 Proof Lab: 14-language dataset_300 stress-testing & NVIDIA direct deductions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

DATASET_LANGS: Tuple[str, ...] = (
    "bam",
    "bmq",
    "boz",
    "dtm",
    "ful",
    "hsy",
    "mku",
    "mlq",
    "myk",
    "seq",
    "ses",
    "snk",
    "taq",
    "bbo",
)


@dataclass(frozen=True)
class DatasetLangSpec:
    code: str
    name: str
    family: str
    dabax_baseline: str
    ood_cross_baseline: str
    is_ood: bool
    alias_note: str


LANG_SPECS: Dict[str, DatasetLangSpec] = {
    "bam": DatasetLangSpec("bam", "Bambara (Bamanankan)", "Niger-Congo · Mande", "bam", "taq", False, "Direct MULTI13 anchor baseline"),
    "bmq": DatasetLangSpec("bmq", "Bomu", "Niger-Congo · Gur", "bmq", "taq", False, "Direct MULTI13 Gur baseline"),
    "boz": DatasetLangSpec("boz", "Tiéyaxo Bozo", "Niger-Congo · Mande", "boz", "taq", False, "Direct MULTI13 Bozo baseline"),
    "dtm": DatasetLangSpec("dtm", "Tomo Kan Dogon", "Niger-Congo · Dogon", "dtm", "taq", False, "Direct MULTI13 Dogon baseline"),
    "ful": DatasetLangSpec("ful", "Maasina Fulfulde", "Niger-Congo · Atlantic", "ful", "taq", False, "Direct MULTI13 Atlantic baseline"),
    "hsy": DatasetLangSpec("hsy", "Hassaniya Arabic", "Afro-Asiatic · Semitic", "mey", "bam", False, "dataset_300 code `hsy` mapped to MULTI13 `mey` baseline"),
    "mku": DatasetLangSpec("mku", "Konyanka Maninka", "Niger-Congo · Mande", "mku", "taq", False, "Direct MULTI13 Maninka baseline"),
    "mlq": DatasetLangSpec("mlq", "Western Maninkakan", "Niger-Congo · Mande", "mlq", "taq", False, "Direct DabaX resolution via Maninkakan/Kassonke baseline"),
    "myk": DatasetLangSpec("myk", "Mamara Senufo", "Niger-Congo · Senufo", "myk", "taq", False, "Direct MULTI13 Senufo baseline"),
    "seq": DatasetLangSpec("seq", "Supyire Senufo", "Niger-Congo · Senufo", "spp", "taq", False, "dataset_300 code `seq` resolved via `spp` Senufo baseline"),
    "ses": DatasetLangSpec("ses", "Koyraboro Senni", "Nilo-Saharan · Songhay", "ses", "taq", False, "Direct MULTI13 Songhay baseline"),
    "snk": DatasetLangSpec("snk", "Soninke", "Niger-Congo · Mande", "snk", "taq", False, "Direct MULTI13 Soninke baseline"),
    "taq": DatasetLangSpec("taq", "Tamasheq", "Afro-Asiatic · Berber", "taq", "bam", False, "Direct MULTI13 two-stage Berber baseline"),
    "bbo": DatasetLangSpec("bbo", "Northern Bobo Madaré", "Niger-Congo · Mande (OOD)", "bam", "taq", True, "OOD outlier (`BBO_OUTLIER` in SebeniProofs) zero-shot parsed with `bam` Mande baseline"),
}


@dataclass(frozen=True)
class NimModelSpec:
    slug: str
    display_name: str
    model_id: str
    role: str
    params_label: str
    reasoning_effort: str | None = None
    max_tokens: int = 2400
    timeout_s: float = 150.0


NIM_MODELS: Tuple[NimModelSpec, ...] = (
    NimModelSpec(
        slug="mistral-large-v4",
        display_name="Mistral Large V4",
        model_id="mistralai/mistral-large-4-1t-a52b",
        role="Sparse MoE Frontier Reasoner",
        params_label="52B act / 1T MoE",
        reasoning_effort=None,
        max_tokens=6144,
        timeout_s=150.0,
    ),
    NimModelSpec(
        slug="nemotron-3-ultra-550b",
        display_name="Nemotron-3-Ultra-550B",
        model_id="nvidia/nemotron-3-ultra-550b-a55b",
        role="Coordinator & Formal Deduction",
        params_label="55B act / 550B",
        reasoning_effort="low",
        max_tokens=6144,
        timeout_s=150.0,
    ),
    NimModelSpec(
        slug="minimax-m3-1-flash",
        display_name="Minimax M3.1 Flash",
        model_id="minimaxai/minimax-m3.1-flash",
        role="High-Speed Deduction Panelist",
        params_label="Flash MoE (models.list)",
        reasoning_effort=None,
        max_tokens=6144,
        timeout_s=120.0,
    ),
    NimModelSpec(
        slug="nemotron-3-5-super-120b",
        display_name="Nemotron-3.5-Super-120B",
        model_id="nvidia/nemotron-3.5-super-vl-120b-a12b",
        role="Fast Structured Verifier (Muse EOL sub.)",
        params_label="12B act / 120B",
        reasoning_effort=None,
        max_tokens=6144,
        timeout_s=120.0,
    ),
)

NIM_MODEL_BY_SLUG: Dict[str, NimModelSpec] = {m.slug: m for m in NIM_MODELS}


LEAN_THEOREMS: List[Dict[str, str]] = [
    {
        "file": "SebeniProofs/Basic.lean",
        "name": "stageToPhi_nonneg_bound",
        "statement": "stageToPhiScaled lat s g ≤ phiScale",
        "meaning": "Bounds Φ within [0, 1] across Manding, MULTI13, BBO_OUTLIER, and isolating typologies.",
    },
    {
        "file": "SebeniProofs/Morphotactics.lean",
        "name": "pfvSandhi_unique_licensing",
        "statement": "pfvSandhiLicensed tail s₁ ∧ pfvSandhiLicensed tail s₂ → s₁ = s₂",
        "meaning": "Every stem tail class licenses a unique PFV.INTR allomorph (-na / -la / -ra).",
    },
    {
        "file": "SebeniProofs/Morphotactics.lean",
        "name": "phi_inflation_without_fsa_validity",
        "statement": "fsaAccepts [.infl] = false ∧ stageToPhiScaled mandingBamLattice 0 false = phiScale",
        "meaning": "Raw Φ alone can saturate to 1.0 on invalid bare suffixes unless MorphotacticFSA + strict lookup gate it.",
    },
    {
        "file": "SebeniProofs/Metrics.lean",
        "name": "mcs_shift_collapse_witness",
        "statement": "mcsMatches ((-1) :: [1,2,1,2]) [1,2,1,2] = 0",
        "meaning": "A single leading insertion collapses index-paired MCS to 0 despite preserving all 4 content stages.",
    },
    {
        "file": "SebeniProofs/Metrics.lean",
        "name": "stageLcs_resists_shift_witness",
        "statement": "stageLcs ((-1) :: [1,2,1,2]) [1,2,1,2] = 4 ∧ mcsMatches = 0",
        "meaning": "Order-preserving LCS alignment (aligned_mcs) recovers 100% of aligned stages under insertions/clitics.",
    },
    {
        "file": "SebeniProofs/Metrics.lean",
        "name": "mcs_corpus_denom_asymmetry",
        "statement": "pred.length < ref.length → mcsCorpusDenom pred ref < mcsSentenceDenom pred ref",
        "meaning": "Corpus MCS (|pred| denom) over-credits truncated hypotheses compared to sentence MCS (max(|pred|, |ref|)).",
    },
    {
        "file": "SebeniProofs/Metrics.lean",
        "name": "mer_overgen_exceeds_one_witness",
        "statement": "levDist [\"a\",\"b\"] [\"a\",\"b\",\"x\",\"y\",\"z\"] > 2",
        "meaning": "When |hyp| > 2|ref|, unbounded MER > 1.0 saturates clipped GRPO rewards to zero; bounded_mer fixes this.",
    },
    {
        "file": "SebeniProofs/Metrics.lean",
        "name": "bounded_mer_always_unit_bounded",
        "statement": "levDist ref hyp ≤ max ref.length hyp.length",
        "meaning": "Symmetric denominator max(|ref|, |hyp|) guarantees bounded_mer ∈ [0, 1] without losing error gradient.",
    },
    {
        "file": "SebeniProofs/Metrics.lean",
        "name": "uwec_current_indicator_prefers_unknown",
        "statement": "s ≠ -1 → uwecIndicator (-1) < uwecIndicator s",
        "meaning": "Baseline I(stage != -1) penalizes analyzed tokens more than unknown (-1); inverted I(stage == -1) + KL separates OOD drift.",
    },
    {
        "file": "SebeniProofs/SafetyAndOOD.lean",
        "name": "strictOOD_blocks_mcs_regression",
        "statement": "mcs' < mcs → allowPromoteStrictOOD x mcs mcs' = false",
        "meaning": "Blocks distilling grammars that inflate Φ while regressing morphosyntactic stage consistency.",
    },
]
