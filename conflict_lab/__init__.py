"""Volume II · Edition No. 1: Mali Conflict OSINT Observatory & Mathematical Proof Engine."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

VOL2_STATE_REPO = os.environ.get("VOL2_STATE_REPO", "s7d11/chronicles-vol2-mali")
DEFAULT_FIRESTORE_PROJECT = os.environ.get("VOL2_FIRESTORE_PROJECT", "mlgramv1")
DEFAULT_FIRESTORE_DB = os.environ.get("VOL2_FIRESTORE_DB", "chronicles-live")


@dataclass(frozen=True)
class OsintAgentSpec:
    agent_id: str
    name: str
    role: str
    model_id: str
    provider: str  # nvidia | openrouter_free | vertex | deterministic
    fallback_model_id: str = ""
    fallback_provider: str = ""


COORDINATOR_SPEC = OsintAgentSpec(
    agent_id="a0-coordinator",
    name="Bamako Watch Coordinator",
    role="Cross-Source Coordinator & OSINT Arbiter",
    model_id="mistralai/mistral-large-4-1t-a52b",
    provider="nvidia",
    fallback_model_id="nvidia/nemotron-3.5-super-vl-120b-a12b",
    fallback_provider="nvidia",
)

CRITIC_SPEC = OsintAgentSpec(
    agent_id="a9-adjudicator",
    name="Gemini 3.1 Flash-Lite Critic",
    role="Batched Corroboration & Mathematical Audit Critic",
    model_id=os.environ.get("VOL2_CRITIC_MODEL", "gemini-3.1-flash-lite"),
    provider="vertex",
    fallback_model_id="nvidia/nemotron-3.5-lightning-30b-a3b",
    fallback_provider="nvidia",
)

AGENT_CAST: Tuple[OsintAgentSpec, ...] = (
    COORDINATOR_SPEC,
    OsintAgentSpec(
        agent_id="a1-ucdp-ingestor",
        name="UCDP GED Ingestor",
        role="HDX UCDP Mali Disaggregated Event Ingestor",
        model_id="deterministic",
        provider="deterministic",
    ),
    OsintAgentSpec(
        agent_id="a2-acled-hdx",
        name="ACLED HDX Reconciler",
        role="HDX Mali Conflict Aggregate Benchmark",
        model_id="deterministic",
        provider="deterministic",
    ),
    OsintAgentSpec(
        agent_id="a3-reliefweb-envoy",
        name="ReliefWeb UN Analyst",
        role="ReliefWeb v1 Report Extractor & Digest",
        model_id="baidu/ernie-5.5-flash-preview:free",
        provider="openrouter_free",
        fallback_model_id="nvidia/nemotron-3.5-super-vl-120b-a12b",
        fallback_provider="nvidia",
    ),
    OsintAgentSpec(
        agent_id="a4-gdelt-wire",
        name="GDELT Sahel Wire",
        role="GDELT 2.0 French/English Conflict Stream",
        model_id="nvidia/nemotron-3.5-lightning-30b-a3b",
        provider="nvidia",
        fallback_model_id="nvidia/nemotron-3.5-bolt-e4b",
        fallback_provider="nvidia",
    ),
    OsintAgentSpec(
        agent_id="a5-fama-dirpa",
        name="FAMa DIRPA Sentinel",
        role="Official Malian Armed Forces Portal (fama.mil.ml)",
        model_id="nvidia/nemotron-3.5-super-vl-120b-a12b",
        provider="nvidia",
        fallback_model_id="mistralai/mistral-large-4-1t-a52b",
        fallback_provider="nvidia",
    ),
    OsintAgentSpec(
        agent_id="a6-public-x-osint",
        name="Sahel Public X Scout",
        role="Sanctioned Public X / Press Cross-Reference",
        model_id="recursive/flint-v1.0:free",
        provider="openrouter_free",
        fallback_model_id="nvidia/nemotron-3.5-bolt-e4b",
        fallback_provider="nvidia",
    ),
    OsintAgentSpec(
        agent_id="a7-georesolver",
        name="Sahel Geo-Dedup Reasoner",
        role="Malian Toponym Normalization & (±2d, 15km) Dedup",
        model_id="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        provider="nvidia",
        fallback_model_id="nvidia/nemotron-3.5-lightning-30b-a3b",
        fallback_provider="nvidia",
    ),
    OsintAgentSpec(
        agent_id="a8-math-theorist",
        name="Cluster Hypothesis & Proof Analyst",
        role="Hawkes / NB / Markov / Beta Hypothesis Formulation",
        model_id="stepfun-ai/step-5-preview",
        provider="nvidia",
        fallback_model_id="openrouter/free",
        fallback_provider="openrouter_free",
    ),
    CRITIC_SPEC,
)

# Runtime guard: ONLY $0 OpenRouter free models permitted
ALLOWED_OPENROUTER_FREE_MODELS = frozenset({
    "baidu/ernie-5.5-flash-preview:free",
    "recursive/flint-v1.0:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "openrouter/free",
})
