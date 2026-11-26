"""Edition No. 4 (`Vol. I · No. 4`): The Living Lexicon Village across MALI13."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from swarm_lab import ALL_LANGS, normalize_lang_code

TARGET_DICT_ENTRIES = 50_000
DEFAULT_LAUNCH_LANG = "bam"
ED4_STATE_REPO = "s7d11/chronicles-edition4-state"

LANGUAGE_NATIVE_NAMES: Dict[str, str] = {
    "bam": "Bambara (Bamanankan)",
    "bmq": "Bomu",
    "boz": "Tiéyaxo Bozo",
    "dtm": "Tomo Kan Dogon",
    "ful": "Maasina Fulfulde",
    "kao": "Xaasongaxango (Kassonke)",
    "mey": "Hassaniya Arabic",
    "mku": "Konyanka Maninka (Malinké)",
    "myk": "Mamara Senoufo (Minyanka)",
    "ses": "Koyraboro Senni Songhay",
    "snk": "Soninke (Sarakolé)",
    "spp": "Supyire Senoufo",
    "taq": "Tamasheq (Tuareg)",
}


@dataclass(frozen=True)
class ModelCastSpec:
    agent_id: str
    name: str
    role: str  # orchestrator | villager | specialist | council | critic
    title: str
    model_id: str
    provider: str  # nim | adaption | vertex
    home_zone: str
    domain: str
    seat: int = 0
    color: str = "#8a5a18"


ORCHESTRATOR_SPEC = ModelCastSpec(
    agent_id="mande",
    name="Mande",
    role="orchestrator",
    title="Chief Orchestrator (Baobab Plaza)",
    model_id="nvidia/nemotron-3-ultra-550b-a55b",
    provider="nim",
    home_zone="baobab_plaza",
    domain="Cross-Domain Coordination & Dynamic Staffing",
    color="#1b4d6e",
)

CRITIC_SPEC = ModelCastSpec(
    agent_id="argon-critic",
    name="Flash Critic",
    role="critic",
    title="Primary Standardization Critic (Gemini 3.5 Flash)",
    model_id=os.environ.get("ED4_CRITIC_MODEL", "gemini-3.5-flash"),
    provider="vertex",
    home_zone="critic_bench",
    domain="Final DNAFLA Standardization & Anti-Calque Gate",
    color="#8c1d24",
)

BASE_VILLAGERS: Tuple[ModelCastSpec, ...] = (
    ModelCastSpec(
        agent_id="v1-sugula",
        name="Sugula",
        role="villager",
        title="Market Trader (Sugu)",
        model_id="gemma-4-31b-it",
        provider="adaption",
        home_zone="market_square",
        domain="Commerce, Weights, Currency & Pottery",
        color="#c2781a",
    ),
    ModelCastSpec(
        agent_id="v2-jeli",
        name="Jeli",
        role="villager",
        title="Elder Griot & Proverb Keeper",
        model_id="qwen-SEA-LION-v5-397b-a17b-plus",
        provider="adaption",
        home_zone="baobab_plaza",
        domain="Oral History, Proverbial Discourse & Kinship",
        color="#7b3f1c",
    ),
    ModelCastSpec(
        agent_id="v3-senekela",
        name="Sɛnɛkɛla",
        role="villager",
        title="Agronomist & Savanna Botanist",
        model_id="glm-5.3-flash",
        provider="adaption",
        home_zone="millet_fields",
        domain="Agriculture, Soil, Botany & Monsoon Calendar",
        color="#1e5c3a",
    ),
    ModelCastSpec(
        agent_id="v4-baamogo",
        name="Baa-mɔgɔ",
        role="villager",
        title="Riverbank & Water Keeper",
        model_id="deepseek-v4-flash",
        provider="adaption",
        home_zone="riverbank_well",
        domain="Hydrology, Fishing, Ferrying & Cisterns",
        color="#236478",
    ),
    ModelCastSpec(
        agent_id="v5-somogo",
        name="So-mɔgɔ",
        role="villager",
        title="Compound Host & Mason",
        model_id="nemotron-SEA-LION-v4.8-120b-a12b",
        provider="adaption",
        home_zone="banco_compounds",
        domain="Banco Architecture, Domestic Craft & Hospitality",
        color="#a34828",
    ),
)

COUNCIL_SEATS: Tuple[ModelCastSpec, ...] = (
    ModelCastSpec(
        agent_id="c1-orthography",
        name="Chair Qwen-397B",
        role="council",
        title="Seat 1 · Orthography & DNAFLA Standardization Chair",
        model_id="qwen3.5-397b",
        provider="adaption",
        home_zone="toguna_council",
        domain="DNAFLA/AMALAN Alphabet, Nasalization & NFC Hygiene",
        seat=1,
        color="#181613",
    ),
    ModelCastSpec(
        agent_id="c2-morphotactics",
        name="Architect GLM-5.2",
        role="council",
        title="Seat 2 · Morphotactics & Grammar Architect",
        model_id="glm-5.2",
        provider="adaption",
        home_zone="toguna_council",
        domain="Lemma vs. .gram Rule Boundary & Affix Patterns",
        seat=2,
        color="#2f4858",
    ),
    ModelCastSpec(
        agent_id="c3-mdf-scribe",
        name="Scribe Qwen-27B",
        role="council",
        title="Seat 3 · MDF Lexicographer & Scribe",
        model_id="qwen3.8-27b",
        provider="adaption",
        home_zone="toguna_council",
        domain="MDF \\lx/\\va/\\ps/\\ge/\\gf/\\xv/\\xf Structure & Glossing",
        seat=3,
        color="#5c4727",
    ),
    ModelCastSpec(
        agent_id="c4-dabax-verifier",
        name="Verifier Gemma-26B",
        role="council",
        title="Seat 4 · Corpus & DabaX Verifier",
        model_id="gemma-4-26b-a4b-it",
        provider="adaption",
        home_zone="toguna_council",
        domain="DabaX Strict Parse Compilation & Zero Regression",
        seat=4,
        color="#1e5c3a",
    ),
)

# All 9 live Adaption Labs model IDs used across the 5 villagers, 4 Council seats, and dynamic specialists
ALL_ADAPTION_MODELS: Tuple[str, ...] = tuple(
    [v.model_id for v in BASE_VILLAGERS] + [c.model_id for c in COUNCIL_SEATS]
)

SPECIALIST_TEMPLATES: Tuple[ModelCastSpec, ...] = (
    ModelCastSpec(
        agent_id="sp-numu",
        name="Numu",
        role="specialist",
        title="Master Blacksmith (Forge)",
        model_id="qwen3.5-397b",
        provider="adaption",
        home_zone="artisan_forge",
        domain="Metallurgy, Smelting, Iron Implements & Guild Taboos",
        color="#b5441c",
    ),
    ModelCastSpec(
        agent_id="sp-furabola",
        name="Furabɔla",
        role="specialist",
        title="Traditional Herbalist & Healer",
        model_id="gemma-4-26b-a4b-it",
        provider="adaption",
        home_zone="millet_fields",
        domain="Medicinal Bark, Roots, Remedies & Ethnobotany",
        color="#2c7a4b",
    ),
    ModelCastSpec(
        agent_id="sp-gesedala",
        name="Gesedala",
        role="specialist",
        title="Bògòlan & Cotton Loom Weaver",
        model_id="glm-5.2",
        provider="adaption",
        home_zone="banco_compounds",
        domain="Weaving, Mud-Cloth Dyeing & Textile Crafts",
        color="#7b5221",
    ),
    ModelCastSpec(
        agent_id="sp-somono",
        name="Somono",
        role="specialist",
        title="River Fisher & Canoewright",
        model_id="qwen3.8-27b",
        provider="adaption",
        home_zone="riverbank_well",
        domain="Ichthyology, Nets, Dugout Pirogues & Flood Seasons",
        color="#1f6f8b",
    ),
    ModelCastSpec(
        agent_id="sp-koroba",
        name="Kɔrɔba",
        role="specialist",
        title="Customary Civic Elder",
        model_id="gemma-4-31b-it",
        provider="adaption",
        home_zone="baobab_plaza",
        domain="Customary Law, Mediation, Land Tenure & Markets",
        color="#6a3b52",
    ),
)

VILLAGE_ZONES: Dict[str, Dict[str, object]] = {
    "toguna_council": {
        "label": "Council Toguna (Palaver Pavilion)",
        "x": 0.0,
        "z": -3.5,
        "neighbors": ["baobab_plaza", "critic_bench", "market_square", "banco_compounds"],
    },
    "critic_bench": {
        "label": "High Critic Bench (Argon)",
        "x": -4.5,
        "z": -4.5,
        "neighbors": ["toguna_council", "baobab_plaza"],
    },
    "baobab_plaza": {
        "label": "Central Baobab Plaza",
        "x": 0.0,
        "z": 2.2,
        "neighbors": ["toguna_council", "critic_bench", "market_square", "banco_compounds", "artisan_forge", "riverbank_well", "millet_fields"],
    },
    "market_square": {
        "label": "Market Square (Sugu)",
        "x": 9.5,
        "z": 3.5,
        "neighbors": ["baobab_plaza", "toguna_council", "artisan_forge"],
    },
    "banco_compounds": {
        "label": "Banco Compounds & Granaries",
        "x": -10.5,
        "z": 2.5,
        "neighbors": ["baobab_plaza", "toguna_council", "millet_fields"],
    },
    "artisan_forge": {
        "label": "Numu Artisan Forge",
        "x": 11.5,
        "z": -6.0,
        "neighbors": ["market_square", "baobab_plaza"],
    },
    "millet_fields": {
        "label": "Sahelian Millet & Sorghum Fields",
        "x": -11.5,
        "z": -8.0,
        "neighbors": ["banco_compounds", "baobab_plaza", "riverbank_well"],
    },
    "riverbank_well": {
        "label": "Niger Tributary & Stone Well",
        "x": 3.5,
        "z": 12.5,
        "neighbors": ["baobab_plaza", "market_square", "millet_fields"],
    },
}

SEMANTIC_DOMAINS: Tuple[str, ...] = (
    "Agriculture & Savanna Botany",
    "Market Trade, Weights & Pottery",
    "Blacksmithing, Carpentry & Weaving",
    "Banco Masonry & Domestic Crafts",
    "Hydrology, Fishing & Monsoon Seasons",
    "Kinship, Hospitality & Proverbs",
    "Ethnomedicine & Herbal Lore",
    "Civic Discourse, Mediation & Law",
    "Culinary Grain, Shea & Spices",
    "Flora, Fauna & Pastoral Ecology",
    "Timekeeping, Astronomy & Travel",
    "Abstract Stative Verbs & Ideophones",
)

__all__ = [
    "ALL_ADAPTION_MODELS",
    "ALL_LANGS",
    "BASE_VILLAGERS",
    "COUNCIL_SEATS",
    "CRITIC_SPEC",
    "DEFAULT_LAUNCH_LANG",
    "ED4_STATE_REPO",
    "LANGUAGE_NATIVE_NAMES",
    "ModelCastSpec",
    "ORCHESTRATOR_SPEC",
    "SEMANTIC_DOMAINS",
    "SPECIALIST_TEMPLATES",
    "TARGET_DICT_ENTRIES",
    "VILLAGE_ZONES",
    "normalize_lang_code",
]
