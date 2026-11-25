"""Edition No. 2 self-organizing polyglot swarm lab for The Sebeni Chronicle."""

from __future__ import annotations

from bench import LANGS, LANG_NAMES

DEFAULT_LANGS = ("taq", "ful", "dtm", "bam")
ALL_LANGS = tuple(LANGS)
ISO_ALIASES = {
    "mlq": "mku",
    "seq": "spp",
}

COORDINATOR_SLUG = "nemotron-3-ultra-550b-a55b"
CLUSTER_SLUGS = (
    "glm-5-3-flash",
    "gemma-4-31b-it",
    "qwen3-8-27b",
    "glm-5-2",
    "nemotron-sea-lion-v4-8-120b-a12b",
    "deepseek-v4-flash",
    "qwen-sea-lion-v5-397b-a17b-plus",
    "gemma-4-26b-a4b-it",
    "qwen3-5-397b",
)

TOPOLOGIES = ("star", "ring", "mesh", "hierarchical", "pipeline")
MIN_CLUSTER_AGENTS = 2
MAX_CLUSTER_AGENTS = 5


def normalize_lang_code(code: str) -> str:
    """Resolve dataset/CSV ISO aliases into Sebeni canonical MULTI13 group codes."""
    c = str(code or "").strip().lower()
    return ISO_ALIASES.get(c, c)
