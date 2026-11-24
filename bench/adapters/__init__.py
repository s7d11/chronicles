"""Unified ModelAdapter exports."""

from bench.adapters.base import AdapterCapabilities, GenerationResult, ModelAdapter
from bench.adapters.registry import MODEL_REGISTRY, get_model_spec, list_model_specs

__all__ = [
    "AdapterCapabilities",
    "GenerationResult",
    "ModelAdapter",
    "MODEL_REGISTRY",
    "get_model_spec",
    "list_model_specs",
]
