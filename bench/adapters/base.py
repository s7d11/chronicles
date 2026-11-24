"""Base ModelAdapter protocol and telemetry structures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class AdapterCapabilities:
    supports_logprobs: bool = False
    supports_tokenizer: bool = True
    supports_bpb: bool = False
    is_api: bool = True
    active_params_b: Optional[float] = None
    price_in_per_1m: float = 0.0
    price_out_per_1m: float = 0.0


@dataclass
class GenerationResult:
    text: str
    raw_text: str = ""
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None
    cloze_mode: str = "generate"
    logprob_scores: Optional[Dict[str, float]] = None


class ModelAdapter(ABC):
    """Protocol implemented by Adaption, NVIDIA NIM, Vertex AI, Local Sebeni, and Mock adapters."""

    capabilities: AdapterCapabilities

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> GenerationResult:
        """Execute a single chat completion."""

    async def score_cloze(
        self,
        messages: List[Dict[str, str]],
        choices: Dict[str, str],
    ) -> GenerationResult:
        """Default MCQ scoring via constrained chat generation."""
        return await self.generate(messages, max_tokens=256, temperature=0.0, json_mode=True)

    @abstractmethod
    async def count_tokens_batch(self, texts: Sequence[str]) -> Tuple[List[int], str]:
        """Return (token_counts, mode_label) for a list of strings."""

    def tokenize_pieces(self, word: str) -> Optional[List[str]]:
        """Optional subword pieces for a single word (for Subword Shatter Inspector)."""
        return None

    async def compute_bpb(self, texts: Sequence[str]) -> Optional[float]:
        """Causal bits-per-byte for local HF adapters; None for chat APIs."""
        return None

    def vram_mb(self) -> Optional[float]:
        return None

    async def close(self) -> None:
        return None
