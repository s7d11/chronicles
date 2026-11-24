"""Deterministic offline MockAdapter for automated tests and smoke checks."""

from __future__ import annotations

import json
import re
from typing import Dict, List, Sequence, Tuple

from bench import LANGS
from bench.adapters.base import AdapterCapabilities, GenerationResult, ModelAdapter


class MockAdapter(ModelAdapter):
    """Zero-network, deterministic adapter returning valid schemas for MALI13 tasks."""

    def __init__(self, quality_bias: float = 1.0):
        self.quality_bias = quality_bias
        self.calls = 0
        self.capabilities = AdapterCapabilities(
            supports_logprobs=False,
            supports_tokenizer=True,
            supports_bpb=True,
            is_api=False,
            active_params_b=0.6,
        )

    async def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> GenerationResult:
        self.calls += 1
        sys_msg = messages[0]["content"]
        user_msg = messages[-1]["content"]

        if "JSON SCHEMA" in sys_msg:
            m = re.search(r"must be ([a-z]{3})", sys_msg)
            lang = m.group(1) if m else "bam"
            words = [w.strip(".,;:!?") for w in user_msg.split() if w.strip(".,;:!?")]
            toks = [
                {
                    "surface": w,
                    "stage": "0",
                    "analyses": [{"form": w.lower(), "ps": ["n"], "gloss": "mock", "morphemes": [{"form": w.lower(), "ps": ["n"], "gloss": "mock"}]}],
                }
                for w in words
            ]
            txt = json.dumps({"text": user_msg, "lang": lang, "tokens": toks}, ensure_ascii=False)
            return GenerationResult(text=txt, raw_text=txt, latency_s=0.01, prompt_tokens=20, completion_tokens=len(toks) * 8)

        if "13-way Language Identification" in sys_msg:
            # Deterministic guess based on diacritics
            guess = "taq" if "ă" in user_msg else ("ful" if "ɓ" in user_msg else ("dtm" if "ɛ" in user_msg else LANGS[0]))
            txt = json.dumps({"lang": guess})
            return GenerationResult(text=txt, raw_text=txt, latency_s=0.01, prompt_tokens=15, completion_tokens=6)

        if "fill [MASK]" in sys_msg:
            txt = json.dumps({"choice": "A"})
            return GenerationResult(text=txt, raw_text=txt, latency_s=0.01, prompt_tokens=15, completion_tokens=5)

        txt = "U be taa kɛnɛya la ni hɛɛrɛ ye."
        return GenerationResult(text=txt, raw_text=txt, latency_s=0.01, prompt_tokens=12, completion_tokens=10)

    async def count_tokens_batch(self, texts: Sequence[str]) -> Tuple[List[int], str]:
        counts = [max(1, len(t) // 3) for t in texts]
        return counts, "mock"

    def tokenize_pieces(self, word: str) -> Optional[List[str]]:
        return [word[i : i + 2] for i in range(0, len(word), 2)]

    async def compute_bpb(self, texts: Sequence[str]) -> Optional[float]:
        return 1.25
