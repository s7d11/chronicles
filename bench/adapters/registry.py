"""Model registry mapping models.list and Google Vertex/Gemma models to adapters."""

from __future__ import annotations

from typing import Dict, List

from bench.schema import ModelSpec


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "sebeni-gemma-4-e2b-grpo-50": ModelSpec(
        slug="sebeni-gemma-4-e2b-grpo-50",
        display_name="Sebeni-Gemma-4-E2B-GRPO-50",
        provider="local_sebeni",
        model_id="google/gemma-4-E2B",
        lora_repo="mlsftwrs/sebeni-gemma-4-e2b-grpo-50",
        hf_tokenizer_id="mlsftwrs/sebeni-gemma-4-e2b-grpo-50",
        track="Sebeni",
        active_params_b=2.0,
        total_params_b=2.0,
        compute_label="Local RTX 3060 (4-bit LoRA)",
    ),
    "gemini-3-1-pro-preview": ModelSpec(
        slug="gemini-3-1-pro-preview",
        display_name="Gemini 3.1 Pro",
        provider="vertex",
        model_id="gemini-3.1-pro-preview",
        track="Google",
        price_in_per_1m=2.00,
        price_out_per_1m=12.00,
        compute_label="Vertex AI (gravityx)",
    ),
    "gemini-3-flash-preview": ModelSpec(
        slug="gemini-3-flash-preview",
        display_name="Gemini 3 Flash",
        provider="vertex",
        model_id="gemini-3-flash-preview",
        track="Google",
        price_in_per_1m=0.50,
        price_out_per_1m=3.00,
        compute_label="Vertex AI (gravityx)",
    ),
    "gemma-4-31b-it": ModelSpec(
        slug="gemma-4-31b-it",
        display_name="gemma-4-31b-it",
        provider="adaption",
        model_id="gemma-4-31b-it",
        hf_tokenizer_id="google/gemma-4-31b-it",
        track="Google",
        active_params_b=31.0,
        total_params_b=31.0,
        compute_label="Adaption API · 31B",
    ),
    "gemma-4-26b-a4b-it": ModelSpec(
        slug="gemma-4-26b-a4b-it",
        display_name="gemma-4-26b-a4b-it",
        provider="adaption",
        model_id="gemma-4-26b-a4b-it",
        hf_tokenizer_id="google/gemma-4-26B-A4B-it",
        track="Google",
        active_params_b=4.0,
        total_params_b=26.0,
        compute_label="Adaption API · 4B act / 26B",
    ),
    "qwen-sea-lion-v5-397b-a17b-plus": ModelSpec(
        slug="qwen-sea-lion-v5-397b-a17b-plus",
        display_name="qwen-SEA-LION-v5-397b-a17b+",
        provider="adaption",
        model_id="qwen-SEA-LION-v5-397b-a17b-plus",
        hf_tokenizer_id="Qwen/Qwen3.5-397B-A17B",
        track="Open Frontier",
        active_params_b=17.0,
        total_params_b=397.0,
        compute_label="Adaption API · 17B act / 397B",
    ),
    "nemotron-sea-lion-v4-8-120b-a12b": ModelSpec(
        slug="nemotron-sea-lion-v4-8-120b-a12b",
        display_name="nemotron-SEA-LION-v4.8-120b",
        provider="adaption",
        model_id="nemotron-SEA-LION-v4.8-120b-a12b",
        hf_tokenizer_id="aisingapore/Nemotron-SEA-LION-v4.8-120B-A12B",
        track="Open Frontier",
        active_params_b=12.0,
        total_params_b=120.0,
        compute_label="Adaption API · 12B act / 120B",
    ),
    "glm-5-3-flash": ModelSpec(
        slug="glm-5-3-flash",
        display_name="glm-5.3-flash",
        provider="adaption",
        model_id="glm-5.3-flash",
        hf_tokenizer_id="zai-org/GLM-5.3-Flash",
        track="Open Frontier",
        compute_label="Adaption API",
    ),
    "glm-5-2": ModelSpec(
        slug="glm-5-2",
        display_name="glm-5.2",
        provider="adaption",
        model_id="glm-5.2",
        hf_tokenizer_id="zai-org/GLM-5.2",
        track="Open Frontier",
        compute_label="Adaption API",
    ),
    "deepseek-v4-flash": ModelSpec(
        slug="deepseek-v4-flash",
        display_name="deepseek-v4-flash",
        provider="adaption",
        model_id="deepseek-v4-flash",
        hf_tokenizer_id="deepseek-ai/DeepSeek-V4-Flash",
        track="Open Frontier",
        compute_label="Adaption API",
    ),
    "qwen3-8-27b": ModelSpec(
        slug="qwen3-8-27b",
        display_name="qwen3.8-27b",
        provider="adaption",
        model_id="qwen3.8-27b",
        hf_tokenizer_id="Qwen/Qwen3.8-27B",
        track="Open Frontier",
        active_params_b=27.0,
        total_params_b=27.0,
        compute_label="Adaption API · 27B",
    ),
    "qwen3-5-397b": ModelSpec(
        slug="qwen3-5-397b",
        display_name="qwen3.5-397b",
        provider="adaption",
        model_id="qwen3.5-397b",
        hf_tokenizer_id="Qwen/Qwen3.5-397B-A17B",
        track="Open Frontier",
        active_params_b=17.0,
        total_params_b=397.0,
        compute_label="Adaption API · 17B act / 397B",
    ),
    "mistral-large-v4": ModelSpec(
        slug="mistral-large-v4",
        display_name="Mistral Large V4",
        provider="nim",
        model_id="mistralai/mistral-large-4-1t-a52b",
        hf_tokenizer_id="mistralai/Mistral-Large-4-1T-A52B",
        track="Open Frontier",
        active_params_b=52.0,
        total_params_b=1000.0,
        compute_label="NVIDIA NIM · 52B act / 1T",
    ),
    "minimax-m3": ModelSpec(
        slug="minimax-m3",
        display_name="Minimax M3",
        provider="nim",
        model_id="minimaxai/minimax-m3.1-flash",
        hf_tokenizer_id="MiniMaxAI/MiniMax-M3.1-Flash",
        track="Open Frontier",
        compute_label="NVIDIA NIM",
    ),
    "muse-glimmer-30": ModelSpec(
        slug="muse-glimmer-30",
        display_name="Muse Glimmer 30",
        provider="nim",
        model_id="meta/muse-glimmer-30b",
        hf_tokenizer_id="meta-models/Muse-Glimmer-30B",
        track="Open Frontier",
        active_params_b=30.0,
        total_params_b=30.0,
        compute_label="NVIDIA NIM · 30B",
    ),
    "nemotron-3-ultra-550b-a55b": ModelSpec(
        slug="nemotron-3-ultra-550b-a55b",
        display_name="nemotron-3-ultra-550b-a55b",
        provider="nim",
        model_id="nvidia/nemotron-3-ultra-550b-a55b",
        hf_tokenizer_id="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
        track="Open Frontier",
        active_params_b=55.0,
        total_params_b=550.0,
        compute_label="NVIDIA NIM · 55B act / 550B",
    ),
}


def get_model_spec(slug: str) -> ModelSpec:
    if slug not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model slug '{slug}'. Available: {', '.join(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[slug]


def list_model_specs(selector: str = "all") -> List[ModelSpec]:
    """Expand 'all', 'all-api', provider name, track name, or comma-separated slugs."""
    sel = selector.strip()
    if sel == "all":
        return list(MODEL_REGISTRY.values())
    if sel == "all-api":
        return [m for m in MODEL_REGISTRY.values() if m.provider != "local_sebeni"]
    if sel in {"local_sebeni", "vertex", "adaption", "nim"}:
        return [m for m in MODEL_REGISTRY.values() if m.provider == sel]
    if sel.lower() == "google":
        return [m for m in MODEL_REGISTRY.values() if m.track == "Google"]
    slugs = [s.strip() for s in sel.split(",") if s.strip()]
    return [get_model_spec(s) for s in slugs]
