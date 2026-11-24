"""Local RTX 3060 4-bit NF4 + PEFT LoRA adapter exclusively for Sebeni.

Streams google/gemma-4-E2B text-only weights directly from safetensors:
- Keeps the 4.7 GB `embed_tokens_per_layer` lookup table on CPU with a 1-line device bridge.
- Quantizes all 35 transformer Linear layers to 4-bit NF4 on `cuda:0` (~2.2 GB total VRAM).
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from beni.core.srl.prompting import strip_end_of_turn
from bench.adapters.base import AdapterCapabilities, GenerationResult, ModelAdapter
from bench.metrics import strip_thinking


def _build_streaming_text_model(base_model_id: str, token: Optional[str]):
    """Load Gemma4ForCausalLM text backbone in 4-bit on cuda:0 + CPU embed_tokens_per_layer."""
    import bitsandbytes as bnb
    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig, Gemma4ForCausalLM

    cfg = AutoConfig.from_pretrained(base_model_id, token=token, trust_remote_code=True)
    text_cfg = getattr(cfg, "text_config", cfg)
    has_cuda = torch.cuda.is_available()
    target_device = torch.device("cuda:0" if has_cuda else "cpu")

    # Initialize skeleton on meta device (0 RAM / 0 VRAM)
    with torch.device("meta"):
        model = Gemma4ForCausalLM(text_cfg)

    src_weights = hf_hub_download(base_model_id, "model.safetensors", token=token)

    def _set_submodule_param(root: nn.Module, param_name: str, tensor: torch.Tensor, quantize_4bit: bool) -> None:
        parts = param_name.split(".")
        mod = root
        for p in parts[:-1]:
            if not hasattr(mod, p):
                return
            mod = getattr(mod, p)
        leaf = parts[-1]
        if not hasattr(mod, leaf):
            return
        if quantize_4bit and has_cuda and isinstance(mod, nn.Linear) and leaf == "weight":
            # Replace nn.Linear with bnb.nn.Linear4bit directly on GPU
            parent_parts = parts[:-2]
            attr_name = parts[-2]
            parent = root
            for p in parent_parts:
                parent = getattr(parent, p)
            bias_val = None
            l4 = bnb.nn.Linear4bit(
                mod.in_features,
                mod.out_features,
                bias=mod.bias is not None,
                compute_dtype=torch.bfloat16,
                compress_statistics=True,
                quant_type="nf4",
            )
            l4.weight = bnb.nn.Params4bit(
                tensor.to(torch.bfloat16).contiguous(),
                requires_grad=False,
                compress_statistics=True,
                quant_type="nf4",
            ).to(target_device)
            if bias_val is not None:
                l4.bias = nn.Parameter(bias_val.to(device=target_device, dtype=torch.bfloat16), requires_grad=False)
            setattr(parent, attr_name, l4)
            return

        dev = torch.device("cpu") if "embed_tokens_per_layer" in param_name else target_device
        dt = torch.bfloat16
        if leaf in getattr(mod, "_buffers", {}):
            mod.register_buffer(leaf, tensor.to(device=dev, dtype=dt), persistent=False)
        else:
            setattr(mod, leaf, nn.Parameter(tensor.to(device=dev, dtype=dt), requires_grad=False))

    # Stream every text tensor directly into GPU 4-bit / CPU lookup table
    with safe_open(src_weights, framework="pt", device="cpu") as f:
        for k in f.keys():
            if not k.startswith("model.language_model."):
                continue
            nk = "model." + k[len("model.language_model.") :]
            t = f.get_tensor(k)
            is_linear_weight = (
                nk.endswith(".weight")
                and "layers." in nk
                and any(seg in nk for seg in (".self_attn.", ".mlp.", ".per_layer_projection."))
            )
            # Only quantize 2D Linear weights (skip 1D norms/scalars)
            _set_submodule_param(model, nk, t, quantize_4bit=(is_linear_weight and t.ndim == 2))

    # Properly materialize Gemma-4 rotary embeddings and embedding scale buffers
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    model.model.rotary_emb = Gemma4TextRotaryEmbedding(text_cfg, device=target_device)

    emb = model.model.embed_tokens
    emb.register_buffer(
        "embed_scale",
        torch.tensor(float(emb.scalar_embed_scale), dtype=torch.float32, device=target_device),
        persistent=False,
    )

    # Tie lm_head.weight to embed_tokens.weight on target_device
    model.lm_head.weight = model.model.embed_tokens.weight

    # Bridge CPU embed_tokens_per_layer lookup seamlessly back to cuda:0
    if hasattr(model.model, "embed_tokens_per_layer"):
        epl = model.model.embed_tokens_per_layer
        epl_dev = torch.device("cpu") if has_cuda else target_device
        epl.register_buffer(
            "embed_scale",
            torch.tensor(float(epl.scalar_embed_scale), dtype=torch.float32, device=epl_dev),
            persistent=False,
        )
        if has_cuda:
            orig_forward = epl.forward

            def _cpu_to_gpu_forward(input_ids: torch.Tensor) -> torch.Tensor:
                out = orig_forward(input_ids.cpu())
                return out.to(device=target_device, dtype=torch.bfloat16)

            epl.forward = _cpu_to_gpu_forward  # type: ignore[method-assign]

    leftover_meta = [n for n, p in model.named_parameters() if p.is_meta] + [
        n for n, b in model.named_buffers() if b.is_meta
    ]
    if leftover_meta:
        raise RuntimeError(f"Unmaterialized meta tensors in Gemma4ForCausalLM: {leftover_meta[:10]}")

    return model


class LocalSebeniAdapter(ModelAdapter):
    """Loads google/gemma-4-E2B text backbone in 4-bit NF4 + mlsftwrs/sebeni-gemma-4-e2b-grpo-50 LoRA."""

    def __init__(
        self,
        *,
        base_model_id: str = "google/gemma-4-E2B",
        lora_repo: str = "mlsftwrs/sebeni-gemma-4-e2b-grpo-50",
        active_params_b: float = 2.0,
    ):
        self.base_model_id = base_model_id
        self.lora_repo = lora_repo
        self.capabilities = AdapterCapabilities(
            supports_logprobs=True,
            supports_tokenizer=True,
            supports_bpb=True,
            is_api=False,
            active_params_b=active_params_b,
        )
        self._model = None
        self._tokenizer = None
        self._peak_vram_mb: Optional[float] = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        import inspect
        import torch
        from huggingface_hub import hf_hub_download
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        token = os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        try:
            tok = AutoTokenizer.from_pretrained(self.lora_repo, token=token, trust_remote_code=True)
        except Exception:
            tok = AutoTokenizer.from_pretrained(self.base_model_id, token=token, trust_remote_code=True)
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token

        base_model = _build_streaming_text_model(self.base_model_id, token)

        cfg_path = hf_hub_download(self.lora_repo, "adapter_config.json", token=token)
        wt_path = hf_hub_download(self.lora_repo, "adapter_model.safetensors", token=token)
        raw_acfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))

        valid_lora_params = set(inspect.signature(LoraConfig.__init__).parameters.keys())
        lora_kwargs = {k: v for k, v in raw_acfg.items() if k in valid_lora_params and v is not None}
        if isinstance(lora_kwargs.get("target_modules"), list):
            lora_kwargs["target_modules"] = [
                m.replace("model.language_model.", "model.") for m in lora_kwargs["target_modules"]
            ]
        lora_cfg = LoraConfig(**lora_kwargs)
        model = get_peft_model(base_model, lora_cfg)

        raw_sd = load_file(wt_path)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        remapped_sd: Dict[str, Any] = {
            k.replace("base_model.model.model.language_model.", "base_model.model.model."): v.to(
                device=device, dtype=torch.bfloat16
            )
            for k, v in raw_sd.items()
        }
        incompatible = set_peft_model_state_dict(model, remapped_sd)
        if getattr(incompatible, "unexpected_keys", None):
            raise RuntimeError(f"Unexpected LoRA keys after remap: {incompatible.unexpected_keys[:5]}")
        model.eval()

        self._tokenizer = tok
        self._model = model
        if torch.cuda.is_available():
            self._peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), 1)

    def _render_prompt(self, messages: List[Dict[str, str]]) -> str:
        self._ensure_loaded()
        tok = self._tokenizer
        if getattr(tok, "chat_template", None):
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        parts = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
        return "\n\n".join(parts) + "\n\nAssistant: "

    def _generate_sync(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> GenerationResult:
        import torch

        self._ensure_loaded()
        prompt = self._render_prompt(messages)
        tok = self._tokenizer
        model = self._model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
        in_len = int(inputs["input_ids"].shape[1])

        eos_ids = sorted(
            {
                tid
                for tid in (tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>"))
                if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id
            }
        )
        t0 = time.perf_counter()
        with torch.inference_mode():
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0.0,
                "pad_token_id": tok.pad_token_id,
                "use_cache": True,
                "stop_strings": ["<|im_end|>", "<|im_start|>", "<turn|>"],
                "tokenizer": tok,
            }
            if eos_ids:
                gen_kwargs["eos_token_id"] = eos_ids
            if temperature > 0.0:
                gen_kwargs["temperature"] = temperature
            out = model.generate(**inputs, **gen_kwargs)
        elapsed = time.perf_counter() - t0
        new_ids = out[0][in_len:]
        raw_decoded = tok.decode(new_ids, skip_special_tokens=False)
        cleaned = strip_end_of_turn(tok.decode(new_ids, skip_special_tokens=True), tok)
        cleaned = strip_thinking(cleaned)
        if torch.cuda.is_available():
            self._peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), 1)
        return GenerationResult(
            text=cleaned,
            raw_text=raw_decoded,
            latency_s=elapsed,
            prompt_tokens=in_len,
            completion_tokens=int(new_ids.shape[0]),
        )

    async def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> GenerationResult:
        return await asyncio.to_thread(self._generate_sync, messages, max_tokens, temperature)

    def _score_cloze_sync(self, messages: List[Dict[str, str]]) -> GenerationResult:
        import torch

        self._ensure_loaded()
        tok = self._tokenizer
        model = self._model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        prompt = self._render_prompt(messages) + '{"choice": "'
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(**inputs)
            logits = out.logits[0, -1, :]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
        elapsed = time.perf_counter() - t0
        scores: Dict[str, float] = {}
        for letter in ("A", "B", "C", "D"):
            ids = tok.encode(letter, add_special_tokens=False)
            scores[letter] = float(log_probs[ids[0]].item()) if ids else -1e9
        best = max(scores.items(), key=lambda kv: kv[1])[0]
        return GenerationResult(
            text=f'{{"choice": "{best}"}}',
            raw_text=f'{{"choice": "{best}"}}',
            latency_s=elapsed,
            prompt_tokens=int(inputs["input_ids"].shape[1]),
            completion_tokens=1,
            cloze_mode="logprob",
            logprob_scores=scores,
        )

    async def score_cloze(
        self,
        messages: List[Dict[str, str]],
        choices: Dict[str, str],
    ) -> GenerationResult:
        return await asyncio.to_thread(self._score_cloze_sync, messages)

    async def count_tokens_batch(self, texts: Sequence[str]) -> Tuple[List[int], str]:
        self._ensure_loaded()
        counts = [len(self._tokenizer.encode(t, add_special_tokens=False)) for t in texts]
        return counts, "exact-hf"

    def tokenize_pieces(self, word: str) -> Optional[List[str]]:
        self._ensure_loaded()
        ids = self._tokenizer.encode(word, add_special_tokens=False)
        return [self._tokenizer.decode([tid], skip_special_tokens=False) or f"[{tid}]" for tid in ids]

    def _compute_bpb_sync(self, texts: Sequence[str]) -> Optional[float]:
        import torch

        self._ensure_loaded()
        tok = self._tokenizer
        model = self._model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        total_nll = 0.0
        total_bytes = 0
        with torch.inference_mode():
            for txt in texts:
                enc = tok(txt, return_tensors="pt", truncation=True, max_length=512)
                ids = enc["input_ids"].to(device)
                n_toks = int(ids.shape[1])
                if n_toks < 2:
                    continue
                out = model(input_ids=ids, labels=ids)
                loss = float(out.loss.item())
                total_nll += loss * (n_toks - 1)
                total_bytes += len(txt.encode("utf-8"))
        if total_bytes <= 0:
            return None
        return round(total_nll / (math.log(2.0) * total_bytes), 4)

    async def compute_bpb(self, texts: Sequence[str]) -> Optional[float]:
        return await asyncio.to_thread(self._compute_bpb_sync, texts)

    def vram_mb(self) -> Optional[float]:
        return self._peak_vram_mb
