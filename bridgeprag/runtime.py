"""High-level BridgePRAG runtime wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import BridgePRAGConfig
from .memory import (
    BridgeKVGenerator,
    build_prompt,
    deterministic_generation_config,
    encode_merged_memory,
    make_memory_hook,
    model_num_heads,
)


def _torch_dtype(name: str):
    normalized = str(name or "").lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    return torch.float16


def _target_layer(model, index: int):
    body = getattr(model, "model", None)
    layers = getattr(body, "layers", None)
    if layers is None:
        raise ValueError("BridgePRAG currently expects a decoder-only Hugging Face model with model.layers.")
    return layers[int(index)]


class BridgePRAG:
    """Run BridgePRAG memory encoding and generation with a HF causal LM."""

    def __init__(
        self,
        model,
        tokenizer,
        hypernet: BridgeKVGenerator,
        config: BridgePRAGConfig,
        device: torch.device | str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.hypernet = hypernet
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.hypernet.to(self.device)
        self.model.eval()
        self.hypernet.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def from_base_model(
        cls,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        *,
        config: BridgePRAGConfig | None = None,
        dtype: str = "float16",
        device: str | torch.device | None = None,
    ) -> "BridgePRAG":
        config = config or BridgePRAGConfig(model_name=model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=_torch_dtype(dtype),
        )
        d_model = int(model.config.hidden_size)
        feature_dim = d_model * (2 if config.use_contextual_memory else 1)
        hypernet = BridgeKVGenerator(
            d_model=d_model,
            num_kv=config.num_kv,
            hidden_dim=config.hidden_dim,
            feature_dim=feature_dim,
            question_fusion=config.question_fusion,
        )
        return cls(model, tokenizer, hypernet, config, device=device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        model_name: str | None = None,
        dtype: str = "float16",
        device: str | torch.device | None = None,
    ) -> "BridgePRAG":
        payload = torch.load(checkpoint, map_location="cpu")
        config = BridgePRAGConfig.from_dict(payload.get("config"))
        if model_name:
            config.model_name = model_name
        runtime = cls.from_base_model(config.model_name, config=config, dtype=dtype, device=device)
        runtime.hypernet.load_state_dict(payload["hypernet"], strict=True)
        runtime.hypernet.eval()
        return runtime

    def save_checkpoint(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config.to_dict(),
            "hypernet": self.hypernet.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)

    @torch.no_grad()
    def encode(self, passages: list[str] | str, question: str) -> dict[str, torch.Tensor]:
        if isinstance(passages, str):
            passages = [passages]
        return encode_merged_memory(
            self.model,
            self.tokenizer,
            self.hypernet,
            list(passages),
            question,
            self.device,
            self.config,
        )

    @torch.no_grad()
    def generate(
        self,
        question: str,
        passages: list[str] | str,
        *,
        max_new_tokens: int = 64,
        alpha: float | None = None,
    ) -> str:
        memory = self.encode(passages, question)
        prompt = build_prompt(question)
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        layer = _target_layer(self.model, self.config.critical_layer)
        hook = layer.register_forward_hook(
            make_memory_hook(
                memory["K"],
                memory["V"],
                model_num_heads(self.model),
                alpha=self.config.alpha if alpha is None else alpha,
                injection_mode=self.config.injection_mode,
            )
        )
        try:
            output = self.model.generate(
                **encoded,
                generation_config=deterministic_generation_config(self.tokenizer, max_new_tokens),
            )
        finally:
            hook.remove()
        generated = output[0, encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

