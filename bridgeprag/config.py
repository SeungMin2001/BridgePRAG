"""Configuration objects for BridgePRAG."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class BridgePRAGConfig:
    """Hyperparameters for FiD-inspired HyperKV memory.

    The defaults match the compact public research setup. Larger experiments can
    use more slots, a later critical layer, and a larger base model.
    """

    model_name: str = "Qwen/Qwen2.5-0.5B"
    num_kv: int = 16
    hidden_dim: int = 1024
    alpha: float = 1.0
    max_memory_tokens: int = 256
    max_seq_len: int = 512
    critical_layer: int = 9
    question_conditioned_memory: bool = True
    question_fusion: str = "kv_adapter"
    use_contextual_memory: bool = False
    injection_mode: str = "attention"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "BridgePRAGConfig":
        if not values:
            return cls()
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

