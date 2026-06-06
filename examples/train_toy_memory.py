"""Train BridgePRAG on the tiny example dataset."""

from __future__ import annotations

from bridgeprag import BridgePRAG, BridgePRAGConfig, load_memory_examples
from bridgeprag.trainer import train_bridgeprag


def main() -> None:
    config = BridgePRAGConfig(
        model_name="Qwen/Qwen2.5-0.5B",
        num_kv=8,
        hidden_dim=512,
        critical_layer=8,
        alpha=1.0,
    )
    bridge = BridgePRAG.from_base_model(config.model_name, config=config)
    examples = load_memory_examples("examples/data/tiny_memory.jsonl")
    train_bridgeprag(
        bridge,
        examples,
        output="runs/tiny_bridgeprag.pt",
        epochs=1,
        lr=5e-5,
        max_steps=3,
    )


if __name__ == "__main__":
    main()

