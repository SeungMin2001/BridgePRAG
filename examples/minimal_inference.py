"""Minimal BridgePRAG inference example.

Run after training or downloading a checkpoint:

python examples/minimal_inference.py --checkpoint runs/bridgeprag.pt
"""

from __future__ import annotations

import argparse

from bridgeprag import BridgePRAG


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    bridge = BridgePRAG.from_checkpoint(args.checkpoint, model_name=args.model)
    answer = bridge.generate(
        question="What does the KV adapter correct?",
        passages=[
            "The KV adapter applies lightweight linear corrections to generated key and value slots.",
        ],
        max_new_tokens=48,
    )
    print(answer)


if __name__ == "__main__":
    main()

