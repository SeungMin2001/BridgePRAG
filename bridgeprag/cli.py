"""Command line interface for BridgePRAG."""

from __future__ import annotations

import argparse

from .config import BridgePRAGConfig
from .data import load_memory_examples
from .runtime import BridgePRAG
from .trainer import train_bridgeprag


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="Base HF causal LM.")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--num-kv", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--critical-layer", type=int, default=9)
    parser.add_argument("--alpha", type=float, default=1.0)


def cmd_train(args: argparse.Namespace) -> None:
    config = BridgePRAGConfig(
        model_name=args.model,
        num_kv=args.num_kv,
        hidden_dim=args.hidden_dim,
        critical_layer=args.critical_layer,
        alpha=args.alpha,
        question_fusion=args.question_fusion,
        use_contextual_memory=args.use_contextual_memory,
    )
    runtime = BridgePRAG.from_base_model(args.model, config=config, dtype=args.dtype)
    examples = load_memory_examples(args.data, max_rows=args.max_rows)
    stats = train_bridgeprag(
        runtime,
        examples,
        output=args.output,
        epochs=args.epochs,
        lr=args.lr,
        answer_target=args.answer_target,
        max_steps=args.max_steps,
    )
    print(f"saved checkpoint: {args.output}")
    print(stats)


def cmd_infer(args: argparse.Namespace) -> None:
    runtime = BridgePRAG.from_checkpoint(args.checkpoint, model_name=args.model, dtype=args.dtype)
    answer = runtime.generate(args.question, args.passage, max_new_tokens=args.max_new_tokens, alpha=args.alpha)
    print(answer)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bridgeprag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a BridgePRAG HyperKV generator.")
    _add_runtime_args(train)
    train.add_argument("--data", required=True, help="JSONL/JSON memory supervision file.")
    train.add_argument("--output", default="runs/bridgeprag.pt")
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--lr", type=float, default=5e-5)
    train.add_argument("--max-rows", type=int)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--answer-target", choices=["answer", "full_answer"], default="answer")
    train.add_argument(
        "--question-fusion",
        default="kv_adapter",
        choices=["none", "text_concat", "feature_concat", "kv_adapter"],
    )
    train.add_argument("--use-contextual-memory", action="store_true")
    train.set_defaults(func=cmd_train)

    infer = subparsers.add_parser("infer", help="Generate with a trained BridgePRAG checkpoint.")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--model", default=None, help="Override checkpoint base model.")
    infer.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    infer.add_argument("--question", required=True)
    infer.add_argument("--passage", required=True, action="append", help="Passage to encode as HyperKV memory.")
    infer.add_argument("--max-new-tokens", type=int, default=64)
    infer.add_argument("--alpha", type=float)
    infer.set_defaults(func=cmd_infer)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

