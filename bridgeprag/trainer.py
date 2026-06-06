"""Compact trainer for BridgePRAG HyperKV memory."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from .data import MemoryExample
from .memory import answer_loss, encode_memory, make_memory_hook, model_num_heads, tokenize_qa
from .runtime import BridgePRAG, _target_layer


def train_bridgeprag(
    runtime: BridgePRAG,
    examples: list[MemoryExample],
    *,
    output: str | Path,
    epochs: int = 1,
    lr: float = 5e-5,
    answer_target: str = "answer",
    max_steps: int | None = None,
) -> dict:
    """Train only the BridgePRAG HyperKV generator.

    The base LLM remains frozen. This function is intentionally small so users
    can adapt it to new datasets without reverse engineering a large script.
    """

    if not examples:
        raise ValueError("Need at least one MemoryExample.")

    runtime.model.eval()
    runtime.hypernet.train()
    for param in runtime.model.parameters():
        param.requires_grad = False

    optimizer = AdamW(runtime.hypernet.parameters(), lr=lr)
    target_layer = _target_layer(runtime.model, runtime.config.critical_layer)
    num_heads = model_num_heads(runtime.model)
    total_steps = 0
    history = []

    for epoch in range(int(epochs)):
        progress = tqdm(examples, desc=f"epoch {epoch + 1}/{epochs}")
        for example in progress:
            memory = encode_memory(
                runtime.model,
                runtime.tokenizer,
                runtime.hypernet,
                example.passage,
                example.question,
                runtime.device,
                runtime.config,
            )
            tok = tokenize_qa(
                runtime.tokenizer,
                example.question,
                example.target_answer(answer_target),
                runtime.device,
                runtime.config.max_seq_len,
            )
            hook = target_layer.register_forward_hook(
                make_memory_hook(
                    memory["K"],
                    memory["V"],
                    num_heads,
                    runtime.config.alpha,
                    runtime.config.injection_mode,
                )
            )
            try:
                logits = runtime.model(
                    input_ids=tok["input_ids"],
                    attention_mask=tok["attention_mask"],
                    use_cache=False,
                ).logits
            finally:
                hook.remove()

            loss = answer_loss(logits, tok["labels"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_steps += 1
            loss_value = float(loss.detach().cpu())
            history.append({"step": total_steps, "loss": loss_value})
            progress.set_postfix(loss=f"{loss_value:.4f}")

            if max_steps is not None and total_steps >= max_steps:
                runtime.save_checkpoint(output, extra={"history": history})
                return {"steps": total_steps, "loss": loss_value}

    runtime.save_checkpoint(output, extra={"history": history})
    return {"steps": total_steps, "loss": history[-1]["loss"]}

