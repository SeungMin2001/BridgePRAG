"""BridgePRAG HyperKV memory and injection utilities."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationConfig

from .config import BridgePRAGConfig


class BridgeKVGenerator(nn.Module):
    """Generate slot-wise K/V memory from a question-conditioned passage.

    BridgePRAG starts from the MergePRAG HyperKV idea but changes the encoding
    boundary: a memory slot is produced from a FiD-like question-passage encoder
    input, then corrected by lightweight linear K/V adapters.
    """

    def __init__(
        self,
        d_model: int,
        num_kv: int = 16,
        hidden_dim: int = 1024,
        feature_dim: int | None = None,
        question_fusion: str = "kv_adapter",
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_kv = int(num_kv)
        self.feature_dim = int(feature_dim or d_model)
        self.question_fusion = question_fusion or "none"
        if self.question_fusion not in {"none", "text_concat", "feature_concat", "kv_adapter"}:
            raise ValueError(f"Unsupported question_fusion={self.question_fusion!r}")

        if self.question_fusion == "feature_concat":
            self.question_feature_fusion = nn.Sequential(
                nn.LayerNorm(self.feature_dim * 2),
                nn.Linear(self.feature_dim * 2, self.feature_dim),
                nn.GELU(),
                nn.LayerNorm(self.feature_dim),
            )

        self.input_norm = nn.LayerNorm(self.feature_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(self.feature_dim, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.att_pool = nn.Linear(self.d_model, self.num_kv)
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.linear_k = nn.Linear(hidden_dim, self.d_model)
        self.linear_v = nn.Linear(hidden_dim, self.d_model)

        if self.question_fusion == "kv_adapter":
            self.k_adapter = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.d_model))
            self.v_adapter = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.d_model))

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        question_features: torch.Tensor | None = None,
        max_memory_tokens: int = 256,
    ) -> dict[str, torch.Tensor]:
        if max_memory_tokens > 0 and features.size(1) > max_memory_tokens:
            features = features[:, :max_memory_tokens]
            attention_mask = attention_mask[:, :max_memory_tokens] if attention_mask is not None else None

        if question_features is not None:
            if self.question_fusion != "feature_concat":
                raise ValueError("question_features require question_fusion='feature_concat'.")
            if question_features.dim() == 2:
                question_features = question_features.unsqueeze(1)
            if question_features.size(1) == 1:
                question_features = question_features.expand(-1, features.size(1), -1)
            features = self.question_feature_fusion(torch.cat([features, question_features.to(features.dtype)], dim=-1))

        encoded = self.input_proj(self.input_norm(features))
        scores = self.att_pool(encoded)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(-1) == 0, torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores, dim=1)
        pooled = torch.einsum("btd,bts->bsd", encoded, weights)
        hidden = self.mlp(pooled)
        key = self.linear_k(hidden)
        value = self.linear_v(hidden)

        result = {
            "encoded": encoded,
            "pooled": pooled,
            "hidden": hidden,
            "K_pre_adapter": key,
            "V_pre_adapter": value,
            "att_weights": weights,
        }
        if self.question_fusion == "kv_adapter":
            key = key + self.k_adapter(key)
            value = value + self.v_adapter(value)
        result.update({"K": key, "V": value})
        return result


def deterministic_generation_config(tokenizer, max_new_tokens: int) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )


def build_memory_text(passage: str, question: str | None, config: BridgePRAGConfig) -> str:
    if question and config.question_conditioned_memory and config.question_fusion in {"text_concat", "kv_adapter"}:
        return f"Question: {question}\nPassage: {passage}"
    return passage


def _tokenize(tokenizer, text: str, device, max_length: int) -> dict[str, torch.Tensor]:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=False)
    return {key: value.to(device) for key, value in encoded.items()}


def _model_body(model):
    body = getattr(model, "model", None)
    return body if body is not None else model


@torch.no_grad()
def passage_features(model, tokenizer, passage: str, question: str | None, device, config: BridgePRAGConfig):
    memory_text = build_memory_text(passage, question, config)
    encoded = _tokenize(tokenizer, memory_text, device, config.max_seq_len)
    embeddings = model.get_input_embeddings()(encoded["input_ids"]).to(dtype=torch.float32)
    if not config.use_contextual_memory:
        return embeddings, encoded.get("attention_mask"), encoded["input_ids"]

    outputs = _model_body(model)(
        input_ids=encoded["input_ids"],
        attention_mask=encoded.get("attention_mask"),
        use_cache=False,
        return_dict=True,
    )
    contextual = outputs.last_hidden_state.to(dtype=torch.float32)
    return torch.cat([embeddings, contextual], dim=-1), encoded.get("attention_mask"), encoded["input_ids"]


def masked_mean(features: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    if attention_mask is None:
        return features.mean(dim=1)
    mask = attention_mask.to(device=features.device, dtype=features.dtype).unsqueeze(-1)
    return (features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def encode_memory(
    model,
    tokenizer,
    hypernet: BridgeKVGenerator,
    passage: str,
    question: str | None,
    device,
    config: BridgePRAGConfig,
) -> dict[str, torch.Tensor]:
    features, attention_mask, input_ids = passage_features(model, tokenizer, passage, question, device, config)
    question_features = None
    if question and config.question_fusion == "feature_concat":
        q_features, q_mask, _ = passage_features(model, tokenizer, question, None, device, config)
        question_features = masked_mean(q_features, q_mask)
    memory = hypernet(
        features,
        attention_mask=attention_mask,
        question_features=question_features,
        max_memory_tokens=config.max_memory_tokens,
    )
    memory["input_ids"] = input_ids
    memory["attention_mask"] = attention_mask
    return memory


def orthogonal_merge_slots(base: torch.Tensor | None, update: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Merge slots by adding only the update component orthogonal to existing slots."""

    if base is None:
        return update
    if base.shape != update.shape:
        raise ValueError(f"Shape mismatch: {base.shape} vs {update.shape}")
    squeezed = False
    if base.dim() == 2:
        base = base.unsqueeze(0)
        update = update.unsqueeze(0)
        squeezed = True
    merged = []
    for existing, incoming in zip(base, update):
        existing_cols = existing.transpose(0, 1).float()
        incoming_cols = incoming.transpose(0, 1).float()
        q_existing, _ = torch.linalg.qr(existing_cols, mode="reduced")
        projection = q_existing @ (q_existing.transpose(0, 1) @ incoming_cols)
        fused = existing_cols + (incoming_cols - projection)
        merged.append(fused.transpose(0, 1).to(dtype=update.dtype))
    out = torch.stack(merged, dim=0)
    return out.squeeze(0) if squeezed else out


def merge_memory_dicts(memories: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not memories:
        raise ValueError("Need at least one memory.")
    if len(memories) == 1:
        return dict(memories[0])
    key = None
    value = None
    for memory in memories:
        key = orthogonal_merge_slots(key, memory["K"])
        value = orthogonal_merge_slots(value, memory["V"])
    merged = dict(memories[0])
    merged.update({"K": key, "V": value, "merged_count": torch.tensor(len(memories))})
    return merged


def encode_merged_memory(
    model,
    tokenizer,
    hypernet: BridgeKVGenerator,
    passages: list[str],
    question: str | None,
    device,
    config: BridgePRAGConfig,
) -> dict[str, torch.Tensor]:
    memories = [
        encode_memory(model, tokenizer, hypernet, passage, question, device, config)
        for passage in passages
        if str(passage or "").strip()
    ]
    return merge_memory_dicts(memories)


def cross_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, num_heads: int) -> torch.Tensor:
    if query.dim() == 2:
        query = query.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    if key.dim() == 2:
        key = key.unsqueeze(0)
    if value.dim() == 2:
        value = value.unsqueeze(0)
    batch, query_len, d_model = query.shape
    head_dim = d_model // num_heads
    if d_model % num_heads != 0:
        raise ValueError(f"d_model={d_model} is not divisible by num_heads={num_heads}")
    query_h = query.view(batch, query_len, num_heads, head_dim).transpose(1, 2)
    key_h = key.view(batch, key.shape[1], num_heads, head_dim).transpose(1, 2)
    value_h = value.view(batch, value.shape[1], num_heads, head_dim).transpose(1, 2)
    scores = (query_h @ key_h.transpose(-2, -1)) / math.sqrt(head_dim)
    output = scores.softmax(dim=-1) @ value_h
    output = output.transpose(1, 2).contiguous().view(batch, query_len, d_model)
    return output.squeeze(0) if squeeze else output


def model_num_heads(model) -> int:
    return int(getattr(model.config, "num_attention_heads"))


def make_memory_hook(key: torch.Tensor, value: torch.Tensor, num_heads: int, alpha: float, injection_mode: str):
    def hook_fn(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        key_local = key.to(device=hidden.device, dtype=hidden.dtype)
        value_local = value.to(device=hidden.device, dtype=hidden.dtype)
        if injection_mode == "attention":
            new_hidden = hidden + alpha * cross_attention(hidden, key_local, value_local, num_heads)
        elif injection_mode == "add_last":
            bias = value_local.mean(dim=1, keepdim=True)
            new_hidden = hidden.clone()
            new_hidden[:, -1:, :] = new_hidden[:, -1:, :] + alpha * bias
        else:
            raise ValueError(f"Unsupported injection_mode={injection_mode!r}")
        if isinstance(output, tuple):
            return (new_hidden,) + output[1:]
        return new_hidden

    return hook_fn


def build_prompt(question: str, answer: str = "") -> str:
    prompt = f"Question: {question}\nAnswer:"
    return f"{prompt} {answer}" if answer else prompt


def tokenize_qa(tokenizer, question: str, answer: str, device, max_seq_len: int) -> dict[str, torch.Tensor]:
    prompt = build_prompt(question)
    target = f" {answer}{tokenizer.eos_token or ''}"
    tok_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len)
    tok_target = tokenizer(target, return_tensors="pt", add_special_tokens=False)
    input_ids = torch.cat([tok_prompt["input_ids"], tok_target["input_ids"]], dim=-1).to(device)
    labels = torch.cat(
        [
            torch.full((1, tok_prompt["input_ids"].shape[1]), -100, dtype=torch.long),
            tok_target["input_ids"],
        ],
        dim=-1,
    ).to(device)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": labels}


def answer_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels != -100
    return F.cross_entropy(shift_logits[valid], shift_labels[valid])

