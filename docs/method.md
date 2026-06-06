# BridgePRAG Method Note

BridgePRAG studies a small trainable memory module for decoder-only RAG. The
base language model is frozen. Retrieved evidence is converted into K/V memory
slots, and those slots are injected into one decoder layer during answer
generation.

## Motivation

Passage-only HyperKV memory is compact, but it can encode evidence without
knowing which fact the current question needs. BridgePRAG borrows the useful
boundary from Fusion-in-Decoder readers: each retrieved passage is encoded in
the presence of the question before the decoder consumes the fused evidence.

BridgePRAG keeps a decoder-only backbone by moving that idea into the memory
encoder:

```text
question + passage -> token features -> HyperKV slots -> decoder-layer memory injection
```

## Architecture

For each passage, BridgePRAG computes token features from either raw token
embeddings or concatenated embedding/contextual features. A slot-wise attentive
pooling layer maps token features into `num_kv` memory slots:

```text
features -> LayerNorm -> Linear -> GELU -> slot attention -> pooled slots
```

Each slot is passed through an MLP and projected into a key and value:

```text
pooled_slot -> MLP -> linear_K, linear_V
```

The KV adapter then applies a residual linear correction:

```text
K_bridge = K + A_k(K)
V_bridge = V + A_v(V)
```

This adapter is deliberately simple. It keeps the method easy to ablate while
giving the memory generator a calibration path after the question-conditioned
encoding step.

## Injection

At a selected decoder layer, BridgePRAG adds memory attention to the hidden
states:

```text
H' = H + alpha * Attention(H, K_bridge, V_bridge)
```

The base model parameters are not updated. Only the HyperKV generator and the
adapter are trained.

## Multi-Passage Composition

BridgePRAG supports multiple passages by producing one memory dictionary per
passage and composing slots with an orthogonal merge:

```text
incoming_orth = incoming - projection(existing, incoming)
merged = existing + incoming_orth
```

The goal is to preserve slot diversity when several retrieved passages are
compressed into a fixed-size memory.

## Training Objective

The compact public trainer optimizes answer-token cross entropy under injected
memory. The intended full research path can add hard-negative ranking:

```text
positive memory should lower the loss for the positive answer
counterfactual memory should lower the loss for the counterfactual answer
```

This preserves the key BridgePRAG question: does the generated K/V memory carry
the evidence strongly enough that the model answers from memory rather than from
the original text prompt?

## Key Ablations

- Passage-only vs question+passage memory encoding.
- No adapter vs KV adapter.
- Embedding-only features vs contextual memory features.
- Attention injection vs additive last-token memory.
- KV slot count: 8, 16, 32, 64.
- Decoder critical layer sweep.

