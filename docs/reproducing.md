# Reproducing BridgePRAG Experiments

This document keeps exact commands close to the code. Start with the toy run,
then scale the same CLI to larger JSONL datasets.

## Toy Run

```bash
python -m pip install -e ".[dev]"
bash scripts/reproduce_toy.sh
```

Expected output:

- `runs/tiny_bridgeprag.pt`
- one short generated answer from `bridgeprag infer`

## Custom Dataset

Prepare JSONL rows:

```json
{"source_id":"row_001","passage":"...","question":"...","answer":"...","full_answer":"..."}
```

Train:

```bash
bridgeprag train \
  --data data/my_memory_train.jsonl \
  --output runs/my_bridgeprag.pt \
  --model Qwen/Qwen2.5-0.5B \
  --num-kv 16 \
  --hidden-dim 1024 \
  --critical-layer 9 \
  --epochs 1 \
  --lr 5e-5
```

Infer:

```bash
bridgeprag infer \
  --checkpoint runs/my_bridgeprag.pt \
  --question "Your question?" \
  --passage "Retrieved evidence passage."
```

## Larger Research Configuration

The branch that produced the included figures used the same mechanism with a
larger Qwen family base model, more KV slots, and a question-conditioned
`kv_adapter` configuration:

```bash
bridgeprag train \
  --data data/entity_memory_train.jsonl \
  --output runs/bridgeprag_entity_kv64.pt \
  --model Qwen/Qwen2.5-3B \
  --num-kv 64 \
  --hidden-dim 1024 \
  --critical-layer 23 \
  --question-fusion kv_adapter \
  --epochs 10 \
  --lr 2e-5
```

## Reporting Checklist

When publishing a new BridgePRAG run, include:

- base model and exact checkpoint id
- number of KV slots
- critical layer
- alpha and injection mode
- dataset split sizes
- passage-only baseline
- question+passage result
- command used to reproduce the table

