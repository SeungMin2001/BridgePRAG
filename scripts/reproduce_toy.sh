#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e ".[dev]"
bridgeprag train \
  --data examples/data/tiny_memory.jsonl \
  --output runs/tiny_bridgeprag.pt \
  --model Qwen/Qwen2.5-0.5B \
  --num-kv 8 \
  --hidden-dim 512 \
  --critical-layer 8 \
  --epochs 1 \
  --max-steps 3

bridgeprag infer \
  --checkpoint runs/tiny_bridgeprag.pt \
  --question "What does the KV adapter correct?" \
  --passage "The KV adapter applies lightweight linear corrections to generated key and value slots."

