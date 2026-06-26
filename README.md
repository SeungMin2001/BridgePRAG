# BridgePRAG

**FiD-inspired question-conditioned HyperKV memory for decoder-only Retrieval-Augmented Generation.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Research Preview](https://img.shields.io/badge/status-research%20preview-purple)](#research-status)
[![Paper](https://img.shields.io/badge/paper-accepted%20%40%20KDCS-0a7c66)](#research-status)

BridgePRAG is a compact research codebase for encoding retrieved evidence into
learned **K/V memory slots** and injecting them into a frozen decoder-only LLM.
It starts from the MergePRAG/HyperKV direction, then changes the memory encoder
with a Fusion-in-Decoder-style question-passage boundary and a lightweight
linear **KV adapter** for calibration.

A paper based on this project has been accepted by the **Korean Digital Contents
Society** (**한국디지털콘텐츠학회**). Publication details will be added after the
proceedings are released.

<div align="center">
  <img src="assets/bridgeprag_architecture.svg" alt="BridgePRAG architecture: question-conditioned evidence is encoded into calibrated K/V memory slots" width="960">
  <br>
  <sub><b>Figure 1.</b> FiD-style question-passage encoding generates calibrated K/V memory slots for decoder-only RAG.</sub>
</div>

## At a Glance

BridgePRAG asks a simple question: can retrieved passages be compressed into
small trainable K/V memories that are already aware of the user question?

Instead of generating memory from a passage alone, BridgePRAG encodes
`question + passage`, creates HyperKV slots, calibrates them with a lightweight
adapter, and injects the resulting memory into a frozen decoder-only model.
The goal is to provide a compact reference implementation for researchers and
engineers exploring decoder-only RAG memory.

**Current validation snapshot**

- +43.00 hit accuracy over passage-only memory on the entity-style validation setup.
- +24.67 F1 score over passage-only memory.
- Faster average inference time in the reported comparison.
- Training curves show faster convergence and higher validation accuracy than the passage-only baseline.

## Why BridgePRAG?

- **Question-conditioned memory**: encode `question + passage` before generating HyperKV slots.
- **FiD-inspired boundary**: move from passage-only memory toward query-aware evidence encoding.
- **KV adapter correction**: learn linear residual corrections for generated key/value slots.
- **Decoder-only compatible**: inject memory through a selected transformer layer without fine-tuning the base LLM.
- **Reproducible research layout**: installable package, toy data, training CLI, inference CLI, tests, figures, citation file.

## Install

```bash
git clone https://github.com/SeungMin2001/BridgePRAG.git
cd BridgePRAG
python -m pip install -e ".[dev]"
```

For GPU runs, install the PyTorch build that matches your CUDA version first:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e ".[dev]"
```

## Quickstart

Train a tiny BridgePRAG memory generator:

```bash
bridgeprag train \
  --data examples/data/tiny_memory.jsonl \
  --output runs/tiny_bridgeprag.pt \
  --model Qwen/Qwen2.5-0.5B \
  --num-kv 8 \
  --hidden-dim 512 \
  --critical-layer 8 \
  --epochs 1 \
  --max-steps 3
```

Run question-conditioned memory inference:

```bash
bridgeprag infer \
  --checkpoint runs/tiny_bridgeprag.pt \
  --question "What does the KV adapter correct?" \
  --passage "The KV adapter applies lightweight linear corrections to generated key and value slots."
```

Python API:

```python
from bridgeprag import BridgePRAG

bridge = BridgePRAG.from_checkpoint("runs/tiny_bridgeprag.pt")
answer = bridge.generate(
    question="What does the KV adapter correct?",
    passages=[
        "The KV adapter applies lightweight linear corrections to generated key and value slots."
    ],
)
print(answer)
```

## Method

BridgePRAG trains only a small HyperKV generator while the base LLM stays frozen.

1. Encode retrieved evidence as either `passage` or `question + passage`.
2. Project token features into slot-wise hidden states with attentive pooling.
3. Produce `K` and `V` memory slots.
4. Apply a linear KV adapter: `K = K + A_k(K)`, `V = V + A_v(V)`.
5. Inject memory into a selected decoder layer with cross-attention.
6. Merge multiple passages through orthogonal slot composition.

The main contrast is:

| Variant | Memory encoder input | Slot correction | Intended effect |
| --- | --- | --- | --- |
| Passage-only HyperKV | passage | none | compact external memory |
| BridgePRAG | question + passage | linear KV adapter | query-aware, calibrated memory |

See [docs/method.md](docs/method.md) for the full research note.

## Results Snapshot

The current BridgePRAG branch reports the following question+passage memory vs
passage-only memory comparison on an entity-style validation setup.

| Metric | Question+Passage | Passage-only | Delta |
| --- | ---: | ---: | ---: |
| Accuracy (Hit) | 84.00 | 41.00 | +43.00 |
| Precision | 69.78 | 45.58 | +24.21 |
| Recall | 86.73 | 63.28 | +23.45 |
| F1 Score | 76.19 | 51.52 | +24.67 |
| QA Score | 38.26 | 26.16 | +12.10 |
| Avg. Time (s) | 4.17 | 4.83 | -0.66 |

<div align="center">
  <img src="assets/bridgeprag_evaluation_metrics_table.png" alt="BridgePRAG evaluation metrics table comparing question-passage memory with passage-only memory" width="860">
  <br>
  <sub><b>Table 1.</b> Question-conditioned memory improves hit accuracy, F1, and QA score while reducing average latency.</sub>
</div>

### Training Curve Comparison

The two training visualizations below compare BridgePRAG against the
passage-only baseline under the same epoch budget. BridgePRAG reduces
validation loss more aggressively and reaches near-saturated validation accuracy,
while the passage-only baseline improves more slowly and remains accuracy-limited.

| BridgePRAG: question-conditioned K/V memory | Passage-only HyperKV memory |
| --- | --- |
| <img src="assets/bridgeprag_train_val_loss_accuracy.png" alt="BridgePRAG training and validation loss/accuracy curves" width="480"> | <img src="assets/passage_only_train_val_loss_accuracy.png" alt="Passage-only training and validation loss/accuracy curves" width="480"> |

<div align="center">
  <sub><b>Figure 2.</b> Question-conditioned memory learns a sharper loss curve and reaches substantially higher validation accuracy than passage-only memory.</sub>
</div>

## Dataset Format

Minimal JSONL:

```json
{"source_id":"ex1","passage":"...","question":"...","answer":"...","full_answer":"..."}
```

Augmented rows may include `qas`, `atomic_qas`, `final_qas`, and
`hard_negatives`; the loader expands them into independent memory supervision
examples.

## Repository Layout

```text
bridgeprag/          # installable research package
  config.py          # BridgePRAGConfig
  memory.py          # HyperKV generator, FiD-style encoding, injection hooks
  runtime.py         # high-level BridgePRAG wrapper
  trainer.py         # compact frozen-LLM training loop
  cli.py             # bridgeprag train / infer
examples/            # tiny runnable examples
docs/                # method and reproduction notes
assets/              # architecture and experiment figures
tests/               # shape/data tests for fast CI
```

## Research Status

This repository is a research preview. The code is intended to make the
BridgePRAG mechanism easy to inspect, run, and adapt. Large checkpoints are not
stored in git; release checkpoints should be attached through GitHub Releases or
Hugging Face Hub.

Paper status:

- Accepted by the Korean Digital Contents Society (한국디지털콘텐츠학회).
- Publication, volume, pages, and DOI are pending.
- The citation block will be updated when the official proceedings entry is available.

## Roadmap

- [ ] Release a compact pretrained BridgePRAG checkpoint.
- [ ] Add a full benchmark script for passage-only vs question+passage memory.
- [ ] Add Hugging Face model card metadata for released checkpoints.
- [ ] Update the citation after the accepted paper is published.

## Citation

```bibtex
@software{bridgeprag2026,
  title = {BridgePRAG: FiD-Inspired Question-Conditioned HyperKV Memory for Decoder-Only RAG},
  author = {SeungMin2001},
  year = {2026},
  url = {https://github.com/SeungMin2001/BridgePRAG}
}
```

## References

- MergePRAG-style HyperKV memory for retrieval-augmented generation.
- Fusion-in-Decoder-style question-passage encoding for retrieved evidence.
- Decoder-only KV-cache and cross-attention memory injection.

## License

MIT. See [LICENSE](LICENSE).
