# Contributing

Contributions are welcome, especially clean ablations and reproducibility
improvements.

Good first contributions:

- add a new benchmark dataset loader
- add a passage-only baseline script
- add a critical-layer sweep
- add a Hugging Face checkpoint loading example
- improve documentation for a failed or successful reproduction

Before opening a pull request:

```bash
python -m pip install -e ".[dev]"
pytest -q
```

Please keep this repository focused on research code, method explanation,
reproducibility, and evaluation.

