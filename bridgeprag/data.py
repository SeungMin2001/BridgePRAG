"""Dataset utilities for BridgePRAG memory training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class MemoryExample:
    """One passage-question-answer memory supervision item."""

    source_id: str
    passage: str
    question: str
    answer: str
    full_answer: str = ""
    negative_passage: str | None = None
    negative_answer: str | None = None
    negative_full_answer: str | None = None

    def target_answer(self, answer_target: str = "answer") -> str:
        if answer_target == "full_answer" and self.full_answer:
            return self.full_answer
        return self.answer


def iter_json_records(path: str | Path) -> Iterable[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON array in {path}")
        yield from payload
        return
    raise ValueError(f"Unsupported dataset format: {path}")


def _first_text(row: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_qas(row: dict) -> list[dict]:
    qas = []
    for key in ("qas", "atomic_qas", "final_qas"):
        value = row.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            question = _first_text(item, ("question", "sub_question"))
            answer = _first_text(item, ("answer", "sub_answer"))
            if not question or not answer:
                continue
            qas.append(
                {
                    "question": question,
                    "answer": answer,
                    "full_answer": _first_text(item, ("full_answer", "response")),
                    "passage": _first_text(item, ("sub_passage", "evidence", "passage")),
                }
            )
    return qas


def load_memory_examples(path: str | Path, max_rows: int | None = None) -> list[MemoryExample]:
    """Load BridgePRAG examples from JSONL/JSON.

    Minimal row format:

    {"source_id": "ex1", "passage": "...", "question": "...", "answer": "..."}

    Augmented rows may also contain qas, atomic_qas, final_qas, and
    hard_negatives. The loader expands each QA into an independent memory item.
    """

    examples: list[MemoryExample] = []
    for row_idx, row in enumerate(iter_json_records(path)):
        if max_rows is not None and row_idx >= max_rows:
            break
        passage = _first_text(row, ("passage", "text", "content", "utterance"))
        if not passage:
            continue
        source_id = str(row.get("source_id") or row.get("id") or f"row_{row_idx}")

        qas = _normalize_qas(row)
        if not qas and row.get("question") and row.get("answer"):
            qas = [
                {
                    "question": str(row["question"]).strip(),
                    "answer": str(row["answer"]).strip(),
                    "full_answer": _first_text(row, ("full_answer", "response")),
                    "passage": passage,
                }
            ]

        negatives = row.get("hard_negatives") if isinstance(row.get("hard_negatives"), list) else []
        negative = negatives[0] if negatives and isinstance(negatives[0], dict) else {}
        negative_passage = _first_text(negative, ("passage", "text", "content"))
        negative_answer = _first_text(negative, ("answer", "target"))
        negative_full_answer = _first_text(negative, ("full_answer", "response"))

        for qa_idx, qa in enumerate(qas):
            examples.append(
                MemoryExample(
                    source_id=f"{source_id}:{qa_idx}",
                    passage=qa.get("passage") or passage,
                    question=qa["question"],
                    answer=qa["answer"],
                    full_answer=qa.get("full_answer") or qa["answer"],
                    negative_passage=negative_passage or None,
                    negative_answer=negative_answer or None,
                    negative_full_answer=negative_full_answer or None,
                )
            )
    return examples

