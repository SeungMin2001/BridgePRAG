from bridgeprag.data import load_memory_examples


def test_load_memory_examples_tiny_dataset():
    examples = load_memory_examples("examples/data/tiny_memory.jsonl")

    assert len(examples) == 3
    assert examples[0].question
    assert examples[0].passage
    assert examples[0].target_answer("full_answer")

