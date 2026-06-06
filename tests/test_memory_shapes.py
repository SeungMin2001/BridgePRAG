import torch

from bridgeprag.memory import BridgeKVGenerator, cross_attention, orthogonal_merge_slots


def test_bridge_kv_generator_shapes():
    generator = BridgeKVGenerator(d_model=32, num_kv=4, hidden_dim=64, question_fusion="kv_adapter")
    features = torch.randn(2, 7, 32)
    mask = torch.ones(2, 7, dtype=torch.long)

    memory = generator(features, mask)

    assert memory["K"].shape == (2, 4, 32)
    assert memory["V"].shape == (2, 4, 32)
    assert memory["att_weights"].shape == (2, 7, 4)


def test_cross_attention_shape():
    query = torch.randn(2, 5, 32)
    key = torch.randn(2, 4, 32)
    value = torch.randn(2, 4, 32)

    output = cross_attention(query, key, value, num_heads=4)

    assert output.shape == query.shape


def test_orthogonal_merge_slots_shape():
    base = torch.randn(2, 4, 32)
    update = torch.randn(2, 4, 32)

    merged = orthogonal_merge_slots(base, update)

    assert merged.shape == update.shape

