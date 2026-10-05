from mb_explainer.catalog import QWEN, head_dim
from mb_explainer.kv import ceiling, kv_bytes_per_token, traffic_bytes


def test_claim_kv_bytes_are_two_tensors() -> None:
    assert kv_bytes_per_token(24, 2, 64, 2) == 2 * 24 * 2 * 64 * 2


def test_claim_ceiling_falls_as_context_grows() -> None:
    row = next(item for item in QWEN if item["id"] == "Qwen/Qwen2.5-7B")
    per_token = kv_bytes_per_token(row["layers"], row["kvHeads"], head_dim(row), 2)
    weights = row["safetensorsElements"] * 2
    short = ceiling(3.35e12, weights, per_token, 1)
    long = ceiling(3.35e12, weights, per_token, row["maxPositionEmbeddings"])
    assert long < short
    assert traffic_bytes(weights, per_token, 1024) == weights + per_token * 1024


def test_claim_qwen_half_b_head_dim() -> None:
    row = next(item for item in QWEN if item["id"] == "Qwen/Qwen2.5-0.5B")
    assert head_dim(row) == 64
    assert row["kvHeads"] < row["heads"]
