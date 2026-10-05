import pytest

from mb_explainer.catalog import GPT2, QWEN, head_dim
from mb_explainer.models import (
    gpt2_mask_elements,
    gpt2_parameters,
    qwen_parameters,
    qwen_projection_shape,
)


@pytest.mark.parametrize("row", GPT2, ids=lambda row: row["id"])
def test_claim_gpt2_parameters_match_the_file_minus_the_mask(row: dict) -> None:
    parameters = gpt2_parameters(
        row["nLayer"], row["nEmbd"], row["nPositions"], row["vocab"], row["nInner"]
    )
    mask = gpt2_mask_elements(row["nLayer"], row["nPositions"])
    assert parameters + mask == row["safetensorsElements"]
    assert mask == row["nLayer"] * row["nPositions"] ** 2


@pytest.mark.parametrize("row", QWEN, ids=lambda row: row["id"])
def test_claim_qwen_parameters_match_the_published_file(row: dict) -> None:
    dim = head_dim(row)
    count = qwen_parameters(
        row["hidden"],
        row["intermediate"],
        row["layers"],
        row["vocab"],
        row["kvHeads"],
        dim,
        row["tied"],
    )
    assert count == row["safetensorsElements"]


def test_claim_qwen7b_layer_shape_has_the_projection_count() -> None:
    row = next(item for item in QWEN if item["id"] == "Qwen/Qwen2.5-7B")
    dim = head_dim(row)
    inner, cols = qwen_projection_shape(row["hidden"], row["intermediate"], row["kvHeads"], dim)
    assert inner == 3584
    assert inner * cols == 233_046_016
