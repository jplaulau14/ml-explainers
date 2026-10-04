import pytest

from lora_explainer.counts import (
    adapter_bytes,
    adapter_params,
    full_multiply_adds,
    full_params,
    lora_params,
    unmerged_layer_multiply_adds,
    update_multiply_adds,
)


def test_claim_4_gpt3_sized_matrix() -> None:
    assert full_params(12288, 12288) == 150_994_944
    assert lora_params(12288, 12288, 4) == 98_304
    assert full_params(12288, 12288) // lora_params(12288, 12288, 4) == 1536


@pytest.mark.parametrize(("r", "expected"), [(1, 4_718_592), (4, 18_874_368), (8, 37_748_736)])
def test_claim_5_gpt3_adapter_size(r: int, expected: int) -> None:
    assert adapter_params(12288, 96, 2, r) == expected


def test_lora_params_at_d_4096() -> None:
    assert lora_params(4096, 4096, 1) == 8_192
    assert lora_params(4096, 4096, 16) == 131_072


def test_claim_9_multiply_adds() -> None:
    d, k, r = 768, 512, 8
    assert update_multiply_adds(d, k, r) == r * (d + k)
    assert full_multiply_adds(d, k) == d * k
    assert unmerged_layer_multiply_adds(d, k, r) == d * k + r * (d + k)


def test_adapter_bytes_in_fp16() -> None:
    assert adapter_bytes(18_874_368, 2) == 37_748_736
