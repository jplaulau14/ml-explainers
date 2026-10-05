import random

import pytest

from mb_explainer.arithmetic import (
    arithmetic_intensity,
    dtype_bytes,
    expected_draft_tokens,
    flops_per_byte,
    flops_per_weight,
    gemm_bytes,
    gemm_flops,
    ridge_intensity,
    ridge_tokens,
    tokens_per_second,
)


def test_claim_two_flops_per_weight() -> None:
    assert flops_per_weight(1) == 2
    assert flops_per_weight(8) == 16


def test_claim_fp16_batch_one_is_one_flop_per_byte() -> None:
    assert flops_per_byte(1, dtype_bytes("float16")) == 1
    assert flops_per_byte(1, dtype_bytes("bfloat16")) == 1


def test_claim_fp32_batch_one_is_half_a_flop_per_byte() -> None:
    assert flops_per_byte(1, dtype_bytes("float32")) == 0.5


def test_claim_int8_batch_one_is_two_flops_per_byte() -> None:
    assert flops_per_byte(1, dtype_bytes("int8")) == 2


def test_claim_gemm_counts_multiply_and_add() -> None:
    assert gemm_flops(1, 4, 8) == 64
    assert gemm_bytes(1, 4, 8, 2) == (4 + 32 + 8) * 2


def test_claim_intensity_rises_when_tokens_share_the_weights() -> None:
    inner, cols, width = 128, 256, 2
    low = arithmetic_intensity(gemm_flops(1, inner, cols), gemm_bytes(1, inner, cols, width))
    high = arithmetic_intensity(gemm_flops(32, inner, cols), gemm_bytes(32, inner, cols, width))
    assert high > low


def test_claim_tokens_per_second_is_bandwidth_over_bytes() -> None:
    assert tokens_per_second(2_000, 4) == 500


def test_claim_ridge_is_peak_over_bandwidth() -> None:
    assert ridge_intensity(312e12, 2039e9) == pytest.approx(312e12 / 2039e9)


def test_claim_ridge_tokens_match_the_intensity_equation() -> None:
    inner, cols, width = 64, 64, 4
    ridge = 8.0
    tokens = ridge_tokens(inner, cols, width, ridge)
    assert tokens is not None
    intensity = arithmetic_intensity(gemm_flops(1, inner, cols), gemm_bytes(1, inner, cols, width))
    solved = arithmetic_intensity(
        gemm_flops(round(tokens), inner, cols), gemm_bytes(round(tokens), inner, cols, width)
    )
    assert solved == pytest.approx(ridge, rel=0.05)
    assert intensity < ridge


def test_claim_speculative_edges() -> None:
    assert expected_draft_tokens(0, 4) == 1
    assert expected_draft_tokens(1, 4) == 5


def test_claim_speculative_matches_a_draw() -> None:
    acceptance, drafts, trials = 0.7, 3, 20_000
    rng = random.Random(0)
    total = 0
    for _ in range(trials):
        accepted = 0
        for _draft in range(drafts):
            if rng.random() > acceptance:
                break
            accepted += 1
        total += accepted + 1
    assert total / trials == pytest.approx(expected_draft_tokens(acceptance, drafts), abs=0.05)
