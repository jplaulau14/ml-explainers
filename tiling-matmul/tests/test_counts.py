import pytest

from tiling_explainer.counts import loads_per_multiply_add, register_step


@pytest.mark.parametrize("tile", [1, 2, 4, 8, 16, 32])
def test_claim_tile_loads_two_values_per_tile_width(tile: int) -> None:
    assert loads_per_multiply_add(tile) == 2 / tile


def test_claim_32_wide_tile_loads_one_value_per_16_multiply_adds() -> None:
    assert loads_per_multiply_add(32) == 1 / 16


def test_claim_4x4_register_tile_does_16_multiply_adds_from_8_loads() -> None:
    step = register_step(4, 4)
    assert (step.loads, step.multiply_adds) == (8, 16)


def test_8x8_register_tile_does_64_multiply_adds_from_16_loads() -> None:
    step = register_step(8, 8)
    assert (step.loads, step.multiply_adds) == (16, 64)
