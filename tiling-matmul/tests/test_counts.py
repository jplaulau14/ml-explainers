import pytest

from tiling_explainer.counts import loads_per_multiply_add, register_step


@pytest.mark.parametrize(
    ("tile", "loads"), [(1, 2.0), (2, 1.0), (4, 0.5), (8, 0.25), (16, 0.125), (32, 0.0625)]
)
def test_claim_tile_loads_two_values_per_tile_width(tile: int, loads: float) -> None:
    assert loads_per_multiply_add(tile) == loads


def test_claim_32_wide_tile_loads_one_value_per_16_multiply_adds() -> None:
    assert loads_per_multiply_add(32) == 0.0625


def test_claim_4x4_register_tile_does_16_multiply_adds_from_8_loads() -> None:
    step = register_step(4, 4)
    assert (step.loads, step.multiply_adds) == (8, 16)


def test_8x8_register_tile_does_64_multiply_adds_from_16_loads() -> None:
    step = register_step(8, 8)
    assert (step.loads, step.multiply_adds) == (16, 64)
