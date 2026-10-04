import pytest
import torch

from lora_explainer.spectrum import (
    energy_kept,
    rank_for_energy,
    relative_error,
    singular_values,
    truncate,
    truncation_error,
)


def random_matrix(rows: int = 40, cols: int = 30, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=generator, dtype=torch.float64)


def test_claim_8_truncation_error_is_tail_of_spectrum() -> None:
    m = random_matrix()
    s = singular_values(m)
    for r in range(len(s) + 1):
        error = torch.linalg.matrix_norm(m - truncate(m, r)).item()
        assert error == pytest.approx(truncation_error(s, r), abs=1e-9)


@pytest.mark.parametrize("r", [1, 3, 10])
def test_no_random_rank_r_matrix_beats_truncation(r: int) -> None:
    m = random_matrix()
    best = truncation_error(singular_values(m), r)
    generator = torch.Generator().manual_seed(r)
    for _ in range(200):
        left = torch.randn(40, r, generator=generator, dtype=torch.float64)
        right = torch.randn(r, 30, generator=generator, dtype=torch.float64)
        coefficients = torch.linalg.lstsq(left, m).solution
        candidates = [left @ right, left @ coefficients]
        assert all(torch.linalg.matrix_norm(m - c).item() >= best for c in candidates)


def test_rank_for_energy_is_monotonic() -> None:
    s = singular_values(random_matrix())
    thresholds = torch.linspace(0.01, 1.0, 100).tolist()
    ranks = [rank_for_energy(s, t) for t in thresholds]
    assert ranks == sorted(ranks)
    pairs = list(zip(ranks, thresholds, strict=True))
    assert all(energy_kept(s, r) >= t - 1e-12 for r, t in pairs)
    assert all(energy_kept(s, r - 1) < t for r, t in pairs)


def test_relative_and_energy_agree() -> None:
    s = singular_values(random_matrix())
    for r in range(len(s) + 1):
        assert relative_error(s, r) ** 2 + energy_kept(s, r) == pytest.approx(1.0)
