import mlx.core as mx
import pytest

from tiling_explainer.harness import ATOL, RTOL, inputs, launcher
from tiling_explainer.metal import KERNELS

SIZES = (64, 100, 256)
RUNS = [(k, n) for k, spec in KERNELS.items() for n in SIZES if n % spec.multiple == 0]
REJECTS = [(k, n) for k, spec in KERNELS.items() for n in SIZES if n % spec.multiple]
metal = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal runs only on macOS")


@metal
@pytest.mark.parametrize(("kernel", "n"), RUNS)
def test_claim_kernel_matches_mx_matmul(kernel: str, n: int) -> None:
    a, b = inputs(n)
    result = launcher(kernel, a, b)()
    assert mx.allclose(result, mx.matmul(a, b), rtol=RTOL, atol=ATOL).item()


@pytest.mark.parametrize(("kernel", "n"), REJECTS)
def test_kernel_rejects_n_off_its_tile(kernel: str, n: int) -> None:
    a, b = inputs(n)
    with pytest.raises(ValueError, match=f"{kernel} needs n to be a positive multiple"):
        launcher(kernel, a, b)()
