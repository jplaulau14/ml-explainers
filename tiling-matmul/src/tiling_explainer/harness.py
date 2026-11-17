import time
from collections.abc import Callable
from functools import cache
from statistics import median_high

import mlx.core as mx

from tiling_explainer.metal import KERNELS, MetalKernel

WARMUPS = 5
BATCH = 8
SAMPLES = 10
RTOL = 1e-4
ATOL = 1e-3
SEED = 0
ORDER = ("naive", "tile32", "reg4x4", "reg8x8", "vendor")

Launch = Callable[[], mx.array]


@cache
def compiled(kernel: str) -> Callable[..., list[mx.array]]:
    spec = KERNELS[kernel]
    return mx.fast.metal_kernel(
        name=spec.kernel,
        input_names=["A", "B"],
        output_names=["C"],
        source=spec.source,
        header=spec.header,
    )


def run_metal(spec: MetalKernel, a: mx.array, b: mx.array) -> mx.array:
    n = a.shape[0]
    grid, threadgroup = spec.launch(n)
    kernel = compiled(spec.kernel)
    outputs = kernel(
        inputs=[a, b],
        template=[("N", n), *spec.template],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(n, n)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]


def launcher(kernel: str, a: mx.array, b: mx.array) -> Launch:
    if kernel == "vendor":
        return lambda: mx.matmul(a, b)
    spec = KERNELS[kernel]
    return lambda: run_metal(spec, a, b)


def inputs(n: int) -> tuple[mx.array, mx.array]:
    mx.random.seed(SEED)
    a = mx.random.normal((n, n), dtype=mx.float32)
    b = mx.random.normal((n, n), dtype=mx.float32)
    mx.eval(a, b)
    return a, b


def max_abs_error(launch: Launch, reference: mx.array, kernel: str) -> float:
    result = launch()
    mx.eval(result)
    if not mx.allclose(result, reference, rtol=RTOL, atol=ATOL).item():
        raise RuntimeError(f"{kernel} does not match mx.matmul within rtol {RTOL}, atol {ATOL}")
    return float(mx.abs(result - reference).max().item())


def median_ms(launch: Launch) -> float:
    for _ in range(WARMUPS):
        mx.eval(launch())
    samples = []
    for _ in range(SAMPLES):
        start = time.perf_counter()
        mx.eval(*[launch() for _ in range(BATCH)])
        samples.append((time.perf_counter() - start) / BATCH)
    return median_high(samples) * 1e3


def run_process(n: int) -> dict[str, dict[str, float]]:
    a, b = inputs(n)
    reference = mx.matmul(a, b)
    mx.eval(reference)
    results = {}
    for kernel in ORDER:
        launch = launcher(kernel, a, b)
        error = max_abs_error(launch, reference, kernel)
        results[kernel] = {"ms": median_ms(launch), "maxAbsErr": error}
    return results
