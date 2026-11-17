import os
import platform
from collections.abc import Callable
from statistics import median_high

import cupy as cp
import numpy

from tiling_explainer.a10.spec import ATOL, KERNEL_ORDER, RTOL, SAMPLES, SEED, WARMUPS
from tiling_explainer.cuda import KERNELS

Launch = Callable[[], object]
Timing = dict[str, float | list[float]]


class KernelMismatch(RuntimeError):
    pass


def inputs(n: int) -> tuple[cp.ndarray, cp.ndarray]:
    cp.random.seed(SEED)
    a = cp.random.randn(n, n, dtype=cp.float32)
    b = cp.random.randn(n, n, dtype=cp.float32)
    return a, b


def compiled(kernel: str) -> cp.RawKernel:
    return cp.RawModule(code=KERNELS[kernel].source).get_function(kernel)


def launcher(kernel: str, a: cp.ndarray, b: cp.ndarray, c: cp.ndarray) -> Launch:
    n = a.shape[0]
    if kernel == "vendor":
        return lambda: cp.matmul(a, b, out=c)
    grid, block = KERNELS[kernel].launch(n)
    function = compiled(kernel)
    return lambda: function(grid, block, (a, b, c, cp.int32(n)))


def max_abs_error(launch: Launch, c: cp.ndarray, reference: cp.ndarray, kernel: str) -> float:
    launch()
    cp.cuda.Device().synchronize()
    if not bool(cp.allclose(c, reference, rtol=RTOL, atol=ATOL)):
        raise KernelMismatch(f"{kernel} does not match cuBLAS within rtol {RTOL}, atol {ATOL}")
    return float(cp.abs(c - reference).max())


def samples_ms(launch: Launch) -> list[float]:
    for _ in range(WARMUPS):
        launch()
    cp.cuda.Device().synchronize()
    start, end = cp.cuda.Event(), cp.cuda.Event()
    samples = []
    for _ in range(SAMPLES):
        start.record()
        launch()
        end.record()
        end.synchronize()
        samples.append(float(cp.cuda.get_elapsed_time(start, end)))
    return samples


def time_kernels(n: int) -> dict[str, Timing]:
    a, b = inputs(n)
    reference = cp.matmul(a, b)
    results: dict[str, Timing] = {}
    for kernel in KERNEL_ORDER:
        c = cp.empty_like(a)
        launch = launcher(kernel, a, b, c)
        error = max_abs_error(launch, c, reference, kernel)
        samples = samples_ms(launch)
        results[kernel] = {"ms": median_high(samples), "samplesMs": samples, "maxAbsErr": error}
    return results


def kernel_attributes() -> dict[str, dict[str, int]]:
    functions = {kernel: compiled(kernel) for kernel in KERNELS}
    return {
        kernel: {
            "registers": function.num_regs,
            "localBytes": function.local_size_bytes,
            "sharedBytes": function.shared_size_bytes,
        }
        for kernel, function in functions.items()
    }


def cuda_version(code: int) -> str:
    return f"{code // 1000}.{code % 1000 // 10}"


def device_facts() -> dict[str, int]:
    properties = cp.cuda.runtime.getDeviceProperties(0)
    return {"smCount": properties["multiProcessorCount"], "l2Bytes": properties["l2CacheSize"]}


def software_versions() -> dict[str, str]:
    return {
        "cudaRuntime": cuda_version(cp.cuda.runtime.runtimeGetVersion()),
        "cudaDriver": cuda_version(cp.cuda.runtime.driverGetVersion()),
        "cupy": cp.__version__,
        "numpy": numpy.__version__,
        "python": platform.python_version(),
        "cupyTf32": os.environ.get("CUPY_TF32", "unset"),
    }
