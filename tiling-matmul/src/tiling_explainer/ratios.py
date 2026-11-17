from collections.abc import Mapping, Sequence
from statistics import median

from tiling_explainer.contract import KernelRow

Timings = Mapping[str, float]


def ratios_to(reference: str, timings: Timings) -> dict[str, float]:
    return {kernel: timings[reference] / ms for kernel, ms in timings.items()}


def gflops(n: int, ms: float) -> float:
    return 2 * n**3 / (ms * 1e-3) / 1e9


def summarize(kernel: str, n: int, runs: Sequence[Timings]) -> KernelRow:
    if not runs:
        raise ValueError("summarize needs at least one run")
    ratios = [ratios_to("naive", timings)[kernel] for timings in runs]
    return {
        "kernel": kernel,
        "ratioMin": min(ratios),
        "ratioMax": max(ratios),
        "gflopsThisRun": median(gflops(n, timings[kernel]) for timings in runs),
    }
