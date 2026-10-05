import time
from collections.abc import Callable


def samples(fn: Callable[[], None], repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    took: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        took.append(time.perf_counter() - start)
    return took


def median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def fastest(values: list[float]) -> float:
    return min(values)
