import sys

import cupy as cp

from tiling_explainer.a10.bench import inputs, launcher
from tiling_explainer.a10.ncu import LAUNCHES
from tiling_explainer.a10.spec import PROFILED


def main(n: int) -> None:
    a, b = inputs(n)
    c = cp.empty_like(a)
    for kernel in PROFILED:
        launch = launcher(kernel, a, b, c)
        for _ in range(LAUNCHES):
            launch()
    cp.cuda.Device().synchronize()


if __name__ == "__main__":
    main(int(sys.argv[1]))
