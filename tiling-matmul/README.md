# tiling-matmul

## What this is

Companion code for the essay [Tiling a Matrix Multiply on a Mac and an NVIDIA GPU](https://www.patslaurel.com/writing/tiling-a-matrix-multiply).
It holds four hand-written fp32 matrix multiply kernels, each in Metal and in CUDA C, and a harness that times the Metal versions on an Apple silicon Mac against `mx.matmul`.

| Kernel id | What each thread does | Threads per group | Sizes it accepts |
| --- | --- | --- | --- |
| `naive` | one output, reading A and B from device memory | 16 x 16 (Metal), 32 x 32 (CUDA) | any n (Metal), multiples of 32 (CUDA) |
| `tile32` | one output, from 32 x 32 tiles of A and B staged in threadgroup memory | 32 x 32 | multiples of 32 |
| `reg4x4` | a 4 x 4 block of a 64 x 64 output tile, 16 values of k at a time | 256 | multiples of 64 |
| `reg8x8` | an 8 x 8 block of a 128 x 128 output tile, 8 values of k at a time | 256 | multiples of 128 |
| `vendor` | `mx.matmul` on the Mac | set by MLX | any n |

The Metal sources are in `src/tiling_explainer/metal.py` and run through `mx.fast.metal_kernel`.
The CUDA sources are in `src/tiling_explainer/cuda.py`, written for CuPy's `RawKernel`, with the thread shapes in the table; nothing in this folder runs them yet.
A kernel launched with a size it does not accept raises `ValueError` before it reaches the GPU.

## Run it

The timing run needs an Apple silicon Mac with macOS 14 or later, the oldest macOS that MLX supports.
Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from this folder:

```bash
uv sync
uv run tiling-explainer mac
```

uv installs Python 3.12 and MLX 0.32.4.
The run starts 3 separate measurement processes at n = 4096 and took about 3.5 minutes on an M3 Pro.
For each process it prints one line per kernel with the time in ms, GFLOP/s, the ratio to naive, the percentage of `mx.matmul` speed and the largest absolute error against `mx.matmul`.
It ends with `wrote out/speeds.json`, `wrote out/challenge.json` and `wrote out/mac-runs.json`.

The run refuses to start when the Mac is on battery, Low Power Mode is on, or files in this folder outside `out/` have uncommitted changes.
It then prints `refusing a publication run:` and the reason, and exits with status 2.
It reads the power state again before and after every process.
If any reading fails those checks, it writes the results to `runs/` with `publication` set to false, leaves `out/` alone and exits with status 2.

The run takes three options.

- `--processes 5` runs more processes. The minimum is 3.
- `--n 2048` times the smaller size. 2048 and 4096 are the only sizes.
- `--allow-battery` skips the checks for a test run.

Only an n = 4096 run without `--allow-battery` is checked and writes to `out/`.
Every other run writes to `runs/`.

`uv run pytest` checks the kernels against `mx.matmul`, the counting and ratio maths, the committed files and their recorded commit.
The test that compares each kernel with `mx.matmul` skips on Linux, where MLX has no Metal backend; the size-rejection test still runs.

## What you get

- `out/speeds.json`: ratios to naive for `naive`, `tile32`, `reg4x4` and `vendor`, one entry per machine.
- `out/challenge.json`: the entry for `reg8x8`.
- `out/mac-runs.json`: every process's times, errors against `mx.matmul`, ratios to naive and to `mx.matmul`, and power readings.

`speeds.json` and `challenge.json` carry the same fields and differ only in the kernels they list.

| Field | Meaning |
| --- | --- |
| `source.commit` | the commit checked out when the run started |
| `source.code` | this folder on GitHub at that commit |
| `machines[].id` | the chip name in lower case without "Apple", with hyphens: `m3-pro` on an M3 Pro |
| `machines[].name` | the chip name from `sysctl -n machdep.cpu.brand_string` |
| `machines[].n` | matrix size |
| `machines[].runs` | number of processes |
| `machines[].conditions` | `acPower`, `lowPowerMode`, `publication`, `dtype` (`float32`) and `timing` (`warmups`, `launchesPerSample`, `samples`) |
| `machines[].versions` | commit, macOS, MLX and Python versions |
| `kernels[].kernel` | the kernel id from the first table |
| `kernels[].ratioMin`, `kernels[].ratioMax` | lowest and highest per-process ratio, naive time / kernel time |
| `kernels[].gflopsThisRun` | median over the processes of 2n³ / time |

The validator in `contract.py` accepts the machine ids `m3-pro` and `a10g`; `a10g` is for an NVIDIA A10G entry, which nothing in this folder writes yet.
A rerun replaces only the entry with the same machine id and keeps the others.

Each process makes A and B with `mx.random.normal` and seed 0 and checks every kernel against `mx.matmul` with `allclose(rtol=1e-4, atol=1e-3)` before timing it.
A kernel then gets 5 warm-up launches and 10 samples.
Each sample queues 8 launches, evaluates them with one `mx.eval`, and divides the wall time by 8; the kernel's time is the upper median of the 10 samples.

## Results

Pending the publication run.

## How this maps to the essay

- A T-wide tile loads 2/T values per multiply-add: `tests/test_counts.py::test_claim_tile_loads_two_values_per_tile_width`
- The 32-wide tile loads one value per 16 multiply-adds: `tests/test_counts.py::test_claim_32_wide_tile_loads_one_value_per_16_multiply_adds`
- A 4 x 4 register block does 16 multiply-adds from 8 loads per step of k: `tests/test_counts.py::test_claim_4x4_register_tile_does_16_multiply_adds_from_8_loads`
- Every kernel computes the same product as `mx.matmul`: `tests/test_kernels.py::test_claim_kernel_matches_mx_matmul`
- Speeds are ratios to naive, shown as a low-to-high range over the processes: `tests/test_ratios.py::test_summary_range_is_the_per_run_extremes`

## Notes

- The kernel tests compare with the same tolerance as the harness and do not test exact equality, because summing k in a different order changes fp32 results in the last digits.
- The Metal `tile32` has no `#pragma unroll` on its inner loop; the CUDA one does.
- `reg8x8` is the essay's reader exercise. Change its block shape in `metal.py` and run `uv run tiling-explainer mac --allow-battery`, because a checked run refuses to start with uncommitted changes. Then compare `reg8x8` in `runs/challenge.json` with `reg4x4` in `runs/speeds.json`.
- The harness records the power state but cannot control clocks or background load.
- `tests/test_provenance.py` accepts a recorded commit that is the last commit touching `out/` or that commit's parent, so a rerun that edits `out/` still passes.

## Sources

- Simon Boehm, [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM), the source of the 8 x 8 register tile shape.
- MLX documentation, [Build and Install](https://ml-explore.github.io/mlx/build/html/install.html) and [Custom Metal Kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html).
- CuPy documentation, [User-Defined Kernels](https://docs.cupy.dev/en/stable/user_guide/kernel.html).
