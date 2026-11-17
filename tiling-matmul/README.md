# tiling-matmul

## What this is

Companion code for the essay [Tiling a Matrix Multiply on a Mac and an NVIDIA GPU](https://www.patslaurel.com/writing/tiling-a-matrix-multiply).
It holds four hand-written fp32 matrix multiply kernels, each in Metal and in CUDA C.
A harness times the Metal versions on an Apple silicon Mac against `mx.matmul`, and a Modal app times and profiles the CUDA versions on an NVIDIA A10 against cuBLAS.

| Kernel id | What each thread does | Threads per group | Sizes it accepts |
| --- | --- | --- | --- |
| `naive` | one output, reading A and B from device memory | 16 x 16 (Metal), 32 x 32 (CUDA) | any n (Metal), multiples of 32 (CUDA) |
| `tile32` | one output, from 32 x 32 tiles of A and B staged in threadgroup memory | 32 x 32 | multiples of 32 |
| `reg4x4` | a 4 x 4 block of a 64 x 64 output tile, 16 values of k at a time | 256 | multiples of 64 |
| `reg8x8` | an 8 x 8 block of a 128 x 128 output tile, 8 values of k at a time | 256 | multiples of 128 |
| `vendor` | `mx.matmul` on the Mac, cuBLAS through `cupy.matmul` on the A10 | set by the library | any n |

The Metal sources are in `src/tiling_explainer/metal.py` and run through `mx.fast.metal_kernel`.
The CUDA sources are in `src/tiling_explainer/cuda.py`, written for CuPy's `RawKernel`, with the thread shapes in the table.
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

`uv run pytest` checks the Metal kernels against `mx.matmul`, the counting and ratio maths, the chip guard, the committed files and their recorded commits.
The CUDA kernels are checked against cuBLAS inside every Modal timing call.
The test that compares each kernel with `mx.matmul` skips on Linux, where MLX has no Metal backend; the size-rejection test still runs.

## Run it on Modal

The A10 run needs a [Modal](https://modal.com/) account, set up once with `uv run modal setup`.
From this folder:

```bash
uv sync
uv run tiling-explainer a10
```

The command reads the current commit and runs `uv lock --check` before it uploads anything.
It refuses to start, with exit status 2, when `uv.lock` does not match `pyproject.toml`, or when files in this folder outside `out/` have uncommitted changes and n is 4096.
It then starts 3 timing calls and 1 Nsight Compute call in parallel, each in its own container with `gpu="A10"`.

Each call first reads `nvidia-smi`.
A call that lands on any GPU other than "NVIDIA A10", an NVIDIA A10G included, times nothing and returns a refusal.
The run records each refusal in `a10-runs.json` and starts a replacement call, up to 6 per run, and past that it exits with an error.

Each timing call compiles the four CUDA kernels with CuPy, checks every hand-written kernel against cuBLAS with `allclose(rtol=1e-4, atol=1e-3)`, and fails if any kernel misses.
It then gives each kernel 3 warm-up launches and 10 single-launch samples timed with CUDA events; the kernel's time is the upper median of the 10.
The Nsight Compute call profiles `naive`, `tile32`, `reg4x4` and cuBLAS twice each and keeps the second launch.
It first runs `ncu --query-metrics` and adds the first tensor-pipe metric the device lists, preferring `sm__inst_executed_pipe_tensor_op_hmma.sum`, and fails if the device lists none of them.
The run exits with status 2 and leaves `out/` alone when the cuBLAS kernel ran a nonzero tensor-pipe count, or when the calls report different GPU facts or software versions.
It does the same when the cuBLAS kernel name contains `tensorop`, `tf32`, `hmma`, `xmma`, `s1688`, `s16816` or `s884`, in upper or lower case.

`--n 2048` and `--n 1024` run smaller sizes and write to `runs/`, and `--calls 5` runs more timing calls; the minimum is 3.
The cost of a full n = 4096 run is not yet confirmed.

## What you get

- `out/speeds.json`: ratios to naive for `naive`, `tile32`, `reg4x4` and `vendor`, one entry per machine.
- `out/challenge.json`: the entry for `reg8x8`.
- `out/mac-runs.json`: every process's times, errors against `mx.matmul`, ratios to naive and to `mx.matmul`, and power readings.
- `out/bytes.json`: Nsight Compute counts for `naive`, `tile32`, `reg4x4` and `vendor` on the A10.
- `out/a10-runs.json`: every A10 call's container, GPU UUID, device facts, versions, times, errors against cuBLAS and kernel register counts, plus the raw Nsight Compute rows.

`speeds.json` and `challenge.json` carry the same fields and differ only in the kernels they list.

| Field | Meaning |
| --- | --- |
| `source.commit` | the commit checked out when the newest run that wrote the file started |
| `source.code` | this folder on GitHub at that commit |
| `machines[].id` | `m3-pro` on an M3 Pro (the chip name in lower case without "Apple", with hyphens), `a10` on the A10 |
| `machines[].name` | the chip name from `sysctl -n machdep.cpu.brand_string` on the Mac, from `nvidia-smi` on the A10 |
| `machines[].n` | matrix size |
| `machines[].runs` | number of processes on the Mac, of timing calls on the A10 |
| `machines[].conditions` | flat values only: `publication`, `dtype` (`float32`), `timingMethod`, `timingWarmups`, `timingLaunchesPerSample` and `timingSamples` on both machines; `acPower` and `lowPowerMode` on the Mac; `maxSmClockMHz`, `powerLimitW`, `smCount`, `l2Bytes` and `distinctGpus` on the A10 |
| `machines[].versions` | strings only: the commit the machine's run started from, and macOS, MLX and Python on the Mac; GPU name, driver, CUDA runtime, CUDA driver, CuPy, Nsight Compute, Modal and Python on the A10 |
| `kernels[].kernel` | the kernel id from the first table |
| `kernels[].ratioMin`, `kernels[].ratioMax` | lowest and highest per-process ratio, naive time / kernel time |
| `kernels[].gflopsThisRun` | median over the processes of 2n³ / time |

A rerun replaces only the entry with the same machine id and keeps the others, so `source.commit` names the newest run and each machine's `versions.commit` names its own.

| `bytes.json` field | Meaning |
| --- | --- |
| `gpu` | `NVIDIA A10` |
| `n` | matrix size of the profiled launches |
| `vendorKernel` | the cuBLAS kernel name, which must not contain the tensor-core name parts listed in Run it on Modal |
| `rows[].globalLoadRequests` | `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`, global-load requests into L1 |
| `rows[].sharedLoads` | `smsp__inst_executed_op_shared_ld.sum`, warp-level shared-memory load instructions |
| `rows[].l2Bytes` | `lts__t_bytes.sum`, bytes through L2 |
| `rows[].dramReadBytes` | `dram__bytes_read.sum`, bytes read from DRAM |

`a10-runs.json` adds each kernel's shared-load wavefronts, L1 hit rate, duration and tensor-pipe count from the same profile.
Its `vendorKernel` holds the cuBLAS kernel name, the tensor-pipe metric's name and its value, which must be 0.

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
- `tests/test_provenance.py` checks each file's `source.commit` against the last commit that touched that file, and each machine's `versions.commit` against the last commit that touched its own run file (`out/mac-runs.json` or `out/a10-runs.json`). Either commit or its parent passes, so a rerun committed after the run still passes.

## Sources

- Simon Boehm, [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM), the source of the 8 x 8 register tile shape.
- MLX documentation, [Build and Install](https://ml-explore.github.io/mlx/build/html/install.html) and [Custom Metal Kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html).
- CuPy documentation, [User-Defined Kernels](https://docs.cupy.dev/en/stable/user_guide/kernel.html).
- NVIDIA, [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html).
- Modal documentation, [GPU acceleration](https://modal.com/docs/guide/gpu).
