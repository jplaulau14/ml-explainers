# memory-bound

## What this is

Companion code for the essay [Why AI is mostly a memory problem](https://www.patslaurel.com/writing/why-ai-is-mostly-a-memory-problem).

A decoder reads every weight to produce one new token, and a multiply-add is two FLOPs, so a token does about two FLOPs per weight. In 16-bit that is one FLOP per byte of weight. A modern GPU can do hundreds of FLOPs for each byte it can load, so at batch size 1 the token rate sits near memory bandwidth divided by the bytes read per token. Wider batches share that read. Narrower dtypes shrink it. The KV cache adds bytes that grow with context. Prompt processing is the other side of the same ratio: many tokens share one weight read, so the limit moves to compute.

This folder measures that on a CPU and writes the numbers the essay widgets read.

## Run it

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from this folder:

```bash
uv sync
uv run mb-explainer all
```

That probes bandwidth and peak matmul, sweeps a Qwen2.5-7B-sized layer, runs Qwen2.5-0.5B decode and prefill, and writes `out/`. The model download is about 1 GB. No GPU and no login. `--quick` shortens the sweep for a smoke test. `uv run pytest` checks the arithmetic and the committed JSON without downloading.

Pieces, if you want them one at a time: `probe`, `roofline`, `decode`, `kv`, `specs`, `report`. `roofline`, `decode`, `kv`, and `report` read `out/machine.json`.

## What you get

- `out/machine.json`: CPU, caches, RAM, library versions, STREAM-style copy and triad, square-gemm peak.
- `out/roofline.json`: one layer of Qwen2.5-7B projections as a single GEMM, token counts 1 through 1024, achieved FLOP/s, arithmetic intensity, and the measured ceilings.
- `out/decode.json`: Qwen2.5-0.5B prefill and decode tokens/s against the bandwidth ceiling, plus a batch sweep.
- `out/kv_cache.json`: KV bytes per token and the bandwidth ceiling at several context lengths, from published configs.
- `out/specs.json`: a few chips with cited bandwidth, memory, and dense FP16 rates.
- `out/report.md`: the same numbers in tables, with how each one was measured.

Every JSON file has `version`, `gitCommit`, and `quick`. `quick` is false on the committed run. Timings move if you rerun. The FLOP and byte counts do not.

## Results

Committed `out/` is one full run on the guest that wrote `out/machine.json`: a 4-core Intel Xeon, 335544320 bytes of L3, 16791945216 bytes of RAM. Rates below are SI (1 GB/s = 1e9 bytes/s). The same tables are in `out/report.md`.

| Kernel | Best GB/s | Median GB/s |
| --- | --- | --- |
| float64 copy | 61.397 | 60.427 |
| float64 triad | 65.594 | 54.216 |
| float32 triad | 65.913 | 64.281 |

The ceiling used everywhere below is the best float64 triad, 65.594 GB/s. The median of those repeats was 54.216 GB/s, so the best call is a noisy peak.

| Square gemm | Best GFLOP/s | Median GFLOP/s |
| --- | --- | --- |
| float32, n = 4096 | 860.800 | 849.963 |
| bfloat16, n = 4096 | 4805.891 | 3452.043 |

Ridge intensity from the float32 peak and the triad is 13.123 FLOP/byte. The analytical crossing for the 3584 × 65024 layer is 26.451 tokens. The first measured point at or above 80% of that float32 peak is 512 tokens.

| Tokens | FLOP/byte | GFLOP/s | Traffic GB/s |
| --- | --- | --- | --- |
| 1 | 0.500 | 22.814 | 45.641 |
| 32 | 15.851 | 362.661 | 22.880 |
| 512 | 222.467 | 758.988 | 3.412 |
| 1024 | 393.404 | 795.507 | 2.022 |

One token is 0.5 FLOP/byte, as the arithmetic says for float32. Its traffic on this run was 45.641 GB/s, under the best triad. From there, achieved FLOP/s climbs toward the float32 gemm.

Qwen2.5-0.5B, batch 1, 128 prompt tokens, 32 new tokens. Decode does not include the prefill.

| Dtype | Decode tokens/s | Weight ceiling | Ratio | Prefill tokens/s |
| --- | --- | --- | --- | --- |
| bfloat16 | 23.825 | 66.387 | 0.359 | 1563.818 |
| float32 | 23.331 | 33.193 | 0.703 | 505.482 |
| int8 | 51.424 | 132.715 | 0.387 | 947.482 |

bfloat16 decode did not pull ahead of float32. Prefill did, and so did the square bfloat16 gemm. int8 decode is a bit more than twice float32, while the ceiling bytes dropped from 1976131072 to 494247424. It is still well under that ceiling. Adding the KV cache at the mean context (143.5 tokens) barely moves the ceilings: 66.268, 33.134, and 131.775 tokens/s.

| Dtype | Batch 1 | Batch 2 | Batch 4 | Batch 8 |
| --- | --- | --- | --- | --- |
| bfloat16 | 23.825 | 48.238 | 88.025 | 171.610 |
| float32 | 23.331 | 42.414 | 51.801 | 89.760 |
| int8 | 51.424 | 90.606 | 149.626 | 249.913 |

Those are total tokens/s across the batch.

In float16, Qwen2.5-0.5B's KV cache is 12288 bytes/token. On this bandwidth the weight-only ceiling is 66.386 tokens/s at context 1 and 47.166 at context 32768. Qwen2.5-7B is 4.307 tokens/s at context 1 and 2.884 at context 131072. The other models and dtypes are in `out/kv_cache.json`.

| Chip | Memory | Bandwidth | Dense FP16 | FLOP/byte |
| --- | --- | --- | --- | --- |
| NVIDIA H100 SXM | 80GB | 3.35 TB/s | 989.5 TFLOP/s | 295.373 |
| NVIDIA A100 80GB SXM | 80GB HBM2e | 2039 GB/s | 312 TFLOP/s | 153.016 |
| NVIDIA GeForce RTX 4090 | 24 GB GDDR6X | 1008 GB/s | 330.3 TFLOP/s | 327.679 |

H100 989.5 is 1979/2. The datasheet's rounded dense figure is 1000 TFLOP/s at 3 TB/s. RTX 4090 is the FP16-accumulate tensor number. Apple M2 Ultra is omitted: the newsroom post gives 800 GB/s and up to 192 GB, and no dense 16-bit FLOP/s.

## How this maps to the essay

- Two FLOPs per weight, and one FLOP per byte at batch 1 in 16-bit: `tests/test_arithmetic.py`
- KV bytes and the context-length ceiling: `tests/test_kv.py`
- Parameter counts match the published safetensors totals: `tests/test_models.py`
- Chip figures match the cited sheets, dense rate with sparsity halved where the vendor says so: `tests/test_specs.py`
- A tied embedding is counted once, and the int8 ceiling skips the leftover fp32 table: `tests/test_weights.py`
- Committed files agree with those formulas: `tests/test_outputs.py`

## Notes

- Bandwidth is the best float64 triad. The working set is about four times the L3 size reported by the guest, and the kernel reads the result so the copy is not deleted.
- Peak FLOP/s is the best square `float32` gemm.
- The roofline matrix has the same weight count and the same multiply-adds as one Qwen2.5-7B layer of q, k, v, o, and MLP projections. It leaves out the small norm and bias tensors, and it is not the attention score math.
- Decode times new tokens only. The prompt forward is a separate prefill measurement.
- The token ceiling is bandwidth divided by bytes. Those bytes are the matmul weights, the biases, and the norms. On this tied model the output projection is the embedding matrix, so that matrix is in the count. int8 leaves the fp32 embedding in memory and packs a second copy. The ceiling uses the packed copy.
- int8 is `torch.ao.quantization.quantize_dynamic` on `nn.Linear`. If that does not pack the linears, the file says so and there is no speed claim.
- Speculative rows use the expectation from Leviathan et al. if each draft token is accepted with a fixed probability and drafting is free. They are not a measurement on this machine.
- Apple M2 Ultra is listed under `omitted` because the newsroom post gives bandwidth and memory and does not give dense 16-bit FLOP/s.
- The H100 dense FP16 figure is 1979/2. The product page prints 1,979 with a sparsity footnote. The datasheet rounds the same chip to 2,000 TFLOP/s sparse and 3 TB/s.

## Sources

- NVIDIA H100 product page: https://www.nvidia.com/en-us/data-center/h100/
- NVIDIA tensor core datasheet: https://resources.nvidia.com/en-us-data-center-overview/nvidia-tensor-core-gpu-datasheet
- NVIDIA A100 datasheet: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
- NVIDIA Ada GPU architecture (RTX 4090 appendix): https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
- Apple M2 Ultra newsroom: https://www.apple.com/newsroom/2023/06/apple-introduces-m2-ultra/
- Leviathan, Kalman, Matias 2023, Fast Inference from Transformers via Speculative Decoding. https://arxiv.org/abs/2211.17192
- Configs and parameter totals: the Hugging Face `config.json` and model API URLs recorded in `out/kv_cache.json`
