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

## How this maps to the essay

- Two FLOPs per weight, and one FLOP per byte at batch 1 in 16-bit: `tests/test_arithmetic.py`
- KV bytes and the context-length ceiling: `tests/test_kv.py`
- Parameter counts match the published safetensors totals: `tests/test_models.py`
- Chip figures match the cited sheets, dense rate with sparsity halved where the vendor says so: `tests/test_specs.py`

## Notes

- Bandwidth is the best float64 triad. The working set is about four times the L3 size reported by the guest, and the kernel reads the result so the copy is not deleted.
- Peak FLOP/s is the best square `float32` gemm.
- The roofline matrix has the same weight count and the same multiply-adds as one Qwen2.5-7B layer of q, k, v, o, and MLP projections. It leaves out the small norm and bias tensors, and it is not the attention score math.
- Decode times new tokens only. The prompt forward is a separate prefill measurement.
- The token ceiling is bandwidth divided by bytes. The bytes are unique stored parameters, then those plus KV at the mean context during the decode.
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
