# Memory-bound measurements



## Machine

CPU: Intel(R) Xeon(R) Processor (family 6, model 207, stepping 2), 4 cores.
L3: 335544320 bytes. Flags of interest: avx2, avx512f, avx512_bf16, amx_bf16.
RAM: 16791945216 bytes total, 9252442112 bytes available when the probe started.
Python 3.11.17, numpy 2.4.6, torch 2.14.1+cpu, transformers 5.18.0.
git commit b1c888febd2184f3a2152b9878e2d2638e704128.

Bandwidth is a STREAM-style kernel in torch. Each call moves the whole array. The working set is about four times L3, capped by available RAM. The best call is the achievable figure. The median is next to it.

| Kernel | Working set | Exceeds L3 | Best GB/s | Median GB/s |
| --- | --- | --- | --- | --- |
| float64 copy | 894784848 | true | 61.397 | 60.427 |
| float64 triad | 1342177272 | true | 65.594 | 54.216 |
| float32 triad | 1342177272 | true | 65.913 | 64.281 |

Peak compute is a square gemm, best of the timed calls.

| Dtype | N | Status | Best GFLOP/s | Median GFLOP/s |
| --- | --- | --- | --- | --- |
| float32 | 4096 | ok | 860.800 | 849.963 |
| bfloat16 | 4096 | ok | 4805.891 | 3452.043 |

## Roofline

Matrix: Qwen/Qwen2.5-7B one layer of projections, 3584 by 65024, float32, 233046016 weight elements.
Config: https://huggingface.co/Qwen/Qwen2.5-7B/raw/main/config.json at d149729398750b98c0af14eb82c78cfe92750796.
Left out of the matrix: 4608 bias elements and 7168 norm elements.
Bandwidth ceiling 65.594 GB/s. Compute ceiling 860.800 GFLOP/s. Ridge intensity 13.123 FLOP/byte.
Analytical token count where this matrix crosses the ridge: 26.451. Smallest measured point at or above 80% of the compute ceiling: 512.
That compute ceiling is the float32 square gemm. The bfloat16 square gemm is a separate row in the machine table.

Arithmetic intensity counts both inputs and the output. Achieved FLOP/s uses the median.

| Tokens | FLOP/byte | GFLOP/s | Traffic GB/s |
| --- | --- | --- | --- |
| 1 | 0.500 | 22.814 | 45.641 |
| 2 | 0.999 | 31.130 | 31.148 |
| 4 | 1.998 | 61.770 | 30.921 |
| 8 | 3.991 | 127.769 | 32.018 |
| 16 | 7.962 | 222.998 | 28.006 |
| 32 | 15.851 | 362.661 | 22.880 |
| 64 | 31.408 | 470.815 | 14.990 |
| 128 | 61.676 | 590.646 | 9.577 |
| 256 | 119.029 | 516.597 | 4.340 |
| 512 | 222.467 | 758.988 | 3.412 |
| 1024 | 393.404 | 795.507 | 2.022 |

FLOPs per byte if a token touched each weight once and did two FLOPs on it:

| Dtype | Bytes | FLOP/byte at batch 1 |
| --- | --- | --- |
| float32 | 4 | 0.500 |
| float16 | 2 | 1.000 |
| bfloat16 | 2 | 1.000 |
| int8 | 1 | 2.000 |

## Decode

Model: Qwen/Qwen2.5-0.5B, catalog revision 060db6499f32faf8b98477b0a26969ef7d8b9987.
Config: https://huggingface.co/Qwen/Qwen2.5-0.5B/raw/main/config.json.
Prompt: "The capital of France is Paris.", repeated out to 128 tokens. Then 32 new tokens, greedy, cache on. The prefill is not inside the decode timer. Median of 2 calls after one warmup. 4 threads.
Published parameter count 494032768 from https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B.
Bandwidth used for the ceilings: 65.594 GB/s.

### bfloat16

Ceiling bytes 988065536, of which linear weights 987922432. Elements in that count 494032768. Quantized linear layers: 0.
The stored weights are larger than L3.
Prefill 1563.818 tokens/s.
Decode 23.825 tokens/s.
Weight-only ceiling 66.387 tokens/s (ratio 0.359).
Weights plus KV at mean context 143.500: ceiling 66.268 tokens/s (ratio 0.360).
KV cache measured 12288 bytes/token, formula 12288 bytes/token, element size 2.
Measured decode is below that ceiling. One token at a time does not keep the memory bus as busy as the triad kernel.

### float32

Ceiling bytes 1976131072, of which linear weights 1975844864. Elements in that count 494032768. Quantized linear layers: 0.
The stored weights are larger than L3.
Prefill 505.482 tokens/s.
Decode 23.331 tokens/s.
Weight-only ceiling 33.193 tokens/s (ratio 0.703).
Weights plus KV at mean context 143.500: ceiling 33.134 tokens/s (ratio 0.704).
KV cache measured 24576 bytes/token, formula 24576 bytes/token, element size 4.
Measured decode is below that ceiling. One token at a time does not keep the memory bus as busy as the triad kernel.

### int8

Ceiling bytes 494247424, of which linear weights 494071808. Elements in that count 494032768. Quantized linear layers: 169.
The stored weights are larger than L3.
Prefill 947.482 tokens/s.
Decode 51.424 tokens/s.
Weight-only ceiling 132.715 tokens/s (ratio 0.387).
Weights plus KV at mean context 143.500: ceiling 131.775 tokens/s (ratio 0.390).
KV cache measured 24576 bytes/token, formula 24576 bytes/token, element size 4.
Measured decode is below that ceiling. One token at a time does not keep the memory bus as busy as the triad kernel.

## Batch

Total tokens/s counts every sequence. Per sequence divides that by the batch.

| Dtype | Batch | Total tokens/s | Per sequence |
| --- | --- | --- | --- |
| bfloat16 | 1 | 23.825 | 23.825 |
| bfloat16 | 2 | 48.238 | 24.119 |
| bfloat16 | 4 | 88.025 | 22.006 |
| bfloat16 | 8 | 171.610 | 21.451 |
| float32 | 1 | 23.331 | 23.331 |
| float32 | 2 | 42.414 | 21.207 |
| float32 | 4 | 51.801 | 12.950 |
| float32 | 8 | 89.760 | 11.220 |
| int8 | 1 | 51.424 | 51.424 |
| int8 | 2 | 90.606 | 45.303 |
| int8 | 4 | 149.626 | 37.406 |
| int8 | 8 | 249.913 | 31.239 |

## KV cache

Bytes per token = 2 * layers * kvHeads * headDim * bytesPerElement. The ceiling is bandwidth / (weight bytes + kv bytes per token * context).
Bandwidth 65.594 GB/s. parameters times the dtype width, embeddings and norms included.

### openai-community/gpt2

124439808 parameters, 12 layers, 12 KV heads, head dim 64, max position 1024.
Config https://huggingface.co/openai-community/gpt2/raw/main/config.json at 607a30d783dfa663caf39e06633721c8d4cfcd7e.
The safetensors file has 137022720 elements. 12582912 of those are causal masks (attn.bias), not parameters.

| Context | KV bytes/token | Bytes/token with weights | Tokens/s ceiling |
| --- | --- | --- | --- |
| 1 | 36864 | 248916480 | 263.519 |
| 1024 | 36864 | 286628352 | 228.848 |
| 4096 | 36864 | 399874560 | 164.037 |
| 32768 | 36864 | 1456839168 | 45.025 |

### openai-community/gpt2-medium

354823168 parameters, 24 layers, 16 KV heads, head dim 64, max position 1024.
Config https://huggingface.co/openai-community/gpt2-medium/raw/main/config.json at 6dcaa7a952f72f9298047fd5137cd6e4f05f41da.
The safetensors file has 379988992 elements. 25165824 of those are causal masks (attn.bias), not parameters.

| Context | KV bytes/token | Bytes/token with weights | Tokens/s ceiling |
| --- | --- | --- | --- |
| 1 | 98304 | 709744640 | 92.420 |
| 1024 | 98304 | 810309632 | 80.950 |
| 4096 | 98304 | 1112299520 | 58.972 |
| 32768 | 98304 | 3930871808 | 16.687 |

### Qwen/Qwen2.5-0.5B

494032768 parameters, 24 layers, 2 KV heads, head dim 64, max position 32768.
Config https://huggingface.co/Qwen/Qwen2.5-0.5B/raw/main/config.json at 060db6499f32faf8b98477b0a26969ef7d8b9987.

| Context | KV bytes/token | Bytes/token with weights | Tokens/s ceiling |
| --- | --- | --- | --- |
| 1 | 12288 | 988077824 | 66.386 |
| 1024 | 12288 | 1000648448 | 65.552 |
| 4096 | 12288 | 1038397184 | 63.169 |
| 32768 | 12288 | 1390718720 | 47.166 |

### Qwen/Qwen2.5-7B

7615616512 parameters, 28 layers, 4 KV heads, head dim 128, max position 131072.
Config https://huggingface.co/Qwen/Qwen2.5-7B/raw/main/config.json at d149729398750b98c0af14eb82c78cfe92750796.

| Context | KV bytes/token | Bytes/token with weights | Tokens/s ceiling |
| --- | --- | --- | --- |
| 1 | 57344 | 15231290368 | 4.307 |
| 1024 | 57344 | 15289953280 | 4.290 |
| 4096 | 57344 | 15466114048 | 4.241 |
| 32768 | 57344 | 17110281216 | 3.834 |
| 131072 | 57344 | 22747425792 | 2.884 |

### Qwen/Qwen2.5-72B

72706203648 parameters, 80 layers, 8 KV heads, head dim 128, max position 131072.
Config https://huggingface.co/Qwen/Qwen2.5-72B/raw/main/config.json at efba10c8e54e91e0d9570ab5f7b51a958474d4cb.

| Context | KV bytes/token | Bytes/token with weights | Tokens/s ceiling |
| --- | --- | --- | --- |
| 1 | 327680 | 145412734976 | 0.451 |
| 1024 | 327680 | 145747951616 | 0.450 |
| 4096 | 327680 | 146754584576 | 0.447 |
| 32768 | 327680 | 156149825536 | 0.420 |
| 131072 | 327680 | 188362080256 | 0.348 |

## Speculative decoding

Expected tokens from one target-model step when each draft token is accepted with the given probability and drafting is free. Upper bound from the paper, not a run on this machine.
Source: https://arxiv.org/abs/2211.17192.

| Acceptance | Draft tokens | Expected tokens per step |
| --- | --- | --- |
| 0.600 | 1 | 1.600 |
| 0.600 | 2 | 1.960 |
| 0.600 | 4 | 2.306 |
| 0.600 | 8 | 2.475 |
| 0.800 | 1 | 1.800 |
| 0.800 | 2 | 2.440 |
| 0.800 | 4 | 3.362 |
| 0.800 | 8 | 4.329 |
| 0.900 | 1 | 1.900 |
| 0.900 | 2 | 2.710 |
| 0.900 | 4 | 4.095 |
| 0.900 | 8 | 6.126 |

## Hardware specs

Bandwidth is converted with SI units: 1 GB/s = 1e9 bytes/s and 1 TB/s = 1e12 bytes/s. Memory capacity is the vendor's string.

| Chip | Memory | Bandwidth | Dense FP16 | FLOP/byte |
| --- | --- | --- | --- | --- |
| NVIDIA H100 SXM | 80GB | 3.35TB/s | 989.5 TFLOP/s | 295.373 |
| NVIDIA A100 80GB SXM | 80GB HBM2e | 2039GB/s | 312 TFLOP/s | 153.016 |
| NVIDIA GeForce RTX 4090 | 24 GB GDDR6X | 1008 GB/sec | 330.3 TFLOP/s | 327.679 |

NVIDIA H100 SXM: The H100 product page lists 1,979 teraFLOPS of FP16 tensor core for H100 SXM with the footnote 'With sparsity'. The H100 datasheet says specifications shown with sparsity are 1/2 lower without sparsity. Sources: https://www.nvidia.com/en-us/data-center/h100/, https://resources.nvidia.com/en-us-data-center-overview/nvidia-tensor-core-gpu-datasheet.

NVIDIA A100 80GB SXM: The datasheet lists FP16 Tensor Core as 312 TFLOPS | 624 TFLOPS*. The footnote says the starred number is with sparsity. Sources: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf.

NVIDIA GeForce RTX 4090: Appendix A lists Peak FP16 Tensor TFLOPS with FP16 Accumulate as 330.3/660.6. Footnote 2 says the second number uses the sparsity feature. The same table lists FP32-accumulate FP16 tensor core as 165.2/330.4. Sources: https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf.

Apple M2 Ultra is not in the table. Apple's newsroom post lists 800GB/s of memory bandwidth and up to 192GB of unified memory. It does not publish dense 16-bit FLOP/s. Source: https://www.apple.com/newsroom/2023/06/apple-introduces-m2-ultra/.

## Caveats

These bandwidth and FLOP figures are what this process achieved.
The CPU name string comes from the guest and can be generic.
Arithmetic intensity is compulsory traffic, not a hardware counter of DRAM bytes.
The decode ceiling counts matmul weights, biases, and norms once per token.
On this tied model the output projection is the embedding matrix, so it is included.
int8 leaves the fp32 embedding in memory and packs a separate output matrix.
The int8 ceiling uses the packed copy, not the leftover table.
Dynamic int8 dequantizes inside the kernel, so fewer bytes may not mean more speed.
bfloat16 decode stayed near the float32 token rate on this CPU.
Prefill and the square bfloat16 gemm were much faster.
The H100 dense FP16 number is half the product page's sparse number.
The datasheet says sparse figures are twice the dense ones.
That datasheet rounds the same chip to 2,000 TFLOP/s sparse and 3 TB/s.
The RTX 4090 dense number is the FP16-accumulate tensor figure.
FP32-accumulate is lower and is in specs.json.
Speculative rows are not a measurement.
The roofline matmul skips the small norm and bias tensors.
It is not the attention score math.
Reruns move the timed numbers. The arithmetic does not.
