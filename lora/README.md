# lora

## What this is

Companion code for the essay [Why LoRA works: fine-tuning a giant model with a tiny matrix](https://www.patslaurel.com/writing/why-lora-works).
It fine-tunes GPT-2 small on SST-2 twice, once in full and once with LoRA, and exports the weight update for two attention matrices so the essay can show how much of it a low-rank matrix can rebuild.

## Run it

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from this folder:

```bash
uv sync
uv run lora-explainer all
```

This downloads GPT-2 small and SST-2 (about 550 MB), trains both presets on the CPU and rewrites `out/`.
It took about 7 minutes on an Apple M3 Pro. No GPU, API key or login is needed.
Run one step at a time with `train --preset full`, `train --preset lora64`, `export` and `report`.
`uv run pytest` checks the math and the committed files without downloading anything.

## What you get

- `out/full-q.json`: spectrum and rank-64 factors of ΔW_q, layer 6, after full fine-tuning.
- `out/full-v.json`: the same for ΔW_v.
- `out/lora64-q.json`: ΔW_q = BA for layer 6 after LoRA with r = 64.
- `out/lora64-v.json`: the same for ΔW_v.
- `out/results.json`: accuracy, error by rank and run time as numbers.
- `out/report.md`: the same results as tables.

Each JSON file holds all 768 singular values, a 64 × 64 crop of ΔW, and the crop rows of U and V.
The essay rebuilds the crop at rank r as `u[:, :r] @ diag(sigma[:r]) @ v[:, :r].T`.

## Results

| Model | SST-2 validation accuracy |
| --- | --- |
| base | 0.6089 |
| full | 0.8888 |
| lora64 | 0.8704 |

Relative Frobenius error of the best rank-r rebuild of ΔW, and the rank needed to keep 90% and 99% of its energy:

| Run | Matrix | r=1 | r=4 | r=8 | r=16 | r=32 | r=64 | 90% energy | 99% energy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | W_q | 0.7972 | 0.6635 | 0.5839 | 0.5127 | 0.4367 | 0.3518 | 84 | 327 |
| full | W_v | 0.8902 | 0.7124 | 0.6237 | 0.5395 | 0.4588 | 0.3710 | 95 | 359 |
| lora64 | W_q | 0.1598 | 0.0713 | 0.0539 | 0.0381 | 0.0218 | 0.0000 | 1 | 3 |
| lora64 | W_v | 0.3003 | 0.1049 | 0.0773 | 0.0524 | 0.0292 | 0.0000 | 1 | 5 |

## How this maps to the essay

- Claim 1, BA has rank at most r: `tests/test_adapter.py::test_claim_1_update_rank_is_at_most_r`
- Claim 2, B = 0 means training starts from the base model: `tests/test_adapter.py::test_claim_2_injected_model_starts_identical`
- Claim 3, BA merges into W with no extra cost: `tests/test_adapter.py::test_claim_3_merged_weight_matches_adapter`
- Claim 4, 1536 times fewer parameters for a 12288 × 12288 matrix at r = 4: `tests/test_counts.py::test_claim_4_gpt3_sized_matrix`
- Claim 5, GPT-3 adapter sizes for r = 1, 4 and 8: `tests/test_counts.py::test_claim_5_gpt3_adapter_size`
- Claim 8, truncation error is the tail of the singular values: `tests/test_spectrum.py::test_claim_8_truncation_error_is_tail_of_spectrum`
- Claim 9, multiply-adds with and without merging: `tests/test_counts.py::test_claim_9_multiply_adds`

## Notes

- The notation follows Hu et al.: W is d × k, B is d × r, A is r × k and h = Wx + BAx.
- α = r, so the scale α/r is 1 and the code adds exactly BAx.
- A starts from Kaiming-uniform as in microsoft/LoRA. The paper describes a Gaussian.
- GPT-2 stores q, k and v in one `Conv1D`. W_q is `c_attn.weight[:, 0:768].T` and W_v is `c_attn.weight[:, 1536:2304].T`.
- For the full run, ΔW is W after training minus W before.
- Reruns on the same machine give the same numbers. Other CPUs can differ in the last digits.

## Sources

- Hu et al. 2021, LoRA: Low-Rank Adaptation of Large Language Models. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Biderman et al. 2024, LoRA Learns Less and Forgets Less. [arXiv:2405.09673](https://arxiv.org/abs/2405.09673)
