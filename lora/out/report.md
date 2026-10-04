# Report

## Accuracy

| Model | SST-2 validation accuracy |
| --- | --- |
| base | 0.6089 |
| full | 0.8888 |
| lora1 | 0.8360 |
| lora8 | 0.8555 |
| lora64 | 0.8704 |

## Relative error by rank

| Run | Matrix | r=1 | r=4 | r=8 | r=16 | r=32 | r=64 | 90% energy | 99% energy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | W_q | 0.7972 | 0.6635 | 0.5839 | 0.5127 | 0.4367 | 0.3518 | 84 | 327 |
| full | W_v | 0.8902 | 0.7124 | 0.6237 | 0.5395 | 0.4588 | 0.3710 | 95 | 359 |
| lora64 | W_q | 0.1598 | 0.0713 | 0.0539 | 0.0381 | 0.0218 | 0.0000 | 1 | 3 |
| lora64 | W_v | 0.3003 | 0.1049 | 0.0773 | 0.0524 | 0.0292 | 0.0000 | 1 | 5 |

## Run time

| Preset | Train seconds |
| --- | --- |
| full | 215 |
| lora1 | 135 |
| lora8 | 134 |
| lora64 | 134 |

CPU: Apple M3 Pro
