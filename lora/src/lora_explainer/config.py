from dataclasses import dataclass
from typing import Literal

Mode = Literal["full", "lora"]


@dataclass(frozen=True)
class RunConfig:
    name: str
    mode: Mode
    learning_rate: float
    rank: int | None = None
    model_id: str = "openai-community/gpt2"
    dataset_id: str = "stanfordnlp/sst2"
    layer: int = 6
    train_examples: int = 4000
    max_length: int = 64
    batch_size: int = 16
    epochs: int = 1
    seed: int = 0
    threads: int = 4


FULL = RunConfig(name="full", mode="full", learning_rate=5e-5)
LORA64 = RunConfig(name="lora64", mode="lora", learning_rate=5e-4, rank=64)
PRESETS = {preset.name: preset for preset in (FULL, LORA64)}
