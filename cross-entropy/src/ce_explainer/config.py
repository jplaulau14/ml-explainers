from dataclasses import dataclass
from typing import Literal

Loss = Literal["ce", "mse"]
Start = Literal["gentle", "confident"]

INIT_STD: dict[str, float] = {"gentle": 0.01, "confident": 4.0}
RATE_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
SWEEP_SEEDS = (0, 1, 2)
FINAL_SEEDS = (0, 1, 2, 3, 4)
EPOCHS = 30
BATCH_SIZE = 32
HIDDEN = 32


@dataclass(frozen=True)
class RunConfig:
    name: str
    loss: Loss
    start: Start
    hidden: int = 0
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE

    @property
    def init_std(self) -> float:
        return INIT_STD[self.start]


def preset(loss: Loss, start: Start, hidden: int = 0) -> RunConfig:
    prefix = "mlp-" if hidden else ""
    return RunConfig(name=f"{prefix}{start}-{loss}", loss=loss, start=start, hidden=hidden)


STARTS: tuple[Start, ...] = ("confident", "gentle")
LOSSES: tuple[Loss, ...] = ("ce", "mse")
LINEAR = tuple(preset(loss, start) for start in STARTS for loss in LOSSES)
MLP = tuple(preset(loss, start, HIDDEN) for start in STARTS for loss in LOSSES)
PRESETS = {config.name: config for config in (*LINEAR, *MLP)}
EXPORTED = tuple(config.name for config in LINEAR)
