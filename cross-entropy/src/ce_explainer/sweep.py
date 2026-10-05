from collections.abc import Iterable
from typing import Any

import numpy as np

from ce_explainer.config import RATE_GRID, SWEEP_SEEDS, RunConfig
from ce_explainer.data import Dataset, Splits
from ce_explainer.metrics import accuracy
from ce_explainer.train import train


def validation_accuracy(
    config: RunConfig, rate: float, seed: int, data: Dataset, validation: Dataset
) -> float:
    return accuracy(train(config, rate, seed, data).snapshots[-1], validation)


def select_rate(
    config: RunConfig,
    data: Dataset,
    validation: Dataset,
    grid: tuple[float, ...] = RATE_GRID,
    seeds: tuple[int, ...] = SWEEP_SEEDS,
) -> dict[str, Any]:
    scores = {
        rate: float(
            np.mean([validation_accuracy(config, rate, s, data, validation) for s in seeds])
        )
        for rate in grid
    }
    best = max(grid, key=lambda rate: (scores[rate], -rate))
    table = [{"rate": rate, "validationAccuracy": score} for rate, score in scores.items()]
    return {"selected": best, "grid": table}


def sweep_all(
    splits: Splits,
    configs: Iterable[RunConfig],
    grid: tuple[float, ...] = RATE_GRID,
    seeds: tuple[int, ...] = SWEEP_SEEDS,
) -> dict[str, dict[str, Any]]:
    return {
        config.name: select_rate(config, splits.train, splits.validation, grid, seeds)
        for config in configs
    }
