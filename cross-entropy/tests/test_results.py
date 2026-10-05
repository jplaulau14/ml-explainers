import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ce_explainer.config import PRESETS, RATE_GRID
from ce_explainer.data import Dataset, load_splits
from ce_explainer.sweep import sweep_all

OUT = Path(__file__).resolve().parents[1] / "out"


def results() -> dict[str, Any]:
    return json.loads((OUT / "results.json").read_text())


def mean_final(name: str) -> float:
    return results()["finalAccuracy"][name]["mean"]


def test_claim_gentle_start_is_ln10() -> None:
    assert abs(results()["start"]["gentle"]["crossEntropy"] - math.log(10)) < 0.01


def test_claim_confident_start_gap() -> None:
    assert mean_final("confident-ce") - mean_final("confident-mse") >= 0.20


def test_claim_gentle_tie() -> None:
    assert abs(mean_final("gentle-ce") - mean_final("gentle-mse")) <= 0.01


def test_selected_rates_are_inside_the_grid() -> None:
    for name, rate in results()["selectedRates"].items():
        assert RATE_GRID[0] < rate < RATE_GRID[-1], name


def test_claim_lr_chosen_on_validation() -> None:
    splits = load_splits()
    rng = np.random.default_rng(0)
    test = splits.test
    scrambled = Dataset(x=rng.random(test.x.shape), y=rng.permutation(test.y), pixels=test.pixels)
    configs = [replace(PRESETS[name], epochs=2) for name in ("gentle-ce", "confident-mse")]
    grid, seeds = (0.3, 3.0), (0,)
    real = sweep_all(splits, configs, grid, seeds)
    assert sweep_all(replace(splits, test=scrambled), configs, grid, seeds) == real
