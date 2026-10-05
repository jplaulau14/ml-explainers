import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ce_explainer.config import EXPORTED, FINAL_SEEDS, INIT_STD
from ce_explainer.data import load_splits
from ce_explainer.export import LIKELIHOOD_EPOCHS, fixed_indices, hook_index
from ce_explainer.model import init_params

OUT = Path(__file__).resolve().parents[1] / "out"
BUDGETS = {"hero.json": 6_000, "replay.json": 120_000, "likelihood.json": 40_000}
EPOCHS = 31


def load(name: str) -> dict[str, Any]:
    return json.loads((OUT / name).read_text())


def shape(value: Any) -> tuple[int, ...]:
    return np.array(value, dtype=float).shape


@pytest.mark.parametrize("name", sorted(BUDGETS))
def test_sizes(name: str) -> None:
    assert (OUT / name).stat().st_size < BUDGETS[name]
    assert load(name)["version"] == 1


def test_keys_shapes_and_sizes() -> None:
    hero, replay, likelihood = load("hero.json"), load("replay.json"), load("likelihood.json")
    assert set(hero) == {"version", "source", "example", "start", "runs"}
    assert len(hero["example"]["pixels"]) == 64
    for run in hero["runs"].values():
        assert shape(run["probs"]) == (EPOCHS, 10) and len(run["testAccuracy"]) == EPOCHS
    assert set(replay["runs"]) == set(EXPORTED) and len(replay["examples"]) == 10
    for run in replay["runs"].values():
        assert shape(run["probs"]) == (EPOCHS, 10, 10)
        assert shape(run["seedAccuracy"]) == (len(FINAL_SEEDS), EPOCHS)
        for column in run["gradientBuckets"].values():
            assert shape(column) == (EPOCHS, len(replay["bucketEdges"]) - 1)
    assert shape(likelihood["pTrue"]) == (len(LIKELIHOOD_EPOCHS), 450)
    assert len(likelihood["hard"]) == 20


def test_probabilities_sum_to_one() -> None:
    rows = [row for run in load("hero.json")["runs"].values() for row in run["probs"]]
    rows += [row for run in load("replay.json")["runs"].values() for e in run["probs"] for row in e]
    assert np.abs(np.array(rows).sum(axis=1) - 1).max() <= 1e-3


def test_hook_digit_rule() -> None:
    test = load_splits().test
    fixed = fixed_indices(test)
    assert [int(test.y[i]) for i in fixed] == list(range(10))
    assert [e["testIndex"] for e in load("replay.json")["examples"]] == fixed
    start = init_params(np.random.default_rng(0), INIT_STD["confident"], 0)
    assert load("hero.json")["example"]["testIndex"] == hook_index(start, test, fixed)


def test_replay_matches_results() -> None:
    results = load("results.json")
    for name, run in load("replay.json")["runs"].items():
        mean = np.mean([curve[-1] for curve in run["seedAccuracy"]])
        assert mean == pytest.approx(results["finalAccuracy"][name]["mean"], abs=1e-4)
