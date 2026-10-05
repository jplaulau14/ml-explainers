import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ce_explainer.config import PRESETS
from ce_explainer.data import Dataset, Splits, source_block
from ce_explainer.field import (
    LEVELS_PER_DECADE,
    MAX_LEVEL,
    SCHEME,
    field_bytes,
    snapshot_bytes,
    to_base64,
)
from ce_explainer.metrics import (
    BUCKET_EDGES,
    gradient_buckets,
    probabilities,
    right_class,
    start_stats,
)
from ce_explainer.model import Params

VERSION = 1
LIKELIHOOD_EPOCHS = (0, 1, 2, 3, 5, 10, 20, 30)
HARD_COUNT = 20
LIKELIHOOD_RUN = "gentle-ce"
HERO_RUNS = {"ce": "confident-ce", "mse": "confident-mse"}
Run = tuple[list[Params], dict[str, Any]]


def round_sig(value: float, digits: int = 4) -> float | None:
    return float(f"{value:.{digits}g}") if math.isfinite(value) else None


def round_dec(value: float, digits: int = 4) -> float | None:
    return round(float(value), digits) if math.isfinite(value) else None


def rounded(value: Any, fn: Any = round_sig) -> Any:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        return {key: rounded(item, fn) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [rounded(item, fn) for item in value]
    return fn(value)


def fixed_indices(test: Dataset) -> list[int]:
    return [int(np.flatnonzero(test.y == digit)[0]) for digit in range(10)]


def hook_index(start: Params, test: Dataset, fixed: list[int]) -> int:
    right = right_class(probabilities(start, test.x[fixed]), test.y[fixed])
    return fixed[int(np.argmin(right))]


def example(test: Dataset, index: int) -> dict[str, Any]:
    return {"testIndex": index, "label": int(test.y[index]), "pixels": test.pixels[index].tolist()}


def example_probs(snapshots: list[Params], x: np.ndarray) -> list[Any]:
    return [rounded(probabilities(params, x), round_dec) for params in snapshots]


def seed_mean_final(metrics: dict[str, Any]) -> float:
    return float(np.mean([curve["accuracy"][-1] for curve in metrics["seeds"].values()]))


def hero_run(run: Run, test: Dataset) -> dict[str, Any]:
    snapshots, metrics = run
    return {
        "learningRate": metrics["learningRate"],
        "testAccuracy": rounded(metrics["seeds"]["0"]["accuracy"], round_dec),
        "finalAccuracyMean": round_dec(seed_mean_final(metrics)),
    }


def hero_payload(splits: Splits, runs: dict[str, Run]) -> dict[str, Any]:
    test = splits.test
    start = runs[HERO_RUNS["ce"]][0][0]
    return {
        "version": VERSION,
        "source": source_block(splits) | {"seed": 0, "runs": HERO_RUNS},
        "start": {"initStd": PRESETS[HERO_RUNS["ce"]].init_std} | rounded(start_stats(start, test)),
        "labels": test.y.tolist(),
        "encoding": {"scheme": SCHEME, "levelsPerDecade": LEVELS_PER_DECADE, "maxLevel": MAX_LEVEL},
        "epoch0": to_base64(snapshot_bytes(start, test)),
        "runs": {loss: hero_run(runs[name], test) for loss, name in HERO_RUNS.items()},
    }


def field_payload(splits: Splits, runs: dict[str, Run]) -> dict[str, Any]:
    test = splits.test
    return {
        "version": VERSION,
        "source": source_block(splits) | {"seed": 0, "runs": HERO_RUNS},
        "count": len(test.y),
        "epochs": len(runs[HERO_RUNS["ce"]][0]),
        "encoding": {"scheme": SCHEME, "levelsPerDecade": LEVELS_PER_DECADE, "maxLevel": MAX_LEVEL},
        "runs": {
            loss: to_base64(field_bytes(runs[name][0], test)) for loss, name in HERO_RUNS.items()
        },
    }


def run_config(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    config = PRESETS[name]
    return {
        "loss": config.loss,
        "start": config.start,
        "initStd": config.init_std,
        "learningRate": metrics["learningRate"],
        "batchSize": config.batch_size,
        "epochs": config.epochs,
    }


def replay_run(name: str, run: Run, splits: Splits, fixed: list[int]) -> dict[str, Any]:
    snapshots, metrics = run
    seed0 = metrics["seeds"]["0"]
    seeds = [metrics["seeds"][key]["accuracy"] for key in sorted(metrics["seeds"], key=int)]
    return {
        "config": run_config(name, metrics),
        "testAccuracy": rounded(seed0["accuracy"], round_dec),
        "testCrossEntropy": rounded(seed0["crossEntropy"]),
        "testSquaredError": rounded(seed0["squaredError"]),
        "seedAccuracy": rounded(seeds, round_dec),
        "probs": example_probs(snapshots, splits.test.x[fixed]),
        "gradientBuckets": bucket_columns([gradient_buckets(p, splits.train) for p in snapshots]),
    }


def bucket_columns(epochs: list[list[dict[str, float]]]) -> dict[str, Any]:
    columns = {key: [[bucket[key] for bucket in row] for row in epochs] for key in epochs[0][0]}
    return {key: value if key == "count" else rounded(value) for key, value in columns.items()}


def replay_payload(splits: Splits, runs: dict[str, Run], fixed: list[int]) -> dict[str, Any]:
    return {
        "version": VERSION,
        "source": source_block(splits),
        "bucketEdges": list(BUCKET_EDGES),
        "examples": [example(splits.test, index) for index in fixed],
        "runs": {name: replay_run(name, run, splits, fixed) for name, run in runs.items()},
    }


def likelihood_payload(splits: Splits, snapshots: list[Params]) -> dict[str, Any]:
    test = splits.test
    probs = [probabilities(snapshots[epoch], test.x) for epoch in LIKELIHOOD_EPOCHS]
    right = [right_class(p, test.y) for p in probs]
    hard = np.argsort(right[-1], kind="stable")[:HARD_COUNT]
    return {
        "version": VERSION,
        "source": source_block(splits) | {"run": LIKELIHOOD_RUN, "seed": 0},
        "epochs": list(LIKELIHOOD_EPOCHS),
        "labels": test.y.tolist(),
        "pTrue": rounded(right),
        "predicted": [p.argmax(axis=1).tolist() for p in probs],
        "hard": [example(test, int(index)) for index in hard],
    }


def write_payload(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    return path
