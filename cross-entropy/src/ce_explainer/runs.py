import json
from pathlib import Path
from typing import Any

import numpy as np

from ce_explainer.data import Dataset
from ce_explainer.metrics import curves
from ce_explainer.model import Params
from ce_explainer.train import Trajectory


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def save_run(directory: Path, trajectories: list[Trajectory], test: Dataset) -> None:
    first = trajectories[0]
    arrays = {
        f"{name}_{epoch}": value
        for epoch, params in enumerate(first.snapshots)
        for name, value in params.items()
    }
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "snapshots.npz", **arrays)
    metrics = {
        "learningRate": first.learning_rate,
        "trainSeconds": sum(t.seconds for t in trajectories),
        "seeds": {str(t.seed): curves(t.snapshots, test) for t in trajectories},
    }
    write_json(directory / "metrics.json", metrics)


def load_snapshots(directory: Path) -> list[Params]:
    with np.load(directory / "snapshots.npz") as archive:
        arrays = {key: archive[key] for key in archive.files}
    epochs = 1 + max(int(key.rsplit("_", 1)[1]) for key in arrays)
    names = sorted({key.rsplit("_", 1)[0] for key in arrays})
    return [{name: arrays[f"{name}_{epoch}"] for name in names} for epoch in range(epochs)]


def load_metrics(directory: Path) -> dict[str, Any]:
    return read_json(directory / "metrics.json")
