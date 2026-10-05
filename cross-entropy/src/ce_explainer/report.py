import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from ce_explainer.config import LINEAR, MLP, PRESETS
from ce_explainer.data import Splits
from ce_explainer.export import Run, fixed_indices, hook_index, round_dec, rounded
from ce_explainer.metrics import probabilities, right_class, start_stats

HOOK_EPOCHS = (0, 1, 2, 3, 5, 10, 20, 30)


def cpu_name() -> str:
    if platform.system() == "Darwin":
        command = ["sysctl", "-n", "machdep.cpu.brand_string"]
        return subprocess.run(command, capture_output=True, text=True).stdout.strip()
    if Path("/proc/cpuinfo").exists():
        lines = Path("/proc/cpuinfo").read_text().splitlines()
        names = [line.split(":", 1)[1].strip() for line in lines if line.startswith("model name")]
        return names[0] if names else platform.processor()
    return platform.processor()


def final_accuracy(metrics: dict[str, Any]) -> dict[str, Any]:
    curves = [metrics["seeds"][key]["accuracy"] for key in sorted(metrics["seeds"], key=int)]
    final = [curve[-1] for curve in curves]
    early = {f"epoch{e}Mean": float(np.mean([c[e] for c in curves])) for e in (1, 5)}
    summary = {"mean": float(np.mean(final)), "min": min(final), "max": max(final)} | early
    return rounded({"seeds": final} | summary, round_dec)


def hook_trajectory(snapshots: list[Any], splits: Splits, index: int) -> list[dict[str, Any]]:
    x, label = splits.test.x[[index]], splits.test.y[[index]]
    rows = []
    for epoch, params in enumerate(snapshots):
        p = probabilities(params, x)
        right = float(right_class(p, label)[0])
        rows.append({"epoch": epoch, "predicted": int(p.argmax()), "pLabel": right})
    return rows


def first_correct(rows: list[dict[str, Any]], label: int) -> int | None:
    return next((row["epoch"] for row in rows if row["predicted"] == label), None)


def hook_summary(runs: dict[str, Run], splits: Splits) -> dict[str, Any]:
    fixed = fixed_indices(splits.test)
    index = hook_index(runs["confident-ce"][0][0], splits.test, fixed)
    label = int(splits.test.y[index])
    trajectories = {
        loss: hook_trajectory(runs[f"confident-{loss}"][0], splits, index) for loss in ("ce", "mse")
    }
    return {
        "testIndex": index,
        "label": label,
        "firstCorrectEpoch": {k: first_correct(v, label) for k, v in trajectories.items()},
        "correctAtEpoch30": {k: v[-1]["predicted"] == label for k, v in trajectories.items()},
        "trajectory": trajectories,
    }


def start_summary(runs: dict[str, Run], splits: Splits) -> dict[str, Any]:
    names = {f"{c.start}{'Mlp' if c.hidden else ''}": c.name for c in (*LINEAR, *MLP)}
    return {key: rounded(start_stats(runs[name][0][0], splits.test)) for key, name in names.items()}


def build_results(
    sweep: dict[str, Any], runs: dict[str, Run], splits: Splits, seconds: dict[str, float]
) -> dict[str, Any]:
    return {
        "selectedRates": {name: sweep[name]["selected"] for name in PRESETS},
        "sweep": {name: rounded(sweep[name]["grid"], lambda v: v) for name in PRESETS},
        "finalAccuracy": {name: final_accuracy(runs[name][1]) for name in PRESETS},
        "start": start_summary(runs, splits),
        "hook": hook_summary(runs, splits),
        "seconds": {key: round(value, 2) for key, value in seconds.items()},
        "cpu": cpu_name(),
    }
