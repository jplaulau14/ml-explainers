import json
from pathlib import Path
from typing import Any

import torch

from lora_explainer.export import round_sig
from lora_explainer.spectrum import rank_for_energy, relative_error

RANKS = (1, 4, 8, 16, 32, 64)
THRESHOLDS = {"rankFor90": 0.9, "rankFor99": 0.99}


def matrix_summary(payload: dict[str, Any]) -> dict[str, Any]:
    s = torch.tensor(payload["singularValues"], dtype=torch.float64)
    errors = {str(r): round_sig(relative_error(s, r)) for r in RANKS}
    ranks = {key: rank_for_energy(s, threshold) for key, threshold in THRESHOLDS.items()}
    return {"run": payload["run"], "matrix": payload["matrix"], "relativeError": errors} | ranks


def build_results(
    base: float, metadata: dict[str, dict[str, Any]], payloads: list[dict[str, Any]]
) -> dict[str, Any]:
    accuracy = {"base": base} | {run: meta["accuracy"] for run, meta in metadata.items()}
    return {
        "accuracy": {name: round(value, 4) for name, value in accuracy.items()},
        "matrices": [matrix_summary(payload) for payload in payloads],
        "trainSeconds": {run: meta["trainSeconds"] for run, meta in metadata.items()},
        "cpu": next(iter(metadata.values()))["cpu"],
    }


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = [header, ["---"] * len(header), *rows]
    return ["| " + " | ".join(cells) + " |" for cells in lines]


def accuracy_table(results: dict[str, Any]) -> list[str]:
    rows = [[name, f"{value:.4f}"] for name, value in results["accuracy"].items()]
    return table(["Model", "SST-2 validation accuracy"], rows)


def error_table(results: dict[str, Any]) -> list[str]:
    header = ["Run", "Matrix", *[f"r={r}" for r in RANKS], "90% energy", "99% energy"]
    rows = [
        [m["run"], f"W_{m['matrix']}"]
        + [f"{m['relativeError'][str(r)]:.4f}" for r in RANKS]
        + [str(m["rankFor90"]), str(m["rankFor99"])]
        for m in results["matrices"]
    ]
    return table(header, rows)


def runtime_table(results: dict[str, Any]) -> list[str]:
    rows = [[run, f"{seconds:.0f}"] for run, seconds in results["trainSeconds"].items()]
    return table(["Preset", "Train seconds"], rows) + ["", f"CPU: {results['cpu']}"]


def render_markdown(results: dict[str, Any]) -> str:
    sections = [
        ["# Report"],
        ["## Accuracy", "", *accuracy_table(results)],
        ["## Relative error by rank", "", *error_table(results)],
        ["## Run time", "", *runtime_table(results)],
    ]
    return "\n\n".join("\n".join(section) for section in sections) + "\n"


def write_report(results: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (out_dir / "report.md").write_text(render_markdown(results))
