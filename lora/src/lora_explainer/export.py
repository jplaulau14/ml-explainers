import json
from pathlib import Path
from typing import Any

import torch

CROP = 64
DIGITS = 6
VERSION = 1


def round_sig(value: float) -> float:
    return float(f"{value:.{DIGITS}g}")


def rounded(t: torch.Tensor) -> Any:
    if t.dim() == 0:
        return round_sig(t.item())
    return [rounded(row) for row in t]


def source_block(metadata: dict[str, Any]) -> dict[str, Any]:
    config = metadata["config"]
    return {
        "model": config["model_id"],
        "modelRevision": metadata["modelRevision"],
        "dataset": config["dataset_id"],
        "datasetRevision": metadata["datasetRevision"],
        "trainExamples": config["train_examples"],
        "epochs": config["epochs"],
        "learningRate": config["learning_rate"],
        "rank": config["rank"],
        "seed": config["seed"],
    }


def spectrum_block(delta: torch.Tensor) -> dict[str, Any]:
    u, s, vh = torch.linalg.svd(delta, full_matrices=False)
    return {
        "frobeniusNorm": round_sig(torch.linalg.matrix_norm(delta).item()),
        "singularValues": rounded(s),
        "crop": {"rows": [0, CROP], "cols": [0, CROP]},
        "maxRank": CROP,
        "u": rounded(u[:CROP, :CROP]),
        "v": rounded(vh.T[:CROP, :CROP]),
        "sigma": rounded(s[:CROP]),
        "original": rounded(delta[:CROP, :CROP]),
    }


def build_payload(
    delta: torch.Tensor, run: str, matrix: str, metadata: dict[str, Any]
) -> dict[str, Any]:
    header = {
        "version": VERSION,
        "run": run,
        "matrix": matrix,
        "layer": metadata["config"]["layer"],
        "shape": list(delta.shape),
        "source": source_block(metadata),
    }
    return header | spectrum_block(delta.to(torch.float64))


def export_path(out_dir: Path, run: str, matrix: str) -> Path:
    return out_dir / f"{run}-{matrix}.json"


def write_payload(payload: dict[str, Any], out_dir: Path) -> Path:
    path = export_path(out_dir, payload["run"], payload["matrix"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    return path
