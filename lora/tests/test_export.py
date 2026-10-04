import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

OUT = Path(__file__).resolve().parents[1] / "out"
EXPORTS = sorted(path for path in OUT.glob("*.json") if path.name != "results.json")
KEYS = {
    "version",
    "run",
    "matrix",
    "layer",
    "shape",
    "source",
    "frobeniusNorm",
    "singularValues",
    "crop",
    "maxRank",
    "u",
    "v",
    "sigma",
    "original",
}
MAX_BYTES = 150_000


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def rebuild(payload: dict[str, Any], r: int) -> torch.Tensor:
    u = torch.tensor(payload["u"], dtype=torch.float64)[:, :r]
    v = torch.tensor(payload["v"], dtype=torch.float64)[:, :r]
    sigma = torch.tensor(payload["sigma"], dtype=torch.float64)[:r]
    return u @ torch.diag(sigma) @ v.T


@pytest.mark.parametrize("path", EXPORTS, ids=lambda p: p.name)
def test_keys_and_shapes(path: Path) -> None:
    payload = load(path)
    assert set(payload) == KEYS
    assert payload["run"] in {"full", "lora64"} and payload["matrix"] in {"q", "v"}
    assert payload["shape"] == [768, 768]
    assert len(payload["singularValues"]) == 768
    assert len(payload["sigma"]) == payload["maxRank"] == 64
    for key in ("u", "v", "original"):
        assert torch.tensor(payload[key]).shape == (64, 64)
    assert path.stat().st_size < MAX_BYTES


@pytest.mark.parametrize("path", EXPORTS, ids=lambda p: p.name)
def test_spectrum_is_consistent(path: Path) -> None:
    payload = load(path)
    s = payload["singularValues"]
    assert s == sorted(s, reverse=True)
    assert payload["sigma"] == s[:64]
    norm = math.sqrt(sum(value**2 for value in s))
    assert payload["frobeniusNorm"] == pytest.approx(norm, rel=1e-5)


@pytest.mark.parametrize("path", EXPORTS, ids=lambda p: p.name)
def test_higher_rank_rebuild_is_closer(path: Path) -> None:
    payload = load(path)
    original = torch.tensor(payload["original"], dtype=torch.float64)
    errors = [torch.linalg.matrix_norm(original - rebuild(payload, r)).item() for r in (1, 64)]
    assert errors[1] < errors[0]
