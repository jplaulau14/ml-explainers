import base64

import numpy as np

from ce_explainer.data import Dataset
from ce_explainer.metrics import probabilities, right_class
from ce_explainer.model import Params

LEVELS_PER_DECADE = 8
MAX_LEVEL = 127
CORRECT_BIT = 128
SCHEME = "byte = 128 * correct + min(127, round(8 * -log10(p))), epoch-major, base64"


def encode_levels(right: np.ndarray, correct: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        decades = -np.log10(right)
    levels = np.minimum(MAX_LEVEL, np.round(LEVELS_PER_DECADE * decades)).astype(np.uint8)
    return levels | (correct.astype(np.uint8) * CORRECT_BIT)


def decode_levels(encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    levels = (encoded & MAX_LEVEL).astype(np.float64)
    return 10.0 ** (-levels / LEVELS_PER_DECADE), encoded >= CORRECT_BIT


def snapshot_bytes(params: Params, test: Dataset) -> np.ndarray:
    p = probabilities(params, test.x)
    return encode_levels(right_class(p, test.y), p.argmax(axis=1) == test.y)


def field_bytes(snapshots: list[Params], test: Dataset) -> np.ndarray:
    return np.concatenate([snapshot_bytes(params, test) for params in snapshots])


def to_base64(encoded: np.ndarray) -> str:
    return base64.b64encode(encoded.astype(np.uint8).tobytes()).decode("ascii")


def from_base64(text: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype=np.uint8)
