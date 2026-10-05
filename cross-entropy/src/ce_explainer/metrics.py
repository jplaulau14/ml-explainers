import numpy as np

from ce_explainer.data import Dataset, one_hot
from ce_explainer.model import Params, ce_logit_grad, forward, se_logit_grad, softmax

BUCKET_EDGES = (0.0, 1e-8, 1e-4, 1e-2, 0.1, 0.5, 0.9, 1.0)


def probabilities(params: Params, x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return softmax(forward(params, x)[0])


def right_class(p: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return p[np.arange(len(labels)), labels]


def accuracy(params: Params, data: Dataset) -> float:
    return float((probabilities(params, data.x).argmax(axis=1) == data.y).mean())


def evaluate(params: Params, data: Dataset) -> dict[str, float]:
    p = probabilities(params, data.x)
    with np.errstate(divide="ignore", invalid="ignore"):
        cross_entropy = -np.log(right_class(p, data.y)).mean()
    return {
        "accuracy": float((p.argmax(axis=1) == data.y).mean()),
        "crossEntropy": float(cross_entropy),
        "squaredError": float((0.5 * ((p - one_hot(data.y)) ** 2).sum(axis=1)).mean()),
    }


def curves(snapshots: list[Params], data: Dataset) -> dict[str, list[float]]:
    rows = [evaluate(params, data) for params in snapshots]
    return {key: [row[key] for row in rows] for key in rows[0]}


def start_stats(params: Params, data: Dataset) -> dict[str, float]:
    p = probabilities(params, data.x)
    right = right_class(p, data.y)
    return {
        "meanTopProbability": float(p.max(axis=1).mean()),
        "wrongShare": float((p.argmax(axis=1) != data.y).mean()),
        "rightBelow001Share": float((right < 0.01).mean()),
        "crossEntropy": float(-np.log(right).mean()),
    }


def bucket_of(right: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.array(BUCKET_EDGES[1:-1]), right, side="right")


def gradient_buckets(params: Params, data: Dataset) -> list[dict[str, float]]:
    p = probabilities(params, data.x)
    targets = one_hot(data.y)
    ce_norm = np.linalg.norm(ce_logit_grad(p, targets), axis=1)
    se_norm = np.linalg.norm(se_logit_grad(p, targets), axis=1)
    buckets = bucket_of(right_class(p, data.y))
    rows = []
    for bucket in range(len(BUCKET_EDGES) - 1):
        mask = buckets == bucket
        count = int(mask.sum())
        rows.append(
            {
                "count": count,
                "meanNormCe": float(ce_norm[mask].mean()) if count else 0.0,
                "meanNormMse": float(se_norm[mask].mean()) if count else 0.0,
            }
        )
    return rows
