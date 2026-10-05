import numpy as np


def entropy(p: np.ndarray, base: float = np.e) -> float:
    nonzero = p[p > 0]
    return float(-(nonzero * np.log(nonzero)).sum() / np.log(base))


def cross_entropy(p: np.ndarray, q: np.ndarray, base: float = np.e) -> float:
    mask = p > 0
    return float(-(p[mask] * np.log(q[mask])).sum() / np.log(base))


def kl_divergence(p: np.ndarray, q: np.ndarray, base: float = np.e) -> float:
    mask = p > 0
    return float((p[mask] * np.log(p[mask] / q[mask])).sum() / np.log(base))


def surprise(p: float, base: float = 2.0) -> float:
    return float(-np.log(p) / np.log(base))


def bits_from_nats(nats: float) -> float:
    return nats / np.log(2.0)
