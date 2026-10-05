import time
from dataclasses import dataclass

import numpy as np

from ce_explainer.config import RunConfig
from ce_explainer.data import Dataset
from ce_explainer.model import Params, batch_grads, init_params


@dataclass(frozen=True)
class Trajectory:
    config: RunConfig
    learning_rate: float
    seed: int
    snapshots: list[Params]
    seconds: float


def sgd_step(params: Params, grads: Params, learning_rate: float) -> Params:
    return {name: value - learning_rate * grads[name] for name, value in params.items()}


def train_epoch(
    params: Params, data: Dataset, config: RunConfig, rate: float, rng: np.random.Generator
) -> Params:
    order = rng.permutation(len(data.y))
    for start in range(0, len(order), config.batch_size):
        batch = order[start : start + config.batch_size]
        grads = batch_grads(params, data.x[batch], data.y[batch], config.loss)
        params = sgd_step(params, grads, rate)
    return params


def train(config: RunConfig, learning_rate: float, seed: int, data: Dataset) -> Trajectory:
    began = time.perf_counter()
    rng = np.random.default_rng(seed)
    params = init_params(rng, config.init_std, config.hidden)
    snapshots = [params]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(config.epochs):
            params = train_epoch(params, data, config, learning_rate, rng)
            snapshots.append(params)
    return Trajectory(config, learning_rate, seed, snapshots, time.perf_counter() - began)
