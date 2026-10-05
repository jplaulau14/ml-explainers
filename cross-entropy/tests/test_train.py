from dataclasses import replace

import numpy as np

from ce_explainer.config import PRESETS
from ce_explainer.data import load_splits
from ce_explainer.model import mean_loss
from ce_explainer.train import train


def test_determinism() -> None:
    data = load_splits().train
    config = replace(PRESETS["gentle-ce"], epochs=2)
    first, second = train(config, 1.0, 0, data), train(config, 1.0, 0, data)
    other = train(config, 1.0, 1, data)
    for a, b in zip(first.snapshots, second.snapshots, strict=True):
        assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(first.snapshots[-1]["W"], other.snapshots[-1]["W"])


def test_training_lowers_the_loss() -> None:
    data = load_splits().train
    for name in ("gentle-ce", "gentle-mse"):
        config = PRESETS[name]
        run = train(replace(config, epochs=3), 1.0, 0, data)
        before, after = run.snapshots[0], run.snapshots[-1]
        assert mean_loss(after, data.x, data.y, config.loss) < mean_loss(
            before, data.x, data.y, config.loss
        )
