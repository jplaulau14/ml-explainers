import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ce_explainer.field import (
    LEVELS_PER_DECADE,
    MAX_LEVEL,
    decode_levels,
    encode_levels,
    from_base64,
    to_base64,
)

OUT = Path(__file__).resolve().parents[1] / "out"
COUNT = 450
EPOCHS = 31


def load(name: str) -> dict[str, Any]:
    return json.loads((OUT / name).read_text())


def field(loss: str) -> np.ndarray:
    return from_base64(load("field.json")["runs"][loss]).reshape(EPOCHS, COUNT)


def test_encoding_round_trip() -> None:
    right = 10.0 ** -np.random.default_rng(0).uniform(0, 15, 1000)
    correct = np.random.default_rng(1).random(1000) < 0.5
    encoded = from_base64(to_base64(encode_levels(right, correct)))
    decoded, flags = decode_levels(encoded)
    assert np.array_equal(flags, correct)
    assert np.abs(np.log10(decoded) - np.log10(right)).max() <= 0.5 / LEVELS_PER_DECADE + 1e-12


def test_encoding_saturates_tiny_and_zero_probabilities() -> None:
    encoded = encode_levels(np.array([1.0, 1e-30, 0.0]), np.array([True, False, False]))
    assert encoded.tolist() == [128, MAX_LEVEL, MAX_LEVEL]


@pytest.mark.parametrize("loss", ["ce", "mse"])
def test_field_agrees_with_accuracies(loss: str) -> None:
    _, correct = decode_levels(field(loss))
    hero = load("hero.json")["runs"][loss]["testAccuracy"]
    replay = load("replay.json")["runs"][f"confident-{loss}"]["testAccuracy"]
    measured = correct.mean(axis=1)
    assert measured == pytest.approx(hero, abs=1e-4)
    assert measured == pytest.approx(replay, abs=1e-4)


def test_hero_inlines_the_shared_first_frame() -> None:
    epoch0 = from_base64(load("hero.json")["epoch0"])
    assert np.array_equal(epoch0, field("ce")[0])
    assert np.array_equal(epoch0, field("mse")[0])


def test_hero_labels_are_the_test_labels() -> None:
    labels = load("hero.json")["labels"]
    assert len(labels) == COUNT and np.bincount(labels).max() <= 48
