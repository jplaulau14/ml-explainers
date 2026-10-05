from dataclasses import dataclass

import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SPLIT_SEED = 0
TEST_SIZE = 0.25
VALIDATION_SIZE = 0.2
PIXEL_MAX = 16.0
CLASSES = 10


@dataclass(frozen=True)
class Dataset:
    x: np.ndarray
    y: np.ndarray
    pixels: np.ndarray


@dataclass(frozen=True)
class Splits:
    train: Dataset
    validation: Dataset
    test: Dataset


def take(pixels: np.ndarray, labels: np.ndarray, index: np.ndarray) -> Dataset:
    return Dataset(x=pixels[index] / PIXEL_MAX, y=labels[index], pixels=pixels[index])


def load_splits() -> Splits:
    digits = load_digits()
    pixels = digits.data.astype(np.int64)
    labels = digits.target.astype(np.int64)
    index = np.arange(len(labels))
    rest, test = train_test_split(
        index, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=labels
    )
    train, validation = train_test_split(
        rest, test_size=VALIDATION_SIZE, random_state=SPLIT_SEED, stratify=labels[rest]
    )
    return Splits(
        train=take(pixels, labels, train),
        validation=take(pixels, labels, validation),
        test=take(pixels, labels, test),
    )


def one_hot(labels: np.ndarray) -> np.ndarray:
    return np.eye(CLASSES)[labels]


def source_block(splits: Splits) -> dict[str, object]:
    return {
        "dataset": "sklearn.datasets.load_digits",
        "sklearnVersion": sklearn.__version__,
        "numpyVersion": np.__version__,
        "splitSeed": SPLIT_SEED,
        "sizes": {
            "train": len(splits.train.y),
            "validation": len(splits.validation.y),
            "test": len(splits.test.y),
        },
        "pixelScale": f"pixel / {PIXEL_MAX:g}",
        "code": "https://github.com/jplaulau14/ml-explainers/tree/main/cross-entropy",
    }
