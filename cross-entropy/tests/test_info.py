import math

import numpy as np
import pytest

from ce_explainer.info import bits_from_nats, cross_entropy, entropy, kl_divergence, surprise


def test_claim_cross_entropy_is_entropy_plus_kl() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        p, q = rng.dirichlet(np.ones(5)), rng.dirichlet(np.ones(5))
        assert cross_entropy(p, q) == pytest.approx(entropy(p) + kl_divergence(p, q))
        assert kl_divergence(p, q) >= 0
    p, q = np.array([0.7, 0.2, 0.1]), np.array([0.5, 0.3, 0.2])
    assert entropy(p, 2) == pytest.approx(1.1568, abs=1e-4)
    assert cross_entropy(p, q, 2) == pytest.approx(1.2796, abs=1e-4)
    assert kl_divergence(p, q, 2) == pytest.approx(0.1228, abs=1e-4)
    assert kl_divergence(q, p, 2) == pytest.approx(0.1328, abs=1e-4)


def test_claim_one_hot_cross_entropy_is_nll() -> None:
    q = np.random.default_rng(1).dirichlet(np.ones(10))
    y = np.eye(10)[4]
    assert entropy(y) == 0
    assert cross_entropy(y, q) == pytest.approx(kl_divergence(y, q))
    assert cross_entropy(y, q) == pytest.approx(-math.log(q[4]))


def test_claim_bits_and_nats() -> None:
    assert surprise(0.5) == pytest.approx(1.0)
    assert surprise(1 / 8) == pytest.approx(3.0)
    assert surprise(0.001) == pytest.approx(9.9658, abs=1e-4)
    assert surprise(0.001, math.e) == pytest.approx(6.9078, abs=1e-4)
    assert bits_from_nats(math.log(10)) == pytest.approx(math.log2(10))
    assert cross_entropy(np.eye(10)[0], np.full(10, 0.1)) == pytest.approx(math.log(10))
    assert entropy(np.full(10, 0.1), 2) == pytest.approx(3.3219, abs=1e-4)
