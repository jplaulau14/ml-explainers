from collections.abc import Callable

import numpy as np
import pytest

from ce_explainer.data import one_hot
from ce_explainer.model import (
    batch_grads,
    bce_logit_grad,
    ce_loss,
    init_params,
    logsumexp,
    mean_loss,
    naive_softmax,
    se_logit_grad,
    se_loss,
    sigmoid,
    sigmoid_se_grad,
    softmax,
)

EPS = 1e-6


def numeric_grad(f: Callable[[np.ndarray], float], z: np.ndarray) -> np.ndarray:
    grad = np.zeros_like(z)
    for i in range(z.size):
        step = np.zeros_like(z)
        step.flat[i] = EPS
        grad.flat[i] = (f(z + step) - f(z - step)) / (2 * EPS)
    return grad


def random_logits(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 3, 10)


def test_claim_softmax_is_a_distribution() -> None:
    p = softmax(np.random.default_rng(0).normal(0, 5, (20, 10)))
    assert (p >= 0).all()
    assert p.sum(axis=1) == pytest.approx(np.ones(20))
    assert softmax(np.array([2.0, 1.0, 0.1])) == pytest.approx([0.6590, 0.2424, 0.0986], abs=1e-4)


@pytest.mark.parametrize("c", [-50.0, 3.0, 100.0])
def test_claim_shift_invariance(c: float) -> None:
    z = random_logits()
    assert softmax(z + c) == pytest.approx(softmax(z))


def binary_ce(z: np.ndarray, y: float) -> float:
    return float(np.log1p(np.exp(-z[0])) if y == 1.0 else np.log1p(np.exp(z[0])))


def test_claim_ce_gradient_is_p_minus_y() -> None:
    z, label = random_logits(), np.array([3])
    expected = softmax(z) - one_hot(label)[0]
    numeric = numeric_grad(lambda v: ce_loss(v[None], label)[0], z)
    assert numeric == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize("y", [0.0, 1.0])
def test_claim_binary_ce_gradient_is_p_minus_y(y: float) -> None:
    numeric = numeric_grad(lambda v: binary_ce(v, y), np.array([-1.7]))[0]
    assert numeric == pytest.approx(bce_logit_grad(-1.7, y))


def test_claim_mse_gradient_formula() -> None:
    z, label = random_logits(1), np.array([0])
    analytic = se_logit_grad(softmax(z), one_hot(label)[0])
    numeric = numeric_grad(lambda v: se_loss(v[None], label)[0], z)
    assert numeric == pytest.approx(analytic, abs=1e-8)
    worked = se_logit_grad(softmax(np.array([0.0, 5.0, 0.0])), np.array([1.0, 0.0, 0.0]))
    assert worked == pytest.approx([-0.01303, 0.01942, -0.00639], abs=1e-5)
    binary = numeric_grad(lambda v: 0.5 * (sigmoid(v[0]) - 1) ** 2, np.array([-5.0]))[0]
    assert binary == pytest.approx(sigmoid_se_grad(-5.0, 1.0))
    assert sigmoid_se_grad(-5.0, 1.0) == pytest.approx(-0.006604, abs=1e-6)
    assert bce_logit_grad(-5.0, 1.0) == pytest.approx(-0.9933, abs=1e-4)


@pytest.mark.parametrize("y", [0.0, 1.0])
def test_claim_sigmoid_mse_gradient_bound(y: float) -> None:
    sizes = np.abs(sigmoid_se_grad(np.linspace(-30, 30, 600_001), y))
    assert sizes.max() <= 4 / 27 + 1e-12
    assert sizes.max() == pytest.approx(4 / 27, abs=1e-8)


def test_claim_logsumexp_is_stable() -> None:
    z = random_logits(2)
    assert logsumexp(z) == pytest.approx(np.log(np.exp(z).sum()))
    big = np.array([1000.0, 999.0, 998.0])
    assert logsumexp(big) == pytest.approx(1000.4076, abs=1e-4)
    assert softmax(big) == pytest.approx([0.6652, 0.2447, 0.0900], abs=1e-4)
    with np.errstate(over="ignore", invalid="ignore"):
        assert np.isnan(naive_softmax(big)).all()
        assert np.isfinite(np.exp(np.float32(88.7)))
        assert np.isinf(np.exp(np.float32(88.8)))


def test_confident_start_is_the_gentle_start_scaled_up() -> None:
    gentle = init_params(np.random.default_rng(0), 0.01, 0)["W"]
    confident = init_params(np.random.default_rng(0), 4.0, 0)["W"]
    assert confident == pytest.approx(400 * gentle)


@pytest.mark.parametrize("loss", ["ce", "mse"])
@pytest.mark.parametrize("hidden", [0, 4])
def test_parameter_gradients_match_finite_differences(loss: str, hidden: int) -> None:
    rng = np.random.default_rng(3)
    params = init_params(rng, 1.0, hidden)
    x, labels = rng.random((6, 64)), rng.integers(0, 10, 6)
    grads = batch_grads(params, x, labels, loss)
    for name, value in params.items():

        def f(v: np.ndarray, name: str = name) -> float:
            return mean_loss(params | {name: v}, x, labels, loss)

        assert numeric_grad(f, value) == pytest.approx(grads[name], abs=1e-7)
