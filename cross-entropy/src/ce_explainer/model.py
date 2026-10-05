import numpy as np

from ce_explainer.data import CLASSES, one_hot

Params = dict[str, np.ndarray]
FEATURES = 64


def logsumexp(z: np.ndarray) -> np.ndarray:
    m = z.max(axis=-1, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=-1, keepdims=True)))[..., 0]


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def naive_softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-z))


def ce_loss(z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return logsumexp(z) - np.take_along_axis(z, labels[:, None], axis=1)[:, 0]


def se_loss(z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return 0.5 * ((softmax(z) - one_hot(labels)) ** 2).sum(axis=1)


def ce_logit_grad(p: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return p - targets


def se_logit_grad(p: np.ndarray, targets: np.ndarray) -> np.ndarray:
    d = p - targets
    return p * (d - (p * d).sum(axis=-1, keepdims=True))


def bce_logit_grad(z: float, y: float) -> float:
    return sigmoid(z) - y


def sigmoid_se_grad(z: float, y: float) -> float:
    p = sigmoid(z)
    return (p - y) * p * (1 - p)


LOSS_FUNCTIONS = {"ce": ce_loss, "mse": se_loss}
LOGIT_GRADS = {"ce": ce_logit_grad, "mse": se_logit_grad}


def init_params(rng: np.random.Generator, std: float, hidden: int) -> Params:
    params: Params = {}
    width = FEATURES
    if hidden:
        params["W1"] = rng.standard_normal((FEATURES, hidden)) / np.sqrt(FEATURES)
        params["b1"] = np.zeros(hidden)
        width = hidden
    params["W"] = rng.standard_normal((width, CLASSES)) * std
    params["b"] = np.zeros(CLASSES)
    return params


def forward(params: Params, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = np.tanh(x @ params["W1"] + params["b1"]) if "W1" in params else x
    return h @ params["W"] + params["b"], h


def backward(params: Params, x: np.ndarray, h: np.ndarray, g: np.ndarray) -> Params:
    grads = {"W": h.T @ g, "b": g.sum(axis=0)}
    if "W1" in params:
        gh = (g @ params["W"].T) * (1 - h**2)
        grads |= {"W1": x.T @ gh, "b1": gh.sum(axis=0)}
    return grads


def batch_grads(params: Params, x: np.ndarray, labels: np.ndarray, loss: str) -> Params:
    z, h = forward(params, x)
    g = LOGIT_GRADS[loss](softmax(z), one_hot(labels)) / len(labels)
    return backward(params, x, h, g)


def mean_loss(params: Params, x: np.ndarray, labels: np.ndarray, loss: str) -> float:
    z, _ = forward(params, x)
    return float(LOSS_FUNCTIONS[loss](z, labels).mean())
