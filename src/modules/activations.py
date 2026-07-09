import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(x.dtype)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    return np.tanh(x)


def tanh_grad(x):
    return 1 - np.tanh(x) ** 2


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def softmax_grad(x):
    s = softmax(x)
    if s.ndim == 1:
        return np.diag(s) - np.outer(s, s)
    return (
        np.einsum("...i,...ij->...ij", s, np.eye(s.shape[-1]))
        - np.einsum("...i,...j->...ij", s, s)
    )
