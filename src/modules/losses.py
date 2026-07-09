import numpy as np
from .activations import sigmoid, softmax


def bce_with_logits(y_true, logits):
    """Numerically stable binary cross-entropy from raw logits."""
    y_true = np.asarray(y_true).reshape(logits.shape)
    loss = (
        np.maximum(logits, 0)
        - logits * y_true
        + np.log1p(np.exp(-np.abs(logits)))
    )
    return np.mean(loss)


def bce_with_logits_grad(y_true, logits):
    y_true = np.asarray(y_true).reshape(logits.shape)
    return (sigmoid(logits) - y_true) / y_true.size


def cross_entropy_with_logits(y_true, logits):
    """Multi-class cross-entropy; y_true holds integer class indices."""
    probs = softmax(logits)
    loss = -np.log(probs[np.arange(len(y_true)), y_true] + 1e-15)
    return np.mean(loss)


def cross_entropy_with_logits_grad(y_true, logits):
    probs = softmax(logits)
    probs[np.arange(len(y_true)), y_true] -= 1
    return probs / len(y_true)
