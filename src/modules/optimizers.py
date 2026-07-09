import numpy as np


class SGD:

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, w, grad):
        return w - self.lr * grad

    def step(self, params, grads):
        """In-place update for a dict of parameters (used by RNN/LSTM models)."""
        for k in params:
            params[k] -= self.lr * grads[k]


class Adam:

    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def update(self, w, grad):
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)

        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * (grad ** 2)

        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)

        return w - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def step(self, params, grads):
        """In-place update for a dict of parameters (used by RNN/LSTM models)."""
        if self.m is None:
            self.m = {k: np.zeros_like(v) for k, v in params.items()}
            self.v = {k: np.zeros_like(v) for k, v in params.items()}

        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g ** 2)

            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)

            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
