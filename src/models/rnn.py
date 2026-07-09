import numpy as np


def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def xavier(shape):
    fan_in, fan_out = shape
    return np.random.randn(*shape) * np.sqrt(2 / (fan_in + fan_out))


class VanillaRNN:
    def __init__(self, vocab_size, emb_dim, hid_dim, out_dim):
        self.params = {
            "E": np.random.randn(vocab_size, emb_dim) * 0.01,
            "Wxh": xavier((hid_dim, emb_dim)),
            "Whh": xavier((hid_dim, hid_dim)),
            "Why": xavier((out_dim, hid_dim)),
            "bh": np.zeros((hid_dim, 1)),
            "by": np.zeros((out_dim, 1)),
        }

    def forward(self, x):
        E = self.params["E"]
        Wxh = self.params["Wxh"]
        Whh = self.params["Whh"]
        Why = self.params["Why"]
        bh = self.params["bh"]
        by = self.params["by"]

        h = np.zeros((Whh.shape[0], 1))
        self.cache = []

        for idx in x:
            if idx == 0:
                break

            h_prev = h.copy()

            x_t = E[idx].reshape(-1, 1)

            h = tanh(Wxh @ x_t + Whh @ h_prev + bh)

            self.cache.append((h_prev, h.copy(), idx))

        self.h_last = h
        self.y = softmax(Why @ h + by)

        return self.y

    def backward(self, y_true):
        E = self.params["E"]
        Wxh = self.params["Wxh"]
        Whh = self.params["Whh"]
        Why = self.params["Why"]

        grads = {k: np.zeros_like(v) for k, v in self.params.items()}

        # Output gradient
        dy = self.y.copy()
        dy[y_true] -= 1

        grads["Why"] += dy @ self.h_last.T
        grads["by"] += dy

        dh = Why.T @ dy

        # Backpropagation Through Time
        for h_prev, h, idx in reversed(self.cache):
            x_t = E[idx].reshape(-1, 1)

            dtanh = (1 - h ** 2) * dh

            grads["Wxh"] += dtanh @ x_t.T
            grads["Whh"] += dtanh @ h_prev.T
            grads["bh"] += dtanh

            # Embedding gradients
            grads["E"][idx] += (Wxh.T @ dtanh).ravel()

            dh = Whh.T @ dtanh

        # Gradient clipping
        for g in grads.values():
            np.clip(g, -5, 5, out=g)

        return grads

    def update(self, optimizer, grads):
        optimizer.step(self.params, grads)
    
    def loss(self, y_true):
        return -np.log(self.y[y_true, 0] + 1e-15)
    
    def predict_proba(self, x):
        return self.forward(x).ravel()
    
    def predict(self, x):
        return np.argmax(self.forward(x))