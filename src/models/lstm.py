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

class LSTM:
    def __init__(self, vocab_size, emb_dim, hid_dim, out_dim):

        d = hid_dim + emb_dim

        self.params = {
            "E": xavier((vocab_size, emb_dim)),

            "Wf": xavier((hid_dim, d)),
            "Wi": xavier((hid_dim, d)),
            "Wc": xavier((hid_dim, d)),
            "Wo": xavier((hid_dim, d)),

            "bf": np.zeros((hid_dim, 1)),
            "bi": np.zeros((hid_dim, 1)),
            "bc": np.zeros((hid_dim, 1)),
            "bo": np.zeros((hid_dim, 1)),

            "Why": xavier((out_dim, hid_dim)),
            "by": np.zeros((out_dim, 1)),
        }

    def forward(self, x):
        p = self.params
        self.cache = []

        h = np.zeros((p["bf"].shape[0], 1))
        c = np.zeros_like(h)

        for idx in x:
            if idx == 0:
                break

            x_t = p["E"][idx].reshape(-1, 1)
            concat = np.vstack((h, x_t))

            f = sigmoid(p["Wf"] @ concat + p["bf"])
            i = sigmoid(p["Wi"] @ concat + p["bi"])
            o = sigmoid(p["Wo"] @ concat + p["bo"])
            g = tanh(p["Wc"] @ concat + p["bc"])

            c_prev = c
            c = f * c + i * g
            h = o * tanh(c)

            self.cache.append((h, c, c_prev, f, i, o, g, concat, idx))

        self.h_last = h
        self.y = softmax(p["Why"] @ h + p["by"])

        return self.y

    def backward(self, y_true):
        p = self.params
        grads = {k: np.zeros_like(v) for k, v in p.items()}

        dy = self.y.copy()
        dy[y_true] -= 1

        grads["Why"] += dy @ self.h_last.T
        grads["by"] += dy

        dh = p["Why"].T @ dy
        dc = np.zeros_like(dh)

        for h, c, c_prev, f, i, o, g, concat, idx in reversed(self.cache):

            do = dh * np.tanh(c)
            do_raw = do * o * (1 - o)

            dc = dc + dh * o * (1 - np.tanh(c) ** 2)

            df = dc * c_prev
            di = dc * g
            dg = dc * i

            df_raw = df * f * (1 - f)
            di_raw = di * i * (1 - i)
            dg_raw = dg * (1 - g ** 2)

            grads["Wf"] += df_raw @ concat.T
            grads["Wi"] += di_raw @ concat.T
            grads["Wc"] += dg_raw @ concat.T
            grads["Wo"] += do_raw @ concat.T

            grads["bf"] += df_raw
            grads["bi"] += di_raw
            grads["bc"] += dg_raw
            grads["bo"] += do_raw

            dconcat = (
                p["Wf"].T @ df_raw +
                p["Wi"].T @ di_raw +
                p["Wc"].T @ dg_raw +
                p["Wo"].T @ do_raw
            )

            grads["E"][idx] += dconcat[dh.shape[0]:].ravel()

            dh = dconcat[:dh.shape[0]]
            dc = dc * f

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