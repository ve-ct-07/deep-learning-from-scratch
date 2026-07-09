import numpy as np
from ..modules import Module
from ..modules.pooling import pooling


class Pool2D(Module):
    """
    2-D pooling layer (max or average).
    """

    def __init__(self, kernel_size=2, stride=None, mode="max"):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.mode = mode

        self.x = None

    # ------------------------------------------------------------------
    def forward(self, x):
        if x.ndim == 3:
            x = x[np.newaxis]
        self.x = x
        self.out = pooling(x, self.kernel_size, self.stride, self.mode)
        return self.out

    # ------------------------------------------------------------------
    def backward(self, grad):
        N, H, W, C = self.x.shape
        k, s = self.kernel_size, self.stride
        _, oH, oW, _ = grad.shape

        dx = np.zeros_like(self.x)

        for i in range(oH):
            hs, he = i * s, i * s + k
            for j in range(oW):
                ws, we = j * s, j * s + k
                window = self.x[:, hs:he, ws:we, :]
                g = grad[:, i, j, :]

                if self.mode == "avg":
                    dx[:, hs:he, ws:we, :] += g[:, np.newaxis, np.newaxis, :] / (k * k)
                else:
                    # max pooling: route to max position(s)
                    flat = window.reshape(N, k * k, C)
                    max_mask = flat == flat.max(axis=1, keepdims=True)
                    # normalise ties
                    max_mask = max_mask / max_mask.sum(axis=1, keepdims=True)
                    dx[:, hs:he, ws:we, :] += (
                        max_mask.reshape(N, k, k, C) * g[:, np.newaxis, np.newaxis, :]
                    )

        return dx

    def parameters(self):
        return []
