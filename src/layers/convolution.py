import numpy as np
from ..modules import Adam, Module
from ..modules.conv2d import conv2d


class Conv2D(Module):
    """
    2-D Convolution layer.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        lr=0.001,
    ):
        kH, kW = kernel_size
        fan_in = kH * kW * in_channels
        self.W = np.random.randn(out_channels, kH, kW, in_channels) * np.sqrt(2 / fan_in)
        self.b = np.zeros(out_channels, dtype=np.float32)

        self.stride = stride
        self.padding = padding

        self.opt_w = Adam(lr)
        self.opt_b = Adam(lr)

        self.x = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x):
        if x.ndim == 3:
            x = x[np.newaxis]
        self.x = x
        self.out = conv2d(x, self.W, self.b, self.stride, self.padding)
        return self.out

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------
    def backward(self, grad):
        """
        grad : same shape as self.out — (N, oH, oW, F)
        Returns dx : (N, H, W, C)
        """
        x = self.x
        N, H, W, C = x.shape
        F, kH, kW, _ = self.W.shape
        _, oH, oW, _ = grad.shape
        pad, stride = self.padding, self.stride

        x_pad = np.pad(
            x,
            ((0, 0), (pad, pad), (pad, pad), (0, 0)),
            mode="constant",
        )

        shape = (N, oH, oW, kH, kW, C)
        strides = (
            x_pad.strides[0],
            x_pad.strides[1] * stride,
            x_pad.strides[2] * stride,
            x_pad.strides[1],
            x_pad.strides[2],
            x_pad.strides[3],
        )
        patches = np.lib.stride_tricks.as_strided(x_pad, shape=shape, strides=strides)
        col = patches.reshape(N, oH * oW, kH * kW * C)
        G = grad.reshape(N, oH * oW, F)

        dW = np.einsum("nqf,nqk->fk", G, col).reshape(F, kH, kW, C)
        db = G.sum(axis=(0, 1))

        dx_pad = np.zeros_like(x_pad)
        K = self.W.reshape(F, kH * kW * C)
        dCol = np.einsum("nqf,fk->nqk", G, K)
        dCol = dCol.reshape(N, oH, oW, kH, kW, C)

        for i in range(oH):
            hs = i * stride
            for j in range(oW):
                ws = j * stride
                dx_pad[:, hs : hs + kH, ws : ws + kW, :] += dCol[:, i, j, :, :, :]

        dx = dx_pad[:, pad : pad + H, pad : pad + W, :] if pad > 0 else dx_pad

        self.W = self.opt_w.update(self.W, dW)
        self.b = self.opt_b.update(self.b, db)

        return dx

    def parameters(self):
        return [self.W, self.b]
