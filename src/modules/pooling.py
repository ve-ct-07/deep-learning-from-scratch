import numpy as np


def pooling(x, kernel_size=2, stride=None, mode="max"):
    """
    NHWC pooling (max or average).
    """
    if x.ndim == 3:
        x = x[np.newaxis]

    assert mode in ("max", "avg"), f"Unknown mode: {mode}"

    if stride is None:
        stride = kernel_size

    N, H, W, C = x.shape
    k = kernel_size
    oH = (H - k) // stride + 1
    oW = (W - k) // stride + 1

    out = np.zeros((N, oH, oW, C), dtype=x.dtype)

    for i in range(oH):
        hs, he = i * stride, i * stride + k
        for j in range(oW):
            ws, we = j * stride, j * stride + k
            window = x[:, hs:he, ws:we, :]
            if mode == "max":
                out[:, i, j, :] = window.reshape(N, k * k, C).max(axis=1)
            else:
                out[:, i, j, :] = window.reshape(N, k * k, C).mean(axis=1)

    return out
