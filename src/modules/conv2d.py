import numpy as np


def conv2d(volume, kernels, bias=None, stride=1, padding=0):
    """
    PyTorch-style Conv2D
    """
    if volume.ndim == 3:
        volume = volume[np.newaxis]

    N, H, W, C = volume.shape
    F, kH, kW, kC = kernels.shape
    assert C == kC, f"Channel mismatch: volume has {C}, kernels expect {kC}"

    if bias is None:
        bias = np.zeros(F, dtype=np.float32)

    oH = (H + 2 * padding - kH) // stride + 1
    oW = (W + 2 * padding - kW) // stride + 1

    x_pad = np.pad(
        volume,
        ((0, 0), (padding, padding), (padding, padding), (0, 0)),
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
    col = patches.reshape(N, oH, oW, -1)
    K = kernels.reshape(F, -1)
    out = col @ K.T + bias
    return out.astype(np.float32)
