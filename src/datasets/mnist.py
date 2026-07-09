"""
MNIST dataset loader.
"""

import numpy as np
import struct
import gzip
import os


def _read_idx(path: str) -> np.ndarray:
    """Read an IDX-format file (optionally gzip-compressed)."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        n_dims = magic & 0xFF
        dims = [struct.unpack(">I", f.read(4))[0] for _ in range(n_dims)]
        dtype_map = {0x08: np.uint8, 0x09: np.int8, 0x0B: np.int16,
                     0x0C: np.int32, 0x0D: np.float32, 0x0E: np.float64}
        dtype = dtype_map[(magic >> 8) & 0xFF]
        data = np.frombuffer(f.read(), dtype=dtype)
    return data.reshape(dims)


def load_mnist(data_dir: str):
    """
    Load MNIST from local IDX files.

    Expected files (raw or .gz):
        train-images-idx3-ubyte[.gz]
        train-labels-idx1-ubyte[.gz]
        t10k-images-idx3-ubyte[.gz]
        t10k-labels-idx1-ubyte[.gz]
    """
    def _find(name):
        for ext in ("", ".gz"):
            p = os.path.join(data_dir, name + ext)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Cannot find {name}[.gz] in {data_dir}")

    X_train = _read_idx(_find("train-images.idx3-ubyte"))
    y_train = _read_idx(_find("train-labels.idx1-ubyte"))
    X_test  = _read_idx(_find("t10k-images.idx3-ubyte"))
    y_test  = _read_idx(_find("t10k-labels.idx1-ubyte"))

    return (X_train, y_train), (X_test, y_test)


def preprocess_mnist(X_train, X_test, channels_last=True):
    """
    Normalise and reshape MNIST images.
    """
    X_train = X_train.astype(np.float32) / 255.0
    X_test  = X_test.astype(np.float32)  / 255.0

    if channels_last:
        X_train = X_train[..., np.newaxis]
        X_test  = X_test[..., np.newaxis]
    else:
        X_train = X_train.reshape(len(X_train), -1)
        X_test  = X_test.reshape(len(X_test),   -1)

    return X_train, X_test
