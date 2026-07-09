import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from src.layers import Pool2D

pool = Pool2D(kernel_size=2)

x = np.random.randn(2,28,28,8)

out = pool.forward(x)

grad = np.random.randn(*out.shape)

dx = pool.backward(grad)

assert dx.shape == x.shape

print("Pooling passed")