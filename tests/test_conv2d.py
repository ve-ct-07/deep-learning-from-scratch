import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from src.layers import Conv2D

conv = Conv2D(1,8,(3,3))

x = np.random.randn(2,28,28,1)

out = conv.forward(x)

assert out.shape[0] == 2
assert out.shape[-1] == 8

grad = np.random.randn(*out.shape)

dx = conv.backward(grad)

assert dx.shape == x.shape

print("Conv2D passed")