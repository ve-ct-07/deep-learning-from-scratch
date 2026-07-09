import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from src.layers import Dense

dense = Dense(10, 5)

x = np.random.randn(4,10)
out = dense.forward(x)

assert out.shape == (4,5)

grad = np.random.randn(4,5)
dx = dense.backward(grad)

assert dx.shape == (4,10)

print("Dense layer passed")