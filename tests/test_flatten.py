import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from src.layers import Flatten

layer = Flatten()

x = np.random.randn(4,7,7,16)

y = layer.forward(x)

assert y.shape == (4,7*7*16)

dx = layer.backward(y)

assert dx.shape == x.shape

print("Flatten passed")