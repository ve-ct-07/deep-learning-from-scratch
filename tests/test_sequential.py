import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from src.layers import Sequential,Dense

model = Sequential([
    Dense(20,10),
    Dense(10,5)
])

x = np.random.randn(8,20)

y = model.forward(x)

assert y.shape == (8,5)

grad = np.random.randn(8,5)

dx = model.backward(grad)

assert dx.shape == (8,20)

print("Sequential passed")