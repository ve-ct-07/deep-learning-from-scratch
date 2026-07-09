import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from src.modules import SGD,Adam

w = np.random.randn(5,3)

grad = np.random.randn(5,3)

sgd = SGD()

new_w = sgd.update(w.copy(),grad)

assert not np.allclose(w,new_w)

adam = Adam()

new_w = adam.update(w.copy(),grad)

assert not np.allclose(w,new_w)

print("Optimizers passed")