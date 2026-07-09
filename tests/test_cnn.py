import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from src.models import CNN

model = CNN()

x = np.random.randn(2,28,28,1)

y = model.forward(x)

assert y.shape == (2,10)

pred = model.predict(x)

assert pred.shape == (2,)

print("CNN passed")