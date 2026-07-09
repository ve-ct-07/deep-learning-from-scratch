import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from src.models import MLP

model = MLP(
    layer_sizes=[20,16,8,3],
    activations=["relu","relu","linear"]
)

x = np.random.randn(5,20)

y = model.forward(x)

assert y.shape == (5,3)

pred = model.predict(x)

assert pred.shape == (5,)

print("MLP passed")