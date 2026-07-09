import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from src.modules import cross_entropy_with_logits, cross_entropy_with_logits_grad

logits = np.random.randn(8,10)

labels = np.random.randint(0,10,size=8)

l = cross_entropy_with_logits(labels,logits)

assert np.isfinite(l)

grad = cross_entropy_with_logits_grad(labels,logits)

assert grad.shape == logits.shape

print("CrossEntropy passed")