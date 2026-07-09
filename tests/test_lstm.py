import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from src.datasets import generate_sequence_dataset
from src.models import LSTM

data = generate_sequence_dataset(
    n_train=100,
    n_val=20,
    seq_len=20
)

model = LSTM(
    vocab_size=data["vocab_size"],
    emb_dim=16,
    hid_dim=32,
    out_dim=2
)

out = model.forward(data["X_train"][0])

assert out.shape == (2,1)

pred = model.predict(data["X_train"][0])

assert pred in [0,1]

print("LSTM passed")