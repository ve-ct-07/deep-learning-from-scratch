# Experiments

Configuration used for each run reported in [results.md](results.md).

## MLP

- Dataset: MNIST
- Architecture: 784 → 128 → 64 → 10
- Activation: ReLU
- Loss: Cross Entropy

---

## CNN

- Dataset: MNIST
- Architecture: Conv(1→8) → Pool → Conv(8→16) → Pool → FC(128) → Output
- Loss: Cross Entropy

---

## RNN

- Dataset: Synthetic dataset
- Task: Sequence Classification

---

## LSTM

- Dataset: Synthetic dataset
- Task: Sequence Classification

---

## Coupled Input-Forget Gate LSTM

- Dataset: Synthetic dataset
- Task: Sequence Classification

---

## RNN on IMDB

- Dataset: IMDB
- Task: Sequence Classification

---

## LSTM on IMDB

- Dataset: IMDB
- Task: Sequence Classification

---

## Coupled Input-Forget Gate LSTM on IMDB

- Dataset: IMDB
- Task: Sequence Classification