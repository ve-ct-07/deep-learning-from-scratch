# Results

## MLP

Training Accuracy: 99.59
Validation Accuracy: 98.00
Test Accuracy: 97.94

Observations:

- Fast convergence
- Stable gradients

---

## CNN

Training Accuracy: 99.33
Validation Accuracy: 98.57
Test Accuracy: 98.24

Observations:

- Better spatial feature extraction
- Outperformed MLP with fewer epochs

---

## RNN on synthetic dataset

Training Accuracy: 100.00
Validation Accuracy: 100.00
Test Accuracy: 100.00

Observations:

- Perfect train and validation accuracy in just three epochs

---

## LSTM on synthetic dataset

Training Accuracy: 100.00
Validation Accuracy: 100.00
Test Accuracy: 100.00

Observations:

- Outperformed RNN by minimizing loss and reaching perfect accuracy in one epoch

---

## Coupled Input-Forget Gate LSTM on synthetic dataset

Training Accuracy: 100.00
Validation Accuracy: 100.00
Test Accuracy: 100.00

Observations:

- Outperformed LSTM by further minimizing loss

---

## RNN on IMDB Dataset

Training Accuracy: 83.46
Validation Accuracy: 59.20
Test Accuracy: 57.44

Observations:

- Cannot retain long-range information well
- Overfitting on training data

---

## LSTM on IMDB Dataset

Training Accuracy: 99.52
Validation Accuracy: 75.60
Test Accuracy: 74.80

Observations:

- Outperformed RNN due to the inclusion of memory cell

---

## Coupled Input-Forget Gate LSTM on IMDB Dataset

Training Accuracy: 98.00
Validation Accuracy: 77.80
Test Accuracy: 85.52

Observations:

- Outperformed LSTM despite a simpler gating mechanism
- Best test accuracy among the sequence models