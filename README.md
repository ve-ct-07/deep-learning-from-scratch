# Deep Learning From Scratch

A modular deep learning framework implemented entirely in **NumPy**, designed to demonstrate the implementation of modern neural network architectures and training algorithms from first principles, without relying on high-level deep learning libraries.

## Highlights

- Implemented from scratch using NumPy
- Modular layer-based framework
- Reusable training pipeline
- Implements five neural architectures: MLP, CNN, RNN, LSTM and Coupled LSTM
- Complete forward and backward propagation for every component

## Project Goals

- Understand neural network fundamentals through implementation
- Build reusable deep learning components
- Compare different neural architectures on common tasks
- Provide an educational codebase for learning deep learning internals

## Architectures

- Multi-Layer Perceptron (MLP)
- Convolutional Neural Network (CNN)
- Vanilla Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)
- Coupled Input-Forget Gate LSTM

### Implemented Components

- Dense Layers
- Convolutional Layers
- Pooling Layers
- Flatten Layers
- Sequential Containers
- SGD Optimizer
- Adam Optimizer
- Cross Entropy Losses
- Xavier Initialization
- He Initialization
- L1/L2 Regularization
- Early Stopping

`src/modules/` contains common utilities including activation functions, loss functions, optimizers and parameter initialization methods.

## Repository Structure

```text
deep-learning-from-scratch/
│
├── configs/
├── docs/
├── notebooks/
├── src/
│   ├── datasets/
│   ├── layers/
│   ├── models/
│   ├── modules/
│   ├── training/
│   └── utils/
└── tests/
```

## Example Usage

```python
model = MLP(
    layer_sizes=[784, 128, 64, 10],
    activations=["relu", "relu", "linear"]
)

trainer = Trainer(model=model, epochs=20, batch_size=32)

history = trainer.fit(
    X_train,
    y_train,
    X_val,
    y_val
)

predictions = model.predict(X_val)
```

See `notebooks/deep_learning_demo.ipynb` for a full walkthrough of every architecture.

## Results

| Architecture | Dataset | Test Accuracy |
| ------------ | ------- | ------------- |
| MLP          | MNIST   | 97.94%        |
| CNN          | MNIST   | 98.24%        |

### Sequence models

The recurrent architectures were evaluated on both a synthetic sequence-ordering task (to verify the correctness of the implementations) and the IMDB sentiment classification dataset (to evaluate performance on a real-world NLP task).

| Architecture  | Synthetic Sequence |  IMDB Test  |
| ------------- | -----------------: | ----------: |
| RNN           |            100.00% |      57.44% |
| LSTM          |            100.00% |      74.80% |
| Coupled LSTM  |            100.00% |  **85.52%** |

### Training curves

<img src="assets/mlp_mnist.png" width="420"> <img src="assets/cnn_mnist.png" width="420">

<img src="assets/rnn_imdb.png" width="420"> <img src="assets/lstm_imdb.png" width="420">

<img src="assets/coupled_lstm_imdb.png" width="420">

All training curves were generated using `notebooks/deep_learning_demo.ipynb` and exported to the `assets/` directory. Additional experiment details and observations are available in `docs/results.md` and `docs/experiments.md`.

## Installation

```bash
git clone https://github.com/ve-ct-07/deep-learning-from-scratch.git

cd deep-learning-from-scratch

pip install -r requirements.txt
```

Datasets aren't included in the repo — see [`data/README.md`](data/README.md) for download instructions.

## Running Experiments

```bash
jupyter notebook notebooks/deep_learning_demo.ipynb
```

or

```python
from src.models import MLP
from src.training import Trainer
```

## Running Tests

```bash
python tests/run_tests.py
```

## Future Improvements

- Batch Normalization
- Dropout
- Transformer-based sequence models

## License

This project is released under the MIT License.

---

If you find this project useful, consider giving it a ⭐ on GitHub.