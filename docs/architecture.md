# Architecture

The project follows a modular deep learning framework design, similar to
how PyTorch/Keras separate layers, models, and training — just implemented
in plain NumPy.

## Layers

Layers are the basic building blocks. Every layer implements:

- `forward()`
- `backward()`

so `Sequential` can chain them together without needing to know the
implementation details of each one.

Implemented layers:

- Dense
- Conv2D
- Pool2D
- Flatten

## Models

Models are just compositions of layers, trained through the same `Trainer`.

Implemented models:

- MLP
- CNN
- RNN (vanilla)
- LSTM
- Coupled Input-Forget Gate LSTM

## Training Pipeline

Training logic is kept separate from the models themselves, so any model
can reuse the same `Trainer`.

Responsibilities:

- batching
- loss computation
- backpropagation
- parameter updates (via `SGD` or `Adam`)
- validation after each epoch
- early stopping

## Folder Layout

```text
src/
├── datasets/    # MNIST / IMDB / synthetic sequence loaders
├── layers/      # Dense, Conv2D, Pool2D, Flatten
├── models/      # MLP, CNN, RNN, LSTM, CoupledLSTM
├── modules/     # activations, losses, optimizers, initializers
├── training/    # Trainer
└── utils/       # visualization helpers
```