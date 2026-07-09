# Mathematical Foundations

Forward and backward pass equations for the core layers.

## Dense Layer

Forward:

$$y = Wx + b$$

Backward:

$$dW = X^T dY \qquad db = \sum dY$$

## ReLU

Forward:

$$f(x) = \max(0, x)$$

Backward:

$$f'(x) = 1 \text{ if } x > 0 \text{ else } 0$$

## Convolution

Forward: cross-correlation is used (kernel is not flipped), matching how
PyTorch/TensorFlow implement it.

Backward: gradients are computed with respect to:

- input
- kernel weights
- biases

## LSTM

Implemented gates:

- Forget Gate
- Input Gate
- Candidate State
- Output Gate

## Coupled Input-Forget Gate LSTM

Same as above, but the input gate is tied to the forget gate
($i_t = 1 - f_t$), so there's one less set of weights to learn.