import numpy as np
from ..modules import (
    cross_entropy_with_logits, cross_entropy_with_logits_grad,
    bce_with_logits, bce_with_logits_grad,
    sigmoid, softmax,
)
from ..layers.fcn import FullyConnectedNetwork


class MLP:
    """
    Multi-Layer Perceptron with early stopping.
    """

    def __init__(
        self,
        layer_sizes,
        activations,
        loss="cross_entropy",
        learning_rate=0.01,
        optimizer="adam",
        batch_size=32,
        weight_init="he",
        regularization=None,
        lambda_reg=0.01,
    ):
        assert len(layer_sizes) - 1 == len(activations)

        self.batch_size = batch_size

        self.network = FullyConnectedNetwork(
            layer_sizes=layer_sizes,
            activations=activations,
            learning_rate=learning_rate,
            optimizer=optimizer,
            weight_init=weight_init,
            regularization=regularization,
            lambda_reg=lambda_reg,
        )

        if loss == "cross_entropy":
            self.loss_fn = cross_entropy_with_logits
            self.loss_grad = cross_entropy_with_logits_grad
        elif loss == "binary_cross_entropy":
            self.loss_fn = bce_with_logits
            self.loss_grad = bce_with_logits_grad
        else:
            raise ValueError(f"Unsupported loss: {loss!r}")

        self.is_binary = (loss == "binary_cross_entropy")

    # ------------------------------------------------------------------
    def forward(self, X):
        self.logits = self.network.forward(X)
        return self.logits

    def backward(self, y):
        delta = self.loss_grad(y, self.logits)
        return self.network.backward(delta)

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        logits = self.network.forward(X)
        probs = sigmoid(logits) if self.is_binary else softmax(logits)
        return np.squeeze(probs)

    def predict(self, X):
        probs = self.predict_proba(X)
        if self.is_binary:
            return (probs > 0.5).astype(np.int32)
        return np.argmax(probs, axis=-1)
    
    def parameters(self):
        return self.network.get_state()
