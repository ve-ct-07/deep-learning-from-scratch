import numpy as np
from ..modules import (
    cross_entropy_with_logits, cross_entropy_with_logits_grad,
    softmax,
)
from ..layers import Sequential, Conv2D, Pool2D, Flatten, Dense


class CNN:
    """
    CNN: conv-pool blocks → flatten → dense classifier.
    """

    def __init__(
        self,
        input_shape=(28, 28, 1),
        num_classes=10,
        lr=0.001,
    ):
        in_c = input_shape[2]

        self.features = Sequential([
            Conv2D(in_c, 8,  (3, 3), padding=1, lr=lr),
            Pool2D(kernel_size=2, mode="max"),
            Conv2D(8,    16, (3, 3), padding=1, lr=lr),
            Pool2D(kernel_size=2, mode="max"),
            Flatten(),
        ])

        flat_size = self._compute_flat_size(input_shape)

        self.classifier = Sequential([
            Dense(flat_size, 128, activation="relu", lr=lr),
            Dense(128, num_classes, lr=lr),
        ])
        self.loss_fn = cross_entropy_with_logits
        self.loss_grad = cross_entropy_with_logits_grad

    def _compute_flat_size(self, input_shape):
        dummy = np.zeros((1, *input_shape))
        out = self.features.forward(dummy)
        return out.shape[1]

    # ------------------------------------------------------------------
    def forward(self, x):
        x = self.features.forward(x)
        self.logits = self.classifier.forward(x)
        return self.logits

    def backward(self, y):
        grad = self.loss_grad(y, self.logits)
        grad = self.classifier.backward(grad)
        grad = self.features.backward(grad)
        return grad

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        return np.squeeze(softmax(self.forward(X)))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=-1)

    def parameters(self):
        params = {}

        for i, p in enumerate(
            self.features.parameters() +
            self.classifier.parameters()
        ):
            params[f"param_{i}"] = p

        return params
