import numpy as np
from .dense_layer import Dense


class FullyConnectedNetwork:
    """
    Stack of Dense layers.
    """

    def __init__(
        self,
        layer_sizes,
        activations,
        learning_rate=0.01,
        optimizer="adam",
        weight_init="he",
        regularization=None,
        lambda_reg=0.01,
    ):
        assert len(layer_sizes) - 1 == len(activations), (
            "Expected one activation per layer transition"
        )

        self.regularization = regularization
        self.lambda_reg = lambda_reg

        self.layers = [
            Dense(
                in_dim=layer_sizes[i],
                out_dim=layer_sizes[i + 1],
                activation=activations[i],
                lr=learning_rate,
                optimizer=optimizer,
                regularization=regularization,
                lambda_reg=lambda_reg,
            )
            for i in range(len(layer_sizes) - 1)
        ]

    # ------------------------------------------------------------------
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, delta):
        """
        Runs BPTT through all dense layers and returns a diagnostic dict.
        """
        grad_means, grad_stds, weight_updates = [], [], []

        for layer in reversed(self.layers):
            old_w = layer.W.copy()
            delta = layer.backward(delta)
            grad_means.append(np.mean(np.abs(old_w)))
            grad_stds.append(np.std(old_w))
            weight_updates.append(np.mean(np.abs(layer.W - old_w)))

        return {
            "weight_update": float(np.mean(weight_updates)),
        }

    # ------------------------------------------------------------------
    def get_state(self):
        return {
            "weights": [layer.W.copy() for layer in self.layers],
            "biases": [layer.b.copy() for layer in self.layers],
        }

    def load_state(self, state):
        for layer, W, b in zip(self.layers, state["weights"], state["biases"]):
            layer.W = W.copy()
            layer.b = b.copy()

    # ------------------------------------------------------------------
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
