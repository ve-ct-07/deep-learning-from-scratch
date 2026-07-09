import numpy as np
from ..modules import relu, tanh, sigmoid, relu_grad, tanh_grad, sigmoid_grad
from ..modules import Adam, SGD, random_init, he_init, xavier_init, Module


class Dense(Module):
    """
    Fully-connected layer.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        activation=None,
        lr=0.01,
        weight_init="he",
        optimizer="adam",
        regularization=None,
        lambda_reg=0.01,
    ):
        if weight_init == "he":
            self.W = he_init(in_dim, out_dim)
        elif weight_init == "xavier":
            self.W = xavier_init(in_dim, out_dim)
        else:
            self.W = random_init(in_dim, out_dim)
        self.b = np.zeros((1, out_dim))

        self.activation = activation

        self.opt_w = Adam(lr) if optimizer == "adam" else SGD(lr)
        self.opt_b = Adam(lr) if optimizer == "adam" else SGD(lr)

        self.regularization = regularization
        self.lambda_reg = lambda_reg

        self.x = None
        self.z = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x):
        self.x = x
        self.z = x @ self.W + self.b

        if self.activation == "relu":
            return relu(self.z)
        if self.activation == "tanh":
            return tanh(self.z)
        if self.activation == "sigmoid":
            return sigmoid(self.z)
        return self.z

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------
    def backward(self, grad):
        if self.activation == "relu":
            grad = grad * relu_grad(self.z)
        elif self.activation == "tanh":
            grad = grad * tanh_grad(self.z)
        elif self.activation == "sigmoid":
            grad = grad * sigmoid_grad(self.z)

        dW = self.x.T @ grad
        db = np.sum(grad, axis=0, keepdims=True)

        dx = grad @ self.W.T

        if hasattr(self, "regularization") and self.regularization is not None:
            if self.regularization == "l2":
                dW += self.lambda_reg * self.W
            elif self.regularization == "l1":
                dW += self.lambda_reg * np.sign(self.W)

        self.W = self.opt_w.update(self.W, dW)
        self.b = self.opt_b.update(self.b, db)

        return dx

    # ------------------------------------------------------------------
    def parameters(self):
        return [self.W, self.b]
