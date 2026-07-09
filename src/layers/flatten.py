from ..modules import Module


class Flatten(Module):
    """Reshape (N, H, W, C) → (N, H*W*C) and back."""

    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.shape)

    def parameters(self):
        return []
