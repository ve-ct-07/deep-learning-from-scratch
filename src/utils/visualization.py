"""
Visualisation utilities.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------

def plot_training_history(history: dict, title: str = "Training History"):
    """
    Plot loss and accuracy curves from a history dict.
    """

    has_acc = "train_acc" in history and "val_acc" in history
    n_plots = 2 if has_acc else 1

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train", linewidth=1.8)
    axes[0].plot(epochs, history["val_loss"],   label="Val",   linewidth=1.8, linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if has_acc:
        axes[1].plot(epochs, history["train_acc"], label="Train", linewidth=1.8)
        axes[1].plot(epochs, history["val_acc"],   label="Val",   linewidth=1.8, linestyle="--")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy")
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names=None,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
):
    """
    Plot a normalised confusion matrix.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = int(max(y_true.max(), y_pred.max()) + 1)

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(max(5, n_classes), max(4, n_classes)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(n_classes)
    labels = class_names if class_names else [str(i) for i in range(n_classes)]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    thresh = 0.5
    for i in range(n_classes):
        for j in range(n_classes):
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.2f})",
                    ha="center", va="center", color=color, fontsize=8)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sample predictions
# ---------------------------------------------------------------------------

def plot_sample_predictions(
    X,
    y_true,
    y_pred,
    class_names=None,
    n_samples: int = 16,
    img_shape=None,
    title: str = "Sample Predictions",
):
    """
    Display a grid of images with true / predicted labels.

    Args:
        X           : (N, H, W) or (N, H, W, C) or (N, D) float array
        y_true      : (N,) integer labels
        y_pred      : (N,) integer labels
        class_names : list of str (optional)
        n_samples   : number of samples to show (will be rounded to a square)
        img_shape   : (H, W) or (H, W, C) if X is flat — required when X.ndim==2
        title       : figure title

    Returns:
        matplotlib Figure
    """

    n_samples = min(n_samples, len(X))
    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))

    indices = np.random.choice(len(X), n_samples, replace=False)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.0))
    axes = np.array(axes).flatten()

    for ax, idx in zip(axes, indices):
        img = X[idx]
        if img.ndim == 1:
            if img_shape is None:
                raise ValueError("img_shape must be provided when X is flat (2-D).")
            img = img.reshape(img_shape)
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)

        ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=1)

        t = class_names[y_true[idx]] if class_names else str(y_true[idx])
        p = class_names[y_pred[idx]] if class_names else str(y_pred[idx])
        color = "green" if y_true[idx] == y_pred[idx] else "red"
        ax.set_title(f"T:{t}\nP:{p}", fontsize=7, color=color)
        ax.axis("off")

    for ax in axes[n_samples:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# RNN / LSTM training loss
# ---------------------------------------------------------------------------

def plot_rnn_loss(
    losses,
    smooth: int = 50,
    title: str = "RNN Training Loss",
):
    """
    Plot per-step loss curve for RNN/LSTM with optional exponential smoothing.
    """

    losses = np.asarray(losses, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses, alpha=0.25, linewidth=0.8, color="steelblue", label="Raw loss")

    if smooth > 1 and len(losses) >= smooth:
        alpha = 2 / (smooth + 1)
        ema = np.zeros_like(losses)
        ema[0] = losses[0]
        for i in range(1, len(losses)):
            ema[i] = alpha * losses[i] + (1 - alpha) * ema[i - 1]
        ax.plot(ema, linewidth=1.8, color="steelblue", label=f"EMA (span={smooth})")

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
