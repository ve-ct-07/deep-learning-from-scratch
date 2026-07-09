import numpy as np


class Trainer:

    def __init__(self, model, optimizer=None, epochs=20, batch_size=32, patience=None):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience

    def _is_sequence_model(self, x_sample):
        return isinstance(x_sample, list)

    def fit(self, X_train, y_train, X_val=None, y_val=None):

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        best_val = float("inf")
        patience_counter = 0
        n = len(X_train)

        is_seq = self._is_sequence_model(X_train[0])

        for epoch in range(self.epochs):

            idx = np.random.permutation(n)
            losses = []

            # =====================================================
            # SEQUENCE MODELS (RNN / LSTM)
            # =====================================================
            if is_seq:

                for i in idx:

                    x = X_train[i]
                    y = y_train[i]

                    self.model.forward(x)
                    loss = self.model.loss(y)
                    grads = self.model.backward(y)
                    self.model.update(self.optimizer, grads)

                    losses.append(loss)

            # =====================================================
            # BATCH MODELS (MLP / CNN)
            # =====================================================
            else:

                X_epoch = X_train[idx]
                y_epoch = y_train[idx]

                for i in range(0, n, self.batch_size):

                    xb = X_epoch[i:i + self.batch_size]
                    yb = y_epoch[i:i + self.batch_size]

                    logits = self.model.forward(xb)

                    loss = None
                    if hasattr(self.model, "loss_fn"):
                        loss = self.model.loss_fn(yb, logits)

                    self.model.backward(yb)

                    if loss is not None:
                        losses.append(loss)

            # -------------------------
            # METRICS
            # -------------------------
            train_loss = float(np.mean(losses))
            train_pred = self.predict(X_train)
            train_acc = np.mean(train_pred == y_train)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # -------------------------
            # VALIDATION
            # -------------------------
            if X_val is not None and y_val is not None:

                val_losses = []

                for x, y in zip(X_val, y_val):

                    probs = np.atleast_1d(self.model.predict_proba(x))
                    if probs.size == 1:
                        p_true = probs.item() if y == 1 else 1 - probs.item()
                    else:
                        p_true = probs[y]
                    val_losses.append(-np.log(p_true + 1e-15))

                val_loss = float(np.mean(val_losses))
                val_pred = self.predict(X_val)
                val_acc = np.mean(val_pred == y_val)

                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                print(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f} | "
                    f"Val Acc: {val_acc:.4f}"
                )

                # -------------------------
                # EARLY STOPPING
                # -------------------------
                if self.patience is not None:

                    if val_loss < best_val:
                        best_val = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.patience:
                            print(f"Early stopping at epoch {epoch+1}")
                            return history

            else:
                print(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f}"
                )

        return history

    # -----------------------------
    def predict(self, X):

        # sequence model
        if isinstance(X[0], list):
            return np.array([self.model.predict(x) for x in X])

        # batch model
        return self.model.predict(X)