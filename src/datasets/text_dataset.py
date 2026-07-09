"""
Text dataset utilities for IMDB sentiment classification.

Pipeline:
    raw text → clean → tokenize → build vocab → encode → pad → split
"""

import os
import re
import numpy as np
from collections import Counter


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(texts):
    """Return list of token lists."""
    return [clean_text(t).split() for t in texts]


def build_vocab(token_lists, max_vocab_size: int = 10_000):
    """
    Build word→index mapping from a list of token lists.

    Special tokens:
        <PAD> → 0
        <UNK> → 1

    Returns:
        word2idx : dict
        idx2word : dict
    """
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    most_common = counter.most_common(max_vocab_size - 2)

    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for i, (word, _) in enumerate(most_common, start=2):
        word2idx[word] = i

    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word


def encode(tokens, word2idx: dict):
    """Map tokens to indices; unknown words → 1 (<UNK>)."""
    return [word2idx.get(tok, 1) for tok in tokens]


def pad(sequence, max_len: int):
    """Truncate or zero-pad a sequence to exactly max_len."""
    if len(sequence) >= max_len:
        return sequence[:max_len]
    return sequence + [0] * (max_len - len(sequence))


def split_data(X, y, val_ratio: float = 0.1, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(X)

    idx = rng.permutation(n)
    val_size = int(val_ratio * n)

    val_idx = idx[:val_size]
    train_idx = idx[val_size:]

    if isinstance(X, list):
        X = np.array(X, dtype=object)

    y = np.asarray(y)

    return (
        X[train_idx].tolist(),
        y[train_idx],
        X[val_idx].tolist(),
        y[val_idx],
    )

def preprocess_dataset(texts, word2idx=None, max_vocab_size=10_000, max_len=100):
    """
    Tokenize → (build vocab) → encode → pad.
    """
    token_lists = tokenize(texts)

    if word2idx is None:
        word2idx, idx2word = build_vocab(token_lists, max_vocab_size)
    else:
        idx2word = {idx: w for w, idx in word2idx.items()}

    sequences = [
        pad(encode(toks, word2idx), max_len)
        for toks in token_lists
    ]
    return sequences, word2idx, idx2word


def load_imdb_data(data_dir: str):
    """
    Load raw IMDB texts and binary labels from the extracted aclImdb folder.

    Expected structure:
        data_dir/
            pos/*.txt
            neg/*.txt
    """
    texts, labels = [], []
    for label_type in ("pos", "neg"):
        label = 1 if label_type == "pos" else 0
        dir_path = os.path.join(data_dir, label_type)
        for fname in os.listdir(dir_path):
            if fname.endswith(".txt"):
                with open(os.path.join(dir_path, fname), encoding="utf-8") as f:
                    texts.append(f.read())
                labels.append(label)
    return texts, labels


def load_imdb_dataset(
    train_dir: str = "aclImdb/train",
    test_dir:  str = "aclImdb/test",
    max_vocab_size: int = 20_000,
    max_len:        int = 250,
    val_ratio:    float = 0.1,
):
    """
    Full IMDB loading pipeline.
    """
    train_texts, train_labels = load_imdb_data(train_dir)
    test_texts,  test_labels  = load_imdb_data(test_dir)

    X_train_full, word2idx, idx2word = preprocess_dataset(
        train_texts,
        max_vocab_size=max_vocab_size,
        max_len=max_len,
    )
    X_test, _, _ = preprocess_dataset(
        test_texts,
        word2idx=word2idx,
        max_len=max_len,
    )

    X_train, y_train, X_val, y_val = split_data(
        X_train_full, train_labels, val_ratio=val_ratio
    )

    return {
        "X_train":    X_train,
        "y_train":    y_train,
        "X_val":      X_val,
        "y_val":      y_val,
        "X_test":     X_test,
        "y_test":     np.array(test_labels, dtype=np.int32),
        "word2idx":   word2idx,
        "idx2word":   idx2word,
        "vocab_size": len(word2idx),
    }


def generate_sequence_dataset(
    n_train=5000,
    n_val=1000,
    n_test=1000,
    seq_len=30,
    vocab_size=100,
    seed=42,
):
    """
    Binary classification based on order:
        label = 1 if token 7 appears before token 9
        label = 0 if token 9 appears before token 7

    Every sequence contains both tokens exactly once.
    """

    rng = np.random.default_rng(seed)

    def make_split(n):
        X = []
        y = []

        for _ in range(n):
            seq = rng.integers(2, vocab_size, size=seq_len)
            mask = (seq == 7) | (seq == 9)
            while mask.any():
                seq[mask] = rng.integers(2, vocab_size, size=mask.sum())
                mask = (seq == 7) | (seq == 9)

            i = rng.integers(seq_len)
            j = rng.integers(seq_len)

            while j == i:
                j = rng.integers(seq_len)

            seq[i] = 7
            seq[j] = 9

            label = int(i < j)

            X.append(seq.tolist())
            y.append(label)

        return X, np.array(y)

    X_train, y_train = make_split(n_train)
    X_val, y_val = make_split(n_val)
    X_test, y_test = make_split(n_test)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "vocab_size": vocab_size,
    }
