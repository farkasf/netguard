"""From-scratch label encoding and feature scaling. NumPy only — no sklearn.

Both objects persist their fitted state with the model so inference exactly
reproduces the training-time transform.
"""

from __future__ import annotations

import numpy as np


class LabelEncoder:
    """Deterministic string<->int label mapping.

    Classes are ordered by first sorting unique labels, so a given label set
    always yields the same integer assignment regardless of row order.
    """

    def __init__(self) -> None:
        self.classes_: list[str] = []
        self._to_int: dict[str, int] = {}

    def fit(self, labels: list[str] | np.ndarray) -> LabelEncoder:
        uniq = sorted({str(x) for x in labels})
        self.classes_ = uniq
        self._to_int = {label: i for i, label in enumerate(uniq)}
        return self

    def transform(self, labels: list[str] | np.ndarray) -> np.ndarray:
        return np.asarray([self._to_int[str(x)] for x in labels], dtype=np.int64)

    def fit_transform(self, labels: list[str] | np.ndarray) -> np.ndarray:
        return self.fit(labels).transform(labels)

    def inverse_transform(self, ints: np.ndarray) -> list[str]:
        return [self.classes_[int(i)] for i in ints]

    @property
    def n_classes(self) -> int:
        return len(self.classes_)

    def to_dict(self) -> dict:
        return {"classes": self.classes_}

    @classmethod
    def from_dict(cls, d: dict) -> LabelEncoder:
        enc = cls()
        enc.classes_ = list(d["classes"])
        enc._to_int = {label: i for i, label in enumerate(enc.classes_)}
        return enc


class StandardScaler:
    """Per-feature standardization to mean 0 / std 1 (zero-variance safe)."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> StandardScaler:
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0.0] = 1.0  # avoid divide-by-zero on constant features
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None, "scaler not fitted"
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self) -> dict:
        assert self.mean_ is not None and self.std_ is not None
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> StandardScaler:
        sc = cls()
        sc.mean_ = np.asarray(d["mean"], dtype=np.float64)
        sc.std_ = np.asarray(d["std"], dtype=np.float64)
        return sc
