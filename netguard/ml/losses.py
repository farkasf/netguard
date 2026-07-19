"""Loss gradients and hessians. NumPy only — no sklearn.

Two paths:
  * softmax cross-entropy for K-class GBM (the production path), and
  * binary logistic (sigmoid) for the 2-class case and unit tests.

For both, the GBM maintains additive raw scores ``F`` and converts them to
probabilities; the per-class gradient is ``p − y`` and the hessian ``p(1−p)``.
"""

from __future__ import annotations

import numpy as np


def softmax(raw: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax over a ``[n, K]`` matrix."""
    raw = np.asarray(raw, dtype=np.float64)
    shifted = raw - raw.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def sigmoid(raw: np.ndarray) -> np.ndarray:
    """Numerically stable elementwise sigmoid."""
    raw = np.asarray(raw, dtype=np.float64)
    out = np.empty_like(raw)
    pos = raw >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-raw[pos]))
    ex = np.exp(raw[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def softmax_grad_hess(
    F: np.ndarray, Y_onehot: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (gradients, hessians) for softmax cross-entropy.

    ``F`` and ``Y_onehot`` are ``[n, K]``. ``g = p − y``, ``h = p(1−p)``.
    """
    p = softmax(F)
    g = p - Y_onehot
    h = p * (1.0 - p)
    return g, h


def logistic_grad_hess(
    f: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (gradients, hessians) for binary logistic loss.

    ``f`` is the length-``n`` raw score, ``y`` in {0,1}. ``g = p − y``,
    ``h = p(1−p)``.
    """
    p = sigmoid(f)
    g = p - y
    h = p * (1.0 - p)
    return g, h


def cross_entropy(F: np.ndarray, Y_onehot: np.ndarray) -> float:
    """Mean multiclass cross-entropy loss (for convergence monitoring)."""
    p = softmax(F)
    eps = 1e-15
    return float(-np.sum(Y_onehot * np.log(p + eps)) / F.shape[0])
