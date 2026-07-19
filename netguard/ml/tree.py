"""CART regression tree, from scratch. NumPy only — no sklearn.

The tree is the weak learner of the GBM. It fits the Newton step of a
second-order objective using per-sample gradients ``g`` and hessians ``h``.
Split quality uses the modern GBM gain:

    gain = 0.5 * (G_L^2/(H_L+λ) + G_R^2/(H_R+λ) − G_p^2/(H_p+λ)) − γ

and leaf values use the Newton step  leaf = −G / (H + λ).

The per-feature threshold scan is vectorized with prefix sums over the sorted
order: sort once per feature, then derive (G_L, H_L, G_R, H_R) for every
candidate threshold in O(n). No pure-Python sample×threshold double loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Node:
    """A tree node. Internal nodes set feature/threshold/left/right; leaves set value."""

    feature: int = -1
    threshold: float = 0.0
    left: Node | None = None
    right: Node | None = None
    value: float = 0.0  # leaf prediction (Newton step)
    is_leaf: bool = True


class RegressionTree:
    """A single CART regression tree trained on (gradient, hessian) targets."""

    def __init__(
        self,
        max_depth: int = 4,
        min_samples_leaf: int = 5,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        min_split_gain: float = 0.0,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.min_split_gain = min_split_gain
        self.root: Node | None = None
        # Accumulated split gain per feature index (for feature importance).
        self.feature_importances_: np.ndarray | None = None
        self._n_features = 0

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        X: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        feature_indices: np.ndarray | None = None,
    ) -> RegressionTree:
        """Fit the tree.

        ``X`` is ``[n, d]``; ``g`` and ``h`` are length-``n`` gradient/hessian
        vectors. ``feature_indices`` optionally restricts the columns considered
        (column subsampling, supplied by the GBM).
        """
        X = np.asarray(X, dtype=np.float64)
        g = np.asarray(g, dtype=np.float64)
        h = np.asarray(h, dtype=np.float64)
        self._n_features = X.shape[1]
        self.feature_importances_ = np.zeros(self._n_features, dtype=np.float64)
        if feature_indices is None:
            feature_indices = np.arange(self._n_features)
        self.root = self._build(X, g, h, feature_indices, depth=0)
        return self

    def _leaf_value(self, g_sum: float, h_sum: float) -> float:
        return float(-g_sum / (h_sum + self.reg_lambda))

    def _build(
        self, X: np.ndarray, g: np.ndarray, h: np.ndarray, feats: np.ndarray, depth: int
    ) -> Node:
        g_sum = float(g.sum())
        h_sum = float(h.sum())
        leaf = Node(value=self._leaf_value(g_sum, h_sum), is_leaf=True)

        n = X.shape[0]
        if depth >= self.max_depth or n < 2 * self.min_samples_leaf:
            return leaf

        best = self._best_split(X, g, h, feats, g_sum, h_sum)
        if best is None:
            return leaf

        feat, thr, gain = best
        self.feature_importances_[feat] += gain  # type: ignore[index]
        mask = X[:, feat] <= thr
        left = self._build(X[mask], g[mask], h[mask], feats, depth + 1)
        right = self._build(X[~mask], g[~mask], h[~mask], feats, depth + 1)
        return Node(feature=feat, threshold=thr, left=left, right=right, is_leaf=False)

    def _best_split(
        self,
        X: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        feats: np.ndarray,
        g_sum: float,
        h_sum: float,
    ) -> tuple[int, float, float] | None:
        lam = self.reg_lambda
        parent_term = (g_sum * g_sum) / (h_sum + lam)
        best_gain = self.min_split_gain
        best: tuple[int, float, float] | None = None
        n = X.shape[0]

        for feat in feats:
            col = X[:, feat]
            order = np.argsort(col, kind="mergesort")
            col_sorted = col[order]
            g_sorted = g[order]
            h_sorted = h[order]

            # Prefix sums: G_L/H_L are sums of the first k samples (k = 1..n-1).
            g_cum = np.cumsum(g_sorted)
            h_cum = np.cumsum(h_sorted)

            # Candidate splits only between distinct consecutive feature values.
            distinct = col_sorted[1:] != col_sorted[:-1]
            # Index k means: left = samples [0..k], right = [k+1..n-1].
            k = np.arange(n - 1)
            valid = distinct.copy()
            # Honour min_samples_leaf on both sides.
            valid &= (k + 1) >= self.min_samples_leaf
            valid &= (n - (k + 1)) >= self.min_samples_leaf
            if not valid.any():
                continue

            g_l = g_cum[:-1]
            h_l = h_cum[:-1]
            g_r = g_sum - g_l
            h_r = h_sum - h_l

            # min_child_weight: minimum sum of hessians per child.
            valid &= h_l >= self.min_child_weight
            valid &= h_r >= self.min_child_weight
            if not valid.any():
                continue

            gain = 0.5 * (
                (g_l * g_l) / (h_l + lam) + (g_r * g_r) / (h_r + lam) - parent_term
            )
            gain = np.where(valid, gain, -np.inf)
            j = int(np.argmax(gain))
            if gain[j] > best_gain:
                # Threshold = midpoint between the two distinct feature values.
                thr = float((col_sorted[j] + col_sorted[j + 1]) / 2.0)
                best_gain = float(gain[j])
                best = (int(feat), thr, best_gain)

        return best

    # -------------------------------------------------------------- predict
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vectorized batch prediction by routing rows down the tree."""
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        out = np.zeros(n, dtype=np.float64)
        if self.root is None:
            return out
        idx = np.arange(n)
        self._predict_node(self.root, X, idx, out)
        return out

    def _predict_node(
        self, node: Node, X: np.ndarray, idx: np.ndarray, out: np.ndarray
    ) -> None:
        if node.is_leaf:
            out[idx] = node.value
            return
        col = X[idx, node.feature]
        left_mask = col <= node.threshold
        if node.left is not None:
            self._predict_node(node.left, X, idx[left_mask], out)
        if node.right is not None:
            self._predict_node(node.right, X, idx[~left_mask], out)

    # ---------------------------------------------------------- (de)serialize
    def to_dict(self) -> dict:
        return {
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "min_child_weight": self.min_child_weight,
            "reg_lambda": self.reg_lambda,
            "min_split_gain": self.min_split_gain,
            "root": _node_to_dict(self.root),
        }

    @classmethod
    def from_dict(cls, d: dict) -> RegressionTree:
        tree = cls(
            max_depth=d["max_depth"],
            min_samples_leaf=d["min_samples_leaf"],
            min_child_weight=d["min_child_weight"],
            reg_lambda=d["reg_lambda"],
            min_split_gain=d["min_split_gain"],
        )
        tree.root = _node_from_dict(d["root"])
        return tree


def _node_to_dict(node: Node | None) -> dict | None:
    if node is None:
        return None
    if node.is_leaf:
        return {"leaf": True, "value": node.value}
    return {
        "leaf": False,
        "feature": node.feature,
        "threshold": node.threshold,
        "left": _node_to_dict(node.left),
        "right": _node_to_dict(node.right),
    }


def _node_from_dict(d: dict | None) -> Node | None:
    if d is None:
        return None
    if d["leaf"]:
        return Node(value=float(d["value"]), is_leaf=True)
    return Node(
        feature=int(d["feature"]),
        threshold=float(d["threshold"]),
        left=_node_from_dict(d["left"]),
        right=_node_from_dict(d["right"]),
        is_leaf=False,
    )
