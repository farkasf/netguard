"""Gradient Boosting Machine, from scratch. NumPy only — no sklearn.

Multiclass via one regression tree per class per boosting round (one-vs-rest
softmax). The surface mimics sklearn (``fit``/``predict``/``predict_proba``)
but the body is fully custom: it boosts the from-scratch CART trees in
:mod:`netguard.ml.tree` against softmax gradients/hessians from
:mod:`netguard.ml.losses`.
"""

from __future__ import annotations

import numpy as np

from netguard.ml.losses import cross_entropy, softmax
from netguard.ml.tree import RegressionTree


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro F1 computed from scratch (no sklearn — used during early stopping)."""
    f1s = []
    for k in range(n_classes):
        tp = float(np.sum((y_pred == k) & (y_true == k)))
        fp = float(np.sum((y_pred == k) & (y_true != k)))
        fn = float(np.sum((y_pred != k) & (y_true == k)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


class GradientBoostingClassifier:
    """From-scratch multiclass GBM."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.3,
        max_depth: int = 4,
        min_samples_leaf: int = 5,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        min_split_gain: float = 0.0,
        subsample: float = 1.0,
        colsample: float = 1.0,
        early_stopping_rounds: int = 0,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.min_split_gain = min_split_gain
        self.subsample = subsample
        self.colsample = colsample
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state

        self.n_classes_: int = 0
        self.n_features_: int = 0
        self.init_score_: np.ndarray | None = None  # log priors, shape [K]
        # trees_[m] is a list of K trees (one per class) for round m.
        self.trees_: list[list[RegressionTree]] = []
        self.train_loss_: list[float] = []
        self.best_iteration_: int = 0
        self.feature_importances_: np.ndarray | None = None

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> GradientBoostingClassifier:
        """Fit the model. ``y`` must be integer class labels ``0..K-1``."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        rng = np.random.default_rng(self.random_state)

        n, d = X.shape
        self.n_features_ = d
        self.n_classes_ = int(y.max()) + 1 if y.size else 0
        K = self.n_classes_

        # One-hot targets.
        Y = np.zeros((n, K), dtype=np.float64)
        Y[np.arange(n), y] = 1.0

        # Initialize raw scores to log class priors (stabilizes early rounds).
        priors = np.bincount(y, minlength=K).astype(np.float64)
        priors = np.clip(priors / max(priors.sum(), 1.0), 1e-6, None)
        self.init_score_ = np.log(priors)
        F = np.tile(self.init_score_, (n, 1))

        self.feature_importances_ = np.zeros(d, dtype=np.float64)
        self.trees_ = []
        self.train_loss_ = []

        best_f1 = -np.inf
        best_round = 0
        rounds_since_best = 0

        n_sub = max(1, int(round(self.subsample * n)))
        n_col = max(1, int(round(self.colsample * d)))

        for m in range(self.n_estimators):
            P = softmax(F)
            G = P - Y  # [n, K] gradients
            H = P * (1.0 - P)  # [n, K] hessians

            # Row subsampling for this round (shared across the K class trees).
            if self.subsample < 1.0:
                rows = rng.choice(n, size=n_sub, replace=False)
            else:
                rows = np.arange(n)

            round_trees: list[RegressionTree] = []
            for k in range(K):
                # Column subsampling per tree.
                if self.colsample < 1.0:
                    feats = rng.choice(d, size=n_col, replace=False)
                else:
                    feats = np.arange(d)

                tree = RegressionTree(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    min_child_weight=self.min_child_weight,
                    reg_lambda=self.reg_lambda,
                    min_split_gain=self.min_split_gain,
                )
                tree.fit(X[rows], G[rows, k], H[rows, k], feature_indices=feats)
                # Update raw scores for all rows (not just the subsample).
                F[:, k] += self.learning_rate * tree.predict(X)
                if tree.feature_importances_ is not None:
                    self.feature_importances_ += tree.feature_importances_
                round_trees.append(tree)

            self.trees_.append(round_trees)
            self.train_loss_.append(cross_entropy(F, Y))

            # Optional early stopping on a validation split.
            if self.early_stopping_rounds > 0 and eval_set is not None:
                Xv, yv = eval_set
                f1 = _macro_f1(np.asarray(yv), self.predict(Xv), K)
                if f1 > best_f1:
                    best_f1 = f1
                    best_round = m
                    rounds_since_best = 0
                else:
                    rounds_since_best += 1
                    if rounds_since_best >= self.early_stopping_rounds:
                        break

        self.best_iteration_ = (
            best_round if (self.early_stopping_rounds > 0 and eval_set is not None) else len(self.trees_) - 1
        )
        return self

    # -------------------------------------------------------------- predict
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return raw additive scores ``F`` of shape ``[n, K]``."""
        X = np.asarray(X, dtype=np.float64)
        assert self.init_score_ is not None, "model not fitted"
        n = X.shape[0]
        F = np.tile(self.init_score_, (n, 1))
        limit = self.best_iteration_ + 1 if self.trees_ else 0
        for round_trees in self.trees_[:limit]:
            for k, tree in enumerate(round_trees):
                F[:, k] += self.learning_rate * tree.predict(X)
        return F

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(self.decision_function(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
