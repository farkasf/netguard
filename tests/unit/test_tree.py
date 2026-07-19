"""Unit tests for the from-scratch CART regression tree."""

from __future__ import annotations

import numpy as np

from netguard.ml.tree import RegressionTree


def _fit_squared_error_tree(X, target, **kw):
    """Fit a tree to a continuous target using gradient = -target, hessian = 1.

    With g = -target and h = 1, the leaf value -G/(H+λ) approximates the mean of
    ``target`` in the leaf (for λ small), so the tree fits the target.
    """
    g = -target
    h = np.ones_like(target)
    return RegressionTree(**kw).fit(X, g, h)


def test_fits_separable_step_function():
    # Target is a step: 0 for x<0.5, 1 otherwise. One split suffices.
    X = np.linspace(0, 1, 50).reshape(-1, 1)
    target = (X[:, 0] >= 0.5).astype(float)
    tree = _fit_squared_error_tree(X, target, max_depth=3, min_samples_leaf=1,
                                   min_child_weight=1e-6, reg_lambda=1e-6)
    pred = tree.predict(X)
    # Predictions should separate the two regimes well.
    assert pred[X[:, 0] < 0.5].mean() < 0.1
    assert pred[X[:, 0] >= 0.5].mean() > 0.9


def test_min_samples_leaf_respected():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    target = rng.normal(size=40)
    tree = _fit_squared_error_tree(X, target, max_depth=10, min_samples_leaf=10,
                                   min_child_weight=1e-9, reg_lambda=1e-9)

    leaf_sizes: list[int] = []

    def walk(node, idx):
        if node.is_leaf:
            leaf_sizes.append(len(idx))
            return
        m = X[idx, node.feature] <= node.threshold
        walk(node.left, idx[m])
        walk(node.right, idx[~m])

    walk(tree.root, np.arange(len(X)))
    assert min(leaf_sizes) >= 10


def test_max_depth_respected():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 3))
    target = rng.normal(size=100)
    tree = _fit_squared_error_tree(X, target, max_depth=2, min_samples_leaf=1,
                                   min_child_weight=1e-9, reg_lambda=1e-9)

    def depth(node):
        if node.is_leaf:
            return 0
        return 1 + max(depth(node.left), depth(node.right))

    assert depth(tree.root) <= 2


def test_split_gain_matches_hand_computation():
    # Tiny array, single feature, known gradients/hessians.
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    g = np.array([-1.0, -1.0, 1.0, 1.0])
    h = np.array([1.0, 1.0, 1.0, 1.0])
    lam = 1.0
    tree = RegressionTree(max_depth=1, min_samples_leaf=1, min_child_weight=0.0,
                          reg_lambda=lam, min_split_gain=-1e9)
    tree.fit(X, g, h)
    # Best split is between index 1 and 2 (threshold 1.5).
    assert tree.root.feature == 0
    assert abs(tree.root.threshold - 1.5) < 1e-9

    # Hand gain for split at 1.5: G_L=-2,H_L=2 ; G_R=2,H_R=2 ; G_p=0,H_p=4.
    g_l, h_l, g_r, h_r, g_p, h_p = -2, 2, 2, 2, 0, 4
    expected = 0.5 * (g_l**2 / (h_l + lam) + g_r**2 / (h_r + lam) - g_p**2 / (h_p + lam))
    # Recompute the gain the tree would have stored.
    assert tree.feature_importances_[0] > 0
    assert abs(tree.feature_importances_[0] - expected) < 1e-9


def test_leaf_value_is_newton_step():
    X = np.array([[0.0], [1.0]])
    g = np.array([2.0, 4.0])
    h = np.array([1.0, 1.0])
    lam = 1.0
    # max_depth 0 -> single leaf over all data.
    tree = RegressionTree(max_depth=0, reg_lambda=lam).fit(X, g, h)
    expected = -(g.sum()) / (h.sum() + lam)  # -6/3 = -2
    assert abs(tree.predict(X)[0] - expected) < 1e-12
