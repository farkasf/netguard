"""Unit tests for loss gradients/hessians vs finite differences."""

from __future__ import annotations

import numpy as np

from netguard.ml.losses import (
    cross_entropy,
    logistic_grad_hess,
    sigmoid,
    softmax,
    softmax_grad_hess,
)


def test_softmax_rows_sum_to_one():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(10, 4))
    P = softmax(F)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.all(P > 0)


def test_softmax_numerically_stable_on_large_logits():
    F = np.array([[1000.0, 1001.0, 999.0]])
    P = softmax(F)
    assert np.all(np.isfinite(P))
    assert np.isclose(P.sum(), 1.0)


def test_sigmoid_stable_on_extremes():
    x = np.array([-1000.0, 0.0, 1000.0])
    s = sigmoid(x)
    assert np.all(np.isfinite(s))
    assert np.isclose(s[1], 0.5)
    assert 0.0 <= s[0] < 1e-6
    assert 1.0 - 1e-6 < s[2] <= 1.0


def test_softmax_gradient_matches_finite_difference():
    rng = np.random.default_rng(2)
    n, k = 6, 3
    F = rng.normal(size=(n, k))
    Y = np.zeros((n, k))
    Y[np.arange(n), rng.integers(0, k, n)] = 1.0

    g, h = softmax_grad_hess(F, Y)
    eps = 1e-6
    g_num = np.zeros_like(F)
    for i in range(n):
        for j in range(k):
            Fp, Fm = F.copy(), F.copy()
            Fp[i, j] += eps
            Fm[i, j] -= eps
            # cross_entropy averages over n; gradient per-sample = n * d(mean)/dF.
            g_num[i, j] = (cross_entropy(Fp, Y) - cross_entropy(Fm, Y)) / (2 * eps) * n
    assert np.allclose(g, g_num, atol=1e-4)


def test_softmax_hessian_matches_finite_difference():
    rng = np.random.default_rng(3)
    n, k = 5, 3
    F = rng.normal(size=(n, k))
    Y = np.zeros((n, k))
    Y[np.arange(n), rng.integers(0, k, n)] = 1.0
    _, h = softmax_grad_hess(F, Y)

    eps = 1e-5
    h_num = np.zeros_like(F)
    for i in range(n):
        for j in range(k):
            Fp, Fm = F.copy(), F.copy()
            Fp[i, j] += eps
            Fm[i, j] -= eps
            gp, _ = softmax_grad_hess(Fp, Y)
            gm, _ = softmax_grad_hess(Fm, Y)
            h_num[i, j] = (gp[i, j] - gm[i, j]) / (2 * eps)
    assert np.allclose(h, h_num, atol=1e-4)


def test_logistic_grad_hess_basic():
    f = np.array([0.0])
    y = np.array([1.0])
    g, h = logistic_grad_hess(f, y)
    assert np.isclose(g[0], 0.5 - 1.0)  # p - y = 0.5 - 1
    assert np.isclose(h[0], 0.25)  # p(1-p)
