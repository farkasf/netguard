"""Unit tests for the from-scratch GBM."""

from __future__ import annotations

import numpy as np

from netguard.ml.gbm import GradientBoostingClassifier


def test_training_loss_monotonically_non_increasing():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # linearly separable-ish, 2-class
    clf = GradientBoostingClassifier(n_estimators=40, learning_rate=0.2, max_depth=3)
    clf.fit(X, y)
    losses = np.array(clf.train_loss_)
    # Allow tiny numerical wiggle.
    assert np.all(np.diff(losses) <= 1e-9)


def test_converges_on_toy_3class(toy_3class):
    X, y = toy_3class
    clf = GradientBoostingClassifier(n_estimators=60, learning_rate=0.3, max_depth=3)
    clf.fit(X, y)
    acc = (clf.predict(X) == y).mean()
    assert acc > 0.98


def test_predict_proba_rows_sum_to_one(toy_3class):
    X, y = toy_3class
    clf = GradientBoostingClassifier(n_estimators=20, max_depth=2).fit(X, y)
    P = clf.predict_proba(X)
    assert P.shape == (len(X), 3)
    assert np.allclose(P.sum(axis=1), 1.0)


def test_subsample_and_colsample_run(toy_3class):
    X, y = toy_3class
    clf = GradientBoostingClassifier(
        n_estimators=30, max_depth=3, subsample=0.7, colsample=0.7, random_state=1
    )
    clf.fit(X, y)
    acc = (clf.predict(X) == y).mean()
    assert acc > 0.9  # still learns well with subsampling


def test_early_stopping_limits_used_trees(toy_3class):
    X, y = toy_3class
    # Split for validation.
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    cut = int(0.7 * len(X))
    tr, va = idx[:cut], idx[cut:]
    clf = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.3, max_depth=3, early_stopping_rounds=5
    )
    clf.fit(X[tr], y[tr], eval_set=(X[va], y[va]))
    # Should stop well before 200 rounds on an easy problem.
    assert len(clf.trees_) < 200
    assert clf.best_iteration_ < len(clf.trees_)


def test_feature_importances_populated(toy_3class):
    X, y = toy_3class
    clf = GradientBoostingClassifier(n_estimators=20, max_depth=3).fit(X, y)
    assert clf.feature_importances_ is not None
    assert clf.feature_importances_.shape == (X.shape[1],)
    assert clf.feature_importances_.sum() > 0
