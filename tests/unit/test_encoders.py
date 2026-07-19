"""Unit tests for encoders and scaler."""

from __future__ import annotations

import numpy as np

from netguard.ml.encoders import LabelEncoder, StandardScaler


def test_label_encoder_roundtrip():
    le = LabelEncoder()
    labels = ["DOS", "BENIGN", "PORTSCAN", "BENIGN"]
    ints = le.fit_transform(labels)
    assert le.inverse_transform(ints) == labels
    # Deterministic, sorted class ordering.
    assert le.classes_ == ["BENIGN", "DOS", "PORTSCAN"]


def test_label_encoder_order_independent():
    a = LabelEncoder().fit(["b", "a", "c"]).classes_
    b = LabelEncoder().fit(["c", "b", "a"]).classes_
    assert a == b == ["a", "b", "c"]


def test_scaler_produces_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=5.0, scale=3.0, size=(200, 4))
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    assert np.allclose(Xs.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(Xs.std(axis=0), 1.0, atol=1e-9)


def test_scaler_handles_constant_feature():
    X = np.array([[1.0, 5.0], [1.0, 7.0], [1.0, 9.0]])
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    assert np.all(np.isfinite(Xs))
    assert np.all(Xs[:, 0] == 0.0)  # constant column -> 0 after centering


def test_scaler_roundtrip_dict():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, 3))
    sc = StandardScaler().fit(X)
    sc2 = StandardScaler.from_dict(sc.to_dict())
    assert np.allclose(sc.transform(X), sc2.transform(X))
