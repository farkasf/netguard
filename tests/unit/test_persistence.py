"""Unit tests for model persistence (save -> load -> predict equality)."""

from __future__ import annotations

import numpy as np

from netguard.ml.encoders import LabelEncoder, StandardScaler
from netguard.ml.gbm import GradientBoostingClassifier
from netguard.ml.persistence import NetGuardModel


def _build_model(toy_3class):
    X, y = toy_3class
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    le = LabelEncoder().fit(["A", "B", "C"])
    gbm = GradientBoostingClassifier(n_estimators=25, max_depth=3).fit(Xs, y)
    return NetGuardModel(
        gbm=gbm, scaler=sc, label_encoder=le,
        feature_names=[f"f{i}" for i in range(X.shape[1])], macro_f1=0.95,
    ), X


def test_save_load_predict_proba_equality(tmp_path, toy_3class):
    model, X = _build_model(toy_3class)
    before = model.predict_proba(X)
    model.save(tmp_path / "m")
    loaded = NetGuardModel.load(tmp_path / "m")
    after = loaded.predict_proba(X)
    assert np.array_equal(before, after)  # bit-for-bit


def test_metadata_round_trips(tmp_path, toy_3class):
    model, _ = _build_model(toy_3class)
    model.save(tmp_path / "m")
    loaded = NetGuardModel.load(tmp_path / "m")
    assert loaded.version == model.version
    assert loaded.macro_f1 == model.macro_f1
    assert loaded.feature_names == model.feature_names
    assert loaded.classes == model.classes
