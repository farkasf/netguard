"""Scorer model-resolution fallback.

When the registry has no usable active entry, the scorer must fall back to the
saved artifact with the best macro F1 — never simply the newest directory,
which after a rejected retrain would be the *worse* candidate.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from netguard.config import get_settings
from netguard.ml.encoders import LabelEncoder, StandardScaler
from netguard.ml.gbm import GradientBoostingClassifier
from netguard.ml.persistence import NetGuardModel
from netguard.pipeline.scorer import Scorer


def _save_model(models_dir: Path, version: str, macro_f1: float) -> Path:
    model = NetGuardModel(
        gbm=GradientBoostingClassifier(),
        scaler=StandardScaler().fit(np.zeros((2, 3))),
        label_encoder=LabelEncoder().fit(["BENIGN"]),
        feature_names=["a", "b", "c"],
        version=version,
        macro_f1=macro_f1,
    )
    return model.save(models_dir / version)


def test_fallback_prefers_best_f1_over_newest(tmp_path, tmp_repo, monkeypatch):
    models_dir = tmp_path / "models"
    monkeypatch.setenv("NETGUARD_MODELS_DIR", str(models_dir))
    get_settings.cache_clear()
    try:
        incumbent = _save_model(models_dir, "v20260101-000000-000000", macro_f1=0.95)
        rejected = _save_model(models_dir, "v20260102-000000-000000", macro_f1=0.60)
        # Make the rejected candidate the newest on disk (the old mtime-based
        # fallback would have picked it).
        past = time.time() - 3600
        os.utime(incumbent, (past, past))

        scorer = Scorer(tmp_repo)  # registry is empty -> fallback path
        assert scorer.model is not None
        assert scorer.model.version == "v20260101-000000-000000"
        assert rejected.exists()  # rejected artifact stays for the audit trail
    finally:
        get_settings.cache_clear()


def test_fallback_ties_go_to_newest_version(tmp_path, tmp_repo, monkeypatch):
    models_dir = tmp_path / "models"
    monkeypatch.setenv("NETGUARD_MODELS_DIR", str(models_dir))
    get_settings.cache_clear()
    try:
        _save_model(models_dir, "v20260101-000000-000000", macro_f1=1.0)
        _save_model(models_dir, "v20260102-000000-000000", macro_f1=1.0)

        scorer = Scorer(tmp_repo)
        assert scorer.model is not None
        assert scorer.model.version == "v20260102-000000-000000"
    finally:
        get_settings.cache_clear()


def test_registered_active_model_wins_over_fallback(tmp_path, tmp_repo, monkeypatch):
    models_dir = tmp_path / "models"
    monkeypatch.setenv("NETGUARD_MODELS_DIR", str(models_dir))
    get_settings.cache_clear()
    try:
        _save_model(models_dir, "v20260101-000000-000000", macro_f1=0.5)
        best = _save_model(models_dir, "v20260102-000000-000000", macro_f1=0.99)
        # The registry's active entry points at the *lower*-F1 model; it must
        # still win — the fallback only applies when the registry is unusable.
        tmp_repo.register_model(
            version="v20260101-000000-000000",
            path=str(models_dir / "v20260101-000000-000000"),
            macro_f1=0.5,
            activate=True,
        )

        scorer = Scorer(tmp_repo)
        assert scorer.model is not None
        assert scorer.model.version == "v20260101-000000-000000"
        assert best.exists()
    finally:
        get_settings.cache_clear()
