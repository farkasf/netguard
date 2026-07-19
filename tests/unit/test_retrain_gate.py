"""Tests for the retrain F1 promotion gate."""

from __future__ import annotations

import numpy as np
import pytest

from netguard.config import get_settings
from netguard.training import dataset as ds
from netguard.training.retrain_job import run_retrain


@pytest.fixture
def good_data():
    return ds.make_synthetic(n_per_class=200, random_state=1)


@pytest.fixture
def tmp_models(tmp_path, monkeypatch):
    """Point settings.models_dir at a throwaway directory for the test."""
    monkeypatch.setenv("NETGUARD_MODELS_DIR", str(tmp_path / "models"))
    get_settings.cache_clear()
    yield tmp_path / "models"
    get_settings.cache_clear()


def _scramble(y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = ["BENIGN", "DOS", "PORTSCAN"]
    return np.asarray([labels[i] for i in rng.integers(0, 3, len(y))], dtype=object)


def test_first_candidate_is_promoted(tmp_repo, good_data, tmp_models):
    X, y = good_data
    result = run_retrain(X, y, tmp_repo)
    assert result.status == "promoted_first"
    active = tmp_repo.active_model()
    assert active is not None and active["version"] == result.candidate_version


def test_better_candidate_promoted_and_reloads(tmp_repo, good_data, tmp_models):
    X, y = good_data
    # Seed a deliberately weak incumbent (scrambled labels -> low F1).
    weak = run_retrain(X, _scramble(y, 0), tmp_repo)
    assert weak.status == "promoted_first"
    weak_f1 = tmp_repo.active_model()["macro_f1"]

    reloaded = {"called": False}
    strong = run_retrain(X, y, tmp_repo, on_promote=lambda _p: reloaded.__setitem__("called", True))
    assert strong.status == "promoted"
    assert strong.candidate_f1 > weak_f1
    assert reloaded["called"] is True
    assert tmp_repo.active_model()["version"] == strong.candidate_version


def test_worse_candidate_rejected(tmp_repo, good_data, tmp_models):
    X, y = good_data
    strong = run_retrain(X, y, tmp_repo)
    incumbent_version = tmp_repo.active_model()["version"]

    result = run_retrain(X, _scramble(y, 5), tmp_repo)
    assert result.status == "rejected"
    assert result.candidate_f1 <= strong.candidate_f1
    assert tmp_repo.active_model()["version"] == incumbent_version  # unchanged
