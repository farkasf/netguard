"""Candidate retraining with an F1 promotion gate.

A candidate model is trained on a fresh train/test split and evaluated. It is
promoted (registered + activated, and the live scorer hot-reloaded) **only** if
its macro F1 strictly beats the incumbent active model's. Otherwise the
incumbent is left untouched.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from netguard.config import get_settings
from netguard.store.repository import Repository
from netguard.training import dataset as ds
from netguard.training.train import evaluate, train_model


@dataclass
class RetrainResult:
    status: str  # "promoted" | "rejected" | "promoted_first"
    candidate_version: str
    candidate_f1: float
    incumbent_f1: float | None
    reason: str


def run_retrain(
    X: np.ndarray,
    y: np.ndarray,
    repo: Repository,
    *,
    test_size: float = 0.25,
    on_promote: Callable[[str], None] | None = None,
    gbm_kwargs: dict[str, Any] | None = None,
) -> RetrainResult:
    """Train a candidate, evaluate, and promote only if it beats the incumbent."""
    settings = get_settings()
    X_tr, X_te, y_tr, y_te = ds.train_test_split(X, y, test_size=test_size)

    candidate = train_model(X_tr, y_tr, gbm_kwargs=gbm_kwargs)
    report = evaluate(candidate, X_te, y_te)
    candidate.macro_f1 = report["macro_f1"]

    out_dir = Path(settings.models_dir) / candidate.version
    candidate.save(out_dir)
    (out_dir / "metrics.json").write_text(__import__("json").dumps(report, indent=2))

    incumbent = repo.active_model()
    incumbent_f1 = incumbent["macro_f1"] if incumbent else None

    if incumbent is None:
        repo.register_model(
            version=candidate.version, path=str(out_dir),
            macro_f1=candidate.macro_f1, metrics=report, activate=True,
        )
        if on_promote:
            on_promote(str(out_dir))
        return RetrainResult(
            status="promoted_first",
            candidate_version=candidate.version,
            candidate_f1=candidate.macro_f1,
            incumbent_f1=None,
            reason="no incumbent; activated first model",
        )

    if candidate.macro_f1 > incumbent_f1:  # type: ignore[operator]
        repo.register_model(
            version=candidate.version, path=str(out_dir),
            macro_f1=candidate.macro_f1, metrics=report, activate=True,
        )
        if on_promote:
            on_promote(str(out_dir))
        return RetrainResult(
            status="promoted",
            candidate_version=candidate.version,
            candidate_f1=candidate.macro_f1,
            incumbent_f1=incumbent_f1,
            reason=f"candidate F1 {candidate.macro_f1:.4f} > incumbent {incumbent_f1:.4f}",
        )

    # Rejected: register the candidate (inactive) for the audit trail, leave
    # the incumbent active.
    repo.register_model(
        version=candidate.version, path=str(out_dir),
        macro_f1=candidate.macro_f1, metrics=report, activate=False,
    )
    return RetrainResult(
        status="rejected",
        candidate_version=candidate.version,
        candidate_f1=candidate.macro_f1,
        incumbent_f1=incumbent_f1,
        reason=f"candidate F1 {candidate.macro_f1:.4f} <= incumbent {incumbent_f1:.4f}",
    )


def retrain_from_synthetic(
    repo: Repository, on_promote: Callable[[str], None] | None = None
) -> RetrainResult:
    """Convenience entry point used by the API /retrain background task."""
    X, y = ds.make_synthetic(n_per_class=300, random_state=int(time.time()) % 10_000)
    return run_retrain(X, y, repo, on_promote=on_promote)
