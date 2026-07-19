"""Model persistence. NumPy/stdlib only — no sklearn.

A trained model is serialized to a directory holding:
  * ``meta.json`` — hyperparameters, feature names, label mapping, scaler stats,
    macro F1, version string ``vYYYYMMDD-HHMMSS-ffffff``, and the tree structures.

``load`` reconstructs a :class:`NetGuardModel` whose ``predict_proba`` matches
the original bit-for-bit on the same input (asserted in a round-trip test).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from netguard.ml.encoders import LabelEncoder, StandardScaler
from netguard.ml.gbm import GradientBoostingClassifier
from netguard.ml.tree import RegressionTree


def make_version() -> str:
    # Microsecond suffix so models trained within the same second (e.g. two
    # retrain candidates back-to-back) get distinct, unique versions.
    return datetime.now().strftime("v%Y%m%d-%H%M%S-%f")


@dataclass
class NetGuardModel:
    """A self-contained, deployable model: scaler + GBM + label mapping."""

    gbm: GradientBoostingClassifier
    scaler: StandardScaler
    label_encoder: LabelEncoder
    feature_names: list[str]
    version: str = field(default_factory=make_version)
    macro_f1: float = 0.0

    @property
    def classes(self) -> list[str]:
        return self.label_encoder.classes_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Scale raw features then return class probabilities."""
        Xs = self.scaler.transform(np.asarray(X, dtype=np.float64))
        return self.gbm.predict_proba(Xs)

    def predict_labels(self, X: np.ndarray) -> list[str]:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.label_encoder.inverse_transform(idx)

    # ---------------------------------------------------------------- save
    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "version": self.version,
            "macro_f1": self.macro_f1,
            "feature_names": self.feature_names,
            "label_encoder": self.label_encoder.to_dict(),
            "scaler": self.scaler.to_dict(),
            "gbm": {
                "hyperparams": {
                    "n_estimators": self.gbm.n_estimators,
                    "learning_rate": self.gbm.learning_rate,
                    "max_depth": self.gbm.max_depth,
                    "min_samples_leaf": self.gbm.min_samples_leaf,
                    "min_child_weight": self.gbm.min_child_weight,
                    "reg_lambda": self.gbm.reg_lambda,
                    "min_split_gain": self.gbm.min_split_gain,
                    "subsample": self.gbm.subsample,
                    "colsample": self.gbm.colsample,
                },
                "n_classes": self.gbm.n_classes_,
                "n_features": self.gbm.n_features_,
                "init_score": self.gbm.init_score_.tolist()
                if self.gbm.init_score_ is not None
                else None,
                "best_iteration": self.gbm.best_iteration_,
                "trees": [
                    [tree.to_dict() for tree in round_trees]
                    for round_trees in self.gbm.trees_
                ],
                "feature_importances": self.gbm.feature_importances_.tolist()
                if self.gbm.feature_importances_ is not None
                else None,
            },
        }
        (directory / "meta.json").write_text(json.dumps(meta))
        return directory

    # ---------------------------------------------------------------- load
    @classmethod
    def load(cls, directory: str | Path) -> NetGuardModel:
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())

        scaler = StandardScaler.from_dict(meta["scaler"])
        label_encoder = LabelEncoder.from_dict(meta["label_encoder"])

        g = meta["gbm"]
        hp = g["hyperparams"]
        gbm = GradientBoostingClassifier(**hp)
        gbm.n_classes_ = g["n_classes"]
        gbm.n_features_ = g["n_features"]
        gbm.init_score_ = (
            np.asarray(g["init_score"], dtype=np.float64)
            if g["init_score"] is not None
            else None
        )
        gbm.best_iteration_ = g["best_iteration"]
        gbm.trees_ = [
            [RegressionTree.from_dict(td) for td in round_trees]
            for round_trees in g["trees"]
        ]
        gbm.feature_importances_ = (
            np.asarray(g["feature_importances"], dtype=np.float64)
            if g.get("feature_importances") is not None
            else None
        )

        return cls(
            gbm=gbm,
            scaler=scaler,
            label_encoder=label_encoder,
            feature_names=list(meta["feature_names"]),
            version=meta["version"],
            macro_f1=meta["macro_f1"],
        )
