"""Flow scorer.

Loads the active model, scores closed flows, writes them to the rolling
``flows`` table, and persists an ``anomalies`` row whenever a flow is predicted
non-benign or scored with low confidence. Supports hot-reload so the retrain job
can swap models without restarting the process.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from netguard.capture.flow_assembler import FlowRecord
from netguard.config import get_settings
from netguard.features.extractor import extract
from netguard.ml.persistence import NetGuardModel
from netguard.store.repository import Repository


class Scorer:
    """Scores flows against the active NetGuard model."""

    def __init__(self, repo: Repository, model: NetGuardModel | None = None) -> None:
        self.repo = repo
        self.settings = get_settings()
        self.model: NetGuardModel | None = model
        if self.model is None:
            self.reload_model()

    # ------------------------------------------------------------- loading
    def _resolve_active_path(self) -> Path | None:
        active = self.repo.active_model()
        if active is not None:
            p = Path(active["path"])
            if (p / "meta.json").exists():
                return p
        # Fallback (no usable registry entry): pick the saved artifact with the
        # best macro F1, so a rejected retrain candidate — which is always the
        # most recently written directory — can never outrank the incumbent.
        # Ties go to the newest version (versions sort chronologically).
        models_dir = Path(self.settings.models_dir)
        if not models_dir.exists():
            return None
        best: tuple[float, str, Path] | None = None
        for d in models_dir.iterdir():
            meta_path = d / "meta.json"
            if not d.is_dir() or not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            key = (float(meta.get("macro_f1", 0.0)), str(meta.get("version", "")))
            if best is None or key > (best[0], best[1]):
                best = (key[0], key[1], d)
        return best[2] if best else None

    def reload_model(self) -> bool:
        """(Re)load the active model. Returns True if a model was loaded."""
        path = self._resolve_active_path()
        if path is None:
            return False
        self.model = NetGuardModel.load(path)
        return True

    @property
    def model_version(self) -> str:
        return self.model.version if self.model else "none"

    # ------------------------------------------------------------- scoring
    def score_flow(self, record: FlowRecord) -> dict:
        """Score one closed flow; persist flow row and anomaly if warranted.

        Returns a small dict describing the prediction (handy for tests/logs).
        """
        if self.model is None:
            raise RuntimeError("No active model loaded; train one first.")

        features = extract(record)
        proba = self.model.predict_proba(features.reshape(1, -1))[0]
        idx = int(np.argmax(proba))
        confidence = float(proba[idx])
        predicted = self.model.classes[idx]

        duration = max(record.last_ts - record.start_ts, 0.0)
        total_bytes = record.fwd_bytes + record.bwd_bytes

        self.repo.insert_flow(
            last_ts=record.last_ts,
            src_ip=record.src_ip,
            dst_ip=record.dst_ip,
            src_port=record.src_port,
            dst_port=record.dst_port,
            protocol=record.protocol,
            predicted_class=predicted,
            confidence=confidence,
            duration=duration,
            total_packets=record.total_packets,
            total_bytes=total_bytes,
        )

        is_anomalous = predicted != self.settings.benign_label
        low_conf = confidence < self.settings.low_confidence_threshold
        anomaly_id = None
        if is_anomalous or low_conf:
            anomaly_id = self.repo.insert_anomaly(
                ts=record.last_ts,
                src_ip=record.src_ip,
                dst_ip=record.dst_ip,
                src_port=record.src_port,
                dst_port=record.dst_port,
                protocol=record.protocol,
                predicted_class=predicted,
                confidence=confidence,
                features=features.tolist(),
                model_version=self.model_version,
            )

        return {
            "predicted_class": predicted,
            "confidence": confidence,
            "anomaly": anomaly_id is not None,
            "anomaly_id": anomaly_id,
            "low_confidence": low_conf,
        }
