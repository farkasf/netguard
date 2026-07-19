"""API routes: /health /flows /anomalies /metrics /retrain /model/registry."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Query, Request

from netguard.api.schemas import (
    AnomalyOut,
    FlowOut,
    HealthResponse,
    MetricsResponse,
    ModelRegistryEntry,
    RetrainResponse,
)
from netguard.features.extractor import FEATURE_NAMES
from netguard.pipeline.scorer import Scorer
from netguard.store.repository import Repository
from netguard.training.retrain_job import retrain_from_synthetic

router = APIRouter()


def _repo(request: Request) -> Repository:
    return request.app.state.repo


def _scorer(request: Request) -> Scorer | None:
    return getattr(request.app.state, "scorer", None)


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    scorer = _scorer(request)
    version = scorer.model_version if scorer else "none"
    started = request.app.state.started_at
    return HealthResponse(status="ok", model_version=version, uptime_s=time.time() - started)


@router.get("/flows", response_model=list[FlowOut])
def flows(request: Request, limit: int = Query(100, ge=1, le=1000)) -> list[FlowOut]:
    rows = _repo(request).recent_flows(limit=limit)
    return [FlowOut(**r) for r in rows]


@router.get("/anomalies", response_model=list[AnomalyOut])
def anomalies(
    request: Request,
    limit: int = Query(100, ge=1, le=1000),
    since: float | None = Query(None),
) -> list[AnomalyOut]:
    rows = _repo(request).recent_anomalies(limit=limit, since=since)
    return [AnomalyOut(**r) for r in rows]


@router.get("/metrics", response_model=MetricsResponse)
def metrics(request: Request) -> MetricsResponse:
    repo = _repo(request)
    active = repo.active_model()
    flow_count = repo.count_flows()
    anomaly_count = repo.count_anomalies()

    if active is None:
        return MetricsResponse(
            model_version=None, macro_f1=None,
            feature_names=FEATURE_NAMES,
            flow_count=flow_count, anomaly_count=anomaly_count,
        )

    report: dict[str, Any] = active.get("metrics") or {}
    # Pull feature importances from the saved model meta if present.
    importances: list[float] = []
    meta_path = Path(active["path"]) / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        fi = meta.get("gbm", {}).get("feature_importances")
        if fi:
            importances = list(fi)

    return MetricsResponse(
        model_version=active["version"],
        macro_f1=active["macro_f1"],
        macro_precision=report.get("macro_precision"),
        macro_recall=report.get("macro_recall"),
        accuracy=report.get("accuracy"),
        classes=report.get("classes", []),
        per_class=report.get("per_class", {}),
        confusion_matrix=report.get("confusion_matrix", []),
        feature_names=FEATURE_NAMES,
        feature_importances=importances,
        trained_at=active["promoted_at"],
        flow_count=flow_count,
        anomaly_count=anomaly_count,
    )


@router.post("/retrain", response_model=RetrainResponse, status_code=202)
def retrain(request: Request, background: BackgroundTasks) -> RetrainResponse:
    repo = _repo(request)
    scorer = _scorer(request)
    job_id = uuid.uuid4().hex[:12]

    def _on_promote(_path: str) -> None:
        if scorer is not None:
            scorer.reload_model()  # hot-swap the live model; discard the bool

    def _job() -> None:
        result = retrain_from_synthetic(
            repo, on_promote=_on_promote if scorer else None
        )
        request.app.state.last_retrain = {
            "job_id": job_id,
            "status": result.status,
            "candidate_version": result.candidate_version,
            "candidate_f1": result.candidate_f1,
            "incumbent_f1": result.incumbent_f1,
            "reason": result.reason,
        }

    background.add_task(_job)
    return RetrainResponse(
        job_id=job_id, status="accepted",
        detail="Retraining started in the background; poll /metrics or /model/registry.",
    )


@router.get("/retrain/last")
def retrain_last(request: Request) -> dict[str, Any]:
    return getattr(request.app.state, "last_retrain", {}) or {"status": "none"}


@router.get("/model/registry", response_model=list[ModelRegistryEntry])
def model_registry(request: Request) -> list[ModelRegistryEntry]:
    rows = _repo(request).list_models()
    return [
        ModelRegistryEntry(
            version=r["version"], path=r["path"], macro_f1=r["macro_f1"],
            promoted_at=r["promoted_at"], is_active=r["is_active"],
        )
        for r in rows
    ]
