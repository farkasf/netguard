"""Pydantic response models for the API."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_version: str
    uptime_s: float


class FlowOut(BaseModel):
    id: int
    last_ts: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    predicted_class: str
    confidence: float
    duration: float
    total_packets: int
    total_bytes: int


class AnomalyOut(BaseModel):
    id: int
    ts: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    predicted_class: str
    confidence: float
    features: list[float]
    model_version: str


class PerClassMetric(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int


class MetricsResponse(BaseModel):
    model_version: str | None
    macro_f1: float | None
    macro_precision: float | None = None
    macro_recall: float | None = None
    accuracy: float | None = None
    classes: list[str] = []
    per_class: dict[str, PerClassMetric] = {}
    confusion_matrix: list[list[int]] = []
    feature_names: list[str] = []
    feature_importances: list[float] = []
    trained_at: float | None = None
    flow_count: int = 0
    anomaly_count: int = 0


class RetrainResponse(BaseModel):
    job_id: str
    status: str
    detail: str


class ModelRegistryEntry(BaseModel):
    version: str
    path: str
    macro_f1: float
    promoted_at: float
    is_active: bool
