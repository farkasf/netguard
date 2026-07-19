"""API tests using FastAPI's TestClient against a seeded SQLite DB."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from netguard.api.app import create_app
from netguard.store.repository import Repository

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Keep the /retrain background task's model artifacts out of the repo.
    monkeypatch.setenv("NETGUARD_MODELS_DIR", str(tmp_path / "models"))
    from netguard.config import get_settings

    get_settings.cache_clear()
    repo = Repository(
        db_path=tmp_path / "api.db",
        schema_path=REPO_ROOT / "netguard" / "store" / "schema.sql",
    )
    # Seed fixture rows.
    now = time.time()
    repo.insert_flow(now, "10.0.0.1", "10.0.0.2", 1234, 80, "TCP", "BENIGN", 0.99,
                     1.5, 10, 1000)
    repo.insert_flow(now, "10.0.0.66", "10.0.0.2", 55000, 22, "TCP", "PORTSCAN", 0.95,
                     0.01, 1, 54)
    repo.insert_anomaly(now, "10.0.0.66", "10.0.0.2", 55000, 22, "TCP", "PORTSCAN",
                        0.95, [1.0, 2.0, 3.0], "vfixture")
    repo.register_model("vfixture", "data/fixtures/fixture_model", 0.97,
                        metrics={
                            "classes": ["BENIGN", "PORTSCAN"],
                            "macro_f1": 0.97, "macro_precision": 0.96,
                            "macro_recall": 0.98, "accuracy": 0.97,
                            "per_class": {
                                "BENIGN": {"precision": 1.0, "recall": 0.95, "f1": 0.97, "support": 20},
                                "PORTSCAN": {"precision": 0.93, "recall": 1.0, "f1": 0.96, "support": 70},
                            },
                            "confusion_matrix": [[19, 1], [0, 70]],
                        },
                        activate=True)

    app = create_app(repo=repo, attach_scorer=True)
    with TestClient(app) as c:
        yield c
    repo.close()
    get_settings.cache_clear()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "uptime_s" in body
    assert "model_version" in body


def test_flows(client):
    r = client.get("/api/flows?limit=10")
    assert r.status_code == 200
    flows = r.json()
    assert len(flows) == 2
    classes = {f["predicted_class"] for f in flows}
    assert classes == {"BENIGN", "PORTSCAN"}
    assert all("confidence" in f for f in flows)


def test_anomalies(client):
    r = client.get("/api/anomalies?limit=10")
    assert r.status_code == 200
    anomalies = r.json()
    assert len(anomalies) == 1
    a = anomalies[0]
    assert a["predicted_class"] == "PORTSCAN"
    assert a["features"] == [1.0, 2.0, 3.0]
    assert a["model_version"] == "vfixture"


def test_anomalies_since_filter(client):
    future = time.time() + 10_000
    r = client.get(f"/api/anomalies?since={future}")
    assert r.status_code == 200
    assert r.json() == []


def test_metrics(client):
    r = client.get("/api/metrics")
    assert r.status_code == 200
    m = r.json()
    assert m["model_version"] == "vfixture"
    assert m["macro_f1"] == 0.97
    assert m["classes"] == ["BENIGN", "PORTSCAN"]
    assert "PORTSCAN" in m["per_class"]
    assert m["confusion_matrix"] == [[19, 1], [0, 70]]
    assert m["flow_count"] == 2
    assert m["anomaly_count"] == 1
    assert len(m["feature_names"]) > 0


def test_model_registry(client):
    r = client.get("/api/model/registry")
    assert r.status_code == 200
    models = r.json()
    assert len(models) == 1
    assert models[0]["version"] == "vfixture"
    assert models[0]["is_active"] is True


def test_retrain_accepted(client):
    r = client.post("/api/retrain")
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "accepted"
    assert "job_id" in body


def test_dashboard_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "NetGuard" in r.text
