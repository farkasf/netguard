"""Scheduled retraining: the APScheduler lifecycle inside the app lifespan."""

from __future__ import annotations

from fastapi.testclient import TestClient

from netguard.api.app import create_app
from netguard.config import get_settings


def test_scheduler_runs_when_enabled(tmp_repo, monkeypatch):
    monkeypatch.setenv("NETGUARD_RETRAIN_ENABLED", "true")
    monkeypatch.setenv("NETGUARD_RETRAIN_CRON", "*/5 * * * *")
    get_settings.cache_clear()
    try:
        app = create_app(repo=tmp_repo, attach_scorer=False)
        with TestClient(app):
            scheduler = app.state.retrain_scheduler
            assert scheduler is not None and scheduler.running
            job = scheduler.get_job("retrain")
            assert job is not None
            assert job.next_run_time is not None
        # Shut down with the app.
        assert not scheduler.running
    finally:
        get_settings.cache_clear()


def test_scheduler_absent_when_disabled(tmp_repo):
    app = create_app(repo=tmp_repo, attach_scorer=False)
    with TestClient(app):
        assert app.state.retrain_scheduler is None
