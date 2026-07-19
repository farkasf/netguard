"""FastAPI app factory.

Mounts the API router under ``/api`` and serves the vanilla web UI from
``netguard/web`` at ``/``, so the dashboard and API share an origin (no CORS
pain for the dashboard). CORS is still opened to the configured LAN origins for
external clients. Binds ``0.0.0.0:8000``.

When ``NETGUARD_RETRAIN_ENABLED=true``, a background APScheduler runs the
retrain job (with its F1 promotion gate) on the ``NETGUARD_RETRAIN_CRON``
schedule and hot-reloads the scorer on promotion.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from netguard.api.routes import router
from netguard.config import get_settings
from netguard.pipeline.scorer import Scorer
from netguard.store.repository import Repository
from netguard.training.retrain_job import retrain_from_synthetic

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def create_app(repo: Repository | None = None, attach_scorer: bool = True) -> FastAPI:
    settings = get_settings()
    the_repo = repo or Repository()
    # Scorer with no trained model yet is fine; reload_model returns False.
    scorer = Scorer(the_repo, model=None) if attach_scorer else None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        scheduler: BackgroundScheduler | None = None
        if settings.retrain_enabled:

            def _scheduled_retrain() -> None:
                def _on_promote(_path: str) -> None:
                    if scorer is not None:
                        scorer.reload_model()

                result = retrain_from_synthetic(
                    the_repo, on_promote=_on_promote if scorer else None
                )
                app.state.last_retrain = {
                    "job_id": "cron",
                    "status": result.status,
                    "candidate_version": result.candidate_version,
                    "candidate_f1": result.candidate_f1,
                    "incumbent_f1": result.incumbent_f1,
                    "reason": result.reason,
                }

            scheduler = BackgroundScheduler()
            scheduler.add_job(
                _scheduled_retrain,
                CronTrigger.from_crontab(settings.retrain_cron),
                id="retrain",
            )
            scheduler.start()
        app.state.retrain_scheduler = scheduler
        try:
            yield
        finally:
            if scheduler is not None:
                scheduler.shutdown(wait=False)

    app = FastAPI(title="NetGuard", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.started_at = time.time()
    app.state.repo = the_repo
    app.state.last_retrain = {}
    app.state.scorer = scorer
    app.state.retrain_scheduler = None

    app.include_router(router, prefix="/api")

    # Serve the SPA-less dashboard. html=True makes "/" return index.html.
    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

    return app


def main() -> int:  # pragma: no cover - entrypoint
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "netguard.api.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
