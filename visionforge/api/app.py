"""FastAPI application factory."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from visionforge.api.config import ApiSettings
from visionforge.api.errors import ApiError, api_error_handler
from visionforge.api.jobs.queue import JobQueue, recover_queue
from visionforge.api.jobs.store import JobStore
from visionforge.api.routes import capabilities, health, jobs, media


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    settings = settings or ApiSettings.from_env()
    store = JobStore(settings.jobs_root)
    queue = JobQueue(store, settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.jobs_root.mkdir(parents=True, exist_ok=True)
        queue.start()
        recover_queue(queue, store)
        app.state.settings = settings
        app.state.store = store
        app.state.queue = queue
        yield
        queue.stop()

    app = FastAPI(
        title="VisionForge Job API",
        version="0.5.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = settings
    app.state.store = store
    app.state.queue = queue

    app.add_exception_handler(ApiError, api_error_handler)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "details": exc.errors()[:20],
                }
            },
        )

    app.include_router(health.router)
    app.include_router(capabilities.router)
    app.include_router(jobs.router)
    app.include_router(media.router)
    return app
