r"""FastAPI entry point.

    .\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --app-dir backend
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import diagnose, health, knowledge
from app.config import get_settings
from app.core.logging import configure_logging, get_logger

API_PREFIX = "/api/v1"
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    log.info("Starting Agronomic Vision & Advisory Engine v%s", __version__)
    # Warm the models at boot rather than on the first unlucky request.
    from app.api.deps import get_pipeline

    state = get_pipeline().readiness()
    for note in state.degraded:
        log.warning("DEGRADED: %s", note)
    yield
    log.info("Shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Agronomic Vision & Advisory Engine",
        version=__version__,
        description=(
            "Leaf photograph + local field conditions -> ResNet-50 disease classification, "
            "an OpenCV severity index, pgvector retrieval over agricultural extension "
            "material, and a severity-grounded treatment plan."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        # The PWA is served from a different origin in dev; tighten this before deploying.
        allow_origins=[
            "http://localhost:5173",
            "http://localhost:5174",
            "http://localhost:4173",
            "http://localhost:3000",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(diagnose.router, prefix=API_PREFIX)
    app.include_router(knowledge.router, prefix=API_PREFIX)
    return app


app = create_app()
