from __future__ import annotations

from fastapi import APIRouter, Depends

from app import __version__
from app.api.deps import get_pipeline
from app.services.pipeline import AgronomicPipeline

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict:
    """Liveness only — deliberately does not touch the models, so it stays fast."""
    return {"status": "ok", "version": __version__}


@router.get("/readiness")
def readiness(pipeline: AgronomicPipeline = Depends(get_pipeline)) -> dict:
    """Per-component truth: what is loaded, what is degraded, and why."""
    state = pipeline.readiness()
    return {
        "status": "degraded" if state.degraded else "ok",
        "version": __version__,
        "components": state.model_dump(),
        "device": pipeline.settings.resolve_device(),
        "embedding_model": pipeline.knowledge.embedding.model_name if pipeline.knowledge else None,
    }
