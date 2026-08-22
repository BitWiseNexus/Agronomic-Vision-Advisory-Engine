"""Knowledge-base routes — mostly for inspecting retrieval while tuning the corpus."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_pipeline, reset_pipeline
from app.config import get_settings
from app.core.logging import get_logger
from app.rag.ingest import ingest_knowledge
from app.services.pipeline import AgronomicPipeline

log = get_logger(__name__)
router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.get("/search")
def search(
    q: str = Query(..., min_length=3, description="Free-text query"),
    k: int = Query(4, ge=1, le=20),
    pipeline: AgronomicPipeline = Depends(get_pipeline),
) -> dict:
    if pipeline.knowledge is None or pipeline.knowledge.is_empty:
        raise HTTPException(503, "Knowledge base is empty. POST /knowledge/ingest first.")
    nodes = pipeline.knowledge.index.as_retriever(similarity_top_k=k).retrieve(q)
    return {
        "query": q,
        "backend": pipeline.knowledge.backend,
        "results": [
            {
                "score": round(float(node.score or 0.0), 4),
                "source": (node.node.metadata or {}).get("source", "unknown"),
                "crop": (node.node.metadata or {}).get("crop", ""),
                "disease": (node.node.metadata or {}).get("disease", ""),
                "text": node.node.get_content()[:900],
            }
            for node in nodes
        ],
    }


@router.post("/ingest")
def ingest(reset: bool = Query(False, description="Truncate the table before ingesting")) -> dict:
    """Re-index data/knowledge/. Synchronous on purpose — the corpus is small and a
    background task would only hide failures from whoever triggered the ingest."""
    try:
        _, report = ingest_knowledge(get_settings(), reset=reset)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        log.exception("Ingest failed")
        raise HTTPException(500, f"Ingest failed: {exc}") from exc
    reset_pipeline()
    return {
        "documents": report.documents,
        "nodes": report.nodes,
        "backend": report.backend,
        "embedding_model": report.embedding_model,
        "skipped": report.skipped,
    }
