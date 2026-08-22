"""Query construction and reranking.

The query text is assembled from all three signals — disease, severity tier and local
context — because a document about *managing established late blight in wet conditions*
should outrank one about *identifying late blight* when severity is already 40 %.
"""

from __future__ import annotations

import re

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.rag.index import KnowledgeIndex
from app.schemas.diagnosis import Diagnosis, Evidence, SeverityReport
from app.schemas.environment import EnvironmentContext

log = get_logger(__name__)

_WORD_RE = re.compile(r"[a-z]{3,}")
_STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "are", "was", "were", "has",
    "have", "can", "will", "any", "all", "not", "use", "used", "using", "when", "how",
}


def build_query(
    diagnosis: Diagnosis, severity: SeverityReport, environment: EnvironmentContext
) -> str:
    parts = [
        f"{diagnosis.top.crop} {diagnosis.top.disease}",
        "management and treatment",
        f"{severity.action_tier.value} stage",
        f"{severity.severity_pct:.0f} percent leaf area affected",
    ]
    if environment.region:
        parts.append(environment.region)
    if environment.soil_type.value != "unknown":
        parts.append(f"{environment.soil_type.value} soil")
    if environment.season.value != "unknown":
        parts.append(f"{environment.season.value} season")
    if environment.rainfall_last_7d_mm is not None:
        wet = "high rainfall wet conditions" if environment.rainfall_last_7d_mm >= 30 else "dry conditions"
        parts.append(wet)
    if environment.organic_only:
        parts.append("organic biological non-chemical control")
    return " ".join(parts)


def _keywords(text: str) -> set[str]:
    return {word for word in _WORD_RE.findall(text.lower()) if word not in _STOPWORDS}


def _rerank(nodes: list, diagnosis: Diagnosis, keep: int) -> list:
    """Lexical rerank on top of vector similarity.

    Small corpora make pure cosine similarity noisy — a hard overlap bonus for the
    actual crop and disease name reliably pulls the right bulletin to the front.
    """
    disease_terms = _keywords(f"{diagnosis.top.crop} {diagnosis.top.disease}")
    if not disease_terms:
        return nodes[:keep]

    def score(item) -> float:
        text_terms = _keywords(item.node.get_content())
        meta = item.node.metadata or {}
        overlap = len(disease_terms & text_terms) / len(disease_terms)
        meta_blob = f"{meta.get('crop', '')} {meta.get('disease', '')}".lower()
        meta_bonus = 0.25 * len(disease_terms & _keywords(meta_blob)) / len(disease_terms)
        return float(item.score or 0.0) + 0.5 * overlap + meta_bonus

    return sorted(nodes, key=score, reverse=True)[:keep]


def retrieve_evidence(
    knowledge: KnowledgeIndex,
    diagnosis: Diagnosis,
    severity: SeverityReport,
    environment: EnvironmentContext,
    settings: Settings | None = None,
) -> tuple[list[Evidence], str]:
    """Returns (evidence, query_used). Empty evidence is a valid outcome, not an error —
    the advisor then falls back to generic tier-appropriate guidance."""
    settings = settings or get_settings()
    query = build_query(diagnosis, severity, environment)

    if knowledge.is_empty:
        log.warning("Knowledge index is empty — advisory will be ungrounded.")
        return [], query

    try:
        retriever = knowledge.index.as_retriever(similarity_top_k=settings.retrieval_top_k)
        nodes = retriever.retrieve(query)
    except Exception as exc:
        log.error("Retrieval failed (%s) — continuing without evidence", exc)
        return [], query

    evidence: list[Evidence] = []
    for position, item in enumerate(_rerank(nodes, diagnosis, settings.retrieval_keep), start=1):
        meta = item.node.metadata or {}
        evidence.append(
            Evidence(
                id=f"S{position}",
                text=item.node.get_content().strip(),
                score=round(float(item.score or 0.0), 4),
                source=str(meta.get("source", meta.get("file_name", "unknown"))),
                doc_type=str(meta.get("doc_type", "")),
                crop=str(meta.get("crop", "")),
                disease=str(meta.get("disease", "")),
            )
        )
    return evidence, query
