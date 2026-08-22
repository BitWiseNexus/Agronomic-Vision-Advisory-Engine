"""Retrieval-layer tests. These run against the on-disk index, so they skip cleanly if
the corpus has not been ingested yet."""

from __future__ import annotations

import pytest

from app.rag.embeddings import _hash_embed, get_embedding_model
from app.rag.index import load_index
from app.rag.ingest import _parse_front_matter, load_documents
from app.rag.retriever import build_query, retrieve_evidence
from app.schemas.diagnosis import ClassPrediction, Diagnosis, SeverityReport
from app.schemas.environment import EnvironmentContext, Season, SoilType
from app.vision.severity import classify_tier


@pytest.fixture(scope="module")
def knowledge(settings):
    index = load_index(settings, prefer_pgvector=False)
    if index.is_empty:
        pytest.skip("knowledge index not built — run scripts/ingest_knowledge.py --local")
    return index


def make_diagnosis(crop: str = "Tomato", disease: str = "Late blight") -> Diagnosis:
    return Diagnosis(
        top=ClassPrediction(
            label=f"{crop}___{disease.replace(' ', '_')}",
            crop=crop,
            disease=disease,
            confidence=0.9,
        ),
        checkpoint_loaded=True,
        confidence_reliable=True,
    )


def make_severity(pct: float) -> SeverityReport:
    return SeverityReport(
        severity_pct=pct,
        necrotic_pct=pct * 0.6,
        chlorotic_pct=pct * 0.4,
        healthy_pct=100 - pct,
        lesion_count=5,
        largest_lesion_pct=pct / 4,
        leaf_coverage_pct=40.0,
        action_tier=classify_tier(pct),
    )


# --- front matter -----------------------------------------------------------


def test_front_matter_is_parsed_and_stripped() -> None:
    raw = "---\ncrop: Tomato\ndisease: Late blight\ntags: [a, b]\n---\n\n# Body\ntext here\n"
    metadata, body = _parse_front_matter(raw)

    assert metadata["crop"] == "Tomato"
    assert metadata["disease"] == "Late blight"
    assert metadata["tags"] == "a, b"
    assert body.lstrip().startswith("# Body")


def test_document_without_front_matter_still_loads() -> None:
    metadata, body = _parse_front_matter("no front matter here")
    assert metadata == {}
    assert body == "no front matter here"


def test_corpus_loads_with_metadata(settings) -> None:
    documents, _ = load_documents(settings.knowledge_dir)
    assert documents
    crops = {document.metadata["crop"] for document in documents}
    assert "Tomato" in crops
    assert all(document.metadata["source"] for document in documents)


# --- embeddings -------------------------------------------------------------


def test_hash_embedding_is_deterministic_and_normalised() -> None:
    first = _hash_embed("late blight on tomato", 384)
    second = _hash_embed("late blight on tomato", 384)

    assert first == second
    assert len(first) == 384
    assert sum(value * value for value in first) == pytest.approx(1.0, abs=1e-5)
    assert first != _hash_embed("powdery mildew on grape", 384)


def test_embedding_model_reports_its_dimension(settings) -> None:
    _, info = get_embedding_model(settings)
    assert info.dim > 0


# --- query construction -----------------------------------------------------


def test_query_carries_all_three_signals() -> None:
    query = build_query(
        make_diagnosis(),
        make_severity(35.0),
        EnvironmentContext(region="Nashik", soil_type=SoilType.BLACK, season=Season.KHARIF,
                           rainfall_last_7d_mm=90),
    )

    assert "Tomato" in query
    assert "Late blight" in query
    assert "severe" in query
    assert "Nashik" in query
    assert "black soil" in query
    assert "wet" in query


def test_query_reflects_the_organic_constraint() -> None:
    query = build_query(
        make_diagnosis(), make_severity(20.0), EnvironmentContext(organic_only=True)
    )
    assert "organic" in query


# --- retrieval --------------------------------------------------------------


def test_retrieval_returns_numbered_evidence(knowledge, settings) -> None:
    evidence, query = retrieve_evidence(
        knowledge, make_diagnosis(), make_severity(30.0), EnvironmentContext(), settings
    )

    assert query
    assert evidence
    assert len(evidence) <= settings.retrieval_keep
    assert [item.id for item in evidence] == [f"S{i}" for i in range(1, len(evidence) + 1)]


def test_reranking_surfaces_the_matching_crop(knowledge, settings) -> None:
    evidence, _ = retrieve_evidence(
        knowledge, make_diagnosis("Tomato", "Late blight"), make_severity(30.0),
        EnvironmentContext(), settings,
    )
    joined = " ".join(item.text.lower() for item in evidence)
    assert "late blight" in joined


def test_retrieval_on_an_empty_index_is_not_an_error(settings) -> None:
    from app.rag.index import KnowledgeIndex

    empty = KnowledgeIndex(index=None, backend="empty", embedding=get_embedding_model(settings)[1],
                           document_count=0)
    evidence, query = retrieve_evidence(
        empty, make_diagnosis(), make_severity(10.0), EnvironmentContext(), settings
    )
    assert evidence == []
    assert query
