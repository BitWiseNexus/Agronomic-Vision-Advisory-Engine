"""End-to-end tests in the fully degraded configuration — no checkpoint, no Postgres,
no LLM key. That is what a fresh clone hits, and it must still produce a complete
advisory rather than an exception.
"""

from __future__ import annotations

import pytest

from app.schemas.diagnosis import ActionTier, AdvisoryResponse
from app.schemas.environment import EnvironmentContext, IrrigationType, SoilType
from app.services.pipeline import AgronomicPipeline
from app.vision.preprocess import encode_png_b64
from app.vision.synthetic import make_leaf


@pytest.fixture(scope="module")
def pipeline(settings) -> AgronomicPipeline:
    # Skip the classifier: loading ResNet-50 ImageNet weights costs a download and tens of
    # seconds, and the classifier itself is covered separately. The point of these tests is
    # the orchestration and the degradation path.
    return AgronomicPipeline(settings, load_classifier=False, load_knowledge=True)


@pytest.fixture(scope="module")
def leaf_bytes() -> bytes:
    leaf = make_leaf(damaged_fraction=0.3, lesion_count=18, seed=21)
    import base64

    return base64.b64decode(encode_png_b64(leaf.image))


def test_run_returns_a_complete_advisory(pipeline, leaf_bytes) -> None:
    response = pipeline.run(leaf_bytes)

    assert isinstance(response, AdvisoryResponse)
    assert response.severity.severity_pct > 0
    assert response.plan.summary
    assert response.plan.immediate_actions
    assert response.plan.urgency is response.severity.action_tier
    assert response.disclaimer


def test_readiness_reports_every_degradation_honestly(pipeline) -> None:
    state = pipeline.readiness()

    assert state.checkpoint_loaded is False
    assert state.llm_provider == "template"
    # This fixture skips the classifier entirely, so the note names the classifier rather
    # than a missing checkpoint; either wording is an honest report of the same gap.
    assert any(("classifier" in note or "checkpoint" in note) for note in state.degraded)
    assert any(("LLM" in note or "template" in note) for note in state.degraded)


def test_local_context_reaches_the_plan(pipeline, leaf_bytes) -> None:
    environment = EnvironmentContext(
        region="Nashik",
        soil_type=SoilType.CLAY,
        irrigation=IrrigationType.SPRINKLER,
        rainfall_last_7d_mm=120,
    )
    response = pipeline.run(leaf_bytes, environment)
    practices = " ".join(response.plan.cultural_practices).lower()

    assert response.environment.region == "Nashik"
    # A sprinkler-irrigated, recently-drenched field must draw the wetness advice.
    assert "sprinkler" in practices
    assert "wash off" in practices or "fungal" in practices


def test_organic_only_never_yields_a_chemical(pipeline, leaf_bytes) -> None:
    response = pipeline.run(leaf_bytes, EnvironmentContext(organic_only=True))
    assert response.plan.chemical_options == []


def test_healthy_leaf_produces_no_intervention(pipeline) -> None:
    import base64

    healthy = make_leaf(damaged_fraction=0.0, lesion_count=0, seed=8)
    response = pipeline.run(base64.b64decode(encode_png_b64(healthy.image)))

    assert response.severity.action_tier is ActionTier.NEGLIGIBLE
    assert response.plan.urgency is ActionTier.NEGLIGIBLE
    assert response.plan.chemical_options == []


def test_evidence_is_retrieved_and_numbered(pipeline, leaf_bytes) -> None:
    response = pipeline.run(leaf_bytes)

    if pipeline.knowledge is None or pipeline.knowledge.is_empty:
        pytest.skip("knowledge index not built — run scripts/ingest_knowledge.py")

    assert response.evidence
    assert [item.id for item in response.evidence] == [
        f"S{index}" for index in range(1, len(response.evidence) + 1)
    ]
    assert all(item.text for item in response.evidence)


def test_overlay_is_returned_only_when_asked(pipeline, leaf_bytes) -> None:
    assert pipeline.run(leaf_bytes).debug_overlay_png_b64 is None
    assert pipeline.run(leaf_bytes, include_overlay=True).debug_overlay_png_b64


def test_timings_are_recorded(pipeline, leaf_bytes) -> None:
    response = pipeline.run(leaf_bytes)
    assert {"decode", "severity", "retrieve", "synthesize"} <= set(response.elapsed_ms)


def test_undecodable_upload_raises_value_error(pipeline) -> None:
    with pytest.raises(ValueError):
        pipeline.run(b"this is not an image")
