"""API smoke tests through FastAPI TestClient.

The pipeline dependency is overridden with a classifier-free instance so the suite does
not download ResNet-50 weights just to check routing and validation.
"""

from __future__ import annotations

import base64
import io
import json

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_pipeline
from app.main import create_app
from app.services.pipeline import AgronomicPipeline
from app.vision.preprocess import encode_png_b64
from app.vision.synthetic import make_leaf

PREFIX = "/api/v1"


@pytest.fixture(scope="module")
def client(settings):
    pipeline = AgronomicPipeline(settings, load_classifier=False, load_knowledge=True)
    app = create_app()
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    # `with` runs lifespan; the override is already in place so no weights are fetched.
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture(scope="module")
def leaf_file() -> tuple[str, io.BytesIO, str]:
    leaf = make_leaf(damaged_fraction=0.28, lesion_count=16, seed=33)
    payload = base64.b64decode(encode_png_b64(leaf.image))
    return ("leaf.png", io.BytesIO(payload), "image/png")


def _fresh(leaf_file):
    name, buffer, mime = leaf_file
    return (name, io.BytesIO(buffer.getvalue()), mime)


def test_health(client) -> None:
    response = client.get(f"{PREFIX}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_lists_components(client) -> None:
    body = client.get(f"{PREFIX}/readiness").json()
    assert body["status"] in {"ok", "degraded"}
    assert "components" in body
    assert body["components"]["llm_provider"] == "template"


def test_diagnose_returns_the_full_contract(client, leaf_file) -> None:
    environment = {
        "region": "Pune",
        "soil_type": "black",
        "season": "kharif",
        "rainfall_last_7d_mm": 85,
        "temperature_c": 26,
    }
    response = client.post(
        f"{PREFIX}/diagnose",
        files={"image": _fresh(leaf_file)},
        data={"environment": json.dumps(environment)},
    )
    assert response.status_code == 200

    body = response.json()
    for key in ("diagnosis", "severity", "environment", "plan", "readiness", "disclaimer"):
        assert key in body
    assert body["severity"]["severity_pct"] > 0
    assert body["environment"]["region"] == "Pune"
    assert body["plan"]["urgency"] == body["severity"]["action_tier"]


def test_diagnose_without_environment_is_allowed(client, leaf_file) -> None:
    response = client.post(f"{PREFIX}/diagnose", files={"image": _fresh(leaf_file)})
    assert response.status_code == 200
    assert response.json()["environment"]["soil_type"] == "unknown"


def test_diagnose_rejects_malformed_environment_json(client, leaf_file) -> None:
    response = client.post(
        f"{PREFIX}/diagnose",
        files={"image": _fresh(leaf_file)},
        data={"environment": "{not json"},
    )
    assert response.status_code == 422


def test_diagnose_rejects_out_of_range_values(client, leaf_file) -> None:
    response = client.post(
        f"{PREFIX}/diagnose",
        files={"image": _fresh(leaf_file)},
        data={"environment": json.dumps({"soil_ph": 42})},
    )
    assert response.status_code == 422


def test_diagnose_rejects_a_non_image(client) -> None:
    response = client.post(
        f"{PREFIX}/diagnose",
        files={"image": ("notes.txt", io.BytesIO(b"just text"), "text/plain")},
    )
    assert response.status_code == 400


def test_diagnose_rejects_an_empty_upload(client) -> None:
    response = client.post(
        f"{PREFIX}/diagnose", files={"image": ("empty.png", io.BytesIO(b""), "image/png")}
    )
    assert response.status_code == 400


def test_severity_route_is_cv_only(client, leaf_file) -> None:
    response = client.post(f"{PREFIX}/severity", files={"image": _fresh(leaf_file)})
    assert response.status_code == 200

    body = response.json()
    assert 0 < body["severity_pct"] <= 100
    assert body["action_tier"] in {"negligible", "early", "moderate", "severe", "critical"}
    # No plan, no evidence — this route must not invoke the advisory stack.
    assert "plan" not in body


def test_severity_overlay_returns_an_image(client, leaf_file) -> None:
    response = client.post(f"{PREFIX}/severity/overlay", files={"image": _fresh(leaf_file)})
    assert response.status_code == 200

    body = response.json()
    assert body["overlay_png_b64"]
    assert base64.b64decode(body["overlay_png_b64"])[:8] == b"\x89PNG\r\n\x1a\n"
    assert body["legend"]["red"] == "necrotic"


def test_knowledge_search(client) -> None:
    response = client.get(f"{PREFIX}/knowledge/search", params={"q": "late blight tomato", "k": 3})
    if response.status_code == 503:
        pytest.skip("knowledge index not built — run scripts/ingest_knowledge.py")

    assert response.status_code == 200
    body = response.json()
    assert body["results"]
    assert len(body["results"]) <= 3


def test_knowledge_search_rejects_a_too_short_query(client) -> None:
    assert client.get(f"{PREFIX}/knowledge/search", params={"q": "ab"}).status_code == 422


def test_openapi_schema_is_served(client) -> None:
    schema = client.get("/openapi.json").json()
    assert f"{PREFIX}/diagnose" in schema["paths"]
    assert f"{PREFIX}/severity" in schema["paths"]
