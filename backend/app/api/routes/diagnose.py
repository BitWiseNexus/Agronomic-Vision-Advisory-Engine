"""Diagnosis endpoints.

The PWA posts multipart: the photo plus an `environment` JSON string. Sending the context
as one JSON blob rather than 12 form fields keeps the pydantic model as the single source
of truth for the contract — add a field to `EnvironmentContext` and the API accepts it.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import ValidationError

from app.api.deps import get_pipeline
from app.core.logging import get_logger
from app.schemas.diagnosis import AdvisoryResponse, SeverityReport
from app.schemas.environment import EnvironmentContext
from app.services.pipeline import AgronomicPipeline
from app.vision.preprocess import decode_image, encode_png_b64
from app.vision.severity import render_overlay

log = get_logger(__name__)
router = APIRouter(tags=["diagnosis"])

MAX_UPLOAD_BYTES = 12 * 1024 * 1024


async def _read_upload(image: UploadFile) -> bytes:
    payload = await image.read()
    if not payload:
        raise HTTPException(400, "Empty upload.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    return payload


def _parse_environment(raw: str | None) -> EnvironmentContext:
    if not raw or not raw.strip():
        return EnvironmentContext()
    try:
        return EnvironmentContext.model_validate(json.loads(raw))
    except json.JSONDecodeError as exc:
        raise HTTPException(422, f"`environment` is not valid JSON: {exc}") from exc
    except ValidationError as exc:
        raise HTTPException(422, f"`environment` failed validation: {exc.errors()}") from exc


@router.post("/diagnose", response_model=AdvisoryResponse)
async def diagnose(
    image: UploadFile = File(..., description="Leaf photograph (JPEG/PNG/WebP)"),
    environment: str | None = Form(None, description="EnvironmentContext as a JSON string"),
    debug: bool = Query(False, description="Include the base64 severity mask overlay"),
    pipeline: AgronomicPipeline = Depends(get_pipeline),
) -> AdvisoryResponse:
    payload = await _read_upload(image)
    context = _parse_environment(environment)
    try:
        return pipeline.run(payload, context, include_overlay=debug)
    except ValueError as exc:  # undecodable image
        raise HTTPException(400, str(exc)) from exc


@router.post("/severity", response_model=SeverityReport)
async def severity(
    image: UploadFile = File(...),
    pipeline: AgronomicPipeline = Depends(get_pipeline),
) -> SeverityReport:
    """CV only — no classifier, no retrieval, no LLM. Fast enough for a live camera
    preview in the PWA, which is why it is a separate route."""
    payload = await _read_upload(image)
    try:
        report, _ = pipeline.severity_only(decode_image(payload))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return report


@router.post("/severity/overlay")
async def severity_overlay(
    image: UploadFile = File(...),
    pipeline: AgronomicPipeline = Depends(get_pipeline),
) -> dict:
    """The annotated mask image, for the 'show me why' view in the app."""
    payload = await _read_upload(image)
    try:
        report, masks = pipeline.severity_only(decode_image(payload))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "severity": report.model_dump(),
        "overlay_png_b64": encode_png_b64(render_overlay(masks)),
        "legend": {"green": "healthy", "yellow": "chlorotic", "red": "necrotic"},
    }
