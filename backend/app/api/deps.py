"""Shared singletons. The pipeline holds a ResNet-50 and a sentence-transformer, so it is
built exactly once per process."""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.services.pipeline import AgronomicPipeline


@lru_cache(maxsize=1)
def get_pipeline() -> AgronomicPipeline:
    return AgronomicPipeline(get_settings())


def reset_pipeline() -> None:
    """Called after a re-ingest so the next request sees the new index."""
    get_pipeline.cache_clear()
