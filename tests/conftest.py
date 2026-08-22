from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def settings():
    from app.config import Settings

    # Force the fully-offline configuration: no Postgres, no LLM key. The tests assert on
    # the degraded paths, which is exactly what a fresh clone hits.
    return Settings(database_url="", llm_provider="template", checkpoint_path="does/not/exist.pt")
