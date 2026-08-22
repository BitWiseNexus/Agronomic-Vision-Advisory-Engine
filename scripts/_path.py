"""Put `backend/` on sys.path so `import app...` works from scripts and notebooks
without an editable install."""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap() -> Path:
    root = Path(__file__).resolve().parents[1]
    backend = root / "backend"
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    return root
