"""Notebook bootstrap: put `backend/` on sys.path and hand back useful paths.

Usage in the first cell of any notebook:

    from _bootstrap import setup
    ROOT = setup()
"""

from __future__ import annotations

import sys
from pathlib import Path


def setup(quiet: bool = False) -> Path:
    root = Path.cwd()
    # Tolerate being run from the repo root or from notebooks/.
    if not (root / "backend").exists() and (root.parent / "backend").exists():
        root = root.parent

    backend = root / "backend"
    if not backend.exists():
        raise RuntimeError(f"Could not locate backend/ from {Path.cwd()}")
    for path in (str(backend), str(root / "ml")):
        if path not in sys.path:
            sys.path.insert(0, path)

    if not quiet:
        print(f"project root : {root}")
        print(f"python       : {sys.version.split()[0]}")
    return root


def show_image(bgr, title: str = "", ax=None, size=(6, 5)):
    """Display a BGR OpenCV array with matplotlib without the RGB-swap footgun."""
    import cv2
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=size)
    grey = bgr.ndim == 2
    ax.imshow(bgr if grey else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cmap="gray" if grey else None)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    return ax
