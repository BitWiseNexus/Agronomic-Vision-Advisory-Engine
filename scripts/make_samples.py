r"""Write synthetic sample leaves to data/samples/ so the pipeline is demonstrable
before any real dataset is downloaded.

    .\.venv\Scripts\python.exe scripts\make_samples.py
"""

from __future__ import annotations

import cv2
from _path import bootstrap

ROOT = bootstrap()

from app.vision.synthetic import make_leaf  # noqa: E402

# One sample per action tier, so the demo shows the full escalation ladder.
# Lesion counts rise with the target fraction on purpose: a few huge lesions cannot be
# packed inside the blade without spilling onto the background, so the generator would
# silently fall short of the target.
SPECS = [
    ("leaf_healthy", 0.00, 0, 0.0),
    ("leaf_early_stage", 0.05, 12, 0.4),
    ("leaf_moderate", 0.18, 16, 0.35),
    ("leaf_severe", 0.36, 26, 0.3),
    ("leaf_critical", 0.75, 45, 0.25),
]


def main() -> None:
    out_dir = ROOT / "data" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    for index, (name, fraction, lesions, chlorosis) in enumerate(SPECS):
        leaf = make_leaf(
            damaged_fraction=fraction,
            lesion_count=lesions,
            chlorosis_share=chlorosis,
            seed=17 + index,
        )
        path = out_dir / f"{name}.png"
        cv2.imwrite(str(path), leaf.image)
        print(f"{path.name:24s} target {fraction:5.1%}  realised {leaf.true_damaged_fraction:5.1%}")

    print(f"\n{len(SPECS)} samples written to {out_dir}")


if __name__ == "__main__":
    main()
