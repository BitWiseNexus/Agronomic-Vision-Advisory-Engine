r"""Benchmark the severity index against synthetic leaves whose damage is known exactly.

    .\.venv\Scripts\python.exe ml\benchmark_severity.py
    .\.venv\Scripts\python.exe ml\benchmark_severity.py --json outputs\severity_benchmark.json

Every leaf is rendered at a known damaged fraction, then photographed "badly" — colour
casts, exposure errors, shadows, compression, cluttered backgrounds, natural colour
variation in the tissue itself. For each condition the script reports:

  * MAE   — mean absolute error of the severity %, in percentage points
  * bias  — mean signed error (+ over-reports damage, - under-reports)
  * tier  — share of leaves landing in the correct action tier

Tier accuracy is the number that matters: the tier is what constrains the treatment plan,
so a 3 pp error that crosses a boundary is worse than a 6 pp error that does not.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.vision.severity import analyse, classify_tier  # noqa: E402
from app.vision.synthetic import SyntheticLeaf, make_leaf  # noqa: E402

FRACTIONS = (0.0, 0.03, 0.07, 0.15, 0.30, 0.45, 0.65)
SEEDS = (11, 23, 37)

Perturb = Callable[[np.ndarray, SyntheticLeaf], tuple[np.ndarray, np.ndarray, np.ndarray]]


def _scale(img: np.ndarray, bgr: tuple[float, float, float]) -> np.ndarray:
    return np.clip(img.astype(np.float32) * np.array(bgr, np.float32), 0, 255).astype(np.uint8)


def _same(img: np.ndarray, leaf: SyntheticLeaf):
    return img, leaf.leaf_mask, leaf.lesion_mask


def _recolour(leaf: SyntheticLeaf, region: np.ndarray, bgr: tuple[int, int, int], noise: float = 7.0) -> np.ndarray:
    """Repaint one tissue region with a different (still realistic) colour."""
    out = leaf.image.copy()
    rng = np.random.default_rng(1)
    paint = np.clip(np.array(bgr, np.float32) + rng.normal(0, noise, out.shape), 0, 255).astype(np.uint8)
    out[region] = paint[region]
    return cv2.GaussianBlur(out, (3, 3), 0)


def _healthy_region(leaf: SyntheticLeaf) -> np.ndarray:
    return (leaf.leaf_mask > 0) & (leaf.lesion_mask == 0)


def _necrotic_core(leaf: SyntheticLeaf) -> np.ndarray:
    # The generator paints necrosis dark brown (R ~96); find it by colour inside the lesions.
    lesion = leaf.lesion_mask > 0
    red = leaf.image[:, :, 2].astype(np.int16)
    return lesion & (red < 150)


def _with_background(img: np.ndarray, leaf: SyntheticLeaf, bg: np.ndarray):
    out = img.copy()
    outside = leaf.leaf_mask == 0
    out[outside] = bg[outside]
    return out, leaf.leaf_mask, leaf.lesion_mask


def soil_bg(img, leaf):
    rng = np.random.default_rng(5)
    soil = rng.integers(40, 110, img.shape, dtype=np.int16)
    soil[:, :, 2] = np.clip(soil[:, :, 2] + 55, 0, 255)
    return _with_background(img, leaf, soil.astype(np.uint8))


def white_paper_bg(img, leaf):
    rng = np.random.default_rng(6)
    paper = np.clip(236 + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    return _with_background(img, leaf, paper)


def dark_bg(img, leaf):
    rng = np.random.default_rng(7)
    dark = np.clip(28 + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    return _with_background(img, leaf, dark)


def shadow_on_leaf(img, leaf):
    out = img.copy()
    h, w = out.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (int(w * 0.62), int(h * 0.55)), (int(w * 0.16), int(h * 0.30)), 20, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (41, 41), 0).astype(np.float32) / 255.0
    out = (out.astype(np.float32) * (1 - 0.5 * mask[..., None])).astype(np.uint8)
    return out, leaf.leaf_mask, leaf.lesion_mask


def macro_crop(img, leaf):
    ys, xs = np.nonzero(leaf.leaf_mask)
    sl = (slice(ys.min() + 40, ys.max() - 40), slice(xs.min() + 4, xs.max() - 4))
    return img[sl], leaf.leaf_mask[sl], leaf.lesion_mask[sl]


def jpeg(img, leaf):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), leaf.leaf_mask, leaf.lesion_mask


def blur(img, leaf):
    return cv2.GaussianBlur(img, (7, 7), 2.0), leaf.leaf_mask, leaf.lesion_mask


def yellow_green_leaf(img, leaf):
    # Lighter, yellower healthy tissue (e.g. young cucurbit or maize leaves).
    return _recolour(leaf, _healthy_region(leaf), (40, 165, 110)), leaf.leaf_mask, leaf.lesion_mask


def dark_green_leaf(img, leaf):
    # Deep, low-value green (e.g. mature tomato or potato in shade).
    return _recolour(leaf, _healthy_region(leaf), (30, 82, 28)), leaf.leaf_mask, leaf.lesion_mask


def tan_lesions(img, leaf):
    # Dry, pale-tan necrosis (scorch, late blight that has dried out).
    return _recolour(leaf, _necrotic_core(leaf), (110, 150, 175)), leaf.leaf_mask, leaf.lesion_mask


def grey_lesions(img, leaf):
    # Greyish-black dead tissue — low saturation, mid value.
    return _recolour(leaf, _necrotic_core(leaf), (70, 72, 78)), leaf.leaf_mask, leaf.lesion_mask


CONDITIONS: dict[str, Perturb] = {
    "clean": _same,
    "warm light": lambda i, l: (_scale(i, (0.78, 0.95, 1.22)), l.leaf_mask, l.lesion_mask),
    "cool shade": lambda i, l: (_scale(i, (1.25, 1.02, 0.80)), l.leaf_mask, l.lesion_mask),
    "underexposed": lambda i, l: (_scale(i, (0.45, 0.45, 0.45)), l.leaf_mask, l.lesion_mask),
    "overexposed": lambda i, l: (_scale(i, (1.5, 1.5, 1.5)), l.leaf_mask, l.lesion_mask),
    "shadow on leaf": shadow_on_leaf,
    "jpeg q30": jpeg,
    "motion blur": blur,
    "soil background": soil_bg,
    "white paper": white_paper_bg,
    "dark background": dark_bg,
    "macro (all leaf)": macro_crop,
    "yellow-green leaf": yellow_green_leaf,
    "dark-green leaf": dark_green_leaf,
    "tan lesions": tan_lesions,
    "grey lesions": grey_lesions,
}


def run(conditions: dict[str, Perturb] = CONDITIONS) -> dict[str, dict[str, float]]:
    leaves = [
        make_leaf(damaged_fraction=f, lesion_count=max(int(f * 60), 1) if f else 0, seed=s)
        for f in FRACTIONS
        for s in SEEDS
    ]
    results: dict[str, dict[str, float]] = {}
    for name, perturb in conditions.items():
        errors, hits = [], 0
        for leaf in leaves:
            image, leaf_mask, lesion_mask = perturb(leaf.image, leaf)
            leaf_px = int((leaf_mask > 0).sum())
            truth = 100.0 * int(((lesion_mask > 0) & (leaf_mask > 0)).sum()) / max(leaf_px, 1)
            report, _ = analyse(image)
            errors.append(report.severity_pct - truth)
            hits += report.action_tier is classify_tier(truth)
        errs = np.array(errors)
        results[name] = {
            "mae": float(np.abs(errs).mean()),
            "bias": float(errs.mean()),
            "max_err": float(np.abs(errs).max()),
            "tier_acc": hits / len(leaves),
            "n": len(leaves),
        }
    return results


def summarise(results: dict[str, dict[str, float]]) -> dict[str, float]:
    rows = list(results.values())
    return {
        "mae": float(np.mean([r["mae"] for r in rows])),
        "tier_acc": float(np.mean([r["tier_acc"] for r in rows])),
        "worst_mae": float(max(r["mae"] for r in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the severity index on synthetic leaves")
    parser.add_argument("--json", type=Path, default=None, help="Write results to this file")
    args = parser.parse_args()

    results = run()
    print(f"{'condition':<20}{'MAE pp':>9}{'bias':>8}{'max':>8}{'tier acc':>10}")
    print("-" * 55)
    for name, r in results.items():
        flag = "  <--" if r["tier_acc"] < 0.8 or r["mae"] > 5 else ""
        print(f"{name:<20}{r['mae']:9.2f}{r['bias']:+8.2f}{r['max_err']:8.1f}{r['tier_acc']:9.0%}{flag}")
    s = summarise(results)
    print("-" * 55)
    print(f"{'MEAN':<20}{s['mae']:9.2f}{'':>8}{s['worst_mae']:8.1f}{s['tier_acc']:9.0%}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"summary": s, "conditions": results}, indent=2), encoding="utf-8")
        print(f"\nResults -> {args.json}")


if __name__ == "__main__":
    main()
