"""The severity index is the project's quantitative claim, so it is tested against
synthetic leaves whose damaged fraction is known by construction rather than annotated.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.schemas.diagnosis import ActionTier
from app.vision.severity import analyse, classify_tier, render_overlay, segment_leaf
from app.vision.synthetic import make_leaf


@pytest.mark.parametrize("fraction", [0.0, 0.05, 0.15, 0.30, 0.55])
def test_measured_severity_tracks_ground_truth(fraction: float) -> None:
    leaf = make_leaf(damaged_fraction=fraction, lesion_count=max(int(fraction * 60), 1), seed=3)
    report, _ = analyse(leaf.image)

    truth_pct = leaf.true_damaged_fraction * 100
    # 4 percentage points of slack: the bilateral filter softens lesion edges and the
    # tissue thresholds are hard cutoffs, so exact agreement is not the goal — the goal is
    # that the index is a faithful measure rather than a vibe.
    assert report.severity_pct == pytest.approx(truth_pct, abs=4.0)


def test_healthy_leaf_is_negligible() -> None:
    leaf = make_leaf(damaged_fraction=0.0, lesion_count=0, seed=11)
    report, _ = analyse(leaf.image)

    assert report.severity_pct < 2.0
    assert report.action_tier is ActionTier.NEGLIGIBLE
    assert report.lesion_count == 0
    assert report.healthy_pct > 90.0


def test_severity_components_sum_within_leaf() -> None:
    leaf = make_leaf(damaged_fraction=0.25, lesion_count=15, seed=5)
    report, _ = analyse(leaf.image)

    total = report.healthy_pct + report.chlorotic_pct + report.necrotic_pct
    assert total == pytest.approx(100.0, abs=2.0)
    assert report.severity_pct == pytest.approx(report.chlorotic_pct + report.necrotic_pct, abs=0.05)


def test_leaf_mask_excludes_the_background() -> None:
    leaf = make_leaf(damaged_fraction=0.2, lesion_count=12, seed=9)
    mask = segment_leaf(leaf.image)

    # The leaf ellipse occupies roughly 35-40 % of the frame by construction.
    coverage = float((mask > 0).mean())
    assert 0.2 < coverage < 0.6

    # Agreement with the generator's own leaf mask (IoU), which catches the failure mode
    # where white balancing tints the background into the mask.
    intersection = np.logical_and(mask > 0, leaf.leaf_mask > 0).sum()
    union = np.logical_or(mask > 0, leaf.leaf_mask > 0).sum()
    assert intersection / union > 0.85


def test_severity_is_monotonic_in_damage() -> None:
    measured = [
        analyse(make_leaf(damaged_fraction=f, lesion_count=max(int(f * 60), 1), seed=2).image)[0].severity_pct
        for f in (0.0, 0.08, 0.2, 0.4, 0.6)
    ]
    assert measured == sorted(measured)


@pytest.mark.parametrize(
    ("severity", "expected"),
    [
        (0.0, ActionTier.NEGLIGIBLE),
        (1.9, ActionTier.NEGLIGIBLE),
        (2.0, ActionTier.EARLY),
        (9.99, ActionTier.EARLY),
        (10.0, ActionTier.MODERATE),
        (24.9, ActionTier.MODERATE),
        (25.0, ActionTier.SEVERE),
        (49.9, ActionTier.SEVERE),
        (50.0, ActionTier.CRITICAL),
        (100.0, ActionTier.CRITICAL),
    ],
)
def test_tier_boundaries(severity: float, expected: ActionTier) -> None:
    assert classify_tier(severity) is expected


def test_blank_image_does_not_crash() -> None:
    """A photo of a wall must produce a warning, not a ZeroDivisionError."""
    blank = np.full((240, 320, 3), 200, dtype=np.uint8)
    report, _ = analyse(blank)
    assert report.severity_pct >= 0.0
    assert report.action_tier in set(ActionTier)


def test_overlay_is_renderable() -> None:
    leaf = make_leaf(damaged_fraction=0.3, lesion_count=18, seed=4)
    _, masks = analyse(leaf.image)
    overlay = render_overlay(masks)
    assert overlay.shape == masks.working.shape
    assert overlay.dtype == np.uint8


# --- segmentation robustness ------------------------------------------------
#
# A fixed saturation floor let a tinted or cluttered background into the leaf mask, and
# every off-green background pixel then counted as diseased tissue — around +49 pp of
# severity, two whole action tiers. These tests pin the adaptive-threshold behaviour that
# replaced it.


def _tint(image: np.ndarray, gains: tuple[float, float, float]) -> np.ndarray:
    return np.clip(image.astype(np.float32) * np.array(gains), 0, 255).astype(np.uint8)


@pytest.mark.parametrize(
    ("name", "gains"),
    [("warm light", (0.78, 0.95, 1.22)), ("cool shade", (1.25, 1.02, 0.8))],
)
def test_colour_cast_does_not_break_the_leaf_mask(name: str, gains: tuple) -> None:
    leaf = make_leaf(damaged_fraction=0.25, lesion_count=15, seed=77)
    clean, _ = analyse(leaf.image)
    cast, _ = analyse(_tint(leaf.image, gains))

    # The mask must still find the leaf rather than swallowing the tinted background.
    assert cast.leaf_coverage_pct == pytest.approx(clean.leaf_coverage_pct, abs=5.0)
    assert cast.action_tier is clean.action_tier, f"{name} changed the action tier"


def test_saturated_background_does_not_inflate_severity() -> None:
    """Soil is saturated and reddish, so a fixed saturation floor cannot reject it."""
    leaf = make_leaf(damaged_fraction=0.25, lesion_count=15, seed=77)
    rng = np.random.default_rng(5)

    cluttered = leaf.image.copy()
    soil = rng.integers(40, 110, cluttered.shape, dtype=np.int16)
    soil[:, :, 2] = np.clip(soil[:, :, 2] + 55, 0, 255)  # reddish
    outside = leaf.leaf_mask == 0
    cluttered[outside] = soil[outside].astype(np.uint8)

    truth = leaf.true_damaged_fraction * 100
    report, _ = analyse(cluttered)
    assert report.severity_pct == pytest.approx(truth, abs=6.0)
    assert report.leaf_coverage_pct < 60.0  # background was excluded


def test_macro_shot_falls_back_instead_of_carving_up_the_leaf() -> None:
    """On a frame that is *all* leaf, an adaptive threshold has no background to find and
    would split the leaf itself, cutting the darker lesions out and under-reporting damage.
    The border test must reject that and fall back to measuring the whole frame."""
    leaf = make_leaf(damaged_fraction=0.25, lesion_count=15, seed=77)
    ys, xs = np.nonzero(leaf.leaf_mask)
    macro = leaf.image[ys.min() + 40 : ys.max() - 40, xs.min() + 4 : xs.max() - 4]

    report, _ = analyse(macro)
    assert report.leaf_coverage_pct > 95.0
    assert any("frame" in warning for warning in report.quality_warnings)
    # Crucially it must not UNDER-report: silently dropping lesions is the dangerous failure.
    assert report.severity_pct >= leaf.true_damaged_fraction * 100 - 2.0


def test_border_background_fraction_distinguishes_the_two_cases() -> None:
    from app.vision.severity import _border_background_fraction

    edge_background = np.zeros((60, 60), dtype=np.uint8)
    edge_background[10:50, 10:50] = 255  # subject in the middle
    assert _border_background_fraction(edge_background) == pytest.approx(1.0)

    interior_hole = np.full((60, 60), 255, dtype=np.uint8)
    interior_hole[25:35, 25:35] = 0  # a lesion carved out of a full-frame subject
    assert _border_background_fraction(interior_hole) == pytest.approx(0.0)
