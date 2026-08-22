"""Quantitative severity index — the signal that grounds the LLM.

Deliberately *not* a neural network. Every number here is traceable to a pixel count
under a stated threshold, which is what makes it defensible in a report and auditable
by an agronomist. The pipeline is:

    photo -> leaf mask (isolate the specimen from tray/soil/hand)
          -> per-pixel tissue class inside the mask (healthy / chlorotic / necrotic)
          -> area ratios + lesion morphology
          -> action tier

Thresholds live in `SeverityConfig` so `notebooks/01_severity_lab.ipynb` can sweep them
against annotated leaves rather than having magic numbers buried in the code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from app.schemas.diagnosis import ActionTier, SeverityReport
from app.vision.preprocess import background_white_balance, prepare_for_severity


@dataclass(frozen=True)
class SeverityConfig:
    """OpenCV HSV bands. Hue is 0-179 in OpenCV, not 0-359."""

    # --- leaf isolation ---
    leaf_min_saturation: int = 28
    leaf_min_value: int = 25
    leaf_max_value: int = 250
    morph_kernel: int = 7
    # A candidate leaf mask must occupy a middling share of the frame. Below the floor it
    # is noise; above the ceiling it is the background, not the leaf.
    min_leaf_fraction: float = 0.02
    max_leaf_fraction: float = 0.92
    # ...and the region it excludes must really be at the frame edge. See segment_leaf.
    min_border_background: float = 0.5

    # --- healthy green ---
    green_hue_lo: int = 30
    green_hue_hi: int = 92
    green_min_saturation: int = 45
    green_min_value: int = 35

    # --- chlorotic (yellowing) ---
    yellow_hue_lo: int = 18
    yellow_hue_hi: int = 30
    yellow_min_saturation: int = 60

    # --- necrotic (brown / dead / dark lesion) ---
    brown_hue_hi: int = 18  # 0..18 -> red-brown
    brown_hue_wrap_lo: int = 165  # 165..179 -> the other side of red
    dark_value_max: int = 55  # very dark pixels are dead tissue or deep shadow

    # --- lesion morphology ---
    lesion_noise_floor_frac: float = 0.0005  # 0.05 % of leaf area

    # --- action tiers: (upper bound exclusive, tier) ---
    tier_bounds: tuple[tuple[float, ActionTier], ...] = field(
        default=(
            (2.0, ActionTier.NEGLIGIBLE),
            (10.0, ActionTier.EARLY),
            (25.0, ActionTier.MODERATE),
            (50.0, ActionTier.SEVERE),
            (100.1, ActionTier.CRITICAL),
        )
    )


DEFAULT_CONFIG = SeverityConfig()


@dataclass
class SeverityMasks:
    """Kept separate from the report so the API can stay lean while the notebook
    still gets every intermediate mask for the figures."""

    working: np.ndarray  # preprocessed BGR
    leaf: np.ndarray  # uint8 0/255
    healthy: np.ndarray
    chlorotic: np.ndarray
    necrotic: np.ndarray
    lesions: np.ndarray  # chlorotic | necrotic, cleaned


def _largest_component_mask(mask: np.ndarray) -> np.ndarray:
    """Keep only the biggest blob. A leaf is one connected object; anything else in the
    frame (a second leaf, a finger, tray texture) is noise for this measurement."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    # index 0 is the background label
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Lesions that reach the leaf edge can punch holes in the leaf mask; filling them
    keeps damaged tissue inside the denominator instead of silently deleting it."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _saturation_mask(
    saturation: np.ndarray, value: np.ndarray, threshold: int, config: SeverityConfig
) -> np.ndarray:
    """One threshold, cleaned up: gate, morphological open/close, largest blob, fill holes."""
    mask = (
        (saturation >= threshold)
        & (value >= config.leaf_min_value)
        & (value <= config.leaf_max_value)
    ).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.morph_kernel, config.morph_kernel)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return _fill_holes(_largest_component_mask(mask))


def _border_background_fraction(mask: np.ndarray) -> float:
    """Share of the frame border that the candidate treats as background.

    This is the test that tells a real background separation apart from a mask that has
    merely eaten part of the subject. A photographed leaf does not extend to all four
    edges, so its background dominates the border. If a candidate excludes mostly
    *interior* pixels instead, it is cutting lesions out of the leaf, not removing
    background — and that silently deflates the severity index.
    """
    border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    return float((border == 0).mean())


def segment_leaf(bgr: np.ndarray, config: SeverityConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Isolate the leaf from the background.

    A leaf — healthy or diseased — is a *saturated, mid-brightness* object, while the
    typical background (grey tray, paper, blown-out sky, deep shadow) is either desaturated
    or at a brightness extreme. Saturation gating catches that in one step and, unlike Otsu
    on greyscale, does not fail when leaf and background share a luminance.

    A *fixed* saturation floor is not enough, though. Under a warm colour cast, or against
    soil, the background's saturation climbs past the floor, the mask swallows the whole
    frame, and every off-green background pixel is then counted as diseased tissue — worth
    around +49 percentage points of severity in measurement, which is two whole action
    tiers. So two candidates are generated, the fixed floor and an Otsu threshold that
    adapts to the image, and each is validated on:

      * **plausible area** — a leaf occupies a middling share of the frame, not ~all of it;
      * **border test** — the excluded region must actually be at the frame edge.

    The border test is what makes Otsu safe here. On a macro shot that is *all* leaf, Otsu
    has no background to find and splits the leaf itself, carving out the darker lesions;
    that mask excludes interior pixels while the border stays "leaf", so it is rejected and
    we fall back to the whole frame with a quality warning instead of under-reporting damage.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    frame_area = saturation.size

    otsu_threshold, _ = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresholds = [config.leaf_min_saturation]
    if int(otsu_threshold) > config.leaf_min_saturation:
        thresholds.append(int(otsu_threshold))

    accepted: list[tuple[float, np.ndarray]] = []
    for threshold in thresholds:
        mask = _saturation_mask(saturation, value, threshold, config)
        area_fraction = float((mask > 0).sum()) / frame_area
        if not config.min_leaf_fraction <= area_fraction <= config.max_leaf_fraction:
            continue
        if _border_background_fraction(mask) < config.min_border_background:
            continue
        accepted.append((area_fraction, mask))

    if not accepted:
        # No candidate looks like a leaf-on-background: a macro shot filling the frame, a
        # very dull image, or clutter we cannot separate. Measure the whole frame and let
        # the quality warning say so, rather than trusting a mask we just rejected.
        return np.full(saturation.shape, 255, dtype=np.uint8)

    # Prefer the candidate that keeps the most tissue: an over-eager threshold trims
    # low-saturation necrotic areas off the leaf margin and under-reports severity.
    return max(accepted, key=lambda item: item[0])[1]


def _tissue_masks(
    bgr: np.ndarray, leaf: np.ndarray, config: SeverityConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    inside = leaf > 0

    healthy = (
        inside
        & (hue >= config.green_hue_lo)
        & (hue <= config.green_hue_hi)
        & (saturation >= config.green_min_saturation)
        & (value >= config.green_min_value)
    )

    chlorotic = (
        inside
        & ~healthy
        & (hue >= config.yellow_hue_lo)
        & (hue <= config.yellow_hue_hi)
        & (saturation >= config.yellow_min_saturation)
        & (value >= config.dark_value_max)
    )

    brownish = (hue <= config.brown_hue_hi) | (hue >= config.brown_hue_wrap_lo)
    necrotic = inside & ~healthy & ~chlorotic & (brownish | (value < config.dark_value_max))

    to_u8 = lambda m: m.astype(np.uint8) * 255  # noqa: E731
    return to_u8(healthy), to_u8(chlorotic), to_u8(necrotic)


def _lesion_stats(
    lesions: np.ndarray, leaf_area: int, config: SeverityConfig
) -> tuple[int, float, np.ndarray]:
    """Count discrete lesions and measure the biggest one.

    Lesion *count* separates 'many tiny specks' (Septoria-like, early) from 'one large
    coalesced blight patch' at the same total severity — the LLM is told both because
    they call for different spray coverage.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(lesions, cv2.MORPH_OPEN, kernel, iterations=1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    floor = max(int(config.lesion_noise_floor_frac * leaf_area), 12)

    kept = 0
    largest = 0
    keep_mask = np.zeros_like(cleaned)
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < floor:
            continue
        kept += 1
        largest = max(largest, area)
        keep_mask[labels == index] = 255

    largest_pct = 100.0 * largest / leaf_area if leaf_area else 0.0
    return kept, largest_pct, keep_mask


def classify_tier(severity_pct: float, config: SeverityConfig = DEFAULT_CONFIG) -> ActionTier:
    for upper, tier in config.tier_bounds:
        if severity_pct < upper:
            return tier
    return ActionTier.CRITICAL


def analyse(
    bgr: np.ndarray,
    config: SeverityConfig = DEFAULT_CONFIG,
    balance_white: bool = True,
) -> tuple[SeverityReport, SeverityMasks]:
    """Full severity analysis. Returns the report plus every mask for inspection.

    Order of operations is deliberate: segment on the raw denoised image, then white
    balance using the background the segmentation just identified, then classify tissue.
    Balancing before segmentation tints the background and destroys the leaf mask.
    """
    working = prepare_for_severity(bgr)
    leaf = segment_leaf(working, config)
    if balance_white:
        working = background_white_balance(working, leaf)

    leaf_area = int((leaf > 0).sum())
    frame_area = leaf.shape[0] * leaf.shape[1]
    warnings: list[str] = []

    if leaf_area == 0:  # pathological, but never let it divide by zero
        empty = np.zeros_like(leaf)
        report = SeverityReport(
            severity_pct=0.0,
            necrotic_pct=0.0,
            chlorotic_pct=0.0,
            healthy_pct=0.0,
            lesion_count=0,
            largest_lesion_pct=0.0,
            leaf_coverage_pct=0.0,
            action_tier=ActionTier.NEGLIGIBLE,
            quality_warnings=["No leaf tissue detected — retake the photo against a plain background."],
        )
        return report, SeverityMasks(working, leaf, empty, empty, empty, empty)

    healthy, chlorotic, necrotic = _tissue_masks(working, leaf, config)
    lesions_raw = cv2.bitwise_or(chlorotic, necrotic)
    lesion_count, largest_lesion_pct, lesions = _lesion_stats(lesions_raw, leaf_area, config)

    pct = lambda mask: 100.0 * float((mask > 0).sum()) / leaf_area  # noqa: E731
    healthy_pct = pct(healthy)
    chlorotic_pct = pct(chlorotic)
    necrotic_pct = pct(necrotic)
    severity_pct = min(chlorotic_pct + necrotic_pct, 100.0)

    coverage_pct = 100.0 * leaf_area / frame_area
    if coverage_pct < 8:
        warnings.append(
            f"Leaf fills only {coverage_pct:.0f} % of the frame — move closer; "
            "the severity index is less reliable on small subjects."
        )
    if coverage_pct > 97:
        warnings.append(
            "Leaf fills the whole frame, so background separation was skipped — "
            "severity may be over-estimated if any background is included."
        )
    grey = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    if float(grey[leaf > 0].mean()) < 60:
        warnings.append("Image is underexposed; dark pixels can be mis-read as necrosis.")
    if float(grey[leaf > 0].std()) < 8:
        warnings.append("Very low contrast on the leaf — thresholds may be unstable.")

    report = SeverityReport(
        severity_pct=round(severity_pct, 2),
        necrotic_pct=round(necrotic_pct, 2),
        chlorotic_pct=round(chlorotic_pct, 2),
        healthy_pct=round(healthy_pct, 2),
        lesion_count=lesion_count,
        largest_lesion_pct=round(largest_lesion_pct, 2),
        leaf_coverage_pct=round(coverage_pct, 2),
        action_tier=classify_tier(severity_pct, config),
        quality_warnings=warnings,
    )
    masks = SeverityMasks(working, leaf, healthy, chlorotic, necrotic, lesions)
    return report, masks


def render_overlay(masks: SeverityMasks, alpha: float = 0.45) -> np.ndarray:
    """Green = healthy, yellow = chlorotic, red = necrotic, dim = outside the leaf.

    This is the figure that makes the severity index believable to a human reviewer.
    """
    base = masks.working.copy()
    tint = np.zeros_like(base)
    tint[masks.healthy > 0] = (0, 180, 0)  # BGR
    tint[masks.chlorotic > 0] = (0, 220, 255)
    tint[masks.necrotic > 0] = (0, 0, 230)

    blended = cv2.addWeighted(base, 1 - alpha, tint, alpha, 0)
    outside = masks.leaf == 0
    blended[outside] = (base[outside] * 0.35).astype(np.uint8)

    contours, _ = cv2.findContours(masks.leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (255, 255, 255), 2)
    return blended
