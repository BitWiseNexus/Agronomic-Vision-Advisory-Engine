"""Synthetic leaf generator.

Exists for two reasons: the notebooks and API must be demonstrable before the 1.6 GB
PlantVillage download, and the severity engine needs test fixtures whose *true* damaged
fraction is known by construction. Nothing here is used at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SyntheticLeaf:
    image: np.ndarray  # BGR
    true_damaged_fraction: float  # damaged leaf pixels / leaf pixels
    leaf_mask: np.ndarray
    lesion_mask: np.ndarray


def make_leaf(
    damaged_fraction: float = 0.15,
    *,
    size: tuple[int, int] = (480, 640),
    lesion_count: int = 8,
    chlorosis_share: float = 0.35,
    seed: int = 0,
    background_grey: int = 168,
) -> SyntheticLeaf:
    """Render a leaf with a controlled fraction of damaged tissue.

    `damaged_fraction` is approximate: lesions are placed until the target area is
    reached, and overlaps are counted once, so the realised fraction is returned rather
    than assumed. `chlorosis_share` splits the damage between yellowing (chlorotic) and
    brown-dead (necrotic) tissue.
    """
    rng = np.random.default_rng(seed)
    height, width = size

    # --- background: a dull grey tray with a little noise, i.e. low saturation ---
    # The noise must be applied in a signed dtype: adding it as uint8 wraps the negative
    # values to ~244 and turns the "grey" background into saturated colour noise, which
    # the saturation-gated leaf mask would then happily swallow.
    grey_noise = rng.integers(-10, 11, (height, width, 1))
    image = np.clip(
        np.full((height, width, 3), background_grey, dtype=np.int16) + grey_noise, 0, 255
    ).astype(np.uint8)

    # --- leaf body: rotated ellipse with a vertical green gradient ---
    leaf_mask = np.zeros((height, width), dtype=np.uint8)
    centre = (width // 2, height // 2)
    axes = (int(width * 0.30), int(height * 0.40))
    angle = float(rng.uniform(-25, 25))
    cv2.ellipse(leaf_mask, centre, axes, angle, 0, 360, 255, -1)

    gradient = np.linspace(0.78, 1.14, height, dtype=np.float32)[:, None]
    green = np.zeros((height, width, 3), dtype=np.float32)
    green[..., 0] = 42 * gradient  # B
    green[..., 1] = 138 * gradient  # G
    green[..., 2] = 46 * gradient  # R
    green += rng.normal(0, 5, green.shape).astype(np.float32)
    leaf_bgr = np.clip(green, 0, 255).astype(np.uint8)

    # midrib + a few veins, slightly lighter than the blade
    veins = np.zeros((height, width), dtype=np.uint8)
    top = (centre[0] - int(axes[1] * np.sin(np.radians(angle))), centre[1] - int(axes[1] * np.cos(np.radians(angle))))
    bottom = (centre[0] + int(axes[1] * np.sin(np.radians(angle))), centre[1] + int(axes[1] * np.cos(np.radians(angle))))
    cv2.line(veins, top, bottom, 255, 3)
    for offset in range(-3, 4):
        if offset == 0:
            continue
        y = centre[1] + offset * (axes[1] // 4)
        cv2.line(veins, (centre[0], y), (centre[0] + offset * axes[0] // 2, y - axes[1] // 8), 255, 1)
    leaf_bgr[veins > 0] = np.clip(leaf_bgr[veins > 0].astype(np.int16) + 45, 0, 255).astype(np.uint8)

    leaf_area = int((leaf_mask > 0).sum())

    # Lesions are placed at random, so they overlap. For randomly scattered discs the
    # covered fraction converges to 1 - exp(-p), where p is the total disc area laid down
    # divided by the leaf area. Sizing the discs to sum to the target directly therefore
    # undershoots badly at high severity (a 75 % target lands near 43 %). Inverting the
    # relation — p = -ln(1 - target) — puts the realised fraction on target.
    clamped = min(max(damaged_fraction, 0.0), 0.94)
    target_area = -np.log(1.0 - clamped) * leaf_area if clamped > 0 else 0.0

    necrotic = np.zeros((height, width), dtype=np.uint8)
    chlorotic = np.zeros((height, width), dtype=np.uint8)

    if target_area > 0 and lesion_count > 0:
        # Radius chosen so `lesion_count` lesions sum to the target area. Each lesion is a
        # necrotic core of radius r plus a chlorotic halo out to r*(1+share), so the area
        # per lesion is pi*r^2*(1+share)^2 — ignoring the halo here overshoots the target.
        halo_factor = (1.0 + max(chlorosis_share, 0.0)) ** 2
        radius = max(int(np.sqrt(target_area / (lesion_count * np.pi * halo_factor))), 3)
        ys, xs = np.nonzero(leaf_mask)
        placed = 0
        attempts = 0
        while placed < lesion_count and attempts < lesion_count * 40:
            attempts += 1
            pick = int(rng.integers(0, len(xs)))
            cx, cy = int(xs[pick]), int(ys[pick])
            jitter = float(rng.uniform(0.75, 1.3))
            r = max(int(radius * jitter), 2)
            # A lesion may reach the leaf margin — real blights start there — but must not
            # float on the background. Requiring only a majority overlap (rather than full
            # containment) is what lets high target fractions actually be reached: strict
            # containment starves the packing and silently caps severity near 45 %.
            probe = np.zeros_like(leaf_mask)
            cv2.circle(probe, (cx, cy), r + 2, 255, -1)
            probe_area = int((probe > 0).sum())
            inside_area = int((cv2.bitwise_and(probe, leaf_mask) > 0).sum())
            if probe_area == 0 or inside_area / probe_area < 0.6:
                continue
            cv2.circle(necrotic, (cx, cy), r, 255, -1)
            if chlorosis_share > 0:
                halo = int(round(r * (1 + chlorosis_share)))
                ring = np.zeros_like(leaf_mask)
                cv2.circle(ring, (cx, cy), halo, 255, -1)
                cv2.circle(ring, (cx, cy), r, 0, -1)
                chlorotic = cv2.bitwise_or(chlorotic, cv2.bitwise_and(ring, leaf_mask))
            placed += 1

        necrotic = cv2.bitwise_and(necrotic, leaf_mask)
        chlorotic = cv2.bitwise_and(chlorotic, cv2.bitwise_not(necrotic))

    # brown necrosis and yellow chlorosis, with texture so thresholds are not trivially met
    leaf_bgr[necrotic > 0] = (28, 52, 96)  # BGR ~ dark brown
    leaf_bgr[chlorotic > 0] = (52, 196, 226)  # BGR ~ yellow
    noise = rng.normal(0, 7, leaf_bgr.shape).astype(np.float32)
    leaf_bgr = np.clip(leaf_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    image[leaf_mask > 0] = leaf_bgr[leaf_mask > 0]
    image = cv2.GaussianBlur(image, (3, 3), 0)

    lesion_mask = cv2.bitwise_or(necrotic, chlorotic)
    realised = float((lesion_mask > 0).sum()) / leaf_area if leaf_area else 0.0
    return SyntheticLeaf(image, realised, leaf_mask, lesion_mask)
