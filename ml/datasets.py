"""Dataset and augmentation policy for PlantVillage.

The dataset is expected as an ImageFolder tree:

    data/raw/plantvillage/
        Tomato___Late_blight/xxx.jpg
        Tomato___healthy/yyy.jpg
        ...

Class order comes from `ImageFolder`'s alphabetical sort, and the resulting list is saved
into the checkpoint. `classifier.py` reads it back, so label order can never silently drift
between training and inference.

Splits are train / val / test. Validation picks the best epoch and fits the calibration
temperature, so it is no longer an unbiased estimate; the test split is touched exactly
once, by `ml/evaluate.py`, and is the number to report.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.vision.preprocess import to_classifier_tensor  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ServingTransform:
    """Evaluate through the exact preprocessing the API uses at inference.

    torchvision's Resize/CenterCrop on a PIL image and the OpenCV path in
    `app.vision.preprocess.to_classifier_tensor` differ in resampling filter and resize
    target. Scoring validation and test through the serving function means the reported
    accuracy is the accuracy of the deployed model, not of a near-identical twin.
    (A class, not a lambda, so DataLoader workers can pickle it on Windows.)
    """

    def __call__(self, image) -> torch.Tensor:
        bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        return to_classifier_tensor(bgr)[0]


def build_transforms(config: dict[str, Any], train: bool):
    size = int(config.get("image_size", 224))
    if not train:
        return ServingTransform()

    normalise = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    augment = config.get("augment", {}) or {}
    jitter = augment.get("color_jitter", {}) or {}
    scale = tuple(augment.get("random_resized_crop_scale", [0.7, 1.0]))

    steps: list[Any] = [
        transforms.RandomResizedCrop(size, scale=scale, ratio=(0.85, 1.18)),
        transforms.RandomHorizontalFlip(float(augment.get("horizontal_flip", 0.5))),
        transforms.RandomVerticalFlip(float(augment.get("vertical_flip", 0.2))),
        transforms.RandomRotation(float(augment.get("rotation_degrees", 25))),
        transforms.ColorJitter(
            brightness=float(jitter.get("brightness", 0.35)),
            contrast=float(jitter.get("contrast", 0.35)),
            saturation=float(jitter.get("saturation", 0.35)),
            hue=float(jitter.get("hue", 0.06)),
        ),
    ]
    blur_p = float(augment.get("blur_probability", 0.15))
    if blur_p > 0:
        steps.append(
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.6))], p=blur_p)
        )
    steps.extend([transforms.ToTensor(), normalise])

    erase_p = float(augment.get("erasing_probability", 0.15))
    if erase_p > 0:
        # Occlusion robustness: real photos have a finger, a shadow or another leaf in frame.
        steps.append(transforms.RandomErasing(p=erase_p, scale=(0.02, 0.12)))
    return transforms.Compose(steps)


def stratified_split(
    targets: list[int], val_fraction: float, test_fraction: float, seed: int
) -> tuple[list[int], list[int], list[int]]:
    """Per-class seeded split, so every class — including the 152-image Potato___healthy —
    appears in val and test in proportion. A global random split can leave a small class
    with a handful of test images and a meaningless per-class F1."""
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(targets):
        by_class.setdefault(label, []).append(index)

    train, val, test = [], [], []
    for label in sorted(by_class):
        indices = np.array(by_class[label])
        rng.shuffle(indices)
        n_test = max(1, round(len(indices) * test_fraction))
        n_val = max(1, round(len(indices) * val_fraction))
        test.extend(indices[:n_test].tolist())
        val.extend(indices[n_test : n_test + n_val].tolist())
        train.extend(indices[n_test + n_val :].tolist())
    return train, val, test


def class_counts(targets: list[int], indices: list[int], num_classes: int) -> np.ndarray:
    return np.bincount(np.asarray(targets)[indices], minlength=num_classes)


def build_loaders(
    config: dict[str, Any], root: Path
) -> tuple[DataLoader, DataLoader, DataLoader, list[str], np.ndarray]:
    """Two `ImageFolder` views of one tree — train transforms and serving transforms —
    sharing one stratified index split. Returns (train, val, test, classes, train_counts)."""
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset not found at {root}. Run scripts/download_plantvillage.py, or point "
            "data_dir at your own ImageFolder tree."
        )

    train_view = ImageFolder(str(root), transform=build_transforms(config, train=True))
    eval_view = ImageFolder(str(root), transform=build_transforms(config, train=False))
    classes = list(train_view.classes)

    train_idx, val_idx, test_idx = stratified_split(
        train_view.targets,
        float(config.get("val_split", 0.15)),
        float(config.get("test_split", 0.15)),
        int(config.get("seed", 42)),
    )

    batch_size = int(config.get("batch_size", 32))
    workers = int(config.get("num_workers", 4))
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }

    train_loader = DataLoader(Subset(train_view, train_idx), shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(Subset(eval_view, val_idx), shuffle=False, **common)
    test_loader = DataLoader(Subset(eval_view, test_idx), shuffle=False, **common)
    counts = class_counts(train_view.targets, train_idx, len(classes))
    return train_loader, val_loader, test_loader, classes, counts
