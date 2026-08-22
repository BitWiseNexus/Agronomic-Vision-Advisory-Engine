"""Dataset and augmentation policy for PlantVillage.

The dataset is expected as an ImageFolder tree:

    data/raw/plantvillage/
        Tomato___Late_blight/xxx.jpg
        Tomato___healthy/yyy.jpg
        ...

Class order comes from `ImageFolder`'s alphabetical sort, and the resulting list is saved
into the checkpoint. `classifier.py` reads it back, so label order can never silently drift
between training and inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(config: dict[str, Any], train: bool) -> transforms.Compose:
    size = int(config.get("image_size", 224))
    normalise = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    if not train:
        return transforms.Compose(
            [
                transforms.Resize(int(size * 1.14)),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                normalise,
            ]
        )

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


def build_loaders(config: dict[str, Any], root: Path) -> tuple[DataLoader, DataLoader, list[str]]:
    """Stratified-ish split by a fixed random permutation, then two views of one folder.

    Two `ImageFolder` instances over the same root — one with train transforms, one with
    eval transforms — sharing an index split. This is what keeps validation images free of
    augmentation while still coming from a single directory tree.
    """
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset not found at {root}. Run scripts/download_plantvillage.py, or point "
            "data_dir at your own ImageFolder tree."
        )

    train_view = ImageFolder(str(root), transform=build_transforms(config, train=True))
    eval_view = ImageFolder(str(root), transform=build_transforms(config, train=False))
    classes = list(train_view.classes)

    generator = torch.Generator().manual_seed(int(config.get("seed", 42)))
    total = len(train_view)
    permutation = torch.randperm(total, generator=generator).tolist()
    val_size = int(total * float(config.get("val_split", 0.15)))
    val_indices, train_indices = permutation[:val_size], permutation[val_size:]

    batch_size = int(config.get("batch_size", 32))
    workers = int(config.get("num_workers", 4))
    common = {"batch_size": batch_size, "num_workers": workers, "pin_memory": torch.cuda.is_available()}

    train_loader = DataLoader(
        Subset(train_view, train_indices), shuffle=True, drop_last=True, **common
    )
    val_loader = DataLoader(Subset(eval_view, val_indices), shuffle=False, **common)
    return train_loader, val_loader, classes
