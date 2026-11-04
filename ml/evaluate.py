r"""Evaluate a trained checkpoint on the held-out TEST split.

    .\.venv\Scripts\python.exe ml\evaluate.py
    .\.venv\Scripts\python.exe ml\evaluate.py --robustness

Reports, all through the same preprocessing the API uses at inference:

  * accuracy, macro F1 and weighted F1 — with and without flip test-time augmentation
  * per-class F1, weakest classes first
  * calibration: expected calibration error (ECE) before and after temperature scaling,
    and how often the app's "reliable" confidence floor is actually right
  * a row-normalised confusion matrix figure
  * --robustness: accuracy under phone-photo corruptions (colour casts, low light, blur,
    JPEG, sensor noise). PlantVillage is shot under uniform lab conditions, so clean test
    accuracy is an upper bound; this is the number that says how far it degrades.

Per-class F1 matters more than headline accuracy: PlantVillage is imbalanced, and a model
that nails the 5,000-image classes while failing a 150-image class still posts a
flattering overall score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "ml"))

from datasets import ServingTransform, build_loaders  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.vision.classifier import build_model  # noqa: E402
from app.vision.preprocess import to_classifier_tensor  # noqa: E402


# ── corruptions (applied to the BGR photo before serving preprocessing) ──────────────
def _scale(img, bgr):
    return np.clip(img.astype(np.float32) * np.array(bgr, np.float32), 0, 255).astype(np.uint8)


def _jpeg(img, quality=30):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _noise(img, sigma=12.0):
    rng = np.random.default_rng(0)
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


CORRUPTIONS = {
    "clean": lambda img: img,
    "warm light": lambda img: _scale(img, (0.78, 0.95, 1.22)),
    "cool shade": lambda img: _scale(img, (1.25, 1.02, 0.80)),
    "low light": lambda img: _scale(img, (0.45, 0.45, 0.45)),
    "overexposed": lambda img: _scale(img, (1.5, 1.5, 1.5)),
    "blur": lambda img: cv2.GaussianBlur(img, (7, 7), 2.0),
    "jpeg q30": _jpeg,
    "sensor noise": _noise,
}


class CorruptedServingTransform(ServingTransform):
    def __init__(self, kind: str) -> None:
        self.kind = kind

    def __call__(self, image) -> torch.Tensor:
        bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        return to_classifier_tensor(CORRUPTIONS[self.kind](bgr))[0]


# ── inference ────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def collect_logits(model, loader, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Logits for each image and for its mirror image, plus the labels."""
    model.eval()
    plain, flipped, targets = [], [], []
    for images, labels in loader:
        # Plain fp32: what the API runs, and fastest on GPUs without tensor cores.
        images = images.to(device, non_blocking=True)
        plain.append(model(images).float().cpu())
        flipped.append(model(torch.flip(images, dims=[3])).float().cpu())
        targets.append(labels)
    return torch.cat(plain).numpy(), torch.cat(flipped).numpy(), torch.cat(targets).numpy()


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, truth: np.ndarray, bins: int = 15) -> float:
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == truth
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def plot_confusion(matrix: np.ndarray, classes: list[str], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Row-normalised: absolute counts are unreadable on an imbalanced 38-class problem.
    normalised = matrix.astype(np.float32) / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    figure, axes = plt.subplots(figsize=(14, 12))
    image = axes.imshow(normalised, cmap="viridis", vmin=0, vmax=1)
    axes.set_xticks(range(len(classes)), classes, rotation=90, fontsize=6)
    axes.set_yticks(range(len(classes)), classes, fontsize=6)
    axes.set_xlabel("Predicted")
    axes.set_ylabel("True")
    axes.set_title("Confusion matrix on the held-out test split (row-normalised)")
    figure.colorbar(image, ax=axes, fraction=0.046)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    print(f"Confusion matrix -> {out_path}")


def robustness(model, test_loader: DataLoader, temperature: float, device: str, every: int) -> dict:
    """Accuracy (with TTA) on a stratified-enough subsample of test under each corruption."""
    subset: Subset = test_loader.dataset  # type: ignore[assignment]
    indices = subset.indices[::every]
    results = {}
    print(f"\nRobustness on {len(indices):,} test images (every {every}th), flip TTA on:")
    print(f"  {'condition':<14}{'accuracy':>10}{'mean conf':>11}")
    for kind in CORRUPTIONS:
        view = type(subset.dataset)(subset.dataset.root, transform=CorruptedServingTransform(kind))
        loader = DataLoader(
            Subset(view, indices),
            batch_size=test_loader.batch_size,
            num_workers=test_loader.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        plain, flipped, truth = collect_logits(model, loader, device)
        probs = (softmax(plain, temperature) + softmax(flipped, temperature)) / 2
        acc = float((probs.argmax(1) == truth).mean())
        conf = float(probs.max(1).mean())
        results[kind] = {"accuracy": acc, "mean_confidence": conf}
        print(f"  {kind:<14}{acc:>9.2%}{conf:>11.2f}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the disease classifier on the test split")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "checkpoints" / "resnet50_plantvillage.pt")
    parser.add_argument("--config", type=Path, default=ROOT / "ml" / "configs" / "resnet50_plantvillage.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "evaluation")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--robustness", action="store_true", help="Also score photo corruptions")
    parser.add_argument("--robustness-every", type=int, default=4, help="Use every Nth test image")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"No checkpoint at {args.checkpoint}. Run ml/train.py first.")

    from sklearn.metrics import classification_report, confusion_matrix

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) if args.config.exists() else {}
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes = blob["classes"]
    temperature = float(blob.get("temperature", 1.0))
    # Re-create the exact split the checkpoint was trained with.
    for key, value in (blob.get("split") or {}).items():
        config[{"seed": "seed", "val": "val_split", "test": "test_split"}[key]] = value

    data_dir = Path(config.get("data_dir", "data/raw/plantvillage"))
    data_dir = data_dir if data_dir.is_absolute() else ROOT / data_dir
    _, _, test_loader, folder_classes, _ = build_loaders(config, data_dir)

    if folder_classes != classes:
        # A silent mismatch here would make every metric meaningless.
        raise SystemExit(
            "Class order in the dataset folder does not match the checkpoint.\n"
            f"  checkpoint: {classes[:3]}...\n  folder:     {folder_classes[:3]}..."
        )

    model = build_model(num_classes=len(classes), pretrained=False)
    model.load_state_dict(blob["model_state"])
    model.to(device)
    torch.backends.cudnn.benchmark = True

    plain, flipped, truth = collect_logits(model, test_loader, device)
    p_raw = softmax(plain)
    p_cal = softmax(plain, temperature)
    p_tta = (softmax(plain, temperature) + softmax(flipped, temperature)) / 2
    predicted = p_tta.argmax(1)

    acc_plain = float((p_raw.argmax(1) == truth).mean())
    acc_tta = float((predicted == truth).mean())
    report = classification_report(truth, predicted, target_names=classes, output_dict=True, zero_division=0)
    report_plain = classification_report(truth, p_raw.argmax(1), target_names=classes, output_dict=True, zero_division=0)

    floor = get_settings().classifier_confidence_floor
    confident = p_tta.max(1) >= floor
    ece_raw = expected_calibration_error(p_raw, truth)
    ece_cal = expected_calibration_error(p_tta, truth)

    print(f"\nHeld-out test split: n = {len(truth):,} images, {len(classes)} classes")
    print(f"  {'':<22}{'accuracy':>10}{'macro F1':>10}{'weighted F1':>13}")
    print(f"  {'single view':<22}{acc_plain:>9.2%}{report_plain['macro avg']['f1-score']:>10.4f}"
          f"{report_plain['weighted avg']['f1-score']:>13.4f}")
    print(f"  {'flip TTA (served)':<22}{acc_tta:>9.2%}{report['macro avg']['f1-score']:>10.4f}"
          f"{report['weighted avg']['f1-score']:>13.4f}")
    print(f"\nCalibration: T = {temperature:.3f}   ECE {ece_raw:.4f} (raw) -> {ece_cal:.4f} (calibrated)")
    print(f"Confidence floor {floor:.2f}: {confident.mean():.1%} of test images marked reliable; "
          f"accuracy on those {float((predicted[confident] == truth[confident]).mean()):.2%}, "
          f"on the rest {float((predicted[~confident] == truth[~confident]).mean()) if (~confident).any() else float('nan'):.2%}")

    per_class = sorted(
        ((name, report[name]["f1-score"], int(report[name]["support"])) for name in classes),
        key=lambda row: row[1],
    )
    print("\nWeakest 10 classes by F1:")
    for name, f1, support in per_class[:10]:
        print(f"  {f1:.3f}  (n={support:5d})  {name}")

    results = {
        "n_test": int(len(truth)),
        "temperature": temperature,
        "accuracy_single_view": acc_plain,
        "accuracy_tta": acc_tta,
        "macro_f1_tta": report["macro avg"]["f1-score"],
        "ece_raw": ece_raw,
        "ece_calibrated": ece_cal,
        "confidence_floor": floor,
        "share_reliable": float(confident.mean()),
        "report": report,
    }
    if args.robustness:
        results["robustness"] = robustness(model, test_loader, temperature, device, args.robustness_every)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "classification_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_confusion(
        confusion_matrix(truth, predicted, labels=range(len(classes))),
        classes,
        args.output_dir / "confusion_matrix.png",
    )
    print(f"Report -> {args.output_dir / 'classification_report.json'}")


if __name__ == "__main__":
    main()
