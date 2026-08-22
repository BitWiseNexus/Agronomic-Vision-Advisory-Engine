r"""Evaluate a trained checkpoint: overall accuracy, per-class precision/recall/F1, and a
confusion matrix figure.

    .\.venv\Scripts\python.exe ml\evaluate.py --checkpoint models\checkpoints\resnet50_plantvillage.pt

Per-class F1 is the number that matters here, not overall accuracy: PlantVillage is
imbalanced, and a model that nails the 5,000-image tomato classes while failing a
300-image class still posts a flattering headline accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "ml"))

from datasets import build_loaders  # noqa: E402

from app.vision.classifier import build_model  # noqa: E402


@torch.no_grad()
def collect_predictions(model, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for images, labels in loader:
        outputs = model(images.to(device))
        predictions.append(outputs.argmax(1).cpu().numpy())
        targets.append(labels.numpy())
    return np.concatenate(targets), np.concatenate(predictions)


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
    axes.set_title("Confusion matrix (row-normalised)")
    figure.colorbar(image, ax=axes, fraction=0.046)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    print(f"Confusion matrix -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the disease classifier")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "checkpoints" / "resnet50_plantvillage.pt")
    parser.add_argument("--config", type=Path, default=ROOT / "ml" / "configs" / "resnet50_plantvillage.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "evaluation")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"No checkpoint at {args.checkpoint}. Run ml/train.py first.")

    from sklearn.metrics import classification_report, confusion_matrix

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) if args.config.exists() else {}
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes = blob["classes"]

    data_dir = Path(config.get("data_dir", "data/raw/plantvillage"))
    data_dir = data_dir if data_dir.is_absolute() else ROOT / data_dir
    _, val_loader, folder_classes = build_loaders(config, data_dir)

    if folder_classes != classes:
        # A silent mismatch here would make every metric meaningless.
        raise SystemExit(
            "Class order in the dataset folder does not match the checkpoint.\n"
            f"  checkpoint: {classes[:3]}...\n  folder:     {folder_classes[:3]}..."
        )

    model = build_model(num_classes=len(classes), pretrained=False)
    model.load_state_dict(blob["model_state"])
    model.to(device)

    truth, predicted = collect_predictions(model, val_loader, device)
    accuracy = float((truth == predicted).mean())
    report = classification_report(
        truth, predicted, target_names=classes, output_dict=True, zero_division=0
    )
    print(f"\nOverall accuracy: {accuracy:.4f}  (n={len(truth)})")
    print(f"Macro F1:         {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1:      {report['weighted avg']['f1-score']:.4f}")

    per_class = sorted(
        ((name, report[name]["f1-score"], int(report[name]["support"])) for name in classes),
        key=lambda row: row[1],
    )
    print("\nWeakest 10 classes by F1:")
    for name, f1, support in per_class[:10]:
        print(f"  {f1:.3f}  (n={support:5d})  {name}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "classification_report.json").write_text(
        json.dumps({"accuracy": accuracy, "report": report}, indent=2), encoding="utf-8"
    )
    plot_confusion(
        confusion_matrix(truth, predicted, labels=range(len(classes))),
        classes,
        args.output_dir / "confusion_matrix.png",
    )
    print(f"Report -> {args.output_dir / 'classification_report.json'}")


if __name__ == "__main__":
    main()
