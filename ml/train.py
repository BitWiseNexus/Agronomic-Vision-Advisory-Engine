r"""Two-stage transfer learning for the PlantVillage disease classifier.

    .\.venv\Scripts\python.exe ml\train.py --config ml\configs\resnet50_plantvillage.yaml

Stage 1 freezes the ResNet-50 backbone and trains only the new 38-class head, so the
randomly-initialised head cannot destroy the pretrained features with large early
gradients. Stage 2 unfreezes `layer4` and fine-tunes at a tenth of the learning rate.

The checkpoint stores the class list alongside the weights. `app.vision.classifier`
reads it back, which is the only thing preventing a silent label-order mismatch between
training and inference.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "ml"))

from datasets import build_loaders  # noqa: E402

from app.vision.classifier import build_model  # noqa: E402


def load_config(path: Path | None, overrides: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if path is not None:
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config.update({key: value for key, value in overrides.items() if value is not None})
    return config


def set_trainable(model: nn.Module, stage: str, unfreeze_layer3: bool) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable: list[nn.Parameter] = list(model.fc.parameters())
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    if stage == "finetune":
        blocks = [model.layer4] + ([model.layer3] if unfreeze_layer3 else [])
        for block in blocks:
            for parameter in block.parameters():
                parameter.requires_grad = True
                trainable.append(parameter)
    return trainable


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: str) -> tuple[float, float]:
    model.eval()
    loss_sum, correct, seen = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(images)
        loss_sum += float(criterion(outputs, targets)) * targets.size(0)
        correct += int((outputs.argmax(1) == targets).sum())
        seen += targets.size(0)
    return loss_sum / max(seen, 1), correct / max(seen, 1)


def run_stage(
    stage: str,
    model: nn.Module,
    loaders: tuple,
    config: dict[str, Any],
    device: str,
    history: list[dict],
    best: dict,
) -> None:
    train_loader, val_loader = loaders
    epochs = int(config.get(f"epochs_{'head' if stage == 'head' else 'finetune'}", 3))
    lr = float(config.get(f"lr_{'head' if stage == 'head' else 'finetune'}", 1e-3))
    if epochs <= 0:
        return

    trainable = set_trainable(model, stage, bool(config.get("unfreeze_layer3", False)))
    optimiser = torch.optim.AdamW(
        trainable, lr=lr, weight_decay=float(config.get("weight_decay", 1e-4))
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config.get("label_smoothing", 0.05)))
    use_amp = bool(config.get("amp", True)) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    clip = float(config.get("grad_clip", 1.0))

    parameter_count = sum(p.numel() for p in trainable)
    print(f"\n=== Stage '{stage}' | {epochs} epochs | lr {lr:g} | {parameter_count:,} trainable params ===")

    patience = int(config.get("early_stop_patience", 3))
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.perf_counter()
        loss_sum, correct, seen = 0.0, 0, 0

        for step, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            if clip > 0:
                scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(trainable, clip)
            scaler.step(optimiser)
            scaler.update()

            loss_sum += float(loss) * targets.size(0)
            correct += int((outputs.argmax(1) == targets).sum())
            seen += targets.size(0)
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {loss_sum / seen:.4f}", flush=True)

        scheduler.step()
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        entry = {
            "stage": stage,
            "epoch": epoch,
            "train_loss": round(loss_sum / max(seen, 1), 4),
            "train_acc": round(correct / max(seen, 1), 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_accuracy, 4),
            "seconds": round(time.perf_counter() - started, 1),
        }
        history.append(entry)
        print(
            f"  epoch {epoch}: train {entry['train_acc']:.3f} / val {entry['val_acc']:.3f} "
            f"(loss {entry['val_loss']:.4f}) in {entry['seconds']}s"
        )

        if val_accuracy > best["val_acc"]:
            best.update(val_acc=val_accuracy, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            stale = 0
            print(f"  new best: {val_accuracy:.4f}")
        else:
            stale += 1
            if stale >= patience:
                print(f"  early stop — no improvement for {patience} epochs")
                return


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PlantVillage ResNet-50 classifier")
    parser.add_argument("--config", type=Path, default=ROOT / "ml" / "configs" / "resnet50_plantvillage.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--epochs-head", type=int, default=None)
    parser.add_argument("--epochs-finetune", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = load_config(
        args.config if args.config and args.config.exists() else None,
        {
            "data_dir": args.data_dir,
            "output_path": args.output_path,
            "epochs_head": args.epochs_head,
            "epochs_finetune": args.epochs_finetune,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(config.get("seed", 42)))

    data_dir = Path(config.get("data_dir", "data/raw/plantvillage"))
    data_dir = data_dir if data_dir.is_absolute() else ROOT / data_dir
    train_loader, val_loader, classes = build_loaders(config, data_dir)

    print(f"Device: {device}")
    print(f"Classes: {len(classes)}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = build_model(num_classes=len(classes), pretrained=True).to(device)
    history: list[dict] = []
    best = {"val_acc": 0.0, "state": None}

    run_stage("head", model, (train_loader, val_loader), config, device, history, best)
    run_stage("finetune", model, (train_loader, val_loader), config, device, history, best)

    if best["state"] is None:
        print("No epochs completed — nothing to save.")
        return

    output = Path(config.get("output_path", "models/checkpoints/resnet50_plantvillage.pt"))
    output = output if output.is_absolute() else ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state": best["state"],
            "classes": classes,
            "val_acc": best["val_acc"],
            "arch": "resnet50",
            "image_size": int(config.get("image_size", 224)),
            "history": history,
        },
        output,
    )
    (output.parent / f"{output.stem}_history.json").write_text(
        json.dumps({"best_val_acc": best["val_acc"], "history": history}, indent=2), encoding="utf-8"
    )
    print(f"\nSaved {output} (best val acc {best['val_acc']:.4f})")


if __name__ == "__main__":
    main()
