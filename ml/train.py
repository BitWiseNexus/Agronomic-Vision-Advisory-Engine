r"""Two-stage transfer learning for the PlantVillage disease classifier.

    .\.venv\Scripts\python.exe ml\train.py --config ml\configs\resnet50_plantvillage.yaml

Stage 1 freezes the ResNet-50 backbone and trains only the new 38-class head, so the
randomly-initialised head cannot destroy the pretrained features with large early
gradients. Stage 2 unfreezes `layer4` (and `layer3` when configured) and fine-tunes at a
tenth of the learning rate.

After training, a single temperature is fitted on the validation logits and stored in the
checkpoint. Fine-tuned networks are over-confident; without calibration the classifier's
"confidence" — and the reliability floor the advisor relies on — is not a probability.

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

# Set from the config in main(). Both default OFF: on a GTX 1650 (Turing without tensor
# cores) AMP + channels_last measured 8 img/s against 66 img/s for plain fp32 NCHW.
USE_AMP = False
MEMORY_FORMAT = torch.contiguous_format


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


def class_weights(counts, mode: str) -> torch.Tensor | None:
    """Loss weights against class imbalance (Potato___healthy has 152 images, Orange 5,507).
    'sqrt' is the gentle choice: full inverse frequency over-corrects and costs accuracy on
    the large classes for little gain on the small ones."""
    if mode not in {"sqrt", "inverse"}:
        return None
    freq = torch.as_tensor(counts, dtype=torch.float32).clamp(min=1)
    weights = freq.rsqrt() if mode == "sqrt" else 1.0 / freq
    return weights / weights.mean()


@torch.no_grad()
def collect_logits(model: nn.Module, loader, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits, targets = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True).to(memory_format=MEMORY_FORMAT)
        with torch.amp.autocast("cuda", enabled=USE_AMP and device == "cuda"):
            logits.append(model(images).float().cpu())
        targets.append(labels)
    return torch.cat(logits), torch.cat(targets)


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Temperature scaling (Guo et al., 2017): one scalar T minimising validation NLL of
    softmax(logits / T). It never changes the argmax, so accuracy is untouched — only the
    confidence values become honest."""
    log_t = torch.zeros(1, requires_grad=True)
    optimiser = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)
    nll = nn.CrossEntropyLoss()

    def closure():
        optimiser.zero_grad()
        loss = nll(logits / log_t.exp(), targets)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(log_t.exp())


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: str) -> tuple[float, float]:
    model.eval()
    loss_sum, correct, seen = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        images = images.to(memory_format=MEMORY_FORMAT)
        with torch.amp.autocast("cuda", enabled=USE_AMP and device == "cuda"):
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
    weights = config.get("_class_weights")
    criterion = nn.CrossEntropyLoss(
        weight=weights.to(device) if weights is not None else None,
        label_smoothing=float(config.get("label_smoothing", 0.05)),
    )
    use_amp = USE_AMP and device == "cuda"
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
            images = images.to(device, non_blocking=True).to(memory_format=MEMORY_FORMAT)
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

            loss_sum += float(loss.detach()) * targets.size(0)
            correct += int((outputs.argmax(1) == targets).sum())
            seen += targets.size(0)
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {loss_sum / seen:.4f}", flush=True)

        scheduler.step()
        # Validation loss is reported unweighted, so it stays comparable across settings.
        val_loss, val_accuracy = evaluate(model, val_loader, nn.CrossEntropyLoss(), device)
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
    torch.backends.cudnn.benchmark = True  # fixed 224x224 inputs: let cuDNN pick kernels once

    global USE_AMP, MEMORY_FORMAT
    USE_AMP = bool(config.get("amp", False))
    MEMORY_FORMAT = torch.channels_last if config.get("channels_last", False) else torch.contiguous_format

    data_dir = Path(config.get("data_dir", "data/raw/plantvillage"))
    data_dir = data_dir if data_dir.is_absolute() else ROOT / data_dir
    train_loader, val_loader, test_loader, classes, counts = build_loaders(config, data_dir)
    config["_class_weights"] = class_weights(counts, str(config.get("class_weighting", "none")))

    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"Classes: {len(classes)}")
    print(
        f"Images: train {len(train_loader.dataset):,} | val {len(val_loader.dataset):,} "
        f"| test {len(test_loader.dataset):,} (held out for ml/evaluate.py)"
    )

    model = build_model(num_classes=len(classes), pretrained=True).to(device)
    model = model.to(memory_format=MEMORY_FORMAT)
    history: list[dict] = []
    best = {"val_acc": 0.0, "state": None}

    run_stage("head", model, (train_loader, val_loader), config, device, history, best)
    run_stage("finetune", model, (train_loader, val_loader), config, device, history, best)

    if best["state"] is None:
        print("No epochs completed — nothing to save.")
        return

    model.load_state_dict(best["state"])
    val_logits, val_targets = collect_logits(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_targets)
    nll = nn.CrossEntropyLoss()
    print(
        f"\nCalibration: T = {temperature:.3f}  "
        f"(val NLL {float(nll(val_logits, val_targets)):.4f} -> "
        f"{float(nll(val_logits / temperature, val_targets)):.4f})"
    )

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
            "temperature": temperature,
            "split": {
                "seed": int(config.get("seed", 42)),
                "val": float(config.get("val_split", 0.15)),
                "test": float(config.get("test_split", 0.15)),
            },
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
