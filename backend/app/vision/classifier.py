"""ResNet-50 disease classifier.

Loads a fine-tuned checkpoint when one exists. When it does not, the model still builds
(ImageNet backbone + random 38-class head) and every prediction is stamped
`checkpoint_loaded=False` / `confidence_reliable=False`. The pipeline therefore runs on a
fresh clone, and nothing downstream can mistake a placeholder for a real diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.schemas.diagnosis import ClassPrediction, Diagnosis
from app.vision import labels as label_utils
from app.vision.preprocess import to_classifier_tensor

log = get_logger(__name__)


def build_model(num_classes: int = label_utils.NUM_CLASSES, pretrained: bool = True):
    """torchvision ResNet-50 with the classification head swapped for our taxonomy."""
    import torch.nn as nn
    from torchvision.models import ResNet50_Weights, resnet50

    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@dataclass
class ClassifierStatus:
    checkpoint_loaded: bool
    checkpoint_path: str
    device: str
    note: str = ""


class DiseaseClassifier:
    def __init__(self, settings: Settings | None = None, pretrained_backbone: bool = True) -> None:
        import torch

        self.settings = settings or get_settings()
        self.classes = list(label_utils.PLANTVILLAGE_CLASSES)
        self.device = self.settings.resolve_device()
        self._torch = torch

        self.model = build_model(len(self.classes), pretrained=pretrained_backbone)
        self.status = self._load_checkpoint()
        self.model.to(self.device).eval()

    # -- construction helpers ------------------------------------------------
    def _load_checkpoint(self) -> ClassifierStatus:
        path = self.settings.checkpoint_file
        if not path.exists():
            log.warning(
                "No checkpoint at %s — running with an untrained head. "
                "Predictions are placeholders until ml/train.py has been run.",
                path,
            )
            return ClassifierStatus(
                False, str(path), self.device, "checkpoint missing; predictions are placeholders"
            )
        try:
            blob = self._torch.load(path, map_location="cpu", weights_only=False)
            state = blob.get("model_state", blob) if isinstance(blob, dict) else blob
            saved_classes = blob.get("classes") if isinstance(blob, dict) else None
            if saved_classes:
                self.classes = list(saved_classes)
                if len(self.classes) != self.model.fc.out_features:
                    self.model = build_model(len(self.classes), pretrained=False)
            self.model.load_state_dict(state)
            log.info("Loaded checkpoint %s (%d classes)", path.name, len(self.classes))
            return ClassifierStatus(True, str(path), self.device)
        except Exception as exc:  # a corrupt checkpoint must not take the API down
            log.error("Checkpoint at %s failed to load (%s); using untrained head", path, exc)
            return ClassifierStatus(False, str(path), self.device, f"load failed: {exc}")

    # -- inference -----------------------------------------------------------
    def predict_proba(self, bgr: np.ndarray) -> np.ndarray:
        tensor = to_classifier_tensor(bgr).to(self.device)
        with self._torch.no_grad():
            logits = self.model(tensor)
            probs = self._torch.softmax(logits, dim=1)[0]
        return probs.detach().cpu().numpy()

    def predict(self, bgr: np.ndarray, top_k: int = 3) -> Diagnosis:
        probs = self.predict_proba(bgr)
        order = np.argsort(probs)[::-1][: max(top_k, 1)]

        predictions: list[ClassPrediction] = []
        for index in order:
            label = self.classes[int(index)]
            crop, disease = label_utils.split_label(label)
            predictions.append(
                ClassPrediction(
                    label=label,
                    crop=crop,
                    disease=disease,
                    confidence=round(float(probs[int(index)]), 4),
                )
            )

        top = predictions[0]
        reliable = (
            self.status.checkpoint_loaded
            and top.confidence >= self.settings.classifier_confidence_floor
        )
        return Diagnosis(
            top=top,
            alternatives=predictions[1:],
            checkpoint_loaded=self.status.checkpoint_loaded,
            confidence_reliable=reliable,
            is_healthy=label_utils.is_healthy(top.label),
        )
