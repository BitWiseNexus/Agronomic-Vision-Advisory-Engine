"""PlantVillage taxonomy — 38 classes, the standard folder names from the dataset.

Order matters: it must match the alphabetical ImageFolder ordering used at training
time, otherwise a checkpoint's logits map to the wrong labels.
"""

from __future__ import annotations

PLANTVILLAGE_CLASSES: list[str] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

NUM_CLASSES = len(PLANTVILLAGE_CLASSES)


def _titleise(raw: str) -> str:
    cleaned = raw.replace("_", " ").replace("  ", " ").strip()
    return " ".join(part for part in cleaned.split() if part)


def split_label(label: str) -> tuple[str, str]:
    """'Tomato___Late_blight' -> ('Tomato', 'Late blight')."""
    if "___" in label:
        crop, disease = label.split("___", 1)
    else:
        crop, disease = label, "unknown"
    return _titleise(crop), _titleise(disease)


def is_healthy(label: str) -> bool:
    return label.lower().endswith("healthy")


def crop_names() -> list[str]:
    seen: dict[str, None] = {}
    for label in PLANTVILLAGE_CLASSES:
        seen.setdefault(split_label(label)[0], None)
    return list(seen)
