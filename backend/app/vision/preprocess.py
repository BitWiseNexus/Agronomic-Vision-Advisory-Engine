"""Image decode / resize / normalisation shared by the classifier and the severity engine."""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASSIFIER_SIZE = 224
SEVERITY_MAX_SIDE = 768


def decode_image(data: bytes) -> np.ndarray:
    """Bytes -> BGR uint8 array. Raises ValueError on anything undecodable."""
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image — expected JPEG/PNG/WebP bytes.")
    return image


def read_image(path: str) -> np.ndarray:
    with open(path, "rb") as handle:
        return decode_image(handle.read())


def resize_max_side(image: np.ndarray, max_side: int = SEVERITY_MAX_SIDE) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return cv2.resize(
        image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA
    )


def background_white_balance(image: np.ndarray, subject_mask: np.ndarray) -> np.ndarray:
    """Neutralise the illuminant so the severity index is comparable between a noon shot
    and an overcast one — estimating the illuminant from the BACKGROUND only.

    Why not the two textbook estimators:

    * **Grey-world** (equalise the channel means) fails because a frame dominated by one
      hue drags the means with it. It "corrects" the leaf's own greenness by tinting
      everything else, which lifts a neutral grey background to saturation ~58 — above the
      leaf-mask threshold — and segmentation then swallows the whole frame.
    * **White-patch** (gains from a high percentile) fails on the same image for a
      different reason: bright yellow chlorotic tissue outranks the background in the red
      channel, so the lesions become the reference and the background is tinted again.

    Both break because the subject is the strongest colour in the frame. Estimating from
    the non-subject pixels sidesteps that entirely: the background is the one region we can
    reasonably assume is achromatic. If it is too small, or is itself strongly coloured
    (soil, foliage), the assumption does not hold and the image is returned untouched —
    a wrong correction is worse than none.
    """
    background = subject_mask == 0
    pixel_count = int(background.sum())
    if pixel_count < 0.05 * subject_mask.size:
        return image

    samples = image[background].astype(np.float32)
    means = samples.mean(axis=0)
    reference = float(means.max())
    if reference <= 1e-6:
        return image
    # A strongly coloured background is not a valid white reference.
    if (reference - float(means.min())) / reference > 0.35:
        return image

    gains = np.clip(reference / np.maximum(means, 1e-6), 1.0, 1.8)
    return np.clip(image.astype(np.float32) * gains, 0, 255).astype(np.uint8)


def denoise(image: np.ndarray) -> np.ndarray:
    """Bilateral filter: kills JPEG speckle without smearing lesion boundaries."""
    return cv2.bilateralFilter(image, d=7, sigmaColor=45, sigmaSpace=45)


def prepare_for_severity(image: np.ndarray) -> np.ndarray:
    """Resize and denoise only. White balance happens *after* segmentation — see
    `background_white_balance` for why it cannot happen before."""
    return denoise(resize_max_side(image, SEVERITY_MAX_SIDE))


def to_classifier_tensor(image: np.ndarray):
    """BGR uint8 -> normalised NCHW float32 torch tensor of shape (1, 3, 224, 224)."""
    import torch

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize-shortest-side then centre crop, matching torchvision eval transforms.
    height, width = rgb.shape[:2]
    scale = 256 / min(height, width)
    resized = cv2.resize(
        rgb, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA
    )
    rh, rw = resized.shape[:2]
    top = max((rh - CLASSIFIER_SIZE) // 2, 0)
    left = max((rw - CLASSIFIER_SIZE) // 2, 0)
    crop = resized[top : top + CLASSIFIER_SIZE, left : left + CLASSIFIER_SIZE]
    if crop.shape[:2] != (CLASSIFIER_SIZE, CLASSIFIER_SIZE):
        crop = cv2.resize(crop, (CLASSIFIER_SIZE, CLASSIFIER_SIZE), interpolation=cv2.INTER_AREA)

    array = crop.astype(np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)


def encode_png_b64(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("PNG encoding failed")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def to_pil(image: np.ndarray):
    """For matplotlib / notebook display without the BGR->RGB footgun."""
    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(encode_png_b64(image)))).convert("RGB")
