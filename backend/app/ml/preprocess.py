"""Image decoding and model preprocessing."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

from app.ml.metadata import ModelMetadata


class MediaError(ValueError):
    """Raised when media cannot be decoded or fails validation."""


def decode_image_bytes(data: bytes, max_pixels: int, min_side: int) -> Image.Image:
    try:
        image = Image.open(BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise MediaError("The uploaded file is not a readable image.") from exc
    image = ImageOps.exif_transpose(image)
    if image.mode not in ("RGB", "RGBA", "L"):
        image = image.convert("RGB")
    elif image.mode == "RGBA":
        background = Image.new("RGB", image.size, (0, 0, 0))
        background.paste(image, mask=image.split()[3])
        image = background
    elif image.mode == "L":
        image = image.convert("RGB")
    width, height = image.size
    if width * height > max_pixels:
        raise MediaError("Image dimensions exceed the configured pixel limit.")
    if min(width, height) < min_side:
        raise MediaError(f"Image is too small. Minimum side length is {min_side}px.")
    return image


def image_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return rgb[:, :, ::-1].copy()


def build_transform(metadata: ModelMetadata):
    return transforms.Compose(
        [
            transforms.Resize((metadata.input_size, metadata.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=metadata.image_mean, std=metadata.image_std),
        ]
    )
