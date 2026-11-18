"""OpenCV Haar-cascade face detection and cropping."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

# The installed model was trained on the whole face photo, resized to 128px.
# Recropping a picture that is already a close-up removes the border cues it learned.
CLOSEUP_FACE_FRACTION = 0.35


def face_covers_frame(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    minimum_fraction: float = CLOSEUP_FACE_FRACTION,
) -> bool:
    """True when the detected face is already most of the picture."""
    _x, _y, width, height = box
    area = image_width * image_height
    if area <= 0 or width <= 0 or height <= 0:
        return False
    return (width * height) / area >= minimum_fraction


class FaceDetector:
    def __init__(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError("Failed to load OpenCV Haar face cascade.")

    def detect_largest(self, bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        return int(x), int(y), int(w), int(h)

    def crop(self, image: Image.Image, expand: float = 0.25) -> tuple[Image.Image, bool]:
        bgr = np.array(image.convert("RGB"))[:, :, ::-1]
        box = self.detect_largest(bgr)
        if box is None:
            return image, False
        if face_covers_frame(box, image.width, image.height):
            return image, True
        x, y, w, h = box
        pad_x = int(w * expand)
        pad_y = int(h * expand)
        left = max(0, x - pad_x)
        top = max(0, y - pad_y)
        right = min(image.width, x + w + pad_x)
        bottom = min(image.height, y + h + pad_y)
        return image.crop((left, top, right, bottom)), True
