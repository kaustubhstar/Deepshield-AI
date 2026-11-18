from PIL import Image

from app.ml.faces import FaceDetector, face_covers_frame


def test_closeup_fraction_keeps_the_full_photo():
    assert face_covers_frame((10, 10, 80, 80), 100, 100) is True
    assert face_covers_frame((10, 10, 40, 40), 100, 100) is False


def test_closeup_face_is_not_recropped(monkeypatch):
    detector = FaceDetector()
    image = Image.new("RGB", (200, 200), (10, 20, 30))
    monkeypatch.setattr(detector, "detect_largest", lambda _bgr: (20, 20, 160, 160))
    cropped, found = detector.crop(image)
    assert found is True
    assert cropped.size == (200, 200)


def test_small_face_in_a_scene_is_still_cropped(monkeypatch):
    detector = FaceDetector()
    image = Image.new("RGB", (400, 400), (10, 20, 30))
    monkeypatch.setattr(detector, "detect_largest", lambda _bgr: (10, 10, 80, 80))
    cropped, found = detector.crop(image)
    assert found is True
    assert cropped.size[0] < 400
    assert cropped.size[1] < 400
