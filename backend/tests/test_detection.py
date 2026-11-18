from io import BytesIO

from PIL import Image


def test_image_unavailable_without_checkpoint(client_without_model, rgb_png_bytes):
    response = client_without_model.post(
        "/api/detect/image",
        files={"file": ("face.png", rgb_png_bytes, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["prediction"] == "analysis_unavailable"
    assert body["score"] is None


def test_oversized_image_rejected(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from app.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("MAX_IMAGE_BYTES", "64")
    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/api/detect/image",
            files={"file": ("big.png", b"x" * 128, "image/png")},
        )
    get_settings.cache_clear()
    assert response.status_code == 413


def test_invalid_extension_rejected(client_without_model):
    response = client_without_model.post(
        "/api/detect/image",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 415


def test_corrupted_image_rejected(client_with_fixture_model):
    response = client_with_fixture_model.post(
        "/api/detect/image",
        files={"file": ("bad.png", b"not-an-image", "image/png")},
    )
    assert response.status_code == 400


def test_fixture_model_runs_inference(client_with_fixture_model, rgb_png_bytes):
    response = client_with_fixture_model.post(
        "/api/detect/image",
        files={"file": ("face.png", rgb_png_bytes, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["prediction"] in {"likely_manipulated", "likely_authentic"}
    assert body["verdict"] in {"REAL", "FAKE"}
    assert body["score"] is not None
    assert 0.0 <= body["score"] <= 1.0
    assert body["confidence_percent"] is not None
    assert 50.0 <= body["confidence_percent"] <= 100.0
    if body["verdict"] == "FAKE":
        assert body["score"] >= 0.5
        assert body["confidence_percent"] == round(body["score"] * 100, 1)
    else:
        assert body["score"] < 0.5
        assert body["confidence_percent"] == round((1 - body["score"]) * 100, 1)
    assert body["explanation"]
    assert body["score_type"] == "model_score"
    assert body["processing_time_ms"] >= 0
    assert body["score"] != 0.97
    assert body["confidence_percent"] != 97


def test_empty_upload_rejected(client_without_model):
    response = client_without_model.post(
        "/api/detect/image",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 400


def test_frame_endpoint_matches_image_contract(client_with_fixture_model, rgb_png_bytes):
    response = client_with_fixture_model.post(
        "/api/detect/frame",
        files={"file": ("frame.png", rgb_png_bytes, "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "completed"


def _tiny_jpeg() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (64, 64), color=(10, 10, 10)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_jpeg_accepted(client_with_fixture_model):
    response = client_with_fixture_model.post(
        "/api/detect/image",
        files={"file": ("a.jpg", _tiny_jpeg(), "image/jpeg")},
    )
    assert response.status_code == 200
