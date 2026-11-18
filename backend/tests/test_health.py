def test_health_ok(client_without_model):
    response = client_without_model.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "DeepShield AI"


def test_ready_false_without_checkpoint(client_without_model):
    response = client_without_model.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is False
    assert body["model_loaded"] is False
    assert body["checkpoint_present"] is False
    assert body["reason"]


def test_model_info_explains_missing_checkpoint(client_without_model):
    response = client_without_model.get("/model-info")
    body = response.json()
    assert body["loaded"] is False
    assert any("checkpoint" in note.lower() or "not" in note.lower() for note in body["notes"])


def test_cors_allows_localhost_and_loopback(monkeypatch):
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    get_settings.cache_clear()
    origins = get_settings().cors_origin_list
    get_settings.cache_clear()
    assert "http://localhost:3000" in origins
    assert "http://127.0.0.1:3000" in origins
