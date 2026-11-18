import time


def test_unknown_job_404(client_without_model):
    response = client_without_model.get("/api/detect/video/jobs/not-a-real-id")
    assert response.status_code == 404


def test_video_job_without_checkpoint(client_without_model):
    response = client_without_model.post(
        "/api/detect/video/jobs",
        files={"file": ("clip.mp4", b"not-a-video", "video/mp4")},
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    body = None
    for _ in range(40):
        poll = client_without_model.get(f"/api/detect/video/jobs/{job_id}")
        assert poll.status_code == 200
        body = poll.json()
        if body["job_status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert body is not None
    assert body["job_status"] in {"completed", "failed"}
    if body["job_status"] == "completed":
        assert body["prediction"] == "analysis_unavailable"
        assert body["score"] is None
