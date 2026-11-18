from pathlib import Path

import numpy as np

from app.ml.video import sample_frame_indices


def test_sample_frame_indices_spread():
    indices = sample_frame_indices(100, 5)
    assert indices[0] == 0
    assert indices[-1] == 99
    assert len(indices) == 5


def test_sample_empty():
    assert sample_frame_indices(0, 8) == []


def test_video_invalid_extension(client_without_model):
    response = client_without_model.post(
        "/api/detect/video",
        files={"file": ("clip.txt", b"abc", "text/plain")},
    )
    assert response.status_code == 415


def test_temp_file_cleanup(tmp_path: Path):
    from app.core.security import temp_media_file

    with temp_media_file(".mp4") as path:
        path.write_bytes(b"00")
        assert path.exists()
        stored = path
    assert not stored.exists()


def test_median_aggregation_unit():
    scores = np.array([0.2, 0.8, 0.9])
    assert float(np.median(scores)) == 0.8
