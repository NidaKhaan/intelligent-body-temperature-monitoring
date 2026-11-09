import numpy as np
import pytest

from src.detection import face_detector as fdm


def test_ensure_model_uses_existing_file(tmp_path):
    p = tmp_path / "m.tflite"
    p.write_bytes(b"x" * 20000)
    assert fdm.ensure_model(p) == p


def test_ensure_model_rejects_truncated_download(tmp_path, monkeypatch):
    class R:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"x"
    monkeypatch.setattr(fdm.urllib.request, "urlopen", lambda *a, **k: R())
    with pytest.raises(RuntimeError):
        fdm.ensure_model(tmp_path / "m.tflite")
    assert not (tmp_path / "m.tflite").exists()


def test_real_detector_runs_and_finds_no_face_in_blank_frame():
    try:
        detect = fdm.mediapipe_face_detector()
    except Exception as e:          # model download or mediapipe unavailable offline
        pytest.skip(f"face model or mediapipe unavailable: {e!r}")
    assert detect(np.zeros((200, 300), np.uint8)) == []
