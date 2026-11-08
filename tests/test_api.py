import cv2, numpy as np, pytest
from fastapi.testclient import TestClient
import backend.app as be
from src.pipeline.screener import ThermalScreener, ScreenerConfig, Calibration

def screener(cal=None):
    return ThermalScreener(lambda g: [(100, 50, 200, 150)], ScreenerConfig(calibration=cal))

def frame16(max_c=36.5):
    t = np.full((200, 300), 24.0); t[50:150, 100:200] = 34.0; t[80, 150] = max_c
    return np.round((t + 273.15) / 0.01).astype(np.uint16)

def post(client, arr):
    ok, buf = cv2.imencode(".tiff", arr)
    return client.post("/screen", files={"file": ("f.tiff", buf.tobytes(), "image/tiff")})

@pytest.fixture
def make_client():
    def _make(s):
        be.app.dependency_overrides[be.get_screener] = lambda: s
        return TestClient(be.app)
    yield _make
    be.app.dependency_overrides.clear()

def test_health(make_client):
    assert make_client(screener()).get("/health").json() == {"status": "ok"}

def test_uncalibrated_measures_without_alert(make_client):
    j = post(make_client(screener()), frame16(36.5)).json()
    r = j["readings"][0]
    assert j["calibrated"] is False and r["status"] == "ok"
    assert abs(r["face_max_c"] - 36.5) < 0.01 and r["alert"] is None

def test_calibrated_alert(make_client):
    c = make_client(screener(Calibration(1.0, 1.0)))
    assert post(c, frame16(36.0)).json()["readings"][0]["alert"] is False
    assert post(c, frame16(36.6)).json()["readings"][0]["alert"] is True

def test_rejects_8bit_image(make_client):
    assert post(make_client(screener()), np.zeros((50, 50), np.uint8)).status_code == 422

def test_rejects_garbage(make_client):
    r = make_client(screener()).post("/screen", files={"file": ("x.tiff", b"not an image", "image/tiff")})
    assert r.status_code == 422

def test_rejects_oversize(make_client, monkeypatch):
    monkeypatch.setattr(be, "MAX_BYTES", 10)
    assert post(make_client(screener()), frame16()).status_code == 413
