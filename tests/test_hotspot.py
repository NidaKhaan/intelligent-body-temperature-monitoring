import numpy as np

from src.alignment.registration import HomographyRegistration
from src.pipeline.rgb_guided import RgbGuidedScreener
from src.pipeline.screener import ThermalScreener


def test_hotspot_position_thermal_only():
    t = np.full((200, 300), 24.0); t[50:150, 100:200] = 34.0; t[80, 150] = 36.5
    r = ThermalScreener(lambda g: [(100, 50, 200, 150)]).process_celsius(t)[0]
    assert r.hotspot == (150, 80) and abs(r.face_max_c - 36.5) < 1e-9


def test_hotspot_position_rgb_guided():
    reg = HomographyRegistration(np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]]))
    t = np.full((150, 200), 24.0); t[30:130, 50:150] = 34.0; t[60, 90] = 36.1
    r = RgbGuidedScreener(lambda im: [(100, 60, 300, 260)], reg).process_celsius(np.zeros((300, 400, 3), np.uint8), t)[0]
    assert r.hotspot == (90, 60)
