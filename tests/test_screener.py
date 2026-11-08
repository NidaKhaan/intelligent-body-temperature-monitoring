import numpy as np
from src.pipeline.screener import ThermalScreener, ScreenerConfig, Calibration

def det(box):
    return lambda g: [box] if box else []

def frame(max_c=36.0):
    t = np.full((200, 300), 24.0); t[50:150, 100:200] = 34.0; t[80, 150] = max_c
    return t

BOX = (100, 50, 200, 150)   # 100 px tall

def test_no_face():
    r = ThermalScreener(det(None)).process_celsius(frame())[0]
    assert r.status == "no_face" and r.alert is None

def test_small_roi_flagged_not_alerted():
    cfg = ScreenerConfig(calibration=Calibration(1.0, 5.0))
    r = ThermalScreener(det((100, 50, 140, 90)), cfg).process_celsius(frame())[0]
    assert r.status == "unreliable_roi_small" and r.alert is None

def test_uncalibrated_measures_without_alert():
    r = ThermalScreener(det(BOX)).process_celsius(frame(36.5))[0]
    assert r.status == "ok" and abs(r.face_max_c - 36.5) < 1e-9 and r.alert is None

def test_alert_threshold():
    s = ThermalScreener(det(BOX), ScreenerConfig(calibration=Calibration(1.0, 1.0)))
    assert s.process_celsius(frame(36.0))[0].alert is False
    assert s.process_celsius(frame(36.6))[0].alert is True

def test_wrong_dtype_rejected():
    assert ThermalScreener(det(None)).process_raw(np.zeros((10, 10), np.uint8))[0].status == "invalid_input"

def test_nan_rejected():
    t = frame(); t[0, 0] = np.nan
    assert ThermalScreener(det((1, 1, 50, 50))).process_celsius(t)[0].status == "invalid_input"
