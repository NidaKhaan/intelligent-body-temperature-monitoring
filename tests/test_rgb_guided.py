import numpy as np

from src.alignment.registration import HomographyRegistration
from src.pipeline.rgb_guided import RgbGuidedScreener
from src.pipeline.screener import ScreenerConfig, Calibration

REG = HomographyRegistration(np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]]))   # RGB 400x300 -> thermal 200x150
RGB = np.zeros((300, 400, 3), np.uint8)
BOX = (100, 60, 300, 260)                                                      # maps to thermal (50, 30, 150, 130)


def thermal():
    t = np.full((150, 200), 24.0)
    t[30:130, 50:150] = 34.0
    t[40, 60] = 36.4      # hottest pixel inside the mapped face box
    t[5, 5] = 40.0        # hotter pixel outside it: must be ignored
    return t


def test_maps_box_and_ignores_hot_pixel_outside():
    r = RgbGuidedScreener(lambda im: [BOX], REG).process_celsius(RGB, thermal())[0]
    assert r.box == (50, 30, 150, 130) and abs(r.face_max_c - 36.4) < 1e-9 and r.status == "ok"


def test_small_roi_is_flagged_without_alert():
    cfg = ScreenerConfig(calibration=Calibration(1.0, 5.0))
    r = RgbGuidedScreener(lambda im: [(100, 60, 180, 120)], REG, cfg).process_celsius(RGB, thermal())[0]
    assert r.status == "unreliable_roi_small" and r.alert is None


def test_calibrated_alert():
    cfg = ScreenerConfig(calibration=Calibration(1.0, 1.5), alert_threshold_c=37.5)
    assert RgbGuidedScreener(lambda im: [BOX], REG, cfg).process_celsius(RGB, thermal())[0].alert is True


def test_no_face_and_invalid_input():
    s = RgbGuidedScreener(lambda im: [], REG)
    assert s.process_celsius(RGB, thermal())[0].status == "no_face"
    assert s.process_celsius(np.zeros((10, 10), np.uint8), thermal())[0].status == "invalid_input"
    assert s.process_raw(RGB, np.zeros((150, 200), np.uint8))[0].status == "invalid_input"


def test_box_outside_thermal_frame_is_dropped():
    r = RgbGuidedScreener(lambda im: [(900, 900, 1000, 1000)], REG).process_celsius(RGB, thermal())[0]
    assert r.status == "no_face"
