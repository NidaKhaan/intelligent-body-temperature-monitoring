from pathlib import Path
from frontend.helpers import MODE_SYN, synthetic_frame, reading_row
from src.pipeline.screener import ThermalScreener, ScreenerConfig, Calibration

def test_synthetic_frame_box_and_max():
    t, box = synthetic_frame(36.7, 160)
    x0, y0, x1, y1 = box
    assert 0 <= x0 < x1 <= t.shape[1] and 0 <= y0 < y1 <= t.shape[0]
    assert abs(t[y0:y1, x0:x1].max() - 36.7) < 1e-9

def test_screener_on_synthetic_alerts_only_above_threshold():
    cfg = ScreenerConfig(calibration=Calibration(1.0, 0.0), alert_threshold_c=37.5)
    for fmax, expect in [(37.0, False), (38.0, True)]:
        t, box = synthetic_frame(fmax, 160)
        assert ThermalScreener(lambda g: [box], cfg).process_celsius(t)[0].alert is expect

def test_reading_row_has_no_image_data():
    t, box = synthetic_frame(36.5, 160)
    r = ThermalScreener(lambda g: [box]).process_celsius(t)[0]
    assert set(reading_row("synthetic", r, "t")) == {"time", "source", "status", "face_max_c", "roi_height_px", "alert"}

def test_app_runs_default_and_synthetic():
    from streamlit.testing.v1 import AppTest
    app = str(Path(__file__).resolve().parents[1] / "frontend" / "app.py")
    at = AppTest.from_file(app, default_timeout=30).run()
    assert not at.exception
    at.radio(key="mode").set_value(MODE_SYN).run()
    assert not at.exception and len(at.dataframe) >= 1
