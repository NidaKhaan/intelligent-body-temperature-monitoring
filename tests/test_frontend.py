from pathlib import Path

import pandas as pd

from frontend.helpers import synthetic_frame, reading_row
from src.pipeline.screener import ThermalScreener, ScreenerConfig, Calibration

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "frontend" / "app.py")
DATA = ROOT / "frontend" / "data" / "fda_flir_features.csv"


def test_synthetic_frame_box_and_max():
    t, box = synthetic_frame(36.7, 160)
    x0, y0, x1, y1 = box
    assert 0 <= x0 < x1 <= t.shape[1] and 0 <= y0 < y1 <= t.shape[0]
    assert abs(t[y0:y1, x0:x1].max() - 36.7) < 1e-9


def test_screener_alerts_only_above_threshold():
    cfg = ScreenerConfig(calibration=Calibration(1.0, 0.0), alert_threshold_c=37.5)
    for fmax, expect in [(37.0, False), (38.0, True)]:
        t, box = synthetic_frame(fmax, 160)
        assert ThermalScreener(lambda g: [box], cfg).process_celsius(t)[0].alert is expect


def test_reading_row_has_no_image_data():
    t, box = synthetic_frame(36.5, 160)
    r = ThermalScreener(lambda g: [box]).process_celsius(t)[0]
    assert set(reading_row("x", r, "t")) == {"time", "source", "status", "face_max_c", "roi_height_px", "alert"}


def test_fda_bundle_is_deidentified():
    df = pd.read_csv(DATA)
    assert "oral_c" in df.columns and len(df) > 900
    assert not {"SubjectID", "Gender", "Age", "Ethnicity"} & set(df.columns)


def test_app_default_run_shows_fda_metrics():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert len(at.metric) >= 4


def test_fda_threshold_changes_sensitivity():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=60).run()
    sens = lambda: [m.value for m in at.metric if m.label == "Sensitivity"][0]
    at.slider(key="thr_c").set_value(37.0).run(); low = sens()
    at.slider(key="thr_c").set_value(37.8).run(); high = sens()
    assert not at.exception and low != high
