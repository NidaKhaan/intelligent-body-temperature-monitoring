from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "frontend" / "app.py")


def test_subject_explorer_runs_for_different_subjects():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    for n in (1, 50, 500):
        at.slider(key="subj").set_value(n).run()
        assert not at.exception
    labels = {m.label for m in at.metric}
    assert {"Camera reading (face max)", "Calibrated estimate", "Oral thermometer"} <= labels
