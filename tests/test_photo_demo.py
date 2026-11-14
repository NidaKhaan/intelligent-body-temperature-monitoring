from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "frontend" / "app.py")


def test_photo_demo_runs_and_moves_with_slider():
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception
    at.slider(key="sim_c").set_value(39.0).run()
    assert not at.exception
