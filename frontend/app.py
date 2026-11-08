import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # repo root, for `streamlit run frontend/app.py`

import os
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2 ** 25))
import datetime as dt
import cv2, numpy as np, pandas as pd, streamlit as st
from frontend.helpers import MODE_16, MODE_8, MODE_SYN, synthetic_frame, render_overlay, reading_row
from src.pipeline.screener import Calibration, ScreenerConfig, ThermalScreener
from src.temperature.radiometric import raw_to_celsius

MAX_BYTES = 10 * 1024 * 1024
st.set_page_config(page_title="Thermal Screening Demo", layout="wide")
st.title("Thermal face-temperature screening (research demo)")
st.warning("Research prototype, not a medical device. Face surface temperature is not core body temperature. "
           "The 0.01 K/count scale is an assumption checked on one dataset only; accuracy is not validated for your camera. "
           "Images are processed in memory and not stored.")


@st.cache_resource
def get_detector():
    from src.detection.face_detector import mediapipe_face_detector
    return mediapipe_face_detector()


def decode(upload):
    data = upload.getvalue()
    if len(data) > MAX_BYTES:
        st.error("File too large (max 10 MB)."); return None
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        st.error("Could not decode the image."); return None
    return img


if "history" not in st.session_state:
    st.session_state.history = []

mode = st.sidebar.radio("Input", [MODE_16, MODE_8, MODE_SYN], key="mode")
thr = st.sidebar.number_input("Alert threshold on measured face max (°C, uncalibrated)", 30.0, 45.0, 37.5, 0.1, key="thr")
min_h = st.sidebar.number_input("Minimum face height for a reliable reading (px)", 20, 400, 100, 5, key="minh")
sim = st.sidebar.slider("Simulated ΔT added to the frame (°C)", 0.0, 3.0, 0.0, 0.1, key="sim")
if sim > 0:
    st.error(f"SIMULATION: +{sim:.1f} °C added to the frame. This is not a measurement.")


def screen(temp_c, detector):
    cfg = ScreenerConfig(min_face_height_px=int(min_h), alert_threshold_c=float(thr), calibration=Calibration(1.0, 0.0))
    return ThermalScreener(detector, cfg).process_celsius(temp_c)


temp_c, readings, source = None, None, ""
if mode == MODE_16:
    up = st.file_uploader("16-bit single-channel radiometric TIFF/PNG", type=["tif", "tiff", "png"], key="up16")
    if up is None:
        st.info("Upload a 16-bit radiometric image (assumed 0.01 K per count).")
    else:
        img = decode(up)
        if img is not None:
            if img.ndim != 2 or img.dtype != np.uint16:
                st.error("Expected a single-channel 16-bit (uint16) image.")
            else:
                temp_c = raw_to_celsius(img) + sim
                readings, source = screen(temp_c, get_detector()), "upload_16bit"
elif mode == MODE_8:
    up = st.file_uploader("8-bit thermal image (grayscale or colorized)", type=["png", "jpg", "jpeg", "tif", "tiff"], key="up8")
    if up is None:
        st.info("Face detection only: an 8-bit image carries no temperature information.")
    else:
        img = decode(up)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            if gray.dtype != np.uint8:
                st.error("Expected an 8-bit image.")
            else:
                vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                boxes = get_detector()(gray)
                for x0, y0, x1, y1 in boxes:
                    cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 255), 2)
                st.image(vis, caption=f"{len(boxes)} face(s) detected. No temperature available.")
else:
    st.caption("Synthetic frame: a warm rectangle and one hot pixel. Tests ROI, status and alert logic without a face detector.")
    fmax = st.slider("Hottest pixel in face (°C)", 33.0, 40.0, 36.5, 0.1, key="fmax")
    fh = st.slider("Face height (px)", 20, 300, 160, 5, key="fh")
    temp_c, box = synthetic_frame(fmax + sim, fh)
    readings, source = screen(temp_c, lambda g: [box]), "synthetic"

if readings:
    left, right = st.columns([3, 2])
    left.image(render_overlay(temp_c, readings), caption="Colormap: 20 to 38 °C (white box = reading, orange = unreliable, red = alert)")
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [reading_row(source, r, now) for r in readings]
    right.dataframe(pd.DataFrame(rows), use_container_width=True)
    if any(r.alert for r in readings):
        right.error("Alert: measured face max is at or above the threshold.")
    elif any(r.status == "unreliable_roi_small" for r in readings):
        right.warning("Face too small for a reliable reading at this resolution.")
    if right.button("Log this reading", key="log"):
        st.session_state.history.extend(rows)

if st.session_state.history:
    st.subheader("Session history (in memory only)")
    hist = pd.DataFrame(st.session_state.history)
    st.dataframe(hist, use_container_width=True)
    st.download_button("Download report (CSV)", hist.to_csv(index=False).encode(), "screening_report.csv", "text/csv")
    if st.button("Clear history", key="clear"):
        st.session_state.history = []
        st.rerun()
