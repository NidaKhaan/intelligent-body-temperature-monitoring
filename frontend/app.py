import os
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2 ** 25))
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import datetime as dt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, cross_val_predict

from frontend.helpers import render_overlay, reading_row
from src.pipeline.screener import Calibration, ScreenerConfig, ThermalScreener
from src.temperature.radiometric import raw_to_celsius

MAX_BYTES = 10 * 1024 * 1024
DATA = pathlib.Path(__file__).resolve().parent / "data" / "fda_flir_features.csv"
FEATURES = {"Whole-face maximum": "T_Max", "Inner-canthus maximum": "canthiMax", "Forehead maximum": "T_FH_Max"}

st.set_page_config(page_title="Thermal screening", page_icon="🌡️", layout="wide")
st.title("Thermal face-temperature screening")
st.caption("Research prototype, not a medical device. Face surface temperature is not core body temperature. "
           "Images are processed in memory and never stored.")
tab_scan, tab_fda, tab_method = st.tabs(["Screen an image", "FDA calibration explorer", "Method and limits"])


@st.cache_resource
def get_detector():
    from src.detection.face_detector import mediapipe_face_detector
    return mediapipe_face_detector()


@st.cache_data
def load_fda():
    return pd.read_csv(DATA)


@st.cache_data
def calibrated(base: str, rnd: str):
    df = load_fda()
    x = df[[f"{base}{k}" for k in range(1, 5)]].mean(axis=1) if rnd == "Mean of 4" else df[f"{base}{rnd}"]
    y = df["oral_c"]; ok = x.notna() & y.notna()
    x, y = x[ok].to_numpy(), y[ok].to_numpy()
    p = cross_val_predict(LinearRegression(), x.reshape(-1, 1), y, cv=KFold(5, shuffle=True, random_state=42))
    return x, y, p


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

with tab_scan:
    left, right = st.columns([3, 2])
    with right:
        min_h = st.number_input("Minimum face height for a reliable reading (px)", 20, 400, 100, 5, key="minh")
        thr = st.number_input("Flag when the measured face maximum reaches (°C)", 30.0, 45.0, 37.5, 0.1, key="thr")
        st.caption("The flag applies to the measured surface temperature. No calibration is applied in this tab; "
                   "the next tab shows what calibration can and cannot do.")
    with left:
        up = st.file_uploader("Thermal image: 16-bit radiometric (temperature) or 8-bit (detection only)",
                              type=["tif", "tiff", "png", "jpg", "jpeg"], key="up")
    if up is None:
        st.info("Upload a 16-bit single-channel radiometric image (assumed 0.01 K per count) to get a reading. "
                "An 8-bit image only shows face detection because it carries no temperature.")
    else:
        img = decode(up)
        if img is not None and img.ndim == 2 and img.dtype == np.uint16:
            temp_c = raw_to_celsius(img)
            cfg = ScreenerConfig(min_face_height_px=int(min_h), alert_threshold_c=float(thr), calibration=Calibration(1.0, 0.0))
            readings = ThermalScreener(get_detector(), cfg).process_celsius(temp_c)
            a, b = st.columns([3, 2])
            a.image(render_overlay(temp_c, readings), caption="20 to 38 °C colormap. White box: reading, orange: unreliable, red: flagged")
            r0 = readings[0]; m1, m2 = b.columns(2)
            m1.metric("Face max (measured)", "n/a" if r0.face_max_c is None else f"{r0.face_max_c:.1f} °C")
            m2.metric("Status", r0.status.replace("_", " "))
            rows = [reading_row("upload_16bit", r, dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) for r in readings]
            b.dataframe(pd.DataFrame(rows))
            if any(r.alert for r in readings):
                b.error("Flagged: the measured face maximum is at or above the threshold.")
            elif any(r.status == "unreliable_roi_small" for r in readings):
                b.warning("Face too small for a reliable reading at this resolution.")
            if b.button("Log this reading", key="log"):
                st.session_state.history.extend(rows)
        elif img is not None and img.dtype == np.uint8:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB); boxes = get_detector()(gray)
            for x0, y0, x1, y1 in boxes:
                cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 255), 2)
            st.image(vis, caption=f"{len(boxes)} face(s) detected. No temperature available from an 8-bit image.")
        elif img is not None:
            st.error("Expected a single-channel 16-bit image (temperature) or an 8-bit image (detection only).")
    if st.session_state.history:
        st.subheader("Session history (memory only)")
        hist = pd.DataFrame(st.session_state.history); st.dataframe(hist)
        st.download_button("Download report (CSV)", hist.to_csv(index=False).encode(), "screening_report.csv", "text/csv")
        if st.button("Clear history", key="clear"):
            st.session_state.history = []; st.rerun()

with tab_fda:
    st.markdown("How well does a facial temperature predict an oral thermometer? Data: 1,020 volunteers, one FLIR camera, "
                "about 0.8 m (FDA / PhysioNet, CC0, de-identified). The features are the dataset's own extracted values, "
                "so this shows the calibration problem, not image processing.")
    c1, c2, c3 = st.columns(3)
    feat = c1.selectbox("Facial measurement", list(FEATURES), key="feat")
    rnd = c2.selectbox("Round", ["1", "2", "3", "4", "Mean of 4"], key="rnd")
    thr_c = c3.slider("Flag when the oral-equivalent estimate reaches (°C)", 36.8, 38.0, 37.5, 0.1, key="thr_c")
    x, y, pred = calibrated(FEATURES[feat], rnd)
    err = pred - y; fever = y >= 37.5; flag = pred >= thr_c
    sens = float((fever & flag).sum() / max(fever.sum(), 1)); spec = float((~fever & ~flag).sum() / max((~fever).sum(), 1))
    m = st.columns(4)
    m[0].metric("MAE", f"{np.abs(err).mean():.2f} °C"); m[1].metric("RMSE", f"{np.sqrt((err ** 2).mean()):.2f} °C")
    m[2].metric("Sensitivity", f"{sens:.2f}"); m[3].metric("Specificity", f"{spec:.2f}")
    p1, p2 = st.columns(2)
    fig, ax = plt.subplots(figsize=(5.5, 4.2)); ax.scatter(y, pred, s=6, alpha=.45)
    lim = [min(y.min(), pred.min()) - .1, max(y.max(), pred.max()) + .1]; ax.plot(lim, lim, "k--", lw=1)
    ax.axvline(37.5, c="r", lw=1); ax.axhline(thr_c, c="orange", lw=1)
    ax.set_xlabel("Oral temperature (°C)"); ax.set_ylabel("Calibrated estimate (°C)")
    ax.set_title("Estimate vs oral (red: fever = oral >= 37.5)", fontsize=10); p1.pyplot(fig); plt.close(fig)
    fpr, tpr, _ = roc_curve(fever, x)
    fig, ax = plt.subplots(figsize=(5.5, 4.2)); ax.plot(fpr, tpr); ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.scatter([1 - spec], [sens], c="orange", zorder=3, label="chosen threshold"); ax.legend()
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC for fever detection (AUC {roc_auc_score(fever, x):.2f})", fontsize=10); p2.pyplot(fig); plt.close(fig)
    st.caption("Calibration is a 5-fold cross-validated linear fit. Thresholds chosen here are tuned on the same data, "
               "so treat the operating point as exploratory. The orange line and dot show your threshold.")

with tab_method:
    st.markdown("""
**Pipeline.** 16-bit radiometric frame, conversion to °C (assumed 0.01 K per count), face detection on a fixed 20 to 38 °C window,
maximum temperature inside the face box, optional calibration. No face, a face under the minimum height, invalid input or no
calibration gives a status instead of an alert.

**What was measured.**

| Question | Answer from public data |
|---|---|
| Raw facial temperature vs oral | Facial maximum reads about 0.9 to 1.6 °C below oral |
| After linear calibration (FDA camera, 0.8 m, expert ROI) | MAE 0.21 to 0.29 °C, RMSE 0.27 to 0.39 °C |
| Fever subjects | Under-estimated; sensitivity about 0.76 at 37.5 °C |
| Distance 1 m to 6.6 m | Face maximum falls by 1.15 to 1.49 °C |
| Two different cameras | About 0.6 °C apart at similar distance |

**Limits.** The scale is an assumption. Calibrations do not transfer between cameras. Controlled indoor data, one face per frame.
Reliable readings need a face of about 100 px or more (roughly 1.4 m on this sensor). Not a medical device.
""")
