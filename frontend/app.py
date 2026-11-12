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
from matplotlib.patches import Ellipse, Rectangle
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, cross_val_predict

from frontend.helpers import render_overlay, reading_row
from src.pipeline.screener import Calibration, ScreenerConfig, ThermalScreener
from src.temperature.radiometric import raw_to_celsius
from src.temperature.regions import hotspot_region_shares

HERE = pathlib.Path(__file__).resolve().parent
MAX_BYTES = 10 * 1024 * 1024
DATA = HERE / "data" / "fda_flir_features.csv"
ICON, LOGO = HERE / "assets" / "logo_icon.png", HERE / "assets" / "logo_wide.png"
FEATURES = {"Whole-face maximum": "T_Max", "Inner-canthus maximum": "canthiMax", "Forehead maximum": "T_FH_Max"}

st.set_page_config(page_title="Thermal screening", page_icon=Image.open(ICON) if ICON.exists() else "🌡️", layout="wide")
try:
    if LOGO.exists():
        st.logo(Image.open(LOGO), icon_image=Image.open(ICON) if ICON.exists() else None)
except Exception:
    pass

st.title("Thermal face-temperature screening")
st.caption("Research prototype, not a medical device. Face surface temperature is not core body temperature. "
           "Images are processed in memory and never stored.")
tab_home, tab_real, tab_scan, tab_method = st.tabs(["Start here", "Real subjects (FDA)", "Screen an image", "Method and limits"])


@st.cache_resource
def get_detector():
    from src.detection.face_detector import mediapipe_face_detector
    return mediapipe_face_detector()


@st.cache_data
def load_fda():
    return pd.read_csv(DATA)


@st.cache_data
def subject_table():
    df = load_fda()
    cols = {"whole face": "T_Max", "inner canthi": "canthiMax", "forehead": "T_FH_Max", "mouth": "T_OR_Max"}
    out = pd.DataFrame({"oral": df["oral_c"]})
    for name, base in cols.items():
        out[name] = df[f"{base}1"]
    out = out.dropna().reset_index(drop=True)
    out["estimate"] = cross_val_predict(LinearRegression(), out[["whole face"]].to_numpy(), out["oral"].to_numpy(),
                                        cv=KFold(5, shuffle=True, random_state=42))
    return out


@st.cache_data
def region_shares():
    return hotspot_region_shares(load_fda(), rnd=1)


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


def draw_subject(r):
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.add_patch(Ellipse((0.42, 0.52), 0.56, 0.86, fill=False, lw=2.2, ec="#9ca3af"))
    boxes = {"whole face": ((0.16, 0.10, 0.52, 0.84), "#3b82f6"), "forehead": ((0.24, 0.72, 0.36, 0.13), "#22c55e"),
             "inner canthi": ((0.24, 0.52, 0.36, 0.09), "#ef4444"), "mouth": ((0.29, 0.27, 0.26, 0.08), "#6b7280")}
    top = r["whole face"]
    for i, (name, ((x, y, w, h), colour)) in enumerate(boxes.items()):
        ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=2, ec=colour))
        hot = name != "whole face" and abs(r[name] - top) <= 0.05
        ax.annotate(f"{name}: {r[name]:.1f} C" + ("  (hottest)" if hot else ""), xy=(x + w, y + h / 2), xytext=(0.78, 0.88 - i * 0.2),
                    fontsize=10, color=colour, va="center", arrowprops=dict(arrowstyle="-", color=colour, lw=1.2))
        if hot:
            ax.scatter([x + w / 2], [y + h / 2], marker="*", s=220, c="#facc15", edgecolors="black", zorder=4)
    ax.set_xlim(0, 1.3); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    return fig


if "history" not in st.session_state:
    st.session_state.history = []

with tab_home:
    st.header("Can a thermal camera tell who has a fever?")
    st.markdown(
        "**The problem.** During outbreaks, airports, stations and hospitals try to spot people with a raised temperature. "
        "Taking every temperature by hand is slow and nobody can watch a crowd all day. A thermal camera shows the surface temperature "
        "of everything in view, but it does not know which pixels are a face, and a skin reading is not the temperature a doctor "
        "would measure in the mouth.")
    cols = st.columns(4)
    steps = [("1. Find the face", "The program locates the face in the thermal image, so nobody has to aim a device at each person."),
             ("2. Read the hottest point", "It reads the highest temperature inside the face region: the skin surface, not the body core."),
             ("3. Compare with reality", "On real volunteers the camera reading is compared with an oral thermometer to measure how far off it is."),
             ("4. Decide, or say not sure", "No face, a face too small or too far, or no calibration for this camera: a status is shown, never a fever alert.")]
    for col, (head, text) in zip(cols, steps):
        col.markdown(f"**{head}**")
        col.write(text)
    st.subheader("What the evidence says")
    st.markdown(
        "- On **1,020 volunteers** (FDA data) the camera reads about **1 °C below** an oral thermometer. A simple calibration brings the typical "
        "error to about **0.2 to 0.3 °C**, but people with a fever are under-estimated (next tab).\n"
        "- In the Charlotte-ThermalFace data the same person reads **more than 1 °C cooler** from 6.6 m than from 1 m.\n"
        "- Even an accurate thermometer cannot find everyone. Gostic et al. (eLife, 2020, doi:10.7554/eLife.55570) estimated that, "
        "under best-case assumptions, traveller symptom and risk screening would miss more than half of infected people, mostly because "
        "they have no symptoms yet. Fever screening is one tool, not a guarantee.\n"
        "- The reference data come from the FDA study team and four peer-reviewed papers (cited in the project README).")
    st.info("Next: **Real subjects (FDA)** walks through real volunteers. **Screen an image** runs the full pipeline on a thermal image you upload.")

with tab_real:
    st.subheader("A real volunteer, step by step")
    tbl = subject_table()
    n = st.slider("Volunteer (anonymous row number)", 1, len(tbl), 1, key="subj")
    row = tbl.iloc[n - 1]
    left, right = st.columns([3, 2])
    fig = draw_subject(row); left.pyplot(fig); plt.close(fig)
    m1, m2 = right.columns(2)
    m1.metric("Camera reading (face max)", f"{row['whole face']:.2f} °C")
    m2.metric("Calibrated estimate", f"{row['estimate']:.2f} °C")
    m3, m4 = right.columns(2)
    m3.metric("Oral thermometer", f"{row['oral']:.2f} °C")
    m4.metric("Estimate minus oral", f"{row['estimate'] - row['oral']:+.2f} °C")
    fever, flagged = bool(row["oral"] >= 37.5), bool(row["estimate"] >= 37.5)
    if fever and not flagged:
        right.error("Missed: the oral temperature is 37.5 °C or more, but the estimate is below it.")
    elif fever and flagged:
        right.success("Detected: both the oral reading and the estimate are at or above 37.5 °C.")
    elif flagged:
        right.warning("False alarm: the estimate is 37.5 °C or more but the oral reading is below it.")
    else:
        right.info("No fever by the oral reading, and the estimate agrees.")
    right.caption("Schematic face; box positions are illustrative. Temperatures are the dataset's own extracted values for this anonymous "
                  "volunteer (round 1 of 4). The estimate is cross-validated: this volunteer was not used to fit the calibration behind it.")

    st.subheader("Where on the face is the hottest point?")
    sh = region_shares()
    labels = ["inner canthi", "forehead", "mouth", "none of the three"]
    vals = [sh["matches_any"]["inner canthi"], sh["matches_any"]["forehead"], sh["matches_any"]["mouth"], sh["none_of_the_three"]]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(labels[::-1], vals[::-1], color=["#9ca3af", "#6b7280", "#22c55e", "#ef4444"])
    for yv, v in enumerate(vals[::-1]):
        ax.text(v + 0.01, yv, f"{v:.0%}", va="center", fontsize=9)
    ax.set_xlim(0, 1.1); ax.set_xlabel("share of volunteers")
    st.pyplot(fig); plt.close(fig)
    st.caption(f"{sh['n']} volunteers, round 1. A region contains the hotspot when its own maximum equals the whole-face maximum "
               f"(within {sh['tolerance_c']} °C). Regions overlap, so the shares can add up to more than 100%.")

    st.subheader("How good is the calibration? Try a threshold")
    st.markdown("Data: 1,020 volunteers, one FLIR camera, about 0.8 m (FDA / PhysioNet, CC0, de-identified). The features are the dataset's own "
                "extracted values, so this shows the calibration problem, not image processing.")
    c1, c2, c3 = st.columns(3)
    feat = c1.selectbox("Facial measurement", list(FEATURES), key="feat")
    rnd = c2.selectbox("Round", ["1", "2", "3", "4", "Mean of 4"], key="rnd")
    thr_c = c3.slider("Flag when the oral-equivalent estimate reaches (°C)", 36.8, 38.0, 37.5, 0.1, key="thr_c")
    x, y, pred = calibrated(FEATURES[feat], rnd)
    err = pred - y; fev = y >= 37.5; flag = pred >= thr_c
    sens = float((fev & flag).sum() / max(fev.sum(), 1)); spec = float((~fev & ~flag).sum() / max((~fev).sum(), 1))
    mm = st.columns(4)
    mm[0].metric("MAE", f"{np.abs(err).mean():.2f} °C"); mm[1].metric("RMSE", f"{np.sqrt((err ** 2).mean()):.2f} °C")
    mm[2].metric("Sensitivity", f"{sens:.2f}"); mm[3].metric("Specificity", f"{spec:.2f}")
    p1, p2 = st.columns(2)
    fig, ax = plt.subplots(figsize=(5.5, 4.2)); ax.scatter(y, pred, s=6, alpha=.45)
    lim = [min(y.min(), pred.min()) - .1, max(y.max(), pred.max()) + .1]; ax.plot(lim, lim, "k--", lw=1)
    ax.axvline(37.5, c="r", lw=1); ax.axhline(thr_c, c="orange", lw=1)
    ax.set_xlabel("Oral temperature (°C)"); ax.set_ylabel("Calibrated estimate (°C)")
    ax.set_title("Estimate vs oral (red: fever = oral >= 37.5)", fontsize=10); p1.pyplot(fig); plt.close(fig)
    fpr, tpr, _ = roc_curve(fev, x)
    fig, ax = plt.subplots(figsize=(5.5, 4.2)); ax.plot(fpr, tpr); ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.scatter([1 - spec], [sens], c="orange", zorder=3, label="chosen threshold"); ax.legend()
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC for fever detection (AUC {roc_auc_score(fev, x):.2f})", fontsize=10); p2.pyplot(fig); plt.close(fig)
    st.caption("Calibration is a 5-fold cross-validated linear fit. Thresholds chosen here are tuned on the same data, so treat the operating "
               "point as exploratory.")

with tab_scan:
    st.markdown("Upload a thermal image to run the full pipeline: **face box, measured temperature label, and a cross on the hottest pixel.** "
                "A 16-bit radiometric image (assumed 0.01 K per count) gives a temperature; an 8-bit image only shows face detection because it "
                "carries no temperature. No sample frame is bundled because the radiometric datasets used in this project may not be redistributed; "
                "the *Real subjects* tab shows the same idea with real data.")
    sl, sr = st.columns([3, 2])
    with sr:
        min_h = st.number_input("Minimum face height for a reliable reading (px)", 20, 400, 100, 5, key="minh")
        thr = st.number_input("Flag when the measured face maximum reaches (°C)", 30.0, 45.0, 37.5, 0.1, key="thr")
        st.caption("The flag applies to the measured surface temperature; no calibration is applied in this tab.")
    with sl:
        up = st.file_uploader("Thermal image (16-bit radiometric, or 8-bit for detection only)",
                              type=["tif", "tiff", "png", "jpg", "jpeg"], key="up")
    if up is not None:
        img = decode(up)
        if img is not None and img.ndim == 2 and img.dtype == np.uint16:
            temp_c = raw_to_celsius(img)
            cfg = ScreenerConfig(min_face_height_px=int(min_h), alert_threshold_c=float(thr), calibration=Calibration(1.0, 0.0))
            readings = ThermalScreener(get_detector(), cfg).process_celsius(temp_c)
            a, b = st.columns([3, 2])
            a.image(render_overlay(temp_c, readings), caption="20 to 38 °C colormap. White box: reading, orange: unreliable, red: flagged. Cross: hottest pixel.")
            r0 = readings[0]; k1, k2 = b.columns(2)
            k1.metric("Face max (measured)", "n/a" if r0.face_max_c is None else f"{r0.face_max_c:.1f} °C")
            k2.metric("Status", r0.status.replace("_", " "))
            if r0.hotspot is not None:
                b.caption(f"Hottest pixel at x={r0.hotspot[0]}, y={r0.hotspot[1]}.")
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
Reliable readings need a face of about 100 px or more (roughly 1.4 m on this sensor). Real screening thermographs are specified in
IEC 80601-2-59; this prototype has not been tested against it and must not be used for medical decisions.
""")
