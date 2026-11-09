"""Generate README figures from result files and datasets on Google Drive.
Usage: python scripts/make_figures.py [drive_project_root]
Each figure is independent; failures are reported at the end."""
import json, re, sys, traceback, zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]
DRIVE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/content/drive/MyDrive/Intelligent_Temperature_Monitor")
IMG, RES = REPO / "docs" / "images", REPO / "docs" / "results"
IMG.mkdir(parents=True, exist_ok=True); RES.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO))

GREY, BLUE, GRN, YEL, ORG, RED = "#e5e7eb", "#dbeafe", "#dcfce7", "#fef3c7", "#ffedd5", "#fee2e2"


def box(ax, x, y, w, h, text, fc, ls="-", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec="#1f2937", lw=1.3, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#111827")


def arrow(ax, p, q, txt=None, ls="-"):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#374151", linestyle=ls))
    if txt:
        ax.text((p[0] + q[0]) / 2 + 0.1, (p[1] + q[1]) / 2 + 0.12, txt, fontsize=8, color="#374151")


def diagram_pipeline():
    fig, ax = plt.subplots(figsize=(14, 6.2)); ax.set_xlim(0, 14); ax.set_ylim(0, 7.2); ax.axis("off")
    W, H = 3.1, 1.3; xs = [0.3, 3.8, 7.3, 10.8]; top, mid, low = 5.4, 3.0, 0.6
    box(ax, xs[0], top, W, H, "16-bit radiometric\nthermal frame", BLUE)
    box(ax, xs[1], top, W, H, "Counts to degrees C\n(0.01 K per count, assumed)", BLUE)
    box(ax, xs[2], top, W, H, "Fixed 20-38 C window\nto an 8-bit image", BLUE)
    box(ax, xs[3], top, W, H, "MediaPipe\nface detector", BLUE)
    for i in range(3):
        arrow(ax, (xs[i] + W, top + H / 2), (xs[i + 1], top + H / 2))
    box(ax, xs[3], mid, W, H, "Face found?", YEL)
    box(ax, xs[2], mid, W, H, "ROI height at least\n100 px?", YEL)
    box(ax, xs[1], mid, W, H, "Face maximum\ninside the ROI", BLUE)
    box(ax, xs[0], mid, W, H, "Calibration\nloaded?", YEL)
    arrow(ax, (xs[3] + W / 2, top), (xs[3] + W / 2, mid + H))
    arrow(ax, (xs[3], mid + H / 2), (xs[2] + W, mid + H / 2), "yes")
    arrow(ax, (xs[2], mid + H / 2), (xs[1] + W, mid + H / 2), "yes")
    arrow(ax, (xs[1], mid + H / 2), (xs[0] + W, mid + H / 2))
    box(ax, xs[3], low, W, H, "status: no_face\n(no alert)", ORG)
    box(ax, xs[2], low, W, H, "status: unreliable_roi_small\n(no alert)", ORG)
    box(ax, xs[1], low, W, H, "calibrated: oral-equivalent\nvalue and alert flag", RED)
    box(ax, xs[0], low, W, H, "uncalibrated: measured\nvalue only, no alert", GRN)
    arrow(ax, (xs[3] + W / 2, mid), (xs[3] + W / 2, low + H), "no")
    arrow(ax, (xs[2] + W / 2, mid), (xs[2] + W / 2, low + H), "no")
    arrow(ax, (xs[0] + W / 2, mid), (xs[0] + W / 2, low + H), "no")
    arrow(ax, (xs[0] + W, mid + 0.2), (xs[1] + W / 2, low + H), "yes")
    ax.set_title("Thermal-only screening pipeline: every doubtful case yields a status, never an alert", fontsize=12)
    plt.tight_layout(); plt.savefig(IMG / "pipeline.png", dpi=140); plt.close()


def diagram_architecture():
    fig, ax = plt.subplots(figsize=(15, 7.8)); ax.set_xlim(0, 16); ax.set_ylim(0, 8.7); ax.axis("off")
    for x, t in [(2.1, "Public datasets"), (7.3, "Experiments (Colab + Google Drive)"), (13.3, "Application")]:
        ax.text(x, 8.3, t, ha="center", fontsize=11, fontweight="bold")
    rows = [("FDA / PhysioNet\n(CSV features + oral reference)", "Calibration and error analysis\n(regions, ROC)"),
            ("Charlotte-ThermalFace\n(16-bit radiometric)", "Distance effect and\nROI sensitivity"),
            ("LLVIP\n(visible + infrared)", "Person-detector evaluation"),
            ("SpeakingFaces sample\n(CC BY 4.0)", "RGB-thermal registration\nand face alignment")]
    ys = [6.5, 4.9, 3.3, 1.7]
    for (d, e), y in zip(rows, ys):
        box(ax, 0.3, y, 3.6, 1.2, d, GREY); box(ax, 5.2, y, 4.2, 1.2, e, BLUE)
        arrow(ax, (3.9, y + 0.6), (5.2, y + 0.6))
    box(ax, 11.0, 6.6, 4.6, 1.2, "configs/calibration_fda_flir_tmax.json\n(valid for the FDA camera only)", ORG)
    box(ax, 11.0, 4.5, 4.6, 1.4, "ThermalScreener\n(src/pipeline, fail-safe)", GRN, fs=10)
    box(ax, 11.0, 2.6, 2.2, 1.0, "FastAPI\n/screen", BLUE)
    box(ax, 13.4, 2.6, 2.2, 1.0, "Streamlit\ndashboard", BLUE)
    arrow(ax, (9.4, 7.1), (11.0, 7.2))
    arrow(ax, (13.3, 6.6), (13.3, 5.9))
    arrow(ax, (12.1, 4.5), (12.1, 3.6)); arrow(ax, (14.5, 4.5), (14.5, 3.6))
    arrow(ax, (9.4, 2.3), (11.0, 5.0), "optional", ls="--")
    box(ax, 0.3, 0.2, 15.3, 0.8, "StageTracker: resumable stages with atomic JSON checkpoints on Google Drive "
        "(datasets, results and progress survive Colab disconnects)", GREY)
    plt.tight_layout(); plt.savefig(IMG / "architecture.png", dpi=140); plt.close()


def fda_regions():
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import KFold, cross_val_predict
    fl = pd.read_csv(DRIVE / "datasets/fda_physionet/clean_FLIR_groups1and2.csv")
    y_all = fl["aveOralM"]; pat = re.compile(r"^(.*?)_?([1-4])$"); groups = {}
    for c in fl.columns:
        m = pat.match(c)
        if m:
            groups.setdefault(m.group(1), {})[int(m.group(2))] = c
    groups = {b: d for b, d in groups.items() if len(d) == 4 and b != "T_offset"
              and all(pd.api.types.is_numeric_dtype(fl[c]) for c in d.values())}
    cv = KFold(5, shuffle=True, random_state=42); rows, cache = [], {}
    for b, d in groups.items():
        maes, aucs = [], []
        for r in (1, 2, 3, 4):
            x = fl[d[r]]; ok = x.notna() & y_all.notna()
            if ok.sum() < 200:
                continue
            p = cross_val_predict(LinearRegression(), x[ok].to_frame(), y_all[ok], cv=cv)
            maes.append(float(np.abs(p - y_all[ok]).mean())); aucs.append(float(roc_auc_score(y_all[ok] >= 37.5, x[ok])))
            if r == 1:
                cache[b] = (x[ok].to_numpy(), y_all[ok].to_numpy(), p)
        if maes:
            rows.append(dict(feature=b, mean_MAE=np.mean(maes), mean_AUC=np.mean(aucs)))
    res = pd.DataFrame(rows).sort_values("mean_MAE").reset_index(drop=True)
    res.round(4).to_csv(RES / "fda_region_summary.csv", index=False)
    top = res.head(14)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.8), gridspec_kw={"width_ratios": [1.25, 1, 0.8]})
    colors = ["#ef4444" if f in ("T_Max", "canthiMax") else "#60a5fa" for f in top.feature]
    ax[0].barh(range(len(top)), top.mean_MAE, color=colors); ax[0].set_yticks(range(len(top)))
    ax[0].set_yticklabels(top.feature, fontsize=9); ax[0].invert_yaxis()
    for i, (m, a) in enumerate(zip(top.mean_MAE, top.mean_AUC)):
        ax[0].text(m + 0.002, i, f"AUC {a:.2f}", va="center", fontsize=8)
    ax[0].set_xlabel("Calibrated MAE vs oral thermometer (C); mean of 4 single rounds, 5-fold CV")
    ax[0].set_title("Which facial measurement tracks oral temperature best?\n(14 lowest-error features; red = T_Max, canthiMax)", fontsize=10)
    cand = [b for b in dict.fromkeys(list(top.feature.head(3)) + ["T_Max"]) if b in cache]
    for b in cand:
        x, y, _ = cache[b]; f, t, _ = roc_curve(y >= 37.5, x)
        ax[1].plot(f, t, label=f"{b} (AUC {roc_auc_score(y >= 37.5, x):.2f})")
    ax[1].plot([0, 1], [0, 1], "k--", lw=1); ax[1].legend(fontsize=8, loc="lower right")
    ax[1].set_xlabel("False positive rate"); ax[1].set_ylabel("True positive rate")
    ax[1].set_title("Detecting fever (oral >= 37.5 C), round 1", fontsize=10)
    x, y, p = cache["T_Max"]; fever, flag = y >= 37.5, p >= 37.5
    cm = np.array([[(~fever & ~flag).sum(), (~fever & flag).sum()], [(fever & ~flag).sum(), (fever & flag).sum()]])
    ax[2].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax[2].text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=14,
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["not flagged", "flagged"])
    ax[2].set_yticks([0, 1]); ax[2].set_yticklabels(["oral < 37.5", "oral >= 37.5"])
    ax[2].set_title(f"T_Max, round 1, flag at 37.5 C\nsensitivity {cm[1, 1] / cm[1].sum():.2f}, specificity {cm[0, 0] / cm[0].sum():.2f}", fontsize=10)
    plt.tight_layout(); plt.savefig(IMG / "fda_regions.png", dpi=130); plt.close()


def charlotte_hotspots():
    from src.temperature.radiometric import raw_to_celsius
    OUT = Path("/content/cache/charlotte")
    if not (OUT / ".extracted_ok").exists():
        zipfile.ZipFile(DRIVE / "datasets/charlotte_thermal/Charlotte-ThermalFace.zip").extractall(OUT)
        (OUT / ".extracted_ok").touch()
    scan = pd.read_csv(DRIVE / "experiments/charlotte_scan_v2.csv"); scan["d"] = scan.distance.round(1)
    scan = scan[scan.subject.isin([1, 4, 5, 9]) & (scan.n_lm == 73) & (scan.d <= 1.4)]
    xc, yc = [f"x{i}" for i in range(73)], [f"y{i}" for i in range(73)]
    U, V, T = [], [], []
    for s in [1, 4, 5, 9]:
        sdir = next(q for q in OUT.rglob(f"S{s}") if q.is_dir())
        tiffs = {p.stem: p for p in sdir.glob("*.tiff")}
        ann = pd.read_csv(DRIVE / f"datasets/charlotte_thermal/annotations/S{s}.csv").dropna(subset=["ID"]).drop_duplicates("ID").set_index("ID")
        sub = scan[scan.subject == s]; sub = sub.sample(min(150, len(sub)), random_state=0)
        for r in sub.itertuples():
            if r.id not in tiffs or r.id not in ann.index:
                continue
            row = ann.loc[r.id]; xs, ys = row[xc].to_numpy(float), row[yc].to_numpy(float)
            if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
                continue
            img = cv2.imread(str(tiffs[r.id]), cv2.IMREAD_UNCHANGED); h, w = img.shape
            pts = np.stack([np.clip(xs, 0, w - 1), np.clip(ys, 0, h - 1)], 1).astype(np.int32)
            mask = np.zeros((h, w), np.uint8); cv2.fillConvexPoly(mask, cv2.convexHull(pts), 1)
            c = raw_to_celsius(img); c[mask == 0] = -np.inf
            yy, xx = np.unravel_index(np.argmax(c), c.shape)
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            U.append((xx - x0) / (x1 - x0)); V.append((yy - y0) / (y1 - y0))
            T.append(np.stack([(xs - x0) / (x1 - x0), (ys - y0) / (y1 - y0)], 1))
    assert len(U) > 50, f"too few usable frames: {len(U)}"
    U, V, T = np.array(U), np.array(V), np.mean(T, axis=0)
    fig, ax = plt.subplots(figsize=(5.8, 6.6))
    h2 = ax.hist2d(U, V, bins=[24, 28], range=[[0, 1], [0, 1]], cmap="inferno")
    ax.scatter(T[:, 0], T[:, 1], s=14, c="white", edgecolors="black", linewidths=.4)
    ax.invert_yaxis(); fig.colorbar(h2[3], ax=ax, label="frames")
    ax.set_xlabel("relative x inside the landmark box"); ax.set_ylabel("relative y (0 = top of the box)")
    ax.set_title(f"Where the hottest pixel falls (white dots: mean 73-landmark layout)\n{len(U)} frontal frames, 4 subjects, up to 1.4 m", fontsize=9)
    plt.tight_layout(); plt.savefig(IMG / "charlotte_hotspots.png", dpi=140); plt.close()
    json.dump({"n": int(len(U)), "median_x": float(np.median(U)), "median_y": float(np.median(V))},
              open(RES / "charlotte_hotspot_summary.json", "w"))


def face_alignment():
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    from src.alignment.face_aligner import FaceAligner
    from src.detection.face_detector import ensure_model
    det = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
        base_options=mpt.BaseOptions(model_asset_path=str(ensure_model())), min_detection_confidence=0.3))
    SUB = DRIVE / "datasets/speaking_faces/processed/sample_trial1_every81"
    def idx(f): return {p.stem.rsplit("_", 1)[0]: p for p in (SUB / f).glob("*.png")}
    thr, ali = idx("thr_image"), idx("rgb_image_aligned")
    common = sorted(set(thr) & set(ali)); by_pos = {}
    for s in common:
        by_pos.setdefault(s.split("_")[3], s)
    picks = [by_pos[k] for k in sorted(by_pos)][:: max(len(by_pos) // 3, 1)][:3]
    aligner = FaceAligner(target_size=(256, 256), eye_target_ratio=0.35); rows = []
    for s in picks:
        rgb = cv2.cvtColor(cv2.imread(str(ali[s])), cv2.COLOR_BGR2RGB)
        th = cv2.cvtColor(cv2.imread(str(thr[s])), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        if not res.detections:
            continue
        d = max(res.detections, key=lambda q: q.categories[0].score)
        if not d.keypoints:
            continue
        kp = [(k.x * w, k.y * h) for k in d.keypoints]
        left, right = sorted(kp[:2], key=lambda p: p[0])
        a_rgb, M = aligner.align_face_by_eyes(rgb, left, right)
        bb = d.bounding_box
        rows.append((rgb, kp, (bb.origin_x, bb.origin_y, bb.width, bb.height), a_rgb,
                     cv2.warpAffine(th, M, (256, 256), flags=cv2.INTER_LINEAR)))
    assert rows, "no face detected in the picked frames"
    fig, ax = plt.subplots(len(rows), 3, figsize=(10.5, 3.5 * len(rows)), squeeze=False)
    for i, (rgb, kp, bb, a_rgb, a_th) in enumerate(rows):
        ax[i, 0].imshow(rgb); ax[i, 1].imshow(a_rgb); ax[i, 2].imshow(a_th)
        ax[i, 0].add_patch(plt.Rectangle((bb[0], bb[1]), bb[2], bb[3], fill=False, ec="lime", lw=2))
        ax[i, 0].scatter([p[0] for p in kp], [p[1] for p in kp], s=28, c="red", edgecolors="white", linewidths=.6)
        for a in ax[i]:
            a.axis("off")
    ax[0, 0].set_title("MediaPipe face box and 6 keypoints", fontsize=10)
    ax[0, 1].set_title("Eye-based alignment (RGB, 256x256)", fontsize=10)
    ax[0, 2].set_title("Same transform applied to thermal", fontsize=10)
    fig.suptitle("SpeakingFaces (ISSAI), CC BY 4.0: aligned RGB and 8-bit thermal frames from three camera positions", fontsize=8)
    plt.tight_layout(); plt.savefig(IMG / "face_alignment.png", dpi=130); plt.close()


if __name__ == "__main__":
    jobs = [("pipeline.png", diagram_pipeline), ("architecture.png", diagram_architecture), ("fda_regions.png", fda_regions),
            ("charlotte_hotspots.png", charlotte_hotspots), ("face_alignment.png", face_alignment)]
    failed = []
    for name, fn in jobs:
        try:
            fn(); print("ok  ", name)
        except Exception as e:
            failed.append(name); print("FAIL", name, "->", repr(e)); traceback.print_exc(limit=3)
    print("failed:", failed)
