"""Extra README figures: FDA per-subject temperatures, Charlotte landmark schemes (coordinates only), LLVIP detections.
Usage: python scripts/make_figures_extra.py [drive_project_root]"""
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
DRIVE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/content/drive/MyDrive/Intelligent_Temperature_Monitor")
IMG = REPO / "docs" / "images"
IMG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO))


def fda_examples():
    import src.temperature.fda_baseline as fb
    fl = pd.read_csv(DRIVE / "datasets/fda_physionet/clean_FLIR_groups1and2.csv")
    x, y = fl["T_Max1"], fl["aveOralM"]
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    d = pd.DataFrame({"measured": x, "estimate": fb.cv_calibrate(x, y), "oral": y})
    pick = pd.concat([d[d.oral < 37.5].sample(6, random_state=3), d[d.oral >= 37.5].sample(6, random_state=3)])
    pick = pick.sort_values("oral").reset_index(drop=True)
    yy = np.arange(len(pick))[::-1]
    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    ax.hlines(yy, pick.estimate, pick.oral, color="#9ca3af", lw=1.5, zorder=1)
    ax.scatter(pick.measured, yy, c="#6b7280", s=45, label="measured facial maximum (camera, round 1)", zorder=2)
    ax.scatter(pick.estimate, yy, c="#2563eb", marker="s", s=45, label="calibrated estimate (5-fold CV)", zorder=3)
    ax.scatter(pick.oral, yy, c="#dc2626", marker="D", s=50, label="oral thermometer", zorder=3)
    ax.axvline(37.5, c="#dc2626", ls="--", lw=1)
    lo = min(pick.measured.min(), pick.oral.min()) - 0.3
    hi = max(pick.oral.max(), pick.estimate.max()) + 0.9
    for yi, e, o in zip(yy, pick.estimate, pick.oral):
        ax.text(hi - 0.75, yi, f"{e - o:+.2f} C", va="center", fontsize=8)
    ax.set_xlim(lo, hi)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"Subject {chr(65 + i)}" for i in range(len(pick))])
    ax.set_xlabel("Temperature (C); dashed line: 37.5 C")
    ax.set_title("Temperature detection on 12 random FDA subjects (6 below, 6 at or above 37.5 C oral)\n"
                 "right-hand numbers: estimate minus oral", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.savefig(IMG / "fda_examples.png", dpi=140)
    plt.close()


def fda_study_figure():
    from PIL import Image
    src = DRIVE / "datasets/fda_physionet/raw/_Figure1.png"
    assert src.exists(), "FDA Figure 1 not found"
    im = Image.open(src)
    im.thumbnail((1400, 1400))
    im.save(IMG / "fda_study_figure1.png", optimize=True)


def charlotte_landmarks():
    ch = DRIVE / "datasets/charlotte_thermal/annotations"
    xc, yc = [f"x{i}" for i in range(73)], [f"y{i}" for i in range(73)]
    layouts = {43: [], 73: []}
    for s in range(1, 11):
        ann = pd.read_csv(ch / f"S{s}.csv")
        X, Y = ann[xc].to_numpy(float), ann[yc].to_numpy(float)
        n = (np.isfinite(X) & np.isfinite(Y)).sum(1)
        for k in (43, 73):
            sel = np.where(n == k)[0]
            for i in sel[:: max(len(sel) // 6, 1)][:6]:
                m = np.isfinite(X[i]) & np.isfinite(Y[i])
                xs, ys = X[i][m], Y[i][m]
                layouts[k].append(((xs - xs.min()) / (np.ptp(xs) + 1e-9), (ys - ys.min()) / (np.ptp(ys) + 1e-9)))
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 5.8))
    for a, k in zip(ax, (73, 43)):
        cloud = layouts[k]
        assert cloud, f"no {k}-point rows"
        for xs, ys in cloud[1:]:
            a.scatter(xs, ys, s=4, c="#9ca3af", alpha=.35)
        xs, ys = cloud[0]
        a.scatter(xs, ys, s=22, c="#ef4444", edgecolors="white", linewidths=.5, zorder=3)
        hull = cv2.convexHull(np.stack([xs * 1000, ys * 1000], 1).astype(np.int32)).reshape(-1, 2) / 1000.0
        poly = np.vstack([hull, hull[:1]])
        a.plot(poly[:, 0], poly[:, 1], c="#2563eb", lw=1.6, label="convex hull = measurement ROI")
        a.plot(poly[:, 0], poly[:, 1] + 0.10, c="#2563eb", lw=1.2, ls="--", label="ROI shifted by 10% of its height")
        a.invert_yaxis()
        a.set_aspect("equal")
        a.set_title(f"{k}-point scheme ({len(cloud)} frames overlaid)", fontsize=10)
        a.legend(fontsize=7, loc="lower right")
    fig.suptitle("Charlotte-ThermalFace landmark schemes: annotation coordinates only, normalised to each face's box (no images)", fontsize=9)
    plt.tight_layout()
    plt.savefig(IMG / "charlotte_landmarks.png", dpi=140)
    plt.close()


def llvip_detections():
    import src.detection.person_detector as pdm
    sub = DRIVE / "datasets/llvip/test_subset"
    ids = sorted(p.stem for p in (sub / "infrared").glob("*.jpg"))

    def gt(i):
        r = ET.parse(sub / "annotations" / f"{i}.xml").getroot()
        return [[float(o.find("bndbox").find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
                for o in r.findall("object") if o.find("name").text == "person"]

    sel = []
    for lo, hi in ((3, 6), (1, 8)):
        for i in ids[100::97]:
            g = gt(i)
            if lo <= len(g) <= hi and i not in [s[0] for s in sel]:
                sel.append((i, g))
            if len(sel) == 2:
                break
        if len(sel) == 2:
            break
    assert len(sel) == 2, "not enough suitable LLVIP frames"
    det = pdm.PersonDetector("resnet50", score_thr=0.5)
    fig, ax = plt.subplots(2, 2, figsize=(13, 10.5))
    for r, (i, g) in enumerate(sel):
        for c, mod in enumerate(["visible", "infrared"]):
            im = cv2.cvtColor(cv2.imread(str(sub / mod / f"{i}.jpg")), cv2.COLOR_BGR2RGB)
            b, s = det.detect(im)
            ax[r, c].imshow(im)
            for x0, y0, x1, y1 in g:
                ax[r, c].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="lime", lw=1.8))
            for (x0, y0, x1, y1), sc in zip(b, s):
                ax[r, c].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="red", lw=1.4))
                ax[r, c].text(x0, y0 - 4, f"{sc:.2f}", color="red", fontsize=7)
            ax[r, c].set_title(f"{mod}: {len(g)} annotated, {len(b)} detected (score >= 0.5)", fontsize=10)
            ax[r, c].axis("off")
    fig.suptitle("LLVIP test frames. Green: annotation. Red: Faster R-CNN ResNet50 (COCO weights)", fontsize=10)
    plt.tight_layout()
    plt.savefig(IMG / "llvip_detections.png", dpi=100)
    plt.close()


if __name__ == "__main__":
    failed = []
    for name, fn in [("fda_examples", fda_examples), ("fda_study_figure", fda_study_figure),
                     ("charlotte_landmarks", charlotte_landmarks), ("llvip_detections", llvip_detections)]:
        try:
            fn()
            print("ok  ", name)
        except Exception as e:
            failed.append(name)
            print("FAIL", name, "->", repr(e))
            traceback.print_exc(limit=3)
    print("failed:", failed)
