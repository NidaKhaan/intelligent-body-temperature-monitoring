"""Re-measure ROI placement on Charlotte-ThermalFace with the current face detector.
Same 600-image sample as the v1 experiment (random_state=0, 15 frames per subject and distance).
Usage: python scripts/roi_detector_check.py <drive_project_root>"""
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DRIVE = Path(sys.argv[1])
OUT = Path("/content/cache/charlotte")
SUBJECTS = [1, 4, 5, 9]

from src.detection.face_detector import mediapipe_face_detector
from src.temperature.radiometric import raw_to_celsius


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def measure(csv_out):
    ch = DRIVE / "datasets/charlotte_thermal"
    if not (OUT / ".extracted_ok").exists():
        zipfile.ZipFile(ch / "Charlotte-ThermalFace.zip").extractall(OUT); (OUT / ".extracted_ok").touch()
    xc, yc = [f"x{i}" for i in range(73)], [f"y{i}" for i in range(73)]
    kept = pd.read_csv(DRIVE / "experiments/charlotte_scan_v2.csv")
    kept = kept[kept.subject.isin(SUBJECTS)].copy(); kept["d"] = kept.distance.round(1)
    samp = kept.sample(frac=1, random_state=0).groupby(["subject", "d"]).head(15)
    ann = {s: pd.read_csv(ch / f"annotations/S{s}.csv").drop_duplicates("ID").set_index("ID") for s in SUBJECTS}
    tiffs = {s: {q.stem: q for q in next(z for z in OUT.rglob(f"S{s}") if z.is_dir()).glob("*.tiff")} for s in SUBJECTS}
    detect, rows = mediapipe_face_detector(min_conf=0.3), []
    for r in samp.itertuples():
        row = ann[r.subject].loc[r.id]
        img = cv2.imread(str(tiffs[r.subject][r.id]), cv2.IMREAD_UNCHANGED); h, w = img.shape
        c = raw_to_celsius(img)
        xs, ys = row[xc].to_numpy(float), row[yc].to_numpy(float); ok = np.isfinite(xs) & np.isfinite(ys)
        pts = np.stack([np.clip(xs[ok], 0, w - 1), np.clip(ys[ok], 0, h - 1)], 1).astype(np.int32)
        mask = np.zeros((h, w), np.uint8); cv2.fillConvexPoly(mask, cv2.convexHull(pts), 1)
        ref = c[mask.astype(bool)].max()
        rec = dict(subject=r.subject, d=r.d, ref_max=ref, detected=False)
        boxes = detect(np.clip((c - 20.0) / 18.0 * 255, 0, 255).astype(np.uint8))
        if boxes:
            x0, y0, x1, y1 = [int(round(v)) for v in boxes[0]]
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
            if x1 - x0 >= 3 and y1 - y0 >= 3:
                rec.update(detected=True, box_minus_ref=float(c[y0:y1, x0:x1].max() - ref),
                           iou=iou((x0, y0, x1, y1), (*pts.min(0), *pts.max(0))))
        rows.append(rec)
    pd.DataFrame(rows).to_csv(csv_out, index=False)


if __name__ == "__main__":
    out = DRIVE / "experiments/charlotte_roi_detector_v2.csv"
    if not out.exists():
        measure(out)
    shutil.copy(out, REPO / "docs/results/charlotte_roi_detector_v2.csv")
    new = pd.read_csv(out); old = pd.read_csv(REPO / "docs/results/charlotte_roi_sensitivity_v1.csv")
    for df in (new, old):
        df["abs_diff"] = df.box_minus_ref.abs()
    agg = lambda df: df.groupby("d").agg(detected=("detected", "mean"), iou=("iou", "median"), abs_diff=("abs_diff", "mean"))
    print(pd.concat({"tasks_short_range": agg(new), "legacy_full_range": agg(old)}, axis=1).round(2).to_string())
