"""Evaluate RGB -> thermal box mapping on the SpeakingFaces subset kept on Drive.

Proxy ground truth: the face box MediaPipe finds on the dataset's own aligned RGB frame (same geometry as the thermal frame).
For every camera position one homography is calibrated (SIFT matches between the raw and the aligned RGB of one frame) and
tested on the other frames of that position. Controls: plain resize of the RGB box, and the homography of another position.
Usage: python scripts/alignment_eval.py <drive_project_root>"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DRIVE = Path(sys.argv[1])

from src.alignment.registration import HomographyRegistration
from src.detection.face_detector import mediapipe_face_detector

SUB = DRIVE / "datasets/speaking_faces/processed/sample_trial1_every81"
METHODS = ("homography", "plain_resize", "wrong_position")


def index(folder):
    return {p.stem.rsplit("_", 1)[0]: p for p in (SUB / folder).glob("*.png")}


def load_rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def centre(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def main():
    raw, ali = index("rgb_image"), index("rgb_image_aligned")
    common = sorted(set(raw) & set(ali))
    assert common, "no frames found"
    pos = lambda s: s.split("_")[3]
    detect = mediapipe_face_detector(min_conf=0.3)
    sift, bf = cv2.SIFT_create(), cv2.BFMatcher()

    def calibrate(s):
        a = cv2.cvtColor(load_rgb(raw[s]), cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(load_rgb(ali[s]), cv2.COLOR_RGB2GRAY)
        k1, d1 = sift.detectAndCompute(a, None)
        k2, d2 = sift.detectAndCompute(b, None)
        if d1 is None or d2 is None:
            return None
        m = [p[0] for p in bf.knnMatch(d1, d2, k=2) if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(m) < 12:
            return None
        try:
            reg, err, n = HomographyRegistration.from_points([k1[x.queryIdx].pt for x in m],
                                                             [k2[x.trainIdx].pt for x in m])
        except ValueError:
            return None
        return (reg, err, n) if n >= 40 else None

    by_pos = {}
    for s in common:
        by_pos.setdefault(pos(s), []).append(s)
    cals, calib_frame = {}, {}
    for p, frames in sorted(by_pos.items()):
        if len(frames) < 3:
            continue
        for s in frames:
            c = calibrate(s)
            if c:
                cals[p], calib_frame[p] = c, s
                break
    positions = sorted(cals)
    assert len(positions) >= 2, f"need at least 2 calibrated positions, got {positions}"

    hw_raw, hw_ali = load_rgb(raw[common[0]]).shape[:2], load_rgb(ali[common[0]]).shape[:2]
    sx, sy = hw_ali[1] / hw_raw[1], hw_ali[0] / hw_raw[0]
    rows = []
    for i, p in enumerate(positions):
        reg, wrong = cals[p][0], cals[positions[(i + 1) % len(positions)]][0]
        for s in by_pos[p]:
            if s == calib_frame[p]:
                continue
            br, ba = detect(load_rgb(raw[s])), detect(load_rgb(ali[s]))
            if not br or not ba:
                rows.append(dict(position=p, valid=False))
                continue
            br, ba = br[0], ba[0]
            boxes = {"homography": reg.map_box(br), "plain_resize": (br[0] * sx, br[1] * sy, br[2] * sx, br[3] * sy),
                     "wrong_position": wrong.map_box(br)}
            r = dict(position=p, valid=True)
            for k, b in boxes.items():
                r[f"iou_{k}"] = float(iou(b, ba))
                r[f"centre_{k}"] = float(np.linalg.norm(centre(b) - centre(ba)))
            rows.append(r)
    valid = [r for r in rows if r["valid"]]
    assert len(valid) >= 20, f"too few valid frames: {len(valid)}"
    med = lambda key: {k: float(np.median([r[f"{key}_{k}"] for r in valid])) for k in METHODS}
    share = lambda t: {k: float(np.mean([r[f"iou_{k}"] >= t for r in valid])) for k in METHODS}
    summary = {"n_frames": len(rows), "n_valid": len(valid), "n_positions": len(positions),
               "rgb_size_wh": [hw_raw[1], hw_raw[0]], "thermal_geometry_wh": [hw_ali[1], hw_ali[0]],
               "iou_median": med("iou"), "share_iou_ge_0_5": share(0.5), "share_iou_ge_0_7": share(0.7),
               "centre_error_px_median": med("centre"),
               "calibration_reproj_px_median": float(np.median([cals[p][1] for p in positions])),
               "calibration_inliers_median": float(np.median([cals[p][2] for p in positions]))}
    (REPO / "docs/results").mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(REPO / "docs/results/alignment_eval.json", "w"), indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.2]})
    ax[0].boxplot([[r[f"iou_{k}"] for r in valid] for k in METHODS], showfliers=False)
    ax[0].set_xticklabels(["homography\n(own position)", "plain resize\n(no registration)", "homography of\nanother position"], fontsize=8)
    ax[0].set_ylabel("IoU with the box on the aligned RGB frame"); ax[0].set_ylim(0, 1)
    ax[0].set_title(f"Box mapping, {len(valid)} test frames", fontsize=10)
    per = {p: float(np.median([r["iou_homography"] for r in valid if r["position"] == p]))
           for p in positions if any(r["position"] == p for r in valid)}
    ax[1].bar(range(len(per)), list(per.values()), color="#2563eb")
    ax[1].set_xticks(range(len(per))); ax[1].set_xticklabels([f"pos {p}" for p in per], fontsize=8)
    ax[1].set_ylim(0, 1); ax[1].set_ylabel("median IoU (own-position homography)")
    ax[1].set_title("Per camera position", fontsize=10)
    fig.suptitle("SpeakingFaces (ISSAI), CC BY 4.0, one subject. Proxy ground truth: face box on the dataset's aligned RGB frame", fontsize=8)
    plt.tight_layout()
    (REPO / "docs/images").mkdir(parents=True, exist_ok=True)
    plt.savefig(REPO / "docs/images/alignment_eval.png", dpi=130)
    plt.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
