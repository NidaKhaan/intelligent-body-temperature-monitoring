"""Face box and keypoints found directly on real thermal frames. Usage: python scripts/make_figures_thermal_keypoints.py [drive_root]"""
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
DRIVE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/content/drive/MyDrive/Intelligent_Temperature_Monitor")
sys.path.insert(0, str(REPO))
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
from src.detection.face_detector import ensure_model

SUB = DRIVE / "datasets/speaking_faces/processed/sample_trial1_every81/thr_image"
det = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
    base_options=mpt.BaseOptions(model_asset_path=str(ensure_model())), min_detection_confidence=0.3))
frames, seen = [], set()
for p in sorted(SUB.glob("*.png")):
    pos = p.stem.rsplit("_", 1)[0].split("_")[3]
    if pos in seen:
        continue
    gray = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY)
    res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.dstack([gray] * 3))))
    if res.detections and res.detections[0].keypoints:
        seen.add(pos); frames.append((gray, res.detections[0]))
    if len(frames) == 3:
        break
assert frames, "no thermal frame with a face found"
fig, ax = plt.subplots(1, len(frames), figsize=(4.6 * len(frames), 4.2), squeeze=False)
for a, (gray, d) in zip(ax[0], frames):
    h, w = gray.shape
    a.imshow(gray, cmap="gray")
    bb = d.bounding_box
    a.add_patch(plt.Rectangle((bb.origin_x, bb.origin_y), bb.width, bb.height, fill=False, ec="red", lw=2))
    a.scatter([k.x * w for k in d.keypoints], [k.y * h for k in d.keypoints], s=30, c="red", edgecolors="white", linewidths=.6)
    a.axis("off")
fig.suptitle("Face box and 6 keypoints found directly on real thermal frames (SpeakingFaces, ISSAI, CC BY 4.0; 8-bit, no temperature)", fontsize=9)
plt.tight_layout(); (REPO / "docs/images").mkdir(parents=True, exist_ok=True)
plt.savefig(REPO / "docs/images/thermal_keypoints.png", dpi=130); plt.close()
det.close()
print("ok thermal_keypoints.png")
