"""Photo to pseudo-thermal SIMULATION, for demonstrations only.

An ordinary photo contains no temperature information. The temperatures produced here are values chosen by the caller,
not measurements, and must never be presented as such.
"""
import cv2
import numpy as np

REGION_FRACTIONS = {            # (x, y, w, h) as fractions of the face box; approximate, not detected
    "forehead": (0.20, 0.05, 0.60, 0.18),
    "inner canthi": (0.18, 0.30, 0.64, 0.12),
    "mouth": (0.30, 0.68, 0.40, 0.12),
}


def region_boxes(box):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    return {n: (x0 + fx * w, y0 + fy * h, x0 + (fx + fw) * w, y0 + (fy + fh) * h)
            for n, (fx, fy, fw, fh) in REGION_FRACTIONS.items()}


def simulate_thermal(rgb, box, face_max_c, ambient_c=24.0, skin_offset_c=2.0):
    """Returns (temp_c float32 HxW, (x, y) of the simulated hot spot). The maximum inside the box equals face_max_c."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("expected an HxWx3 uint8 RGB photo")
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = max(int(round(x0)), 0), max(int(round(y0)), 0), min(int(round(x1)), w), min(int(round(y1)), h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        raise ValueError("face box too small")
    bw, bh = x1 - x0, y1 - y0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, (int((x0 + x1) / 2), int((y0 + y1) / 2)), (max(int(bw / 2 * 0.95), 1), max(int(bh / 2 * 0.95), 1)),
                0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, bw * 0.03))
    gray_s = cv2.GaussianBlur(gray, (0, 0), 6.0)
    skin = float(face_max_c) - skin_offset_c
    t = ambient_c + (skin - ambient_c) * (0.12 * gray_s * (1.0 - mask) + mask * (0.85 + 0.15 * gray))
    spots = [(int(x0 + 0.35 * bw), int(y0 + 0.36 * bh)), (int(x0 + 0.65 * bw), int(y0 + 0.36 * bh))]
    yy, xx = np.mgrid[0:h, 0:w]
    for sx, sy in spots:
        blob = np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2.0 * (bw * 0.025) ** 2 + 1e-6))
        t = np.maximum(t, ambient_c + (float(face_max_c) - ambient_c) * blob)
    return t.astype(np.float32), spots[0]
