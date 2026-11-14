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
    """Returns (temp_c float32 HxW, (x, y) of the simulated hot spot). The maximum inside the box equals face_max_c.
    Face detail comes from the photo's local contrast (dark features such as brows and hair read cooler); hair, neck and
    shoulders are set to fixed fractions of the skin temperature. All of it is illustrative."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("expected an HxWx3 uint8 RGB photo")
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = max(int(round(x0)), 0), max(int(round(y0)), 0), min(int(round(x1)), w), min(int(round(y1)), h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        raise ValueError("face box too small")
    bw, bh = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    sigma = max(1.0, bw * 0.03)

    def blob_ellipse(centre, axes, sig):
        m = np.zeros((h, w), np.float32)
        cv2.ellipse(m, (int(centre[0]), int(centre[1])), (max(int(axes[0]), 1), max(int(axes[1]), 1)), 0, 0, 360, 1.0, -1)
        return cv2.GaussianBlur(m, (0, 0), sig)

    face = blob_ellipse((cx, cy), (bw / 2 * 0.95, bh / 2 * 0.95), sigma)
    head = blob_ellipse((cx, cy - 0.10 * bh), (bw / 2 * 1.30, bh / 2 * 1.25), sigma * 2)
    body = np.zeros((h, w), np.float32)
    body[min(int(y1 - 0.15 * bh), h):h, max(int(cx - 0.55 * bw), 0):min(int(cx + 0.55 * bw), w)] = 1.0       # neck
    body[min(int(y1 + 0.35 * bh), h):h, max(int(cx - 1.6 * bw), 0):min(int(cx + 1.6 * bw), w)] = 1.0         # shoulders
    body = cv2.GaussianBlur(body, (0, 0), sigma * 2)
    gray_s = cv2.GaussianBlur(gray, (0, 0), 3.0)
    lo, hi = np.percentile(gray[y0:y1, x0:x1], [2, 98])
    gn = np.clip((gray - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    factor = np.maximum.reduce([body * 0.40, head * (0.55 + 0.10 * gray_s), face * (0.80 + 0.20 * gn)])
    skin = float(face_max_c) - skin_offset_c
    t = ambient_c + (skin - ambient_c) * factor
    spots = [(int(x0 + 0.35 * bw), int(y0 + 0.36 * bh)), (int(x0 + 0.65 * bw), int(y0 + 0.36 * bh))]
    yy, xx = np.mgrid[0:h, 0:w]
    for sx, sy in spots:
        blob = np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2.0 * (bw * 0.025) ** 2 + 1e-6))
        t = np.maximum(t, ambient_c + (float(face_max_c) - ambient_c) * blob)
    return t.astype(np.float32), spots[0]
