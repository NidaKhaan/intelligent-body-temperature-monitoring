import cv2
import numpy as np

MODE_16 = "Upload 16-bit radiometric image"
MODE_8 = "Upload 8-bit thermal image (detection only)"
MODE_SYN = "Synthetic frame (logic demo)"


def synthetic_frame(face_max_c=36.5, face_h_px=160, shape=(348, 464), ambient_c=24.0, face_c=34.0):
    """Synthetic thermal frame: warm rectangle (face) + one hot pixel. Returns (temp_c, box)."""
    h, w = shape
    fh = int(min(max(face_h_px, 10), h - 2)); fw = int(fh * 0.75)
    y0, x0 = (h - fh) // 2, (w - fw) // 2
    t = np.full(shape, ambient_c, dtype=np.float64)
    t[y0:y0 + fh, x0:x0 + fw] = face_c
    t[y0 + fh // 4, x0 + fw // 3] = face_max_c
    return t, (x0, y0, x0 + fw, y0 + fh)


def render_overlay(temp_c, readings, lo=20.0, hi=38.0):
    """Colormapped thermal image with, per face: a box, the measured maximum as a label and a cross on the hottest pixel."""
    g = np.clip((temp_c - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    rgb = cv2.applyColorMap(g, cv2.COLORMAP_INFERNO)[:, :, ::-1].copy()
    for r in readings:
        if r.box is None:
            continue
        color = (255, 60, 60) if r.alert else ((255, 255, 255) if r.status == "ok" else (255, 170, 0))
        x0, y0, x1, y1 = r.box
        cv2.rectangle(rgb, (x0, y0), (x1, y1), color, 2)
        if getattr(r, "hotspot", None) is not None:
            cv2.drawMarker(rgb, tuple(int(v) for v in r.hotspot), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
        if r.face_max_c is not None:
            label = f"{r.face_max_c:.1f} C"
            scale = max(0.5, min(1.2, (x1 - x0) / 220))
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            ty = y0 - 6 if y0 - th - 8 > 0 else y1 + th + 8
            cv2.putText(rgb, label, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
    return rgb


def reading_row(source, r, ts):
    return {"time": ts, "source": source, "status": r.status,
            "face_max_c": None if r.face_max_c is None else round(r.face_max_c, 2),
            "roi_height_px": None if r.box is None else r.box[3] - r.box[1],
            "alert": r.alert}
