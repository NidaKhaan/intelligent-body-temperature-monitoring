"""Face detector built on the MediaPipe Tasks API (BlazeFace short-range model).

The model file (about 230 KB) is downloaded on first use to ~/.cache/thermal_screening/
or to the path in the FACE_MODEL_PATH environment variable.
"""
import os
import urllib.request
from pathlib import Path

import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
             "blaze_face_short_range/float16/1/blaze_face_short_range.tflite")
MIN_MODEL_BYTES = 10_000


def default_model_path() -> Path:
    env = os.getenv("FACE_MODEL_PATH")
    return Path(env) if env else Path.home() / ".cache" / "thermal_screening" / "blaze_face_short_range.tflite"


def ensure_model(path=None, url: str = MODEL_URL, timeout: float = 30.0) -> Path:
    path = Path(path) if path else default_model_path()
    if path.exists() and path.stat().st_size > MIN_MODEL_BYTES:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
        f.write(r.read())
    if tmp.stat().st_size <= MIN_MODEL_BYTES:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("downloaded face model looks truncated")
    tmp.replace(path)
    return path


def mediapipe_face_detector(min_conf: float = 0.3, model_path=None):
    """Returns detect(img) -> list of (x0, y0, x1, y1) in pixels, highest score first.
    img: HxW uint8 (grayscale) or HxWx3 uint8 (RGB)."""
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    det = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
        base_options=mpt.BaseOptions(model_asset_path=str(ensure_model(model_path))),
        min_detection_confidence=min_conf))

    def detect(img):
        rgb = np.dstack([img] * 3) if img.ndim == 2 else img
        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        dets = sorted(res.detections, key=lambda d: -d.categories[0].score)
        return [(float(d.bounding_box.origin_x), float(d.bounding_box.origin_y),
                 float(d.bounding_box.origin_x + d.bounding_box.width),
                 float(d.bounding_box.origin_y + d.bounding_box.height)) for d in dets]

    return detect
