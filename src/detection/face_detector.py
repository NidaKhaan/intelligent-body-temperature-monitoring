import numpy as np
from pathlib import Path
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def mediapipe_face_detector(min_conf=0.3):
    """Returns detect(gray8 HxW uint8) -> list of (x0, y0, x1, y1) in pixels using the modern Tasks API."""
    model_path = Path("/content/cache/detector.tflite")
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
        urllib.request.urlretrieve(url, model_path)

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=min_conf)
    detector = vision.FaceDetector.create_from_options(options)

    def detect(gray8):
        h, w = gray8.shape
        rgb = np.dstack([gray8] * 3)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect(mp_image)
        out = []
        for detection in res.detections:
            box = detection.bounding_box
            xmin = max(0, box.origin_x)
            ymin = max(0, box.origin_y)
            xmax = min(w, box.origin_x + box.width)
            ymax = min(h, box.origin_y + box.height)
            out.append((xmin, ymin, xmax, ymax))
        return out
    return detect
