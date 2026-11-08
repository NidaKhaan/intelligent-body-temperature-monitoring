"""Thermal screening API. Images are processed in memory and never stored."""
import os
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2 ** 25))   # decompression-bomb guard (set before cv2 import)
import json
from dataclasses import asdict
from functools import lru_cache

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from src.pipeline.screener import Calibration, ScreenerConfig, ThermalScreener

MAX_BYTES = 10 * 1024 * 1024
app = FastAPI(title="Thermal Screening API", version="0.1.0")


@lru_cache(maxsize=1)
def get_screener() -> ThermalScreener:
    from src.detection.face_detector import mediapipe_face_detector
    cal = None
    path = os.getenv("CALIBRATION_JSON")        # optional; default = uncalibrated (no alerts)
    if path:
        c = json.load(open(path))
        cal = Calibration(c["slope"], c["intercept"])
    return ThermalScreener(mediapipe_face_detector(), ScreenerConfig(calibration=cal))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/screen")
async def screen(file: UploadFile = File(...), screener: ThermalScreener = Depends(get_screener)):
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "file too large (max 10 MB)")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(422, "could not decode image")
    if img.ndim != 2 or img.dtype != np.uint16:
        raise HTTPException(422, "expected a single-channel 16-bit radiometric image (uint16)")
    return {
        "calibrated": screener.cfg.calibration is not None,
        "scale_assumption": "0.01 K/count, plausible on Charlotte-ThermalFace only, not validated per camera",
        "readings": [asdict(r) for r in screener.process_raw(img)],
    }
