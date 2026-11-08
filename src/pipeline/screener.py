"""Thermal-only screening: radiometric frame -> face box -> face max -> optional calibration -> alert.

Fail-safe rules: no face, tiny ROI, invalid input, or no calibration => no alert, only a status.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from src.temperature.radiometric import raw_to_celsius

@dataclass(frozen=True)
class Calibration:
    slope: float
    intercept: float
    def apply(self, t_c: float) -> float:
        return self.slope * t_c + self.intercept

@dataclass(frozen=True)
class ScreenerConfig:
    window_c: tuple = (20.0, 38.0)        # fixed thermal -> 8-bit window for the face detector (assumption)
    min_face_height_px: int = 100         # provisional, from Charlotte distance analysis (one camera)
    alert_threshold_c: float = 37.5       # applied to calibrated (oral-equivalent) temperature
    calibration: Optional[Calibration] = None

@dataclass
class FaceReading:
    box: Optional[tuple]
    face_max_c: Optional[float]
    status: str                            # ok | no_face | unreliable_roi_small | invalid_input
    oral_equiv_c: Optional[float] = None
    alert: Optional[bool] = None

class ThermalScreener:
    def __init__(self, face_detector, config: ScreenerConfig = ScreenerConfig()):
        self.detect, self.cfg = face_detector, config

    def to_gray8(self, temp_c):
        lo, hi = self.cfg.window_c
        return np.clip((temp_c - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

    def process_celsius(self, temp_c):
        if temp_c.ndim != 2 or not np.isfinite(temp_c).all():
            return [FaceReading(None, None, 'invalid_input')]
        h, w = temp_c.shape
        out = []
        for x0, y0, x1, y1 in self.detect(self.to_gray8(temp_c)):
            x0, y0, x1, y1 = max(int(x0), 0), max(int(y0), 0), min(int(x1), w), min(int(y1), h)
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            r = FaceReading((x0, y0, x1, y1), float(temp_c[y0:y1, x0:x1].max()), 'ok')
            if (y1 - y0) < self.cfg.min_face_height_px:
                r.status = 'unreliable_roi_small'
            elif self.cfg.calibration is not None:
                r.oral_equiv_c = self.cfg.calibration.apply(r.face_max_c)
                r.alert = bool(r.oral_equiv_c >= self.cfg.alert_threshold_c)
            out.append(r)
        return out or [FaceReading(None, None, 'no_face')]

    def process_raw(self, raw16):
        if raw16.dtype != np.uint16:
            return [FaceReading(None, None, 'invalid_input')]
        return self.process_celsius(raw_to_celsius(raw16))
