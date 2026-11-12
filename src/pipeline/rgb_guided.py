"""RGB-guided screening: face detected on the RGB frame, mapped to the thermal frame by a calibrated
homography, temperature read from the radiometric frame inside the mapped box."""
import numpy as np

from src.alignment.registration import HomographyRegistration
from src.pipeline.screener import FaceReading, ScreenerConfig
from src.temperature.radiometric import raw_to_celsius


class RgbGuidedScreener:
    def __init__(self, rgb_face_detector, registration: HomographyRegistration, config: ScreenerConfig = ScreenerConfig()):
        self.detect, self.reg, self.cfg = rgb_face_detector, registration, config

    def process_celsius(self, rgb, temp_c):
        if (rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8
                or temp_c.ndim != 2 or not np.isfinite(temp_c).all()):
            return [FaceReading(None, None, "invalid_input")]
        h, w = temp_c.shape
        out = []
        for box in self.detect(rgb):
            x0, y0, x1, y1 = self.reg.map_box(box)
            x0, y0 = max(int(round(x0)), 0), max(int(round(y0)), 0)
            x1, y1 = min(int(round(x1)), w), min(int(round(y1)), h)
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            sub = temp_c[y0:y1, x0:x1]
            iy, ix = np.unravel_index(int(np.argmax(sub)), sub.shape)
            r = FaceReading((x0, y0, x1, y1), float(sub[iy, ix]), "ok", hotspot=(x0 + int(ix), y0 + int(iy)))
            if (y1 - y0) < self.cfg.min_face_height_px:
                r.status = "unreliable_roi_small"
            elif self.cfg.calibration is not None:
                r.oral_equiv_c = self.cfg.calibration.apply(r.face_max_c)
                r.alert = bool(r.oral_equiv_c >= self.cfg.alert_threshold_c)
            out.append(r)
        return out or [FaceReading(None, None, "no_face")]

    def process_raw(self, rgb, raw16):
        if raw16.dtype != np.uint16:
            return [FaceReading(None, None, "invalid_input")]
        return self.process_celsius(rgb, raw_to_celsius(raw16))
