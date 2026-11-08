"""RGB -> thermal registration with a calibrated homography (fixed camera pair)."""
import numpy as np
import cv2


class HomographyRegistration:
    def __init__(self, H):
        H = np.asarray(H, dtype=np.float64)
        if H.shape != (3, 3) or not np.isfinite(H).all():
            raise ValueError("H must be a finite 3x3 matrix")
        self.H = H

    @classmethod
    def from_points(cls, src, dst, ransac_thr=3.0):
        """Returns (registration, mean reprojection error in px over inliers, number of inliers)."""
        src, dst = np.asarray(src, np.float32), np.asarray(dst, np.float32)
        if len(src) != len(dst) or len(src) < 4:
            raise ValueError("need at least 4 matching points")
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thr)
        if H is None:
            raise ValueError("homography estimation failed")
        reg, mask = cls(H), inl.ravel().astype(bool)
        err = np.linalg.norm(reg.map_points(src[mask]) - dst[mask], axis=1)
        return reg, float(err.mean()), int(mask.sum())

    def map_points(self, pts):
        pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H.astype(np.float64)).reshape(-1, 2)

    def map_box(self, box):
        x0, y0, x1, y1 = box
        c = self.map_points([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        return (float(c[:, 0].min()), float(c[:, 1].min()), float(c[:, 0].max()), float(c[:, 1].max()))
