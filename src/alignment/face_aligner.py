"""Face alignment module using 2D affine transformation on facial landmarks."""
import cv2
import numpy as np

class FaceAligner:
    def __init__(self, target_size=(256, 256), eye_target_ratio=0.35):
        self.target_size = target_size
        self.eye_target_ratio = eye_target_ratio

    def align_face_by_eyes(self, img, left_eye_center, right_eye_center):
        """Performs 2D affine transformation to align and crop face based on eye centers."""
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # Desired coordinates of eyes in output crop
        desired_left_x = self.target_size[0] * self.eye_target_ratio
        desired_right_x = self.target_size[0] * (1.0 - self.eye_target_ratio)
        desired_y = self.target_size[1] * self.eye_target_ratio

        dist = np.sqrt(dx**2 + dy**2)
        desired_dist = desired_right_x - desired_left_x
        scale = desired_dist / max(dist, 1e-6)

        # Center between eyes
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) / 2.0,
                       (left_eye_center[1] + right_eye_center[1]) / 2.0)

        # Get Rotation Matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        # Translate eyes center to target center
        t_x = self.target_size[0] * 0.5
        t_y = desired_y
        M[0, 2] += (t_x - eyes_center[0])
        M[1, 2] += (t_y - eyes_center[1])

        # Apply affine transformation
        aligned = cv2.warpAffine(img, M, self.target_size, flags=cv2.INTER_LINEAR)
        return aligned, M
