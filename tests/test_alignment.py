import numpy as np
from src.alignment.face_aligner import FaceAligner

def test_dummy_alignment():
    # Generate dummy image with a white square mimicking eyes
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    aligner = FaceAligner(target_size=(64, 64))
    
    left_eye = (30.0, 40.0)
    right_eye = (70.0, 40.0)
    
    aligned, M = aligner.align_face_by_eyes(img, left_eye, right_eye)
    assert aligned.shape == (64, 64, 3)
    assert M.shape == (2, 3)
