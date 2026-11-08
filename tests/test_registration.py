import numpy as np
import pytest
from src.alignment.registration import HomographyRegistration

def test_translation_maps_box():
    H = np.array([[1, 0, 10], [0, 1, -5], [0, 0, 1]], float)
    assert np.allclose(HomographyRegistration(H).map_box((0, 0, 20, 30)), (10, -5, 30, 25))

def test_from_points_recovers_known_transform():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 400, (40, 2))
    H = np.array([[0.6, 0.02, 12.0], [-0.01, 0.62, 7.0], [0, 0, 1]])
    truth = HomographyRegistration(H)
    reg, err, n = HomographyRegistration.from_points(src, truth.map_points(src))
    assert err < 1e-2 and n == 40
    assert np.allclose(reg.map_points([[100, 100]]), truth.map_points([[100, 100]]), atol=0.05)

def test_rejects_bad_input():
    with pytest.raises(ValueError):
        HomographyRegistration(np.eye(2))
    with pytest.raises(ValueError):
        HomographyRegistration.from_points([[0, 0]] * 3, [[0, 0]] * 3)
