import numpy as np
import pytest

from src.pipeline.simulation import region_boxes, simulate_thermal

BOX = (120, 60, 280, 260)


def photo():
    return np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8)


def test_max_inside_box_equals_requested_and_outside_is_cooler():
    t, _ = simulate_thermal(photo(), BOX, 37.2)
    x0, y0, x1, y1 = BOX
    assert abs(t[y0:y1, x0:x1].max() - 37.2) < 1e-3
    outside = t.copy(); outside[y0:y1, x0:x1] = -np.inf
    assert outside.max() < 33.0


def test_background_near_ambient_and_regions_inside_face():
    t, _ = simulate_thermal(photo(), BOX, 36.4)
    assert abs(t[5, 5] - 24.0) < 3.0
    x0, y0, x1, y1 = BOX
    for rx0, ry0, rx1, ry1 in region_boxes(BOX).values():
        assert x0 <= rx0 < rx1 <= x1 and y0 <= ry0 < ry1 <= y1


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        simulate_thermal(np.zeros((10, 10), np.uint8), (1, 1, 5, 5), 36.0)
    with pytest.raises(ValueError):
        simulate_thermal(photo(), (10, 10, 12, 12), 36.0)
