import numpy as np
from src.temperature.radiometric import raw_to_celsius

def test_known_value():
    assert abs(raw_to_celsius(29815) - 25.0) < 1e-9

def test_array_and_dtype():
    out = raw_to_celsius(np.array([29315, 30915], dtype=np.uint16))
    assert np.allclose(out, [20.0, 36.0])

def test_no_uint16_overflow():
    
    assert raw_to_celsius(np.uint16(65535)) > 300
