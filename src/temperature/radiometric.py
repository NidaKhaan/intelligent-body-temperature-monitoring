"""Raw 16-bit thermal counts -> degrees Celsius.

SCALE_K_PER_COUNT = 0.01 is an assumption: it gives plausible skin temperatures on
Charlotte-ThermalFace and agrees with FDA raw T_Max readings. It is NOT validated
against a blackbody reference.
"""
import numpy as np

KELVIN_OFFSET = 273.15
SCALE_K_PER_COUNT = 0.01

def raw_to_celsius(raw, scale=SCALE_K_PER_COUNT):
    return np.asarray(raw, dtype=np.float64) * scale - KELVIN_OFFSET
