import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1] / "frontend" / "samples" / "pairs"


def test_sample_pairs_complete_and_homography_valid():
    metas = sorted(BASE.glob("pair_*.json"))
    assert len(metas) >= 2
    for m in metas:
        assert (BASE / f"{m.stem}_rgb.png").exists() and (BASE / f"{m.stem}_thermal.png").exists()
        H = np.array(json.load(open(m))["H"])
        assert H.shape == (3, 3) and np.isfinite(H).all()
