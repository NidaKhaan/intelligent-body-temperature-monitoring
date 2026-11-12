"""FDA hotspot-region figure from the bundled table. Usage: python scripts/make_figures_regions.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.temperature.regions import hotspot_region_shares

df = pd.read_csv(REPO / "frontend/data/fda_flir_features.csv")
res = {str(r): hotspot_region_shares(df, rnd=r) for r in (1, 2, 3, 4)}
(REPO / "docs/results").mkdir(parents=True, exist_ok=True)
json.dump(res, open(REPO / "docs/results/fda_hotspot_regions.json", "w"), indent=2)

labels = ["inner canthi", "forehead", "mouth", "none of the three"]
fig, ax = plt.subplots(figsize=(9, 4.6)); w = 0.2
for i, r in enumerate((1, 2, 3, 4)):
    m = res[str(r)]
    vals = [m["matches_any"]["inner canthi"], m["matches_any"]["forehead"], m["matches_any"]["mouth"], m["none_of_the_three"]]
    ax.bar(np.arange(4) + (i - 1.5) * w, vals, w, label=f"round {r} (n={m['n']})")
ax.set_xticks(np.arange(4)); ax.set_xticklabels(labels); ax.set_ylim(0, 1); ax.set_ylabel("share of volunteers"); ax.legend(fontsize=8)
ax.set_title("Where on the face is the hottest point? (FDA data)\nA region contains the hotspot when its own maximum equals the whole-face "
             "maximum (within 0.05 C); regions overlap", fontsize=9)
plt.tight_layout(); (REPO / "docs/images").mkdir(parents=True, exist_ok=True)
plt.savefig(REPO / "docs/images/fda_hotspot_regions.png", dpi=140); plt.close()
print(json.dumps({k: {"n": v["n"], **{n: round(x, 3) for n, x in v["matches_any"].items()}, "none": round(v["none_of_the_three"], 3)} for k, v in res.items()}))
