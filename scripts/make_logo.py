"""Generate the project logo. A logo already present in frontend/assets is kept unless --force is given.
Usage: python scripts/make_logo.py [--force]"""
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]
ASSETS, DOCS = REPO / "frontend" / "assets", REPO / "docs" / "images"


def draw_icon(ax):
    clip = FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0,rounding_size=0.2", fc="none", ec="none")
    ax.add_patch(clip)
    cmap = LinearSegmentedColormap.from_list("heat", ["#1e3a8a", "#7c3aed", "#f97316"])
    im = ax.imshow(np.linspace(0, 1, 256).reshape(-1, 1), extent=(0, 1, 0, 1), origin="lower", cmap=cmap, aspect="auto")
    im.set_clip_path(clip)
    ax.add_patch(Ellipse((0.5, 0.54), 0.54, 0.72, fill=False, ec="white", lw=3, alpha=0.45))
    ax.add_patch(FancyBboxPatch((0.455, 0.36), 0.09, 0.42, boxstyle="round,pad=0,rounding_size=0.045", fc="white", ec="none"))
    ax.add_patch(Circle((0.5, 0.31), 0.095, fc="white", ec="none"))
    ax.add_patch(FancyBboxPatch((0.48, 0.34), 0.04, 0.26, boxstyle="round,pad=0,rounding_size=0.02", fc="#ef4444", ec="none"))
    ax.add_patch(Circle((0.5, 0.31), 0.06, fc="#ef4444", ec="none"))
    for x, y, dx, dy in [(0.14, 0.86, 1, -1), (0.86, 0.86, -1, -1), (0.14, 0.14, 1, 1), (0.86, 0.14, -1, 1)]:
        ax.plot([x, x + dx * 0.09], [y, y], c="white", lw=4, solid_capstyle="round")
        ax.plot([x, x], [y, y + dy * 0.09], c="white", lw=4, solid_capstyle="round")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")


def main(force=False):
    ASSETS.mkdir(parents=True, exist_ok=True); DOCS.mkdir(parents=True, exist_ok=True)
    icon, wide = ASSETS / "logo_icon.png", ASSETS / "logo_wide.png"
    if icon.exists() and wide.exists() and not force:
        print("logo exists, kept"); return
    fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
    draw_icon(fig.add_axes([0, 0, 1, 1]))
    fig.savefig(icon, dpi=100, transparent=True); plt.close(fig)
    fig = plt.figure(figsize=(11, 3.2), dpi=100)
    bg = fig.add_axes([0, 0, 1, 1]); bg.axis("off"); bg.set_xlim(0, 1); bg.set_ylim(0, 1)
    bg.add_patch(FancyBboxPatch((0.005, 0.02), 0.99, 0.96, boxstyle="round,pad=0,rounding_size=0.08", fc="#0f172a", ec="none"))
    draw_icon(fig.add_axes([0.03, 0.1, 2.56 / 11, 0.8]))
    fig.text(0.30, 0.60, "Thermal Screening", color="white", fontsize=44, fontweight="bold", va="center")
    fig.text(0.30, 0.30, "a face-temperature reading, with its limits", color="#cbd5e1", fontsize=17, va="center")
    fig.savefig(wide, dpi=100, transparent=True); plt.close(fig)
    shutil.copyfile(wide, DOCS / "logo_wide.png")
    print("logo written")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
