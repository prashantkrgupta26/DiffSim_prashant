"""README hero renders from the verified S3b L6 evaporation-quench
run: physical-frame film (height h(t) on a substrate) with an RGB
composition composite and theta-hued crystals."""
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from scipy.interpolate import griddata

import sys

sys.path.insert(0, "/home/bglab/Baskar/DiffSim/src")
from diffsim.physics.multiphase import grain_labels

SRC = "/home/bglab/Baskar/s3_renders_L6"
OUT = "/tmp/claude-1000/-home-bglab-Baskar-DiffSim/" \
      "de8f9b3d-9dd7-48ec-8acc-9f5aeb605975/scratchpad"
NG = 420

C_F = np.array([0.95, 0.55, 0.10])     # fullerene-class SM: amber
C_P = np.array([0.10, 0.18, 0.38])     # polymer: deep navy
C_S = np.array([0.93, 0.95, 0.97])     # solvent: near-white


def load(i):
    d = np.load(f"{SRC}/snap_{i:04d}.npz")
    return d


# distinct grain hues (colorblind-friendly-ish, high sat)
GRAIN_RGB = np.array([[0.24, 0.44, 0.85],    # blue
                      [0.16, 0.65, 0.36],    # green
                      [0.72, 0.26, 0.72],    # purple
                      [0.90, 0.32, 0.20],    # vermilion
                      [0.16, 0.62, 0.66],    # teal
                      [0.85, 0.60, 0.10]])   # gold


def grid_fields(d):
    xy = d["coords"]
    gx, gy = np.meshgrid(np.linspace(0, 1, NG),
                         np.linspace(0, 1, NG))
    out = {}
    for k in ("phi_f", "phi_p", "psi", "theta"):
        out[k] = griddata(xy, d[k], (gx, gy), method="linear")
        nn = griddata(xy, d[k], (gx, gy), method="nearest")
        out[k] = np.where(np.isfinite(out[k]), out[k], nn)
    # grain id = nearest implant marker (theta = 0.3 + 0.4 k): the
    # frozen-marker semantics — marker identity IS grain identity,
    # deterministic and consistent across frames
    th = d["theta"]
    k = np.rint((th - 0.3) / 0.4)
    near = np.abs(th - (0.3 + 0.4 * k)) < 0.19
    gid = np.where(near & (k >= 0) & (k < len(GRAIN_RGB)),
                   k + 1, 0.0)
    gg = griddata(xy, gid, (gx, gy), method="nearest")
    # categorical de-speckle: thin theta-advection bands quantize to
    # neighbouring marker ids — a median filter removes the stripes
    from scipy.ndimage import median_filter
    out["grain"] = median_filter(gg, size=13, mode="nearest")
    return out


def compose(g):
    f = np.clip(g["phi_f"], 0, 1)[..., None]
    p = np.clip(g["phi_p"], 0, 1)[..., None]
    s = np.clip(1.0 - f - p, 0, 1)
    rgb = f * C_F + p * C_P + s * C_S
    # crystals: grain-hued overlay, weight = CRYSTALLINE VOLUME
    # phi_f * psi (psi alone is bookkeeping where phi_f ~ 0)
    psi = np.clip(g["psi"], 0, 1)
    cv = np.clip(g["phi_f"], 0, 1) * psi
    gr = np.rint(g["grain"]).astype(int)
    cry = np.where((gr > 0)[..., None],
                   GRAIN_RGB[np.clip(gr - 1, 0, len(GRAIN_RGB) - 1)],
                   np.array([0.98, 0.80, 0.35]))   # untagged: amber
    w = np.clip((cv - 0.30) / 0.30, 0, 1)[..., None] * 0.92
    return np.clip(rgb * (1 - w) + cry * w, 0, 1)


def hero_strip(frames, fname):
    fig, axes = plt.subplots(1, len(frames), figsize=(15.5, 3.4),
                             dpi=300)
    for ax, i in zip(axes, frames):
        d = load(i)
        h = float(d["h"])
        rgb = compose(grid_fields(d))
        ax.imshow(rgb, origin="lower", extent=[0, 1, 0, h],
                  aspect="auto", interpolation="bilinear")
        # substrate
        ax.axhspan(-0.045, 0.0, color="0.25", zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.045, 1.16)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"t = {float(d['t']):.1f}   h = {h:.2f}",
                     fontsize=11)
        # evaporation arrows over the free surface
        for xa in (0.3, 0.7):
            ax.annotate("", xy=(xa, h + 0.15),
                        xytext=(xa, h + 0.03),
                        arrowprops=dict(arrowstyle="-|>", lw=1.4,
                                        color="0.55"))
    axes[0].set_ylabel("film height", fontsize=11)
    axes[0].set_yticks([])
    fig.suptitle("Evaporation-induced phase separation and "
                 "crystallization (M5 OrgElMorph, verified S3 run)",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def final_pair(i, fname):
    d = load(i)
    h = float(d["h"])
    g = grid_fields(d)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.0), dpi=300)
    rgb = compose(g)
    axes[0].imshow(rgb, origin="lower", extent=[0, 1, 0, h],
                   aspect="auto", interpolation="bilinear")
    axes[0].set_title(f"dry-film morphology  (t = {float(d['t']):.1f},"
                      f"  h = {h:.2f})", fontsize=11)
    # grains-only panel: watershed labels, crystalline-volume weight
    cv = np.clip(g["phi_f"], 0, 1) * np.clip(g["psi"], 0, 1)
    gr = np.rint(g["grain"]).astype(int)
    cry = np.where((gr > 0)[..., None],
                   GRAIN_RGB[np.clip(gr - 1, 0, len(GRAIN_RGB) - 1)],
                   np.array([0.98, 0.80, 0.35]))
    w = np.clip((cv - 0.30) / 0.30, 0, 1)[..., None]
    gray = np.full_like(cry, 0.90)
    img = gray * (1 - w) + cry * w
    axes[1].imshow(img, origin="lower", extent=[0, 1, 0, h],
                   aspect="auto", interpolation="bilinear")
    axes[1].set_title("individual crystals (orientation-marker "
                      "identity)", fontsize=11)
    for ax in axes:
        ax.axhspan(-0.012, 0.0, color="0.25", zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.012, h * 1.15)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


hero_strip([2, 13, 24, 28, 31, 37], "hero_film_drying.png")
final_pair(37, "hero_final_grains.png")
