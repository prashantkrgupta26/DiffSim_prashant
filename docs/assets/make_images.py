#!/usr/bin/env python
"""Render the landing-page morphology images from Wodo-campaign fields.

These are the DiffSim replication of Wodo & Ganapathysubramanian (CMS 2012,
solvent-based OSC fabrication), computed device-bound on one A100 at the
paper's 250x100 mesh (see docs/projects/phase-field.md).

To SWAP the images: drop new .npy polymer-fraction fields into SRC (each
shaped (ny, nx), values in [0,1]) or edit CASES / STRIP below, then rerun

    python docs/assets/make_images.py

Only Pillow + NumPy are required (no matplotlib). Outputs land in img/.
"""
import os
import glob
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
# Source fields: the Nova campaign mirror if present (gitignored runtime data),
# else a local committed copy under img/_fields/.
SRC_CANDIDATES = [
    os.path.join(HERE, "..", "dev", "nova-mirrors", "wodo_campaign"),
    os.path.join(HERE, "img", "_fields"),
]
OUT = os.path.join(HERE, "img")

# --- colormap: a magma-like perceptual ramp (dark -> ember -> pale) ----------
_ANCHORS = np.array([
    [0.001, 0.000, 0.014],
    [0.107, 0.047, 0.240],
    [0.331, 0.062, 0.429],
    [0.553, 0.161, 0.406],
    [0.774, 0.276, 0.298],
    [0.933, 0.472, 0.216],
    [0.988, 0.735, 0.396],
    [0.987, 0.991, 0.750],
])


def _cmap(v):
    v = np.clip(v, 0.0, 1.0)
    x = v * (len(_ANCHORS) - 1)
    lo = np.floor(x).astype(int)
    lo = np.clip(lo, 0, len(_ANCHORS) - 2)
    t = (x - lo)[..., None]
    c = _ANCHORS[lo] * (1 - t) + _ANCHORS[lo + 1] * t
    return (c * 255).astype(np.uint8)


def _find(stem):
    for root in SRC_CANDIDATES:
        hits = glob.glob(os.path.join(root, f"{stem}*.npy"))
        if hits:
            return sorted(hits)[0]
    return None


def _norm(a):
    lo, hi = np.percentile(a, 1.0), np.percentile(a, 99.0)
    return (a - lo) / max(hi - lo, 1e-9)


def _tile(field, scale=4, pad=0):
    """field (ny,nx) in [0,1] -> RGB PIL image, nearest-upscaled."""
    rgb = _cmap(field)
    img = Image.fromarray(rgb, "RGB")
    img = img.resize((field.shape[1] * scale, field.shape[0] * scale),
                     Image.NEAREST)
    return img


def render_gallery():
    """Image 1: a 2x3 grid of final morphologies across regimes."""
    # (stem, label) — chosen to show the regime diversity M4 reproduced.
    CASES = [
        "f4_bi0.03_full_phip_hfinal",   # slow drying
        "f4_bi3_full_phip_hfinal",      # fast drying
        "f5_blend108_full_phip_hfinal",  # off-critical stratification
        "f6_n5_full_phip_hfinal",       # short chains
        "f6_n100_full_phip_hfinal",     # long chains
        "f7_chips_full_phip_hfinal",    # solvent-selective
    ]
    scale, gap, bg = 3, 10, (18, 18, 22)
    tiles = []
    for stem in CASES:
        f = _find(stem)
        if f is None:
            print(f"  [skip] {stem} (no source field found)")
            continue
        tiles.append(_tile(_norm(np.load(f)), scale=scale))
    if not tiles:
        return False
    tw, th = tiles[0].size
    cols, rows = 3, 2
    W = cols * tw + (cols + 1) * gap
    H = rows * th + (rows + 1) * gap
    canvas = Image.new("RGB", (W, H), bg)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas.paste(t, (gap + c * (tw + gap), gap + r * (th + gap)))
    path = os.path.join(OUT, "morphology_gallery.png")
    canvas.save(path)
    print(f"  wrote {path}  ({W}x{H})")
    return True


def render_strip():
    """Image 2: a drying time-strip of one case (evaporation -> coarsening)."""
    STRIP = "f5_blend108_full_phip"     # off-critical -> vertical stratification
    FRAMES = ["h0.90", "h0.70", "h0.50", "hfinal"]
    scale, gap, bg = 4, 8, (18, 18, 22)
    tiles = []
    for fr in FRAMES:
        f = _find(f"{STRIP}_{fr}")
        if f is None:
            print(f"  [skip] {STRIP}_{fr}")
            continue
        tiles.append(_tile(_norm(np.load(f)), scale=scale))
    if not tiles:
        return False
    tw, th = tiles[0].size
    W = len(tiles) * tw + (len(tiles) + 1) * gap
    H = th + 2 * gap
    canvas = Image.new("RGB", (W, H), bg)
    for i, t in enumerate(tiles):
        canvas.paste(t, (gap + i * (tw + gap), gap))
    path = os.path.join(OUT, "drying_strip.png")
    canvas.save(path)
    print(f"  wrote {path}  ({W}x{H})")
    return True


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("Rendering landing images...")
    ok1 = render_gallery()
    ok2 = render_strip()
    if not (ok1 and ok2):
        print("\nSome fields were missing. Point SRC_CANDIDATES at the Wodo "
              "campaign output, or copy .npy fields into docs/assets/img/_fields/.")
