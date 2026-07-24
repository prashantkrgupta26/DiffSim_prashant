"""Chapter 05 — 3-D verdict figure + numbers/c5.tex from a saved harness run.

    python run.py --config configs/sphere.yaml --mode reference --output outputs/sp --overwrite
    python gen_figures.py --run-dir outputs/sp

Writes latex/figures/c5_sphere.png (a small fixture/verdict card) and
latex/numbers/c5.tex.
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, "..", "latex", "figures")
NUMTEX = os.path.join(HERE, "..", "latex", "numbers", "c5.tex")


def load(run_dir):
    with open(os.path.join(run_dir, "results.json")) as fh:
        return json.load(fh)


def sphere_fig(r, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.axis("off")
    lines = [
        "3-D immersed sphere (SBM-NS)",
        f"Re={r['Re']}, level={r['level']}, alpha={r['alpha']}",
        "",
        f"n_free = {r['n_free']}    surrogate faces = {r['n_faces']}",
        f"D/h = {r['D_over_h']:.2f}    SBM dmax = {r['dmax']:.4f} (> 0)",
        "",
        f"MONOLITHIC steady Cd = {r['cd_mono']:+.4f}",
        "(positive, physical — the working 3-D drag)",
    ]
    ax.text(0.02, 0.95, "\n".join(lines), va="top", ha="left",
            family="monospace", fontsize=10, transform=ax.transAxes)
    fig.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    fig.savefig(fname, dpi=150)
    print(f"wrote {fname}")


def write_numbers(r):
    os.makedirs(os.path.dirname(NUMTEX), exist_ok=True)
    clat = r.get("pipe_clat_ratio")
    with open(NUMTEX, "w") as fh:
        fh.write("% Chapter 5 measured numbers (05_three_dimensions)\n")
        fh.write(f"\\newcommand{{\\SpCdMono}}{{{r['cd_mono']:.4f}}}\n")
        fh.write(f"\\newcommand{{\\SpNfree}}{{{r['n_free']}}}\n")
        fh.write(f"\\newcommand{{\\SpFaces}}{{{r['n_faces']}}}\n")
        fh.write(f"\\newcommand{{\\SpDoverh}}{{{r['D_over_h']:.2f}}}\n")
        fh.write(f"\\newcommand{{\\SpDmax}}{{{r['dmax']:.4f}}}\n")
        if clat is not None:
            fh.write(f"\\newcommand{{\\SpClatRatio}}{{{clat:.2e}}}\n")
    print(f"wrote {NUMTEX}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.join(HERE, "outputs", "sp"))
    args = ap.parse_args()
    r = load(args.run_dir)
    sphere_fig(r, os.path.join(FIGDIR, "c5_sphere.png"))
    write_numbers(r)


if __name__ == "__main__":
    main()
