"""Chapter 04 — drag-comparison bar + numbers/c4.tex from a saved harness run.

    python run.py --config configs/shift.yaml --mode reference --output outputs/sh --overwrite
    python gen_figures.py --run-dir outputs/sh

Writes latex/figures/c4_drag.png and latex/numbers/c4.tex.
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, "..", "latex", "figures")
NUMTEX = os.path.join(HERE, "..", "latex", "numbers", "c4.tex")


def load(run_dir):
    with open(os.path.join(run_dir, "results.json")) as fh:
        return json.load(fh)


def drag_fig(r, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.bar([0, 1], [r["cd_proj"], r["cd_mono"]],
           color=["#0a5aa0", "#a83232"], width=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["projection", "monolithic\n(true shift)"])
    ax.set_ylabel(r"$C_d$")
    ax.set_title(f"SBM shift, Re={r['Re']}, offset={r['offset']}\n"
                 f"$d_{{\\max}}/h$={r['dmax_over_h']:.2f}  "
                 f"(rel {r['cd_rel']*100:.2f}%)")
    for x, v in [(0, r["cd_proj"]), (1, r["cd_mono"])]:
        ax.text(x, v + 0.02, f"{v:+.3f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    fig.savefig(fname, dpi=150)
    print(f"wrote {fname}")


def write_numbers(r):
    os.makedirs(os.path.dirname(NUMTEX), exist_ok=True)
    with open(NUMTEX, "w") as fh:
        fh.write("% Chapter 4 measured numbers (04_shifted_boundary)\n")
        fh.write(f"\\newcommand{{\\ShCdProj}}{{{r['cd_proj']:.4f}}}\n")
        fh.write(f"\\newcommand{{\\ShCdMono}}{{{r['cd_mono']:.4f}}}\n")
        fh.write(f"\\newcommand{{\\ShCdRel}}{{{r['cd_rel']*100:.2f}\\%}}\n")
        fh.write(f"\\newcommand{{\\ShDmax}}{{{r['dmax']:.4f}}}\n")
        fh.write(f"\\newcommand{{\\ShDmaxRatio}}{{{r['dmax_over_h']:.2f}}}\n")
        fh.write(f"\\newcommand{{\\ShMuRel}}{{{r['mu_rel']*100:.2f}\\%}}\n")
    print(f"wrote {NUMTEX}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.join(HERE, "outputs", "sh"))
    args = ap.parse_args()
    r = load(args.run_dir)
    drag_fig(r, os.path.join(FIGDIR, "c4_drag.png"))
    write_numbers(r)


if __name__ == "__main__":
    main()
