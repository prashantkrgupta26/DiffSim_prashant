"""Chapter 03 — drag-comparison bar + numbers/c3.tex from a saved harness run.

    python run.py --config configs/square.yaml --mode reference --output outputs/sq --overwrite
    python gen_figures.py --run-dir outputs/sq

Writes latex/figures/c3_drag.png and latex/numbers/c3.tex (the macros the
course document cites for Chapter 3).
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, "..", "latex", "figures")
NUMTEX = os.path.join(HERE, "..", "latex", "numbers", "c3.tex")


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
    ax.set_xticklabels(["projection", "monolithic"])
    ax.set_ylabel(r"$C_d$")
    ax.set_title(f"Flow past a square (d=0)\nRe={r['Re']}, weak Nitsche"
                 f"  (rel {r['cd_rel']*100:.1f}%)")
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
        fh.write("% Chapter 3 measured numbers (03_weak_dirichlet_nitsche)\n")
        fh.write(f"\\newcommand{{\\SqCdProj}}{{{r['cd_proj']:.4f}}}\n")
        fh.write(f"\\newcommand{{\\SqCdMono}}{{{r['cd_mono']:.4f}}}\n")
        fh.write(f"\\newcommand{{\\SqCdRel}}{{{r['cd_rel']*100:.2f}\\%}}\n")
        fh.write(f"\\newcommand{{\\SqMuProj}}{{{r['mean_u_proj']:.4f}}}\n")
        fh.write(f"\\newcommand{{\\SqMuMono}}{{{r['mean_u_mono']:.4f}}}\n")
        fh.write(f"\\newcommand{{\\SqMuRel}}{{{r['mu_rel']*100:.2f}\\%}}\n")
    print(f"wrote {NUMTEX}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.join(HERE, "outputs", "sq"))
    args = ap.parse_args()
    r = load(args.run_dir)
    drag_fig(r, os.path.join(FIGDIR, "c3_drag.png"))
    write_numbers(r)


if __name__ == "__main__":
    main()
