"""OrgElMorph course - P0: hierarchy diagram + nondimensionalization figure.

Renders the model-hierarchy map and the interface-resolution figure, and writes
latex/numbers/p0.tex from the saved harness run.  Run the harness first::

    python run.py --config configs/p0.yaml --device cpu --mode reference \\
        --output outputs/p0 --overwrite
    python gen_figures.py --run-dir outputs/p0
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, "..", "..", "latex", "figures")
NUMTEX = os.path.join(HERE, "..", "..", "latex", "numbers", "p0.tex")


def load(run_dir):
    with open(os.path.join(run_dir, "results.json")) as fh:
        results = json.load(fh)
    hist = dict(np.load(os.path.join(run_dir, "history.npz")))
    return results, hist


# ---------------------------------------------------------------------------
def hierarchy_fig(fname):
    """The eight models of the course, each adding one physical ingredient."""
    rows = [
        ("P1  Binary Cahn-Hilliard", r"conserved $\phi$", "two-component demixing; $F=$ bulk $+$ gradient", "#0a5aa0"),
        ("P3  + Substrate BCs", r"conserved $\phi$", "wetting / no-flux boundaries; contact energy", "#0a5aa0"),
        ("P4  Ternary Cahn-Hilliard", r"conserved $\phi_1,\phi_2$", "solvent + two solutes; Gibbs simplex", "#0a5aa0"),
        ("P5  + Moving-frame evaporation", r"conserved $\phi_i$, $h(t)$", "shrinking film; Biot/Peclet drying", "#0a5aa0"),
        ("P6  Allen-Cahn crystallization", r"non-conserved $\psi$", "crystal grows/melts; Avrami kinetics", "#a83232"),
        ("P7  Coupled CH + AC", r"$\phi_i$ (cons.) $+\ \psi$ (non-cons.)", "crystallization-driven demixing; four-fold $\\chi$", "#7a2aa0"),
        ("P8  + Stochastic nucleation", r"$+$ noise (FDT)", "thermal fluctuations seed nuclei; ensembles", "#7a2aa0"),
        ("P9  Evaporation-conditioned", r"$\phi_i,h(t),\psi$", "drying sets where embryos grow", "#7a2aa0"),
    ]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(9.2, 10.6), dpi=130)
    ax.set_xlim(0, 10); ax.set_ylim(0, n * 1.25 + 0.6); ax.axis("off")
    yv = [n * 1.25 - 0.3 - i * 1.25 for i in range(n)]
    for (title, op, desc, col), y in zip(rows, yv):
        box = FancyBboxPatch((0.4, y - 0.52), 6.0, 1.0,
                             boxstyle="round,pad=0.05", linewidth=1.6,
                             edgecolor=col, facecolor=col + "14")
        ax.add_patch(box)
        ax.text(0.62, y + 0.2, title, fontsize=11, fontweight="bold",
                va="center", ha="left", color=col)
        ax.text(0.62, y - 0.12, desc, fontsize=8.8, va="center", ha="left",
                color="#333")
        ax.text(6.7, y, op, fontsize=9.5, va="center", ha="left")
    for y0, y1 in zip(yv[:-1], yv[1:]):
        ax.add_patch(FancyArrowPatch((3.4, y0 - 0.54), (3.4, y1 + 0.44),
                     arrowstyle="-|>", mutation_scale=13, color="#666",
                     linewidth=1.3))
    ax.text(5.0, n * 1.25 + 0.25,
            "The model hierarchy: each concept adds one ingredient",
            fontsize=12.5, fontweight="bold", ha="center")
    # a legend for the order-parameter colours
    ax.text(0.5, -0.05, "blue = conserved (Cahn-Hilliard)   "
            "red = non-conserved (Allen-Cahn)   purple = coupled",
            fontsize=8.5, color="#555", ha="left")
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def nondim_fig(results, hist, fname):
    """The worked nondimensionalization: interface cells vs mesh level."""
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3), dpi=140)
    lv = np.asarray(hist["levels"]); cells = np.asarray(hist["poly_cells_vs_level"])
    ax[0].plot(lv, cells, "C0o-", lw=1.9, ms=7, label="poly interface (cells)")
    ax[0].axhline(results["target_interface_cells"], color="C3", ls="--",
                  label=f"{results['target_interface_cells']:.0f}-cell resolution floor")
    ax[0].axvline(results["level"], color="0.5", ls=":",
                  label=f"P1 tutorial level {results['level']}")
    ax[0].plot([results["level"]], [results["poly_interface_cells"]], "C0*", ms=15)
    ax[0].set_xlabel("mesh level (cells/side $=2^{\\mathrm{level}}$)")
    ax[0].set_ylabel("interface width (mesh cells)")
    ax[0].set_title(f"Resolution constraint (Cahn number $\\tilde\\kappa={results['cahn_number']:.1e}$)")
    ax[0].legend(fontsize=8.5); ax[0].grid(alpha=0.3)
    # right: the scale-separation schematic
    ax[1].axis("off")
    txt = (r"$\bf{One\ nondimensional\ group}$" "\n\n"
           r"$\tilde\kappa = (\lambda/L)^2 = $"
           f"({results['lam_over_L']:.4f})$^2$ = {results['cahn_number']:.1e}\n\n"
           r"$\lambda$ (interface) = "
           f"{results['interface_length_nm']:.1f} nm\n"
           r"$L$ (domain) = " f"{results['domain_size_nm']:.0f} nm\n\n"
           r"interface width $\tilde\ell = \sqrt{2\tilde\kappa}$ = "
           f"{results['poly_interface_ell_tilde']:.4f}\n"
           f"  -> {results['poly_interface_cells']:.2f} cells at level {results['level']}\n"
           f"  (matches the P1 measured value)\n\n"
           r"$\bf{Crystallization\ drive}$ $\Delta h(T/T_m-1)$:" "\n"
           f"accelerated {results['cryst_drive_accel']:.2f} "
           f"vs physical {results['cryst_drive_phys']:.2f} "
           f"({results['cryst_drive_ratio']:.1f}x)")
    ax[1].text(0.02, 0.98, txt, transform=ax[1].transAxes, fontsize=10.5,
               va="top", ha="left", family="sans-serif",
               bbox=dict(boxstyle="round", fc="#eef4fb", ec="#0a5aa0"))
    fig.suptitle("The one worked nondimensionalization (binary Cahn-Hilliard)",
                 fontsize=12, y=1.02)
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


# ---------------------------------------------------------------------------
def write_numbers(results):
    def mac(name, val):
        return rf"\newcommand{{\{name}}}{{{val}}}"

    def sci(x, sig=2):
        mant, exp = f"{x:.{sig-1}e}".split("e")
        return f"\\ensuremath{{{mant}\\times10^{{{int(exp)}}}}}"

    lines = [
        "% AUTO-GENERATED by gen_figures.py from results.json - do not edit.",
        mac("PzeroLam", f"{results['interface_length_nm']:.1f}"),
        mac("PzeroL", f"{results['domain_size_nm']:.0f}"),
        mac("PzeroLamL", f"{results['lam_over_L']:.4f}"),
        mac("PzeroCahn", sci(results["cahn_number"])),
        mac("PzeroPolyEll", f"{results['poly_interface_ell_tilde']:.4f}"),
        mac("PzeroPolyCells", f"{results['poly_interface_cells']:.2f}"),
        mac("PzeroFHCells", f"{results['fh_interface_cells']:.2f}"),
        mac("PzeroFpp", f"{results['fh_curvature_fpp']:.2f}"),
        mac("PzeroLevel", f"{results['level']}"),
        mac("PzeroMinLevel", f"{results['min_level_for_target']}"),
        mac("PzeroTargetCells", f"{results['target_interface_cells']:.0f}"),
        mac("PzeroTm", f"{results['cryst_Tm_K']:.0f}"),
        mac("PzeroDriveAccel", f"{results['cryst_drive_accel']:.2f}"),
        mac("PzeroDrivePhys", f"{results['cryst_drive_phys']:.2f}"),
        mac("PzeroDriveRatio", f"{results['cryst_drive_ratio']:.1f}"),
    ]
    os.makedirs(os.path.dirname(NUMTEX), exist_ok=True)
    with open(NUMTEX, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", NUMTEX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.join(HERE, "outputs", "p0"))
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    results, hist = load(args.run_dir)
    hierarchy_fig("p0_hierarchy.png")
    nondim_fig(results, hist, "p0_nondim.png")
    write_numbers(results)


if __name__ == "__main__":
    main()
