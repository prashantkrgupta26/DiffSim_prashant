"""OrgElMorph course - C0: figures + numbers from the saved harness run.

Renders the pipeline diagram and the verification figure from
``outputs/c0/{results.json,history.npz}`` and writes ``latex/numbers/c0.tex``
(the document's numbers ARE this run's output).  Run the harness first::

    python run.py --config configs/c0.yaml --device cpu --solver splu \\
        --output outputs/c0 --overwrite --mode reference
    python gen_figures.py --run-dir outputs/c0
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
NUMTEX = os.path.join(HERE, "..", "..", "latex", "numbers", "c0.tex")


def load(run_dir):
    with open(os.path.join(run_dir, "results.json")) as fh:
        results = json.load(fh)
    hist = dict(np.load(os.path.join(run_dir, "history.npz")))
    return results, hist


# ---------------------------------------------------------------------------
def pipeline_fig(fname):
    """The weak-form -> CUDA pipeline: each stage box maps to a diffsim file."""
    stages = [
        ("Strong form", r"$-\nabla^2 u + \alpha u^3 = f$", ""),
        ("Weak form (IBP)", r"$\int\nabla w\!\cdot\!\nabla u + \alpha\!\int w u^3 = \int w f$", ""),
        ("Element residual\n+ Jacobian", r"$R_a,\ J_{ab}$ at Gauss points",
         "physics/cahn_hilliard.py: make_ch_newton\napi/example_bricks.py: PoissonBrick"),
        ("Quadrature", r"$\sum_q(\cdot)\,|J|w_q$",
         "mesh/basis.py: basis_tables, gauss_1d\nassembly/femelm.py: fe_N, fe_dN_s"),
        ("Local -> global\n(scatter)", r"$\mathrm{conn}[e,a]\to$ global dof",
         "api/equation.py: assemble_brick_csr\nassembly/device_assembly.py"),
        ("CSR assembly", r"COO $\to$ CSR,  $T^{\!\top}KT$",
         "assembly/operators.py: assemble_csr\nvolume_triplets, CSROperator"),
        ("Constraints / BCs", r"$T$ (hanging),  Dirichlet rows",
         "mesh/constraints.py: build_constraints\nassembly/dirichlet.py: _BCOperator"),
        ("Newton", r"$J\,\delta u=-R$,  update",
         "solvers/newton.py: NonlinearSolver\ncahn_hilliard.py: step()"),
        ("Linear solve", r"splu / cuDSS / blockch",
         "solvers/linsolve.py: solve_linear\nsolvers/result.py: LinearSolveResult"),
        ("Device update", r"$u\leftarrow u+\delta u$,  sync",
         "warp arrays, wp.launch, .numpy()"),
    ]
    n = len(stages)
    fig, ax = plt.subplots(figsize=(8.4, 12.2), dpi=130)
    ax.set_xlim(0, 10); ax.set_ylim(0, n * 1.2 + 0.4); ax.axis("off")
    yv = [n * 1.2 - 0.2 - i * 1.2 for i in range(n)]
    for (title, math, files), y in zip(stages, yv):
        box = FancyBboxPatch((0.4, y - 0.5), 5.1, 0.95,
                             boxstyle="round,pad=0.06", linewidth=1.4,
                             edgecolor="#0a5aa0", facecolor="#eaf2fb")
        ax.add_patch(box)
        ax.text(0.6, y + 0.13, title, fontsize=10.5, fontweight="bold",
                va="center", ha="left")
        ax.text(0.6, y - 0.26, math, fontsize=9.5, va="center", ha="left",
                color="#333")
        if files:
            ax.text(5.8, y, files, fontsize=7.6, va="center", ha="left",
                    family="monospace", color="#7a2828")
    for y0, y1 in zip(yv[:-1], yv[1:]):
        ax.add_patch(FancyArrowPatch((2.95, y0 - 0.52), (2.95, y1 + 0.46),
                     arrowstyle="-|>", mutation_scale=13, color="#0a5aa0",
                     linewidth=1.3))
    ax.text(5.0, n * 1.2 + 0.1, "one scalar problem, every stage",
            fontsize=12, fontweight="bold", ha="center")
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def verify_fig(results, hist, fname):
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=140)
    # --- left: MMS spatial convergence ---
    for key, mark, lab in (("p1", "o", "p=1"), ("p2", "s", "p=2")):
        lv = np.asarray(hist[f"{key}_levels"]); e = np.asarray(hist[f"{key}_l2"])
        h = 2.0 ** (-lv.astype(float))
        ax[0].loglog(h, e, mark + "-", lw=1.8, ms=6, label=f"{lab} L2")
        p = int(key[1]); c = e[0] / h[0] ** (p + 1)
        ax[0].loglog(h, c * h ** (p + 1), "k--", lw=1.0, alpha=0.6)
    ax[0].set_xlabel("mesh size $h$"); ax[0].set_ylabel(r"$L^2$ error")
    ax[0].set_title(f"MMS convergence: order {results['p1_order']:.2f} (p1), "
                    f"{results['p2_order']:.2f} (p2)")
    ax[0].legend(); ax[0].grid(alpha=0.3, which="both")
    # --- right: Newton quadratic convergence ---
    rh = np.asarray(hist["newton_res_history"])
    ax[1].semilogy(range(len(rh)), rh, "C3o-", lw=1.8, ms=6)
    ax[1].set_xlabel("Newton iteration"); ax[1].set_ylabel("residual norm")
    ax[1].set_title(f"Newton convergence (quadratic; correct Jacobian)")
    ax[1].grid(alpha=0.3, which="both")
    ax[1].text(0.5, 0.9, f"analytic Jacobian vs FD:\ncomplete {results['jac_fd_rel_error']:.1e}\n"
               f"starter {results['jac_fd_rel_error_starter']:.1e}",
               transform=ax[1].transAxes, fontsize=9, va="top", ha="left",
               bbox=dict(boxstyle="round", fc="#f3f3f3", ec="#999"))
    fig.suptitle("C0 verification: the whole pipeline converges at the predicted rate",
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

    p1, p2 = results["p1"], results["p2"]
    lines = [
        "% AUTO-GENERATED by gen_figures.py from results.json - do not edit.",
        mac("CzeroAlpha", f"{results['alpha']:g}"),
        mac("CzeroPoneOrder", f"{results['p1_order']:.2f}"),
        mac("CzeroPtwoOrder", f"{results['p2_order']:.2f}"),
        mac("CzeroPoneErrCoarse", sci(p1["l2"][0])),
        mac("CzeroPoneErrFine", sci(p1["l2"][-1])),
        mac("CzeroPtwoErrCoarse", sci(p2["l2"][0])),
        mac("CzeroPtwoErrFine", sci(p2["l2"][-1])),
        mac("CzeroNewtonIters", f"{results['newton_iters_level5']}"),
        mac("CzeroNewtonResZero", sci(results["newton_res0"])),
        mac("CzeroNewtonResFinal", sci(results["newton_res_final"])),
        mac("CzeroJacOk", sci(results["jac_fd_rel_error"])),
        mac("CzeroJacStarter", sci(results["jac_fd_rel_error_starter"])),
        mac("CzeroStiffDiff",
            r"\ensuremath{0}" if results["stiffness_vs_production_max"] == 0.0
            else sci(results["stiffness_vs_production_max"])),
    ]
    os.makedirs(os.path.dirname(NUMTEX), exist_ok=True)
    with open(NUMTEX, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", NUMTEX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.join(HERE, "outputs", "c0"))
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    results, hist = load(args.run_dir)
    pipeline_fig("c0_pipeline.png")
    verify_fig(results, hist, "c0_convergence.png")
    write_numbers(results)


if __name__ == "__main__":
    main()
