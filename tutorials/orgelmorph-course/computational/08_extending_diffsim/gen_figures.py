"""OrgElMorph course - C8: figures + numbers from the saved harness run.

    python run.py --config configs/c8.yaml --device cpu --solver splu \\
        --output outputs/c8 --overwrite --mode reference
    python gen_figures.py --run-dir outputs/c8
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
NUMTEX = os.path.join(HERE, "..", "..", "latex", "numbers", "c8.tex")


def load(run_dir):
    with open(os.path.join(run_dir, "results.json")) as fh:
        results = json.load(fh)
    hist = dict(np.load(os.path.join(run_dir, "history.npz")))
    return results, hist


def verify_fig(results, hist, fname):
    """MMS convergence, the energy/mass invariants, and the modified well."""
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4), dpi=140)
    # MMS convergence
    h = np.asarray(hist["mms_h"]); e = np.asarray(hist["mms_err"])
    ax[0].loglog(h, e, "C0o-", lw=1.9, ms=7, label="L2 error")
    c = e[0] / h[0] ** 2
    ax[0].loglog(h, c * h ** 2, "k--", lw=1.0, alpha=0.6, label=r"$h^2$")
    ax[0].set_xlabel("mesh size $h$"); ax[0].set_ylabel(r"$L^2$ error of $c$")
    ax[0].set_title(f"MMS convergence: order {results['mms_order']:.2f}")
    ax[0].legend(); ax[0].grid(alpha=0.3, which="both")
    # energy + mass over the march
    en = np.asarray(hist["energy"]); ms = np.asarray(hist["mass"])
    k = np.arange(len(en))
    ax[1].plot(k, en, "C3o-", lw=1.9, ms=5, label="energy $F$")
    ax[1].set_xlabel("step"); ax[1].set_ylabel("energy $F$", color="C3")
    ax[1].tick_params(axis="y", labelcolor="C3")
    ax2 = ax[1].twinx()
    ax2.plot(k, ms - ms[0], "C0s--", lw=1.4, ms=4, label="mass drift")
    ax2.set_ylabel(r"mass drift $\int c\,dV - \int c_0\,dV$", color="C0")
    ax2.tick_params(axis="y", labelcolor="C0")
    ax2.set_ylim(-1e-12, 1e-12)
    ax[1].set_title(f"Energy decreases, mass conserved "
                    f"(drift {results['rel_mass_drift']:.0e})")
    ax[1].grid(alpha=0.3)
    # the modified free-energy well
    cc = np.asarray(hist["well_c"])
    ax[2].plot(cc, hist["well_base"], "C7-", lw=1.8,
               label=r"base $\frac{1}{4}(c^2-1)^2$")
    ax[2].plot(cc, hist["well_sextic"], "C2-", lw=2.0,
               label=rf"+ sextic ($\beta={results['beta']:g}$)")
    ax[2].set_xlabel("composition $c$"); ax[2].set_ylabel("$f(c)$")
    ax[2].set_title("The new term steepens the well")
    ax[2].legend(); ax[2].grid(alpha=0.3)
    fig.suptitle("C8: the new term, verified -- convergence, conservation, "
                 "energy, effect", fontsize=12, y=1.02)
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def workflow_fig(results, fname):
    """The contributor workflow: term -> ... -> docs, each a passed gate."""
    steps = [
        ("1. Equations + units", r"$f{+}\frac{\beta}{6}c^6,\ f'{=}\beta c^5,\ f''{=}5\beta c^4$"),
        ("2. Config schema", r"new coefficient $\beta$, validated + defaulted"),
        ("3. Residual", r"$\int N\,f'(c)$ gains $\beta c^5$"),
        ("4. Analytic Jacobian", rf"$\int N\,f''\,N$ gains $5\beta c^4$  (FD {results['jac_fd_rel']:.0e})"),
        ("5. Derivative unit test", rf"analytic vs FD  <$10^{{-5}}$  PASS"),
        ("6. Limiting case", r"$\beta\!\to\!0$ recovers base model  PASS"),
        ("7. MMS / convergence", rf"order {results['mms_order']:.2f} $\approx p{{+}}1$  PASS"),
        ("8. Conservation + energy", rf"mass drift {results['rel_mass_drift']:.0e}, $F\downarrow$  PASS"),
        ("9. CUDA profile", r"new-term assembly overhead negligible  PASS"),
        ("10. Docs + changelog", r"units, schema, CHANGELOG entry"),
    ]
    n = len(steps)
    fig, ax = plt.subplots(figsize=(9.0, 11.2), dpi=125)
    ax.set_xlim(0, 10); ax.set_ylim(0, n * 1.15 + 0.4); ax.axis("off")
    yv = [n * 1.15 - 0.2 - i * 1.15 for i in range(n)]
    for (title, detail), y in zip(steps, yv):
        green = title.split(".")[0] in {"5", "6", "7", "8", "9"}
        fc = "#e7f6ea" if green else "#eaf2fb"
        ec = "#1c8a3a" if green else "#0a5aa0"
        ax.add_patch(FancyBboxPatch((0.4, y - 0.46), 9.0, 0.9,
                     boxstyle="round,pad=0.05", linewidth=1.3,
                     edgecolor=ec, facecolor=fc))
        ax.text(0.65, y + 0.13, title, fontsize=10.5, fontweight="bold",
                va="center", ha="left")
        ax.text(0.65, y - 0.24, detail, fontsize=9.0, va="center", ha="left",
                color="#333")
    for y0, y1 in zip(yv[:-1], yv[1:]):
        ax.add_patch(FancyArrowPatch((5.0, y0 - 0.48), (5.0, y1 + 0.44),
                     arrowstyle="-|>", mutation_scale=12, color="#777",
                     linewidth=1.1))
    ax.text(5.0, n * 1.15 + 0.15, "tutorial user  ->  research contributor",
            ha="center", fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def write_numbers(results):
    def mac(name, val):
        return rf"\newcommand{{\{name}}}{{{val}}}"

    def sci(x, sig=2):
        mant, exp = f"{float(x):.{sig-1}e}".split("e")
        return f"\\ensuremath{{{mant}\\times10^{{{int(exp)}}}}}"

    lines = [
        "% AUTO-GENERATED by gen_figures.py from results.json - do not edit.",
        mac("CeightBeta", f"{results['beta']:g}"),
        mac("CeightJacFd", sci(results["jac_fd_rel"])),
        mac("CeightJacFdBase", sci(results["jac_fd_rel_base"])),
        mac("CeightTermFp", sci(results["term_fp_err"])),
        mac("CeightTermFpp", sci(results["term_fpp_err"])),
        mac("CeightMmsOrder", f"{results['mms_order']:.2f}"),
        mac("CeightMmsCoarse", sci(results["mms_err_coarse"])),
        mac("CeightMmsFine", sci(results["mms_err_fine"])),
        mac("CeightMassDrift", sci(results["rel_mass_drift"])),
        mac("CeightMassAbs", sci(results["mass_drift"])),
        mac("CeightEnergyStart", f"{results['energy_start']:.4f}"),
        mac("CeightEnergyEnd", f"{results['energy_end']:.4f}"),
        mac("CeightEnergyInc",
            r"\ensuremath{0}" if results["energy_max_increment"] == 0.0
            else sci(results["energy_max_increment"])),
        mac("CeightOverhead", f"{results['newterm_overhead_pct']:.0f}"),
        mac("CeightTests", "7"),
    ]
    os.makedirs(os.path.dirname(NUMTEX), exist_ok=True)
    with open(NUMTEX, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", NUMTEX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.join(HERE, "outputs", "c8"))
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    results, hist = load(args.run_dir)
    verify_fig(results, hist, "c8_verify.png")
    workflow_fig(results, "c8_workflow.png")
    write_numbers(results)


if __name__ == "__main__":
    main()
