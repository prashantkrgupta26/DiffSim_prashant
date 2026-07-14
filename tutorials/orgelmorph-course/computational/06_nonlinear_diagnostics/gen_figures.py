"""OrgElMorph course - C6: figures + numbers from the saved harness run.

Renders three figures from ``outputs/c6/{results.json,history.npz}`` and writes
``latex/numbers/c6.tex`` (the document's numbers ARE this run's output)::

    python run.py --config configs/c6.yaml --device cuda:0 --solver splu \\
        --output outputs/c6 --overwrite --mode reference
    python gen_figures.py --run-dir outputs/c6
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
NUMTEX = os.path.join(HERE, "..", "..", "latex", "numbers", "c6.tex")


def load(run_dir):
    with open(os.path.join(run_dir, "results.json")) as fh:
        results = json.load(fh)
    hist = dict(np.load(os.path.join(run_dir, "history.npz")))
    return results, hist


def newton_fig(results, hist, fname):
    """Healthy (quadratic) vs stagnating (clamp-limited) Newton trajectories."""
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=140)
    hk, hr, hd = hist["healthy_k"], hist["healthy_res"], hist["healthy_dx"]
    sk, sr, sd = hist["stag_k"], hist["stag_res"], hist["stag_dx"]
    clamp = hist["stag_clamp"].astype(bool)
    # left: residual norm per iteration
    ax[0].semilogy(hk, np.maximum(hr, 1e-18), "C2o-", lw=1.9, ms=6,
                   label="healthy (quadratic)")
    ax[0].semilogy(sk, np.maximum(sr, 1e-18), "C3s-", lw=1.9, ms=6,
                   label="stagnating (clamp-limited)")
    ax[0].axhline(1e-10, color="k", ls=":", lw=1.0)
    ax[0].text(hk[-1], 1.6e-10, "Newton tol", fontsize=8, ha="right")
    ax[0].set_xlabel("Newton iteration")
    ax[0].set_ylabel(r"residual norm $\Vert r\Vert$")
    ax[0].set_title("Residual: squared each step vs linear crawl")
    ax[0].legend(); ax[0].grid(alpha=0.3, which="both")
    # right: update norm per iteration, clamp events marked
    ax[1].semilogy(hk, np.maximum(hd, 1e-18), "C2o-", lw=1.9, ms=6,
                   label="healthy")
    ax[1].semilogy(sk, np.maximum(sd, 1e-18), "C3s-", lw=1.9, ms=6,
                   label="stagnating")
    if clamp.any():
        ax[1].semilogy(sk[clamp], sd[clamp], "kx", ms=9, mew=2,
                       label="trust clamp fired")
    ax[1].axhline(2.0, color="C3", ls="--", lw=1.0, alpha=0.6)
    ax[1].text(sk[0], 2.3, "clamp threshold |dc|=2", fontsize=8, color="C3")
    ax[1].set_xlabel("Newton iteration")
    ax[1].set_ylabel(r"update norm $\Vert\delta x\Vert_\infty$")
    ax[1].set_title("Update: converges vs pinned at the clamp")
    ax[1].legend(); ax[1].grid(alpha=0.3, which="both")
    fig.suptitle("C6: anatomy of a healthy Newton solve, and a stagnating one",
                 fontsize=12, y=1.02)
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def conditioning_fig(results, hist, fname):
    """cond(J) vs composition, with f'' overlay — the near-wall blow-up."""
    fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=140)
    c = hist["cond_comp"]; cond = hist["cond_val"]; fpp = np.abs(hist["cond_fpp"])
    ax.loglog(c, cond, "C0o-", lw=2.0, ms=7, label=r"cond$(J)$")
    ax.set_xlabel("mean composition $c$ (distance from the wall $c=0$)")
    ax.set_ylabel(r"condition number cond$(J)$", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax.invert_xaxis()
    ax.grid(alpha=0.3, which="both")
    ax2 = ax.twinx()
    ax2.loglog(c, fpp, "C1s--", lw=1.6, ms=6, label=r"$|f''(c)|$")
    ax2.set_ylabel(r"$|f''(c)|$ (regularized FH)", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax.set_title(f"FH Jacobian conditioning: {results['cond_ratio']:.0f}x "
                 f"worse near the wall")
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def decision_fig(fname):
    """The nonlinear-failure decision tree (spec 8/C6)."""
    fig, ax = plt.subplots(figsize=(9.2, 8.2), dpi=130)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=9.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.04", linewidth=1.3,
                     edgecolor=ec, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, label=""):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                     mutation_scale=13, color="#555", linewidth=1.2))
        if label:
            ax.text((x0 + x1) / 2 + 0.15, (y0 + y1) / 2, label, fontsize=8,
                    color="#333", ha="left")

    box(3.4, 9.1, 3.2, 0.7, "Newton did not converge", "#fde9e9", "#b02020",
        fs=10.5, bold=True)
    # symptom row
    box(0.2, 7.3, 2.5, 0.9, "residual stalls,\nupdate pinned at clamp",
        "#eaf2fb", "#0a5aa0")
    box(2.9, 7.3, 2.3, 0.9, "adaptive dt\nat the floor", "#eaf2fb", "#0a5aa0")
    box(5.4, 7.3, 2.3, 0.9, "cond(J) huge /\nlinear solve fails",
        "#eaf2fb", "#0a5aa0")
    box(7.9, 7.3, 1.9, 0.9, "diverges from\nstep 1", "#eaf2fb", "#0a5aa0")
    arrow(4.2, 9.1, 1.4, 8.25); arrow(4.6, 9.1, 4.0, 8.25)
    arrow(5.4, 9.1, 6.5, 8.25); arrow(5.8, 9.1, 8.8, 8.25)
    # diagnosis row
    box(0.2, 5.5, 2.5, 0.9, "clamped diverging\ndirection (dt too large)",
        "#fff6e0", "#c98a00")
    box(2.9, 5.5, 2.3, 0.9, "LTE tol\nunreachable", "#fff6e0", "#c98a00")
    box(5.4, 5.5, 2.3, 0.9, "state near the\nwall: f'' -> 1/eps",
        "#fff6e0", "#c98a00")
    box(7.9, 5.5, 1.9, 0.9, "wrong Jacobian /\nbad IC", "#fff6e0", "#c98a00")
    for x in (1.45, 4.05, 6.55, 8.85):
        arrow(x, 7.3, x, 6.45)
    # action row
    box(0.2, 3.5, 2.5, 1.1, "reduce dt (or add\nline search); the\nfix is not more iters",
        "#e7f6ea", "#1c8a3a")
    box(2.9, 3.5, 2.3, 1.1, "loosen LTE tol,\nor accept the\nfloor's accuracy",
        "#e7f6ea", "#1c8a3a")
    box(5.4, 3.5, 2.3, 1.1, "keep iterates off\nthe wall (box\nprojection)",
        "#e7f6ea", "#1c8a3a")
    box(7.9, 3.5, 1.9, 1.1, "verify J vs FD\n(C0); seed a\nphysical IC",
        "#e7f6ea", "#1c8a3a")
    for x in (1.45, 4.05, 6.55, 8.85):
        arrow(x, 5.5, x, 4.65)
    # checkpoint footer
    box(2.6, 1.6, 4.8, 0.9, "ALWAYS: write a failure checkpoint\n"
        "(state + residual/update history) -- never silently drop the step",
        "#f0eaf8", "#6a3ca0", fs=9.5, bold=True)
    for x in (1.45, 4.05, 6.55, 8.85):
        arrow(x, 3.5, 5.0, 2.55)
    ax.text(5.0, 0.7, "symptom  ->  diagnosis  ->  action  ->  checkpoint",
            ha="center", fontsize=10, style="italic", color="#444")
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
        mac("CsixHealthyIters", f"{results['healthy_iters']}"),
        mac("CsixHealthyResZero", sci(results["healthy_res0"])),
        mac("CsixHealthyResEnd", sci(results["healthy_res_converged"])),
        mac("CsixStagIters", "15"),
        mac("CsixStagDt", "0.05"),
        mac("CsixStagResFinal", sci(results["stag_res_final"])),
        mac("CsixStagClampFrac", f"{results['stag_clamp_fraction']:.2f}"),
        mac("CsixStagFixedDt", sci(2e-3)),
        mac("CsixStagFixedIters", f"{results['stag_fixed_iters']}"),
        mac("CsixSafeProjDofs", f"{results['safe_proj_dofs']}"),
        mac("CsixSafeProjMax", f"{results['safe_proj_max']:.2f}"),
        mac("CsixMindtSteps", f"{results['mindt_n_steps']}"),
        mac("CsixMindtLte", f"{results['mindt_lte_at_floor']:.2f}"),
        mac("CsixMindtTol", sci(1e-9)),
        mac("CsixCondBulk", sci(results["cond_bulk"])),
        mac("CsixCondWall", sci(results["cond_wall"])),
        mac("CsixCondRatio", f"{results['cond_ratio']:.0f}"),
    ]
    os.makedirs(os.path.dirname(NUMTEX), exist_ok=True)
    with open(NUMTEX, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", NUMTEX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.join(HERE, "outputs", "c6"))
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    results, hist = load(args.run_dir)
    newton_fig(results, hist, "c6_newton.png")
    conditioning_fig(results, hist, "c6_conditioning.png")
    decision_fig("c6_decision.png")
    write_numbers(results)


if __name__ == "__main__":
    main()
