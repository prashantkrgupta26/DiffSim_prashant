"""OrgElMorph course - Physics P6: figures + numbers macros.

    python gen_figures.py --device cuda:0

Writes ../../latex/figures/p6_*.png and ../../latex/numbers/p6.tex from
REAL runs (the same battery run.py checks).  Figures:
  p6_grow_melt.png  quadrature crystallinity <psi>(t) grow vs melt + fields
  p6_kinetics.png   interface velocity v(dT) + critical-radius bracket
  p6_avrami.png     JMAK sigmoid + baseline-corrected double-log fit + grains
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from crystallization import (
    build_mesh_dm, run_grow_melt, run_avrami, sweep_interface_velocity,
    run_critical_radius, drive_of, TM)
from run import IV_TEMPS, CRIT_DEEP_T, CRIT_DEEP_R, CRIT_MILD_T, CRIT_MILD_R

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "..", "latex",
                      "figures")
NUMTEX = os.path.join(os.path.dirname(__file__), "..", "..", "latex",
                      "numbers", "p6.tex")


def fig_grow_melt(grow, melt, fname):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), dpi=150)
    axes[0].plot(grow["t"], grow["qpsi"], "C3-o", ms=3,
                 label="T = 333 K (< Tm): grows")
    axes[0].plot(melt["t"], melt["qpsi"], "C0-s", ms=3,
                 label="T = 700 K (> Tm): melts")
    axes[0].set_xlabel("time $t$")
    axes[0].set_ylabel(r"quadrature crystallinity $\langle\psi\rangle$")
    axes[0].set_title("Growth below Tm, melting above")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    for ax, r, lab in ((axes[1], grow, "grown (T < Tm)"),
                       (axes[2], melt, "melted (T > Tm)")):
        im = ax.imshow(r["psi"].T, origin="lower", cmap="magma",
                       vmin=0, vmax=1)
        ax.set_title(lab, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=[axes[1], axes[2]], fraction=0.02, pad=0.02,
                 label=r"crystallinity $\psi$")
    fig.suptitle(r"Seeded crystal: the driving force changes sign at Tm "
                 r"($\langle\psi\rangle$ by quadrature)",
                 fontsize=12, y=1.02)
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def fig_kinetics(sw, cd, cm, fname):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=150)
    # interface velocity vs undercooling
    axes[0].plot(sw["dT"], sw["v"], "C1-o", ms=5)
    axes[0].set_xlabel(r"undercooling $\Delta T = T_m - T$")
    axes[0].set_ylabel(r"interface velocity $v = \mathrm{d}r_{\rm eff}/\mathrm{d}t$")
    axes[0].set_title("Growth front speeds up with undercooling")
    axes[0].grid(alpha=0.3)
    # critical radius: grow/shrink markers at two undercoolings
    for row, r, c, lab in ((0, cd, "C3", f"T = {cd['T']:.0f} K"),
                           (1, cm, "C0", f"T = {cm['T']:.0f} K")):
        y = np.full(r["radii"].shape, row, float)
        axes[1].scatter(r["radii"][r["grew"]], y[r["grew"]], marker="^",
                        s=70, color=c, label=f"{lab}: grows")
        axes[1].scatter(r["radii"][~r["grew"]], y[~r["grew"]], marker="v",
                        s=70, facecolors="none", edgecolors=c,
                        label=f"{lab}: shrinks")
        axes[1].axvline(r["r_star"], color=c, ls="--", lw=1.2)
        axes[1].annotate(rf"$r^* \approx {r['r_star']:.3f}$",
                         (r["r_star"], row + 0.18), color=c, fontsize=9,
                         ha="center")
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels([f"deep\n(drive {cd['drive']:+.2f})",
                             f"mild\n(drive {cm['drive']:+.2f})"])
    axes[1].set_xlabel(r"seed radius $r_0$")
    axes[1].set_title(r"Critical radius $r^*$ shrinks with undercooling")
    axes[1].legend(fontsize=7, loc="lower right"); axes[1].grid(alpha=0.3)
    fig.suptitle("Interface velocity and the critical radius",
                 fontsize=12, y=1.01)
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def fig_avrami(av, fname):
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), dpi=150)
    t = av["t"]
    axes[0].plot(t, av["X"], "C2-o", ms=3, label=r"area ($\psi>0.5$)")
    axes[0].plot(t, av["Xq"], "C4-s", ms=3,
                 label=r"quadrature $\langle\psi\rangle$")
    axes[0].set_xlabel("time $t$")
    axes[0].set_ylabel("crystalline fraction")
    axes[0].set_title("JMAK sigmoid"); axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    # baseline-corrected Avrami linearization
    f = av["fit"]
    X0 = f["X0"]
    Xs = (av["X"] - X0) / max(1.0 - X0, 1e-9)
    m = (Xs > f["lo"]) & (Xs < f["hi"]) & (t > 1e-6) & np.isfinite(Xs)
    if m.sum() >= 2:
        xx = np.log(t[m])
        yy = np.log(-np.log(1 - Xs[m]))
        axes[1].plot(xx, yy, "C2o", ms=4)
        b = (yy - f["n"] * xx).mean()
        axes[1].plot(xx, f["n"] * xx + b, "k--", lw=1.2,
                     label=(rf"$n = {f['n']:.2f}\pm{f['ci95']:.2f}$, "
                            rf"$R^2={f['r2']:.3f}$"))
        axes[1].legend(fontsize=8)
    axes[1].set_xlabel(r"$\ln t$")
    axes[1].set_ylabel(r"$\ln(-\ln(1-X^*))$")
    axes[1].set_title(r"Avrami exponent ($X^*=(X-X_0)/(1-X_0)$)")
    axes[1].grid(alpha=0.3)
    im = axes[2].imshow(av["theta"].T, origin="lower", cmap="hsv",
                        vmin=0, vmax=1)
    axes[2].contour(av["psi"].T, levels=[0.5], colors="k", linewidths=0.8)
    axes[2].set_title(r"grains (frozen label $\theta$)", fontsize=11)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label=r"$\theta$")
    fig.suptitle("Avrami / JMAK crystallization kinetics and grains",
                 fontsize=12, y=1.02)
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def write_numbers(grow, melt, sw, cd, cm, av):
    f = av["fit"]
    ratio_meas = cm["r_star"] / cd["r_star"]
    ratio_pred = abs(cd["drive"]) / abs(cm["drive"])

    def mac(name, val):
        return rf"\newcommand{{\{name}}}{{{val}}}"
    lines = ["% AUTO-GENERATED by gen_figures.py - do not edit.",
             mac("PsixTm", f"{TM:g}"),
             mac("PsixGrowStart", f"{grow['qpsi0']:.4f}"),
             mac("PsixGrowEnd", f"{grow['qpsi1']:.4f}"),
             mac("PsixGrowRatio",
                 f"{grow['qpsi1'] / max(grow['qpsi0'], 1e-9):.2f}"),
             mac("PsixGrowDrive", f"{grow['drive']:+.3f}"),
             mac("PsixMeltStart", f"{melt['qpsi0']:.4f}"),
             mac("PsixMeltEnd", f"{melt['qpsi1']:.4f}"),
             mac("PsixMeltDrive", f"{melt['drive']:+.3f}"),
             mac("PsixVdeep", f"{sw['v'][-1]:.3f}"),
             mac("PsixVmild", f"{sw['v'][0]:.3f}"),
             mac("PsixdTdeep", f"{sw['dT'][-1]:g}"),
             mac("PsixdTmild", f"{sw['dT'][0]:g}"),
             mac("PsixRstarDeep", f"{cd['r_star']:.3f}"),
             mac("PsixRstarMild", f"{cm['r_star']:.3f}"),
             mac("PsixTdeep", f"{cd['T']:g}"),
             mac("PsixTmild", f"{cm['T']:g}"),
             mac("PsixRstarRatioMeas", f"{ratio_meas:.2f}"),
             mac("PsixRstarRatioPred", f"{ratio_pred:.2f}"),
             mac("PsixAvramiN", f"{f['n']:.2f}"),
             mac("PsixAvramiCI", f"{f['ci95']:.2f}"),
             mac("PsixAvramiRsq", f"{f['r2']:.3f}"),
             mac("PsixAvramiXzero", f"{f['X0']:.3f}"),
             mac("PsixAvramiXend", f"{av['X_end']:.3f}"),
             mac("PsixAvramiXqend", f"{av['Xq_end']:.3f}"),
             mac("PsixSeeds", f"{av['n_seeds']}"),
             mac("PsixGrains", f"{av['n_grains']}")]
    os.makedirs(os.path.dirname(NUMTEX), exist_ok=True)
    with open(NUMTEX, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", NUMTEX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--level", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    dm, mesh, cons = build_mesh_dm(args.level, device=args.device)
    grow = run_grow_melt(dm, mesh, cons, 333.0, device=args.device)
    melt = run_grow_melt(dm, mesh, cons, 700.0, device=args.device)
    sw = sweep_interface_velocity(dm, mesh, cons, IV_TEMPS,
                                  device=args.device)
    cd = run_critical_radius(dm, mesh, cons, CRIT_DEEP_T, CRIT_DEEP_R,
                             device=args.device)
    cm = run_critical_radius(dm, mesh, cons, CRIT_MILD_T, CRIT_MILD_R,
                             device=args.device)
    av = run_avrami(dm, mesh, cons, device=args.device)
    fig_grow_melt(grow, melt, "p6_grow_melt.png")
    fig_kinetics(sw, cd, cm, "p6_kinetics.png")
    fig_avrami(av, "p6_avrami.png")
    write_numbers(grow, melt, sw, cd, cm, av)


if __name__ == "__main__":
    main()
