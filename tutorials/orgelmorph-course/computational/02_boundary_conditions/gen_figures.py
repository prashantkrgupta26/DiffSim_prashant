"""OrgElMorph course - Computational C2: figures + numbers.

Writes four figures into ../../latex/figures/ and the measured macros into
../../latex/numbers/c2.tex.  Same core as run.py.

  c2_fields.png  no-flux vs Dirichlet fields (the boundary reshapes it)
  c2_mass.png    mass(t): conserved vs reservoir
  c2_matrix.png  strong (row-replace) vs weak (penalty) BC imposition
  c2_flux.png    flux balance dm/dt = net boundary flux, and its shut-off

    PYTHONPATH=<repo>/src python gen_figures.py --device cuda:0
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bc import (compare, flux_balance, bc_test_matrix,
                strong_vs_weak_dirichlet, DT)

FIGDIR = os.path.join(os.path.dirname(__file__),
                      "..", "..", "latex", "figures")
NUMTEX = os.path.join(os.path.dirname(__file__),
                      "..", "..", "latex", "numbers", "c2.tex")


def fields_figure(r, fname):
    nf, di = r["noflux"], r["dirichlet"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3), dpi=150)
    for ax, rec, title in ((axes[0], nf, "natural (no-flux)"),
                           (axes[1], di,
                            f"Dirichlet (wall $c={di['wall']:.1f}$)")):
        im = ax.imshow(rec["img"], origin="lower", cmap="RdBu_r",
                       vmin=-1, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.023, pad=0.02, label="c")
    fig.suptitle("Same blend, two boundary conditions "
                 "-- the wall pins the Dirichlet field", y=1.0)
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def mass_figure(r, fname):
    nf, di = r["noflux"], r["dirichlet"]
    fig, ax = plt.subplots(figsize=(6.4, 4.3), dpi=150)
    for rec, col, lab in ((nf, "C0", "no-flux (conserved)"),
                          (di, "C3", "Dirichlet (reservoir)")):
        t = np.arange(len(rec["masses"])) * DT
        ax.plot(t, rec["masses"] - rec["masses"][0], col + "-", lw=2,
                label=lab)
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"mass change $m(t)-m(0)$")
    ax.set_title("Mass conservation depends on the boundary condition")
    ax.grid(alpha=0.3); ax.legend()
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def matrix_figure(w, fname):
    """Left: the tiny stiffness before / after strong row-replacement, with
    the asymmetry it introduces. Right: the weak (penalty) BC converges
    O(1/beta) to the pinned value while keeping the matrix symmetric."""
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(12.6, 4.0), dpi=150)
    vmax = np.abs(w["K"]).max()
    for ax, Mtx, ttl in ((a0, w["K"], "assembled $K$ (SPD)"),
                         (a1, w["A_rr"],
                          f"row-replaced (symmetric: {w['rr_symmetric']})"),
                         (a2, w["A_sym"],
                          f"sym. elimination (symmetric: {w['sym_symmetric']})")):
        im = ax.imshow(Mtx, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks(range(0, w["n"] + 1, 2))
        ax.set_yticks(range(0, w["n"] + 1, 2))
    fig.colorbar(im, ax=(a0, a1, a2), fraction=0.015, pad=0.02)
    fig.suptitle("Strong Dirichlet: row replacement vs symmetric elimination "
                 "on a tiny $-u''=0$ system", y=1.02)
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def penalty_figure(w, fname):
    fig, ax = plt.subplots(figsize=(6.2, 4.3), dpi=150)
    betas = [p["beta"] for p in w["penalty"]]
    bc_err = [p["bc_err"] for p in w["penalty"]]
    ax.loglog(betas, bc_err, "o-", color="C2", lw=2, ms=7,
              label="weak (penalty) boundary error")
    ax.loglog(betas, [betas[0] * bc_err[0] / b for b in betas], "--",
              color="gray", alpha=0.7, label=r"$\propto 1/\beta$")
    ax.axhline(w["strong_err"] + 1e-16, color="C3", ls=":",
               label="strong (row-replace) error $\\approx 0$")
    ax.set_xlabel(r"penalty $\beta$")
    ax.set_ylabel("boundary-value error")
    ax.set_title("Weakly imposed Dirichlet converges as $1/\\beta$")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def flux_figure(fb, fname):
    fig, ax = plt.subplots(figsize=(6.4, 4.3), dpi=150)
    for r, col, lab in ((fb["noflux"], "C0", "no-flux: net flux $=0$"),
                        (fb["dirichlet"], "C3",
                         "Dirichlet: reservoir influx, decaying")):
        ax.plot(r["t"], r["dmdt"], col + "-", lw=2, label=lab)
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$dm/dt$ = net boundary flux $-\oint J\cdot n$")
    ax.set_title("Flux balance: mass changes exactly at the wall-flux rate")
    ax.grid(alpha=0.3); ax.legend()
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def sci(val):
    m, e = f"{val:.1e}".split("e")
    return rf"\ensuremath{{{m}\times10^{{{int(e)}}}}}"


def write_numbers(r, fb, mat, w):
    nf, di = r["noflux"], r["dirichlet"]

    def mac(name, val):
        return rf"\newcommand{{\{name}}}{{{val}}}"
    # penalty at beta = 1e4 (the illustrative point)
    p4 = next(p for p in w["penalty"] if abs(p["beta"] - 1e4) < 1)
    mix = next(m for m in mat if m["name"].startswith("mixed"))
    lines = [
        "% AUTO-GENERATED by gen_figures.py - do not edit.",
        mac("CtwoWall", f"{di['wall']:.1f}"),
        mac("CtwoNfMassDrift", sci(nf["mass_drift"])),
        mac("CtwoNfEdge", f"{nf['edge_mean']:+.2f}"),
        mac("CtwoDirMassDrift", f"{di['mass_drift']:.2f}"),
        mac("CtwoDirEdge", f"{di['edge_mean']:+.2f}"),
        mac("CtwoFieldDiff", f"{r['field_diff']:.2f}"),
        # flux balance
        mac("CtwoNfFlux", sci(fb["noflux"]["flux_abs_max"])),
        mac("CtwoDirFluxEarly", f"{fb['dirichlet']['flux_early']:+.2f}"),
        mac("CtwoDirFluxLate", f"{fb['dirichlet']['flux_late']:+.2f}"),
        # weak vs strong
        mac("CtwoStrongErr", sci(max(w["strong_err"], 1e-16))),
        mac("CtwoWeakBeta", "\\ensuremath{10^{4}}"),
        mac("CtwoWeakErr", sci(p4["bc_err"])),
        mac("CtwoRowReplaceSym", "asymmetric"),
        mac("CtwoSymElimSym", "symmetric"),
        # BC test matrix (mixed case)
        mac("CtwoMixDrift", f"{mix['mass_drift']:.2f}"),
        mac("CtwoMixEdge", f"{mix['edge_mean']:+.2f}"),
    ]
    os.makedirs(os.path.dirname(NUMTEX), exist_ok=True)
    with open(NUMTEX, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", NUMTEX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)

    r = compare(device=args.device)
    fields_figure(r, "c2_fields.png")
    mass_figure(r, "c2_mass.png")

    fb = {m: flux_balance(m, device=args.device) for m in ("noflux",
                                                           "dirichlet")}
    flux_figure(fb, "c2_flux.png")

    mat = bc_test_matrix(device=args.device)
    w = strong_vs_weak_dirichlet()
    matrix_figure(w, "c2_matrix.png")
    penalty_figure(w, "c2_penalty.png")

    write_numbers(r, fb, mat, w)


if __name__ == "__main__":
    main()
