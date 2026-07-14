"""OrgElMorph course - Computational C1: figures + numbers.

Writes three convergence/diagnostic plots into ../../latex/figures/ and the
measured macros into ../../latex/numbers/c1.tex.  Same core as run.py, so the
document's numbers ARE the measured output of this run.

    PYTHONPATH=<repo>/src python gen_figures.py --device cuda:0
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from convergence import (spatial_mms, temporal_convergence, algebraic_control,
                         verify_reference, DELIBERATE_FAILURES)

FIGDIR = os.path.join(os.path.dirname(__file__),
                      "..", "..", "latex", "figures")
NUMTEX = os.path.join(os.path.dirname(__file__),
                      "..", "..", "latex", "numbers", "c1.tex")
SPATIAL = ((1, (3, 4, 5, 6)), (2, (2, 3, 4, 5)))
DTS = (1.6e-2, 8e-3, 4e-3, 2e-3)


def spatial_figure(recs, fname):
    """Two panels: L2 of c and mu (left), H1 seminorm of c (right)."""
    fig, (axL, axH) = plt.subplots(1, 2, figsize=(10.4, 4.3), dpi=150)
    for r, col in zip(recs, ("C0", "C3")):
        h = np.array(r["h"])
        axL.loglog(h, r["l2_c"], "o-", color=col, lw=2, ms=7,
                   label=f"$p={r['p']}$: $L^2(c)$ (order {r['order']:.2f})")
        axL.loglog(h, r["l2_mu"], "s--", color=col, lw=1.4, ms=6, alpha=0.8,
                   label=f"$p={r['p']}$: $L^2(\\mu)$ "
                         f"(order {r['l2_mu_order']:.2f})")
        ref = r["l2_c"][0] * (h / h[0]) ** (r["p"] + 1)
        axL.loglog(h, ref, ":", color=col, alpha=0.5,
                   label=f"ideal $h^{{{r['p'] + 1}}}$")
        axH.loglog(h, r["h1_c"], "o-", color=col, lw=2, ms=7,
                   label=f"$p={r['p']}$: $H^1(c)$ (order {r['h1_order']:.2f})")
        refh = r["h1_c"][0] * (h / h[0]) ** r["p"]
        axH.loglog(h, refh, ":", color=col, alpha=0.5,
                   label=f"ideal $h^{{{r['p']}}}$")
    for ax, ttl, yl in ((axL, "$L^2$ error, both fields", "$L^2$ error"),
                        (axH, "$H^1$ seminorm error of $c$",
                         "$H^1$ seminorm error")):
        ax.set_xlabel("mesh size $h$"); ax.set_ylabel(yl)
        ax.set_title(ttl); ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Spatial convergence (steady manufactured solution)")
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def temporal_figure(recs, fname):
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
    for r, mk, col in zip(recs, ("o", "s"), ("C0", "C3")):
        d, e = np.array(r["dts"]), np.array(r["errs"])
        name = "BDF1" if r["order"] == 1 else "BDF2"
        ax.loglog(d, e, mk + "-", color=col, lw=2, ms=8,
                  label=f"{name} (order {r['order_est']:.2f})")
        ref = e[0] * (d / d[0]) ** r["order"]
        ax.loglog(d, ref, "--", color=col, alpha=0.5,
                  label=f"slope ${r['order']}$ (ideal)")
    ax.set_xlabel(r"time step $\Delta t$")
    ax.set_ylabel(r"relative $L^2$ error vs.\ fine reference")
    ax.set_title("Temporal convergence (fixed over-resolved mesh)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def diagnostics_figure(alg, fails, fname):
    """Left: discretization error is flat vs Newton tol (algebraic control).
    Right: the deliberate failures, each a wrong/erratic order."""
    fig, (axA, axF) = plt.subplots(1, 2, figsize=(10.4, 4.3), dpi=150)
    tols = [r["newton_tol"] for r in alg["rows"]]
    l2 = [r["l2_c"] for r in alg["rows"]]
    axA.semilogx(tols, l2, "o-", color="C2", lw=2, ms=8)
    axA.set_xlabel("Newton tolerance"); axA.set_ylabel("$L^2(c)$ error")
    axA.set_title("Algebraic control: discretization error is invariant\n"
                  f"(moves {alg['spread_frac']:.0e} of itself)")
    axA.grid(True, which="both", alpha=0.3)
    axA.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for f, col in zip(fails, ("C1", "C3", "C4", "C5")):
        e = np.array(f["errs"])
        x = np.arange(1, len(e) + 1)
        axF.semilogy(x, e, "o-", color=col, lw=1.8, ms=6,
                     label=f"{f['name']} ({f['order']:+.2f})")
    axF.set_xlabel("refinement level (coarse -> fine)")
    axF.set_ylabel("measured error")
    axF.set_title("Deliberate failures: wrong-but-plausible orders")
    axF.grid(True, which="both", alpha=0.3); axF.legend(fontsize=6.5)
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def sci(val):
    """Format a small number as LaTeX $m\\times10^{e}$."""
    m, e = f"{val:.1e}".split("e")
    return rf"\ensuremath{{{m}\times10^{{{int(e)}}}}}"


def write_numbers(sp, tp, alg, vref, fails):
    def mac(name, val):
        return rf"\newcommand{{\{name}}}{{{val}}}"
    lines = ["% AUTO-GENERATED by gen_figures.py - do not edit.",
             # spatial orders (L2 and H1, both fields)
             mac("ConeSpaPoneOrder", f"{sp[1]['order']:.2f}"),
             mac("ConeSpaPtwoOrder", f"{sp[2]['order']:.2f}"),
             mac("ConeSpaPoneHoneOrder", f"{sp[1]['h1_order']:.2f}"),
             mac("ConeSpaPtwoHoneOrder", f"{sp[2]['h1_order']:.2f}"),
             mac("ConeSpaPoneMuOrder", f"{sp[1]['l2_mu_order']:.2f}"),
             mac("ConeSpaPtwoMuOrder", f"{sp[2]['l2_mu_order']:.2f}"),
             # spatial L2(c) errors, coarse -> fine
             mac("ConeSpaPoneErrCoarse", sci(sp[1]["l2_c"][0])),
             mac("ConeSpaPoneErrFine", sci(sp[1]["l2_c"][-1])),
             mac("ConeSpaPtwoErrCoarse", sci(sp[2]["l2_c"][0])),
             mac("ConeSpaPtwoErrFine", sci(sp[2]["l2_c"][-1])),
             mac("ConeSpaPoneNlevels", f"{len(sp[1]['levels'])}"),
             mac("ConeSpaPtwoNlevels", f"{len(sp[2]['levels'])}"),
             # algebraic control
             mac("ConeAlgSpread", sci(alg["spread_frac"])),
             mac("ConeAlgTolLo", "\\ensuremath{10^{-4}}"),
             mac("ConeAlgTolHi", "\\ensuremath{10^{-12}}"),
             # temporal orders + errors
             mac("ConeBdfoneOrder", f"{tp[1]['order_est']:.2f}"),
             mac("ConeBdftwoOrder", f"{tp[2]['order_est']:.2f}"),
             mac("ConeBdfoneErrCoarse", sci(tp[1]["errs"][0])),
             mac("ConeBdfoneErrFine", sci(tp[1]["errs"][-1])),
             mac("ConeBdftwoErrCoarse", sci(tp[2]["errs"][0])),
             mac("ConeBdftwoErrFine", sci(tp[2]["errs"][-1])),
             mac("ConeNdt", f"{len(DTS)}"),
             # reference verification
             mac("ConeRefOrderLo", f"{vref['orders'][vref['ref_dts'][0]]:.2f}"),
             mac("ConeRefOrderHi", f"{vref['orders'][vref['ref_dts'][1]]:.2f}"),
             mac("ConeRefStable", f"{vref['order_stable']:.3f}"),
             mac("ConeRefRich", sci(vref["richardson_ref_err"])),
             # deliberate-failure measured orders
             mac("ConeFailBcOrder", f"{fails[0]['order']:.2f}"),
             mac("ConeFailSrcOrder", f"{fails[1]['order']:.2f}"),
             mac("ConeFailFeatOrder", f"{fails[2]['order']:.2f}"),
             mac("ConeFailRefOrder", f"{fails[3]['order']:.2f}")]
    os.makedirs(os.path.dirname(NUMTEX), exist_ok=True)
    with open(NUMTEX, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", NUMTEX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)

    sp = {p: spatial_mms(p, levels, device=args.device)
          for p, levels in SPATIAL}
    spatial_figure([sp[1], sp[2]], "c1_spatial.png")

    tp = {o: temporal_convergence(o, DTS, ref_dt=2e-4, device=args.device)
          for o in (1, 2)}
    temporal_figure([tp[1], tp[2]], "c1_temporal.png")

    alg = algebraic_control(1, 5, device=args.device)
    vref = verify_reference(device=args.device)
    fails = [fn(device=args.device) for fn in DELIBERATE_FAILURES]
    diagnostics_figure(alg, fails, "c1_diagnostics.png")

    write_numbers(sp, tp, alg, vref, fails)


if __name__ == "__main__":
    main()
