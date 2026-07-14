"""OrgElMorph course - Computational C3: figures + numbers.

Writes four figures into ../../latex/figures/ and the measured macros into
../../latex/numbers/c3.tex.  Same core as run.py.

  c3_octree.png    octree refinement along a static circle (hanging nodes)
  c3_transfer.png  conservative-transfer error (injection loses mass)
  c3_ladder.png    the LTE dt ladder over a quench
  c3_cost.png      REAL cost: adaptive vs matched-accuracy fixed sweep

    PYTHONPATH=<repo>/src python gen_figures.py --device cuda:0
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adaptivity import (octree_refinement, transfer_error,
                        transfer_error_sweep, adaptive_time_stepping,
                        adaptive_cost, bdf2_variable_order)

FIGDIR = os.path.join(os.path.dirname(__file__),
                      "..", "..", "latex", "figures")
NUMTEX = os.path.join(os.path.dirname(__file__),
                      "..", "..", "latex", "numbers", "c3.tex")


def octree_figure(o, fname):
    fig, ax = plt.subplots(figsize=(5.4, 5.2), dpi=150)
    c, h, lv = o["centers"], o["hs"], o["levels"]
    for i in np.argsort(lv):
        x0, y0 = c[i] - h[i] / 2
        ax.add_patch(plt.Rectangle((x0, y0), h[i], h[i], fill=False,
                                   ec=plt.cm.viridis(lv[i] / lv.max()),
                                   lw=0.5))
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(0.5 + 0.30 * np.cos(th), 0.5 + 0.30 * np.sin(th), "r-",
            lw=1.5, label="static interface (refined here)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_title(f"Octree refinement (static criterion): "
                 f"{o['adaptive_nodes']} vs {o['uniform_nodes']} nodes\n"
                 f"({o['node_savings']:.1f}x fewer dofs, same finest $h$)")
    ax.legend(loc="upper right", fontsize=9)
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def transfer_figure(te, sweep, fname):
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(12.4, 4.0), dpi=150)
    a0.imshow(te["c_fine"], origin="lower", cmap="viridis")
    a0.set_title(f"fine field ({te['n_drops']} sub-cell droplets)",
                 fontsize=10)
    a1.imshow(te["c_inj"], origin="lower", cmap="viridis")
    a1.set_title(f"injection: {100 * te['inj_err']:.0f}% mass lost",
                 fontsize=10)
    for ax in (a0, a1):
        ax.set_xticks([]); ax.set_yticks([])
    rad = [0.5, 0.7, 1.0, 1.5, 2.5][:len(sweep)]     # transfer_error_sweep radii
    a2.semilogy(rad, [s["inj_err"] + 1e-18 for s in sweep], "o-", color="C3",
                lw=2, label="injection (naive)")
    a2.semilogy(rad, [max(s["avg_err"], 1e-17) for s in sweep], "s-",
                color="C0", lw=2, label="averaging (conservative)")
    a2.set_xlabel("feature radius (fine cells)")
    a2.set_ylabel("relative mass error")
    a2.set_title("transfer error vs feature size", fontsize=10)
    a2.grid(True, which="both", alpha=0.3); a2.legend(fontsize=8)
    fig.suptitle("Conservative transfer: naive restriction loses sub-cell "
                 "mass; averaging is exact", y=1.02)
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def ladder_figure(t, fname):
    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=150)
    ax.plot(t["ts"], t["dts"], "o-", color="C0", ms=4, lw=1.5)
    ax.set_xlabel("time $t$"); ax.set_ylabel(r"accepted step $\Delta t$")
    ax.set_yscale("log")
    ax.set_title(f"Adaptive time step over a quench "
                 f"({t['span']:.0f}x range, {t['n_adaptive']} steps)")
    ax.grid(alpha=0.3, which="both")
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def cost_figure(ac, fname):
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=150)
    labels = ["adaptive"] + [f"fixed\n{f['dt']:.0e}" for f in ac["fixed"]]
    newton = [ac["adaptive"]["newton_iters"]] + [f["newton_iters"]
                                                 for f in ac["fixed"]]
    errs = [ac["adaptive"]["err"]] + [f["err"] for f in ac["fixed"]]
    cols = ["C0"] + ["C7"] * len(ac["fixed"])
    # highlight the matched fixed run
    for i, f in enumerate(ac["fixed"]):
        if abs(f["dt"] - ac["matched"]["dt"]) < 1e-12:
            cols[i + 1] = "C3"
    bars = ax.bar(labels, newton, color=cols)
    ax.set_ylabel("Newton iterations (total work)")
    ax.set_title("Real cost vs matched-accuracy fixed dt "
                 "(red = matched; label = final error)")
    for b, e in zip(bars, errs):
        ax.annotate(f"{e:.1e}", (b.get_x() + b.get_width() / 2,
                    b.get_height()), ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def sci(val):
    m, e = f"{val:.1e}".split("e")
    return rf"\ensuremath{{{m}\times10^{{{int(e)}}}}}"


def write_numbers(o, te, ac, t, b):
    def mac(name, val):
        return rf"\newcommand{{\{name}}}{{{val}}}"
    ad, m = ac["adaptive"], ac["matched"]
    lines = [
        "% AUTO-GENERATED by gen_figures.py - do not edit.",
        # octree
        mac("CthreeAdaptNodes", f"{o['adaptive_nodes']:,}".replace(",", "{,}")),
        mac("CthreeUniformNodes", f"{o['uniform_nodes']:,}".replace(",", "{,}")),
        mac("CthreeNodeSavings", f"{o['node_savings']:.1f}"),
        mac("CthreeElemSavings", f"{o['elem_savings']:.1f}"),
        mac("CthreeMaxLevel", f"{o['max_level']}"),
        # transfer
        mac("CthreeInjErr", f"{100 * te['inj_err']:.0f}\\%"),
        mac("CthreeAvgErr", sci(max(te['avg_err'], 1e-17))),
        # temporal ladder (dt growth)
        mac("CthreeNadaptive", f"{t['n_adaptive']}"),
        mac("CthreeSpan", f"{t['span']:.0f}"),
        mac("CthreeDtMin", sci(t["dt_min"])),
        mac("CthreeDtMax", sci(t["dt_max"])),
        # REAL cost accounting
        mac("CthreeCostTend", f"{ac['t_end']:.2f}"),
        mac("CthreeAdaptAccepted", f"{ad['accepted']}"),
        mac("CthreeAdaptRejected", f"{ad['rejected']}"),
        mac("CthreeAdaptSolves", f"{ad['full_solves'] + ad['half_solves']}"),
        mac("CthreeAdaptNewton", f"{ad['newton_iters']:,}".replace(",", "{,}")),
        mac("CthreeAdaptErr", sci(ad["err"])),
        mac("CthreeMatchedDt", sci(m["dt"])),
        mac("CthreeMatchedNewton", f"{m['newton_iters']:,}".replace(",", "{,}")),
        mac("CthreeMatchedErr", sci(m["err"])),
        mac("CthreeNewtonSavings", f"{ac['newton_savings']:.1f}"),
        mac("CthreeWallSavings", f"{ac['wall_savings']:.1f}"),
        # BDF2 order (both MEASURED)
        mac("CthreeVarOrder", f"{b['order']:.2f}"),
        mac("CthreeConstOrder", f"{b['const_order']:.2f}"),
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
    o = octree_refinement(device=args.device)
    te = transfer_error()
    sweep = transfer_error_sweep()
    t = adaptive_time_stepping(device=args.device)
    ac = adaptive_cost(device=args.device)
    b = bdf2_variable_order(device=args.device)
    octree_figure(o, "c3_octree.png")
    transfer_figure(te, sweep, "c3_transfer.png")
    ladder_figure(t, "c3_ladder.png")
    cost_figure(ac, "c3_cost.png")
    write_numbers(o, te, ac, t, b)


if __name__ == "__main__":
    main()
