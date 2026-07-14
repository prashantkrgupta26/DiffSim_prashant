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
                        adaptive_cost, bdf2_variable_order,
                        fe_conservative_transfer, dynamic_amr_cycle,
                        amr_error_vs_dofs, spacetime_order)

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


def amr_cycle_figure(cy, fname):
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    out = cy["out"]
    mesh = out["mesh"]
    tree = mesh.tree
    centers, hs, lev = tree.centers(), tree.h(), np.asarray(tree.levels)
    cfull = np.asarray(out["cons"].T @ out["final_c"])
    cc = cfull[mesh.conn_of[1]].mean(1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 4.3), dpi=150)
    r1 = [Rectangle((cx - h / 2, cy2 - h / 2), h, h)
          for (cx, cy2), h in zip(centers, hs)]
    pc = PatchCollection(r1, cmap="viridis", edgecolor="k", linewidth=0.15)
    pc.set_array(lev.astype(float)); a1.add_collection(pc)
    a1.set_xlim(0, 1); a1.set_ylim(0, 1); a1.set_aspect("equal")
    a1.set_title("Adaptive mesh (color = octree level)")
    plt.colorbar(pc, ax=a1, fraction=0.046)
    r2 = [Rectangle((cx - h / 2, cy2 - h / 2), h, h)
          for (cx, cy2), h in zip(centers, hs)]
    pc2 = PatchCollection(r2, cmap="coolwarm")
    pc2.set_array(cc); a2.add_collection(pc2)
    a2.set_xlim(0, 1); a2.set_ylim(0, 1); a2.set_aspect("equal")
    a2.set_title("c: refinement tracks the interface")
    plt.colorbar(pc2, ax=a2, fraction=0.046)
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def amr_conservation_figure(cy, fname):
    rec = cy["out"]["rec"]
    t = np.array(rec["t"]); mass = np.array(rec["mass"])
    en = np.array(rec["energy"])
    dm = np.abs(np.array(rec["remesh_mass_after"])
                - np.array(rec["remesh_mass_before"]))
    dE = np.abs(np.array(rec["remesh_E_after"])
                - np.array(rec["remesh_E_before"]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.8), dpi=150)
    a1.plot(t, mass - mass[0], color="#11aa66")
    for rt in rec["remesh_t"]:
        a1.axvline(rt, color="#bbb", lw=0.6, ls="--")
    a1.set_xlabel("t"); a1.set_ylabel(r"$\int c\,dV-\int c_0\,dV$")
    a1.set_title(f"Mass drift (max remesh jump {dm.max():.0e})")
    a1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    a2.plot(t, en, color="#3366cc")
    for rt in rec["remesh_t"]:
        a2.axvline(rt, color="#bbb", lw=0.6, ls="--")
    a2.set_xlabel("t"); a2.set_ylabel("free energy $F$")
    a2.set_title(f"Energy (max remesh jump {dE.max():.1e})")
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def amr_error_dofs_figure(ed, fname):
    fig, ax = plt.subplots(figsize=(5.6, 4.2), dpi=150)
    ud = np.array([u["dofs"] for u in ed["uniform"]], float)
    ue = np.array([u["err"] for u in ed["uniform"]], float)
    o = np.argsort(ud)
    ax.loglog(ud[o], ue[o], "o-", color="#444", label="uniform mesh")
    ax.loglog([ed["amr_dofs"]], [ed["amr_err"]], "*", ms=18, color="#c1272d",
              label="dynamic AMR")
    ax.axhline(ed["amr_err"], ls=":", color="#c1272d", lw=0.8)
    ax.annotate(f"{ed['dof_ratio']:.1f}x fewer dofs\nat equal error",
                xy=(ed["amr_dofs"], ed["amr_err"]),
                xytext=(ed["amr_dofs"] * 1.35, ed["amr_err"] * 2.2),
                fontsize=9, color="#c1272d")
    ax.set_xlabel("degrees of freedom")
    ax.set_ylabel(f"L2 error vs uniform-L{ed['ref_level']} reference")
    ax.set_title("Error vs dofs: AMR reaches uniform-fine\naccuracy at fewer "
                 "dofs")
    ax.legend(); ax.grid(True, which="both", alpha=0.25)
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def sci(val):
    m, e = f"{val:.1e}".split("e")
    return rf"\ensuremath{{{m}\times10^{{{int(e)}}}}}"


def write_numbers(o, te, ac, t, b, ft=None, cy=None, ed=None, so=None):
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
    if ft is not None:
        lines += [
            # DYNAMIC AMR — conservative transfer core (A1)
            mac("CthreeFeMassRefine", sci(max(ft["mass_refine"], 1e-18))),
            mac("CthreeFeMassCoarsen", sci(max(ft["mass_coarsen"], 1e-18))),
            mac("CthreeFeTransOrder", f"{ft['order']:.2f}"),
            mac("CthreeFeConsErr", sci(max(ft["cons_coarsen_err"], 1e-18))),
        ]
    if cy is not None:
        lines += [
            # DYNAMIC AMR — driven cycle (A2)
            mac("CthreeAmrRemeshes", f"{cy['remeshes']}"),
            mac("CthreeAmrMassJump", sci(max(cy["mass_jump"], 1e-18))),
            mac("CthreeAmrEnergyJump", sci(cy["energy_jump"])),
            mac("CthreeAmrOverlap", f"{cy['overlap_min']:.2f}"),
            mac("CthreeAmrDofsPeak", f"{cy['dofs_peak']:,}".replace(",", "{,}")),
        ]
    if ed is not None:
        finest = max(ed["uniform"], key=lambda u: u["dofs"])
        lines += [
            mac("CthreeAmrErr", sci(ed["amr_err"])),
            mac("CthreeAmrDofs", f"{ed['amr_dofs']:,}".replace(",", "{,}")),
            mac("CthreeAmrRefLevel", f"{ed['ref_level']}"),
            mac("CthreeAmrRefDofs",
                f"{ed['ref_dofs']:,}".replace(",", "{,}")),
            mac("CthreeAmrDofRatio", f"{ed['dof_ratio']:.1f}"),
            mac("CthreeAmrWallRatio", f"{ed['wall_ratio']:.1f}"),
            mac("CthreeAmrFineLevel", f"{finest['level']}"),
            mac("CthreeAmrFineErr", sci(finest["err"])),
            mac("CthreeAmrFineDofs",
                f"{finest['dofs']:,}".replace(",", "{,}")),
        ]
    if so is not None:
        lines += [
            # DYNAMIC AMR — space-time interaction (A3)
            mac("CthreeStBothOrder", f"{so['both_order']:.2f}"),
            mac("CthreeStDropOrder", f"{so['drop_order']:.2f}"),
            mac("CthreeStDropPenalty", f"{so['drop_err_penalty']:.1f}"),
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
    # dynamic (solution-adaptive) AMR — the Phase-3 deliverable
    ft = fe_conservative_transfer()
    cy = dynamic_amr_cycle(device=args.device)
    ed = amr_error_vs_dofs(device=args.device)
    so = spacetime_order(device=args.device)
    octree_figure(o, "c3_octree.png")
    transfer_figure(te, sweep, "c3_transfer.png")
    ladder_figure(t, "c3_ladder.png")
    cost_figure(ac, "c3_cost.png")
    amr_cycle_figure(cy, "c3_amr_cycle.png")
    amr_conservation_figure(cy, "c3_amr_conservation.png")
    amr_error_dofs_figure(ed, "c3_amr_error_dofs.png")
    write_numbers(o, te, ac, t, b, ft=ft, cy=cy, ed=ed, so=so)


if __name__ == "__main__":
    main()
