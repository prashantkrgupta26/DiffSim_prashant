"""OrgElMorph course - Computational C5: figures + numbers.

Writes four figures into ../../latex/figures/ and the measured macros into
../../latex/numbers/c5.tex.  Live 2-D/tiny-3-D; cited device-scale points.

  c5_dofgrowth.png  dofs vs level, 2-D vs 3-D
  c5_nnz.png        nnz vs dofs (live + cited + int32 ceiling)
  c5_memory.png     COMPLETE memory: total GB vs n, with 48 GB + int32 walls
  c5_physics.png    2-D vs 3-D morphology at equal resolution

    PYTHONPATH=<repo>/src python gen_figures.py --device cuda:0
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scaling import (measure_scaling, physics_comparison, memory_accounting,
                     CITED_LADDER, INT32_MAX)
from estimate_capacity import estimate

FIGDIR = os.path.join(os.path.dirname(__file__),
                      "..", "..", "latex", "figures")
NUMTEX = os.path.join(os.path.dirname(__file__),
                      "..", "..", "latex", "numbers", "c5.tex")


def dof_growth_figure(s, fname):
    fig, ax = plt.subplots(figsize=(6.4, 4.5), dpi=150)
    for recs, col, lab in ((s["two_d"], "C0", "2-D ($\\times4$/level)"),
                           (s["three_d"], "C3", "3-D ($\\times8$/level)")):
        lv = [r["level"] for r in recs]
        dofs = [r["dofs"] for r in recs]
        ax.semilogy(lv, dofs, "o-", color=col, lw=2, ms=8, label=lab)
    ax.set_xlabel("refinement level"); ax.set_ylabel("degrees of freedom")
    ax.set_title("Degrees of freedom explode faster in 3-D")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def nnz_figure(s, fname):
    fig, ax = plt.subplots(figsize=(6.8, 4.7), dpi=150)
    for recs, col, lab in ((s["two_d"], "C0", "2-D live"),
                           (s["three_d"], "C3", "3-D live")):
        d = [r["dofs"] for r in recs]
        n = [r["nnz"] for r in recs]
        ax.loglog(d, n, "o-", color=col, lw=2, ms=8, label=lab)
    cd = [c[1] for c in CITED_LADDER]
    cn = [c[2] for c in CITED_LADDER]
    ax.loglog(cd, cn, "D", color="C1", ms=9, label="cited (dev notes)")
    for (label, dofs, nnz, note) in CITED_LADDER:
        ax.annotate(label.split(" ")[0], (dofs, nnz), fontsize=7,
                    xytext=(4, -8), textcoords="offset points")
    ax.axhline(INT32_MAX, ls="--", color="k", alpha=0.6)
    ax.text(ax.get_xlim()[0] * 1.2, INT32_MAX * 1.3,
            "int32 nnz ceiling ($2^{31}$)", fontsize=8)
    ax.set_xlabel("degrees of freedom"); ax.set_ylabel("matrix nonzeros (nnz)")
    ax.set_title("Matrix nonzeros vs.\\ problem size (live + cited)")
    ax.grid(True, which="both", alpha=0.3); ax.legend(loc="lower right")
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def memory_figure(fname):
    """Total device memory (COMPLETE accounting) vs n, 2-D and 3-D, with the
    48 GB card and int32-nnz walls."""
    ns = [16, 32, 48, 64, 96, 128, 192, 256]
    fig, ax = plt.subplots(figsize=(6.8, 4.7), dpi=150)
    for dim, col, solver in ((2, "C0", "splu"), (3, "C3", "blockch")):
        gb = [estimate(dim, n, solver=solver)["total_gb"] for n in ns]
        ax.loglog(ns, gb, "o-", color=col, lw=2, ms=7,
                  label=f"{dim}-D ({solver})")
    ax.axhline(48, ls="--", color="k", alpha=0.7)
    ax.text(ns[0], 52, "48 GB card", fontsize=8)
    ax.set_xlabel("cells per side $n$")
    ax.set_ylabel("total device memory (GB, all buffers)")
    ax.set_title("Complete memory footprint (not just CSR values)")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def physics_figure(pc, fname):
    """2-D field and a 3-D mid-slice at equal resolution, with the measured
    interfacial-area density and S(q) wavelength."""
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(12.6, 4.0), dpi=150)
    f2, f3 = pc[2]["field"], pc[3]["field"]
    a0.imshow(f2, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
    a0.set_title(f"2-D (h={pc[2]['dx']:.3f})\n"
                 f"area density {pc[2]['interfacial_area_density']:.1f}, "
                 f"$\\lambda$={pc[2]['peak_wavelength']:.3f}", fontsize=9)
    mid = f3.shape[2] // 2
    a1.imshow(f3[:, :, mid], origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
    a1.set_title(f"3-D mid-slice (h={pc[3]['dx']:.3f})\n"
                 f"area density {pc[3]['interfacial_area_density']:.1f}, "
                 f"$\\lambda$={pc[3]['peak_wavelength']:.3f}", fontsize=9)
    for ax in (a0, a1):
        ax.set_xticks([]); ax.set_yticks([])
    dims = [2, 3]
    dens = [pc[d]["interfacial_area_density"] for d in dims]
    a2.bar(["2-D", "3-D"], dens, color=["C0", "C3"])
    a2.set_ylabel("interfacial-area density (length$^{d-1}$/vol)")
    a2.set_title("Interface budget differs by dimension", fontsize=9)
    a2.grid(axis="y", alpha=0.3)
    fig.suptitle("Physics at equal resolution: 2-D is not a cheap 3-D", y=1.02)
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def write_numbers(s, pc):
    def mac(name, val):
        return rf"\newcommand{{\{name}}}{{{val}}}"
    a = s["two_d"][-1]
    b = s["three_d"][-1]
    # capacity milestones
    est128 = estimate(3, 128, solver="blockch")
    est256 = estimate(3, 256, solver="blockch")
    lines = [
        "% AUTO-GENERATED by gen_figures.py - do not edit.",
        mac("CfiveDofGrowthTwoD", f"{s['dof_growth_2d']:.1f}"),
        mac("CfiveDofGrowthThreeD", f"{s['dof_growth_3d']:.1f}"),
        mac("CfiveNnzPerDofTwoD", f"{s['nnz_per_dof_2d']:.0f}"),
        mac("CfiveNnzPerDofThreeD", f"{s['nnz_per_dof_3d']:.0f}"),
        mac("CfiveTwoDdofs", f"{a['dofs']:,}".replace(",", r"{,}")),
        mac("CfiveThreeDdofs", f"{b['dofs']:,}".replace(",", r"{,}")),
        mac("CfiveNnzRatio", f"{b['nnz'] / a['nnz']:.1f}"),
        mac("CfiveStepRatio", f"{b['step_s'] / a['step_s']:.1f}"),
        # memory accounting
        mac("CfiveMemOneTwoEight", f"{est128['total_gb']:.1f}"),
        mac("CfiveMemTwoFiveSix", f"{est256['total_gb']:.1f}"),
        # physics
        mac("CfivePhysH", f"{pc[2]['dx']:.3f}"),
        mac("CfiveAreaTwoD", f"{pc[2]['interfacial_area_density']:.1f}"),
        mac("CfiveAreaThreeD", f"{pc[3]['interfacial_area_density']:.1f}"),
        mac("CfiveLamTwoD", f"{pc[2]['peak_wavelength']:.3f}"),
        mac("CfiveLamThreeD", f"{pc[3]['peak_wavelength']:.3f}"),
        # cited anchors
        mac("CfiveFilmDofs", f"{CITED_LADDER[1][1]:,}".replace(",", r"{,}")),
        mac("CfiveMkOverflowDofs",
            f"{CITED_LADDER[2][1]:,}".replace(",", r"{,}")),
        mac("CfiveNovaDofs", f"{CITED_LADDER[3][1]:,}".replace(",", r"{,}")),
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
    s = measure_scaling(device=args.device)
    pc = physics_comparison(device=args.device)
    dof_growth_figure(s, "c5_dofgrowth.png")
    nnz_figure(s, "c5_nnz.png")
    memory_figure("c5_memory.png")
    physics_figure(pc, "c5_physics.png")
    write_numbers(s, pc)


if __name__ == "__main__":
    main()
