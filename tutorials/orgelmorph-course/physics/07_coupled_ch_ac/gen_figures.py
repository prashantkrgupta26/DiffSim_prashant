"""OrgElMorph course - Physics P7: figures + numbers macros.

    python gen_figures.py --device cuda:0

Writes ../../latex/figures/p7_*.png and ../../latex/numbers/p7.tex from
REAL runs.  Figures:
  p7_demixing.png  the three commensurate controls + the contrast
                   decomposition (crystal-bulk vs chi-expulsion channel)
  p7_ladder.png    the coupled free-energy budget (monotone) + causality
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import coupled as C
from coupled import chi_eff_limits
from run import three_controls

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "..", "latex",
                      "figures")
NUMTEX = os.path.join(os.path.dirname(__file__), "..", "..", "latex",
                      "numbers", "p7.tex")


def fig_demix(tc, fname):
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0), dpi=150)
    panels = [("ch\\_only ($K{=}0$)", tc["chonly"]),
              ("coupled\\_nochi\n($\\chi_{ca}{=}\\chi_{aa}$)", tc["nochi"]),
              ("full ($\\chi_{ca}{>}\\chi_{aa}$)", tc["full"])]
    im = None
    for ax, (lab, r) in zip(axes[:3], panels):
        im = ax.imshow(r["phi0"].T, origin="lower", cmap="viridis",
                       vmin=0, vmax=1)
        # same shared mask outline (the full crystal footprint) on all three
        ax.contour(tc["full"]["psi"].T, levels=[0.5], colors="r",
                   linewidths=1.2)
        ax.set_title(lab, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes[:3], fraction=0.012, pad=0.02,
                 label=r"$\phi_0$ (crystallizer)")
    # contrast decomposition bar
    ax = axes[3]
    vals = [tc["c_chonly"], tc["cryst_channel"], tc["chi_channel"]]
    labs = ["CH only", "crystal-bulk\nchannel", "chi-expulsion\nchannel"]
    cols = ["#888", "C0", "C3"]
    bottoms = [0, max(tc["c_chonly"], 0), tc["c_nochi"]]
    for v, l, c, b in zip(vals, labs, cols, bottoms):
        ax.bar(0, v, bottom=b, color=c, width=0.6, label=l)
    ax.axhline(tc["c_full"], color="k", ls="--", lw=1,
               label=f"full = {tc['c_full']:.3f}")
    ax.set_xticks([]); ax.set_ylabel("composition contrast (shared mask)")
    ax.set_title("Contrast decomposition", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Commensurate controls (SAME mask + metric): demixing = "
                 "crystal-bulk channel + chi-expulsion channel",
                 fontsize=12, y=1.03)
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def fig_budget_causality(tc, caus, fname):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=150)
    # energy budget
    e = tc["full"]["energy"]
    t = tc["full"]["t"]
    for key, col in (("entropy", "C0"), ("chi", "C3"), ("cryst", "C2"),
                     ("grad_phi", "C4"), ("grad_psi", "C5")):
        axes[0].plot(t, [ee[key] for ee in e], col, lw=1.3, label=key)
    axes[0].plot(t, [ee["total"] for ee in e], "k-", lw=2.2, label="total")
    axes[0].set_xlabel("time $t$"); axes[0].set_ylabel("free energy $F$")
    axes[0].set_title("Coupled energy budget (total decreases)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    # causality bars
    labs = ["frozen\nchi-channel", "kinetics\nfast", "kinetics\nslow"]
    vals = [caus["frozen_chi_channel"], caus["kinetics_fast_contrast"],
            caus["kinetics_slow_contrast"]]
    axes[1].bar(range(3), vals, color=["C3", "C0", "C1"])
    axes[1].set_xticks(range(3)); axes[1].set_xticklabels(labs, fontsize=8)
    axes[1].set_ylabel("contrast")
    axes[1].set_title("Causality: coupling demixes at fixed geometry;\n"
                      "both rates reach a large contrast", fontsize=10)
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("Coupled free-energy budget and causality experiments",
                 fontsize=12, y=1.02)
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def run_causality(dm, mesh, cons, device):
    fz_full = C.run(dm, mesh, cons, mode="full", freeze_psi=True,
                    device=device)
    fz_nochi = C.run(dm, mesh, cons, mode="coupled_nochi", freeze_psi=True,
                     device=device)
    fzmask = fz_full["psi_full"] > 0.5
    fz_chi = (C.contrast_on_mask(fz_full["phi0_full"], fzmask)
              - C.contrast_on_mask(fz_nochi["phi0_full"], fzmask))
    fast = C.run(dm, mesh, cons, mode="full", L_psi=6.0, device=device)
    slow = C.run(dm, mesh, cons, mode="full", L_psi=1.0, device=device)
    return dict(
        frozen_chi_channel=float(fz_chi),
        kinetics_fast_area=fast["area_end"],
        kinetics_slow_area=slow["area_end"],
        kinetics_fast_contrast=C.contrast_on_mask(
            fast["phi0_full"], fast["psi_full"] > 0.5),
        kinetics_slow_contrast=C.contrast_on_mask(
            slow["phi0_full"], slow["psi_full"] > 0.5))


def write_numbers(tc, caus):
    lim = chi_eff_limits(1.2, 2.1, 2.6, 3.0, bulk="p1")
    limr = chi_eff_limits(1.2, 2.1, 2.6, 3.0, bulk="r14")
    et = tc["full"]["energy_total"]
    incr = np.diff(et)

    def mac(name, val):
        return rf"\newcommand{{\{name}}}{{{val}}}"
    lines = ["% AUTO-GENERATED by gen_figures.py - do not edit.",
             mac("PsevenChiAA", f"{lim['aa']:.2f}"),
             mac("PsevenChiCA", f"{lim['ca']:.2f}"),
             mac("PsevenChiAC", f"{lim['ac']:.2f}"),
             mac("PsevenChiCC", f"{lim['cc']:.2f}"),
             mac("PsevenChiCAincr", f"{limr['ca']:.2f}"),
             mac("PsevenCtrlChonly", f"{tc['c_chonly']:+.3f}"),
             mac("PsevenCtrlNochi", f"{tc['c_nochi']:+.3f}"),
             mac("PsevenCtrlFull", f"{tc['c_full']:+.3f}"),
             mac("PsevenCrystChannel", f"{tc['cryst_channel']:+.3f}"),
             mac("PsevenChiChannel", f"{tc['chi_channel']:+.3f}"),
             mac("PsevenRelFullNochi",
                 f"{tc['c_full'] / tc['c_nochi']:.2f}"),
             mac("PsevenEnergyStart", f"{et[0]:.4f}"),
             mac("PsevenEnergyEnd", f"{et[-1]:.4f}"),
             mac("PsevenEnergyMaxPos", f"{max(0.0, incr.max()):.1e}"),
             mac("PsevenFrozenChi", f"{caus['frozen_chi_channel']:+.3f}"),
             mac("PsevenKinFast", f"{caus['kinetics_fast_contrast']:.3f}"),
             mac("PsevenKinSlow", f"{caus['kinetics_slow_contrast']:.3f}")]
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
    dm, mesh, cons = C.build_mesh_dm(args.level, device=args.device)
    tc = three_controls(dm, mesh, cons, args.device)
    caus = run_causality(dm, mesh, cons, args.device)
    fig_demix(tc, "p7_demixing.png")
    fig_budget_causality(tc, caus, "p7_ladder.png")
    write_numbers(tc, caus)


if __name__ == "__main__":
    main()
