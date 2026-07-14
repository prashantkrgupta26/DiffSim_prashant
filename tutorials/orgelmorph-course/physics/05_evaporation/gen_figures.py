"""OrgElMorph course - Physics P5: render ALL figures + number macros from a
SAVED harness run directory (results.json + history.npz).  Nothing is
re-simulated here (spec Phase 1: figures generated from saved data, not
hand-copied), so the document's figures and numbers ARE the checked run.

    PYTHONPATH=<repo>/src python gen_figures.py --run-dir outputs/p5

Figures written to ../../latex/figures/p5_*.png:
  p5_drying.png      phi_s(t) and h(t) drying curves, annotated with exit reason
  p5_matched.png     morphology metric vs mean solvent fraction (matched state)
  p5_morphology.png  fields at matched dryness, FIXED and PHYSICAL coordinates
  p5_regime.png      the Biot regime map (k_e x D_s, Bi contours)
Numbers to ../../latex/numbers/p5.tex.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, "..", "..", "latex", "figures")
NUMTEX = os.path.join(HERE, "..", "..", "latex", "numbers", "p5.tex")
COLORS = ["C0", "C2", "C3", "C1", "C4"]


def load_run(run_dir):
    with open(os.path.join(run_dir, "results.json")) as fh:
        res = json.load(fh)
    hist = dict(np.load(os.path.join(run_dir, "history.npz")))
    return res, hist


def base_rate_keys(hist):
    """The k_e values stored to history (the matched/base rows), sorted
    ascending, as (float, prefix) pairs."""
    ks = set()
    for k in hist:
        if k.startswith("ke") and k.endswith("_t"):
            ks.add(k[2:-2])
    return [(v, f"ke{v:.3f}_") for v in sorted(float(s) for s in ks)]


# ----------------------------------------------------------------------
def fig_drying(res, hist, fname):
    rows = base_rate_keys(hist)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.2, 4.4), dpi=150)
    for (ke, pre), c in zip(rows, COLORS):
        t = hist[pre + "t"]; phis = hist[pre + "phis"]; h = hist[pre + "h"]
        rkey = f"ke{ke:.3f}_ds{res['config']['ds_base']:.3f}"
        reason = res["runs"][rkey]["reason"]
        Bi = res["runs"][rkey]["Bi"]
        ax0.plot(t, phis, c, lw=2, label=f"$k_e$={ke:g} (Bi={Bi:g})")
        ax0.plot(t[-1], phis[-1], c + "o", ms=6)
        ax0.annotate(reason, (t[-1], phis[-1]), textcoords="offset points",
                     xytext=(4, 6), fontsize=8, color=c)
        ax1.plot(t, h, c, lw=2, label=f"$k_e$={ke:g}")
    for ph in res["config"]["matched_phis"]:
        ax0.axhline(ph, color="0.7", ls=":", lw=1)
        ax0.annotate(fr"$\phi_s$={ph:g}", (0, ph), fontsize=8, color="0.4",
                     va="bottom")
    ax0.set_xlabel("time $t$"); ax0.set_ylabel(r"mean solvent $\phi_s$")
    ax0.set_title("Drying curves (dots = march exit)")
    ax0.legend(fontsize=9); ax0.grid(alpha=0.3)
    ax1.set_xlabel("time $t$"); ax1.set_ylabel("film height $h(t)$")
    ax1.set_title(r"Film thins as solvent leaves ($dh/dt=-K$)")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
    fig.suptitle("Evaporation sets the morphology clock — every rate driven to "
                 "the same dryness", fontsize=12, y=1.02)
    _save(fig, fname)


def fig_matched(res, hist, fname):
    """Morphology metrics as a CONTINUOUS function of mean solvent fraction
    (the honest, un-confounded axis), with the matched dryness levels marked."""
    rows = base_rate_keys(hist)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), dpi=150)
    metrics = [("wl_cells", "lateral wavelength (cells)"),
               ("iface", "interfacial length"),
               ("contrast", "phase contrast (std)")]
    for (ke, pre), c in zip(rows, COLORS):
        phis = hist[pre + "phis"]
        rkey = f"ke{ke:.3f}_ds{res['config']['ds_base']:.3f}"
        Bi = res["runs"][rkey]["Bi"]
        for ax, (key, _) in zip(axes, metrics):
            ax.plot(phis, hist[pre + key], c, lw=2,
                    label=f"$k_e$={ke:g} (Bi={Bi:g})")
    for ax, (key, ylab) in zip(axes, metrics):
        for ph in res["config"]["matched_phis"]:
            ax.axvline(ph, color="0.7", ls=":", lw=1)
        ax.set_xlabel(r"mean solvent fraction $\phi_s$  (dryness $\rightarrow$)")
        ax.set_ylabel(ylab)
        ax.invert_xaxis()          # dry to the right
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    axes[0].set_title("wavelength: Bi-controlled but noisy")
    axes[1].set_title("interfacial length vs dryness")
    axes[2].set_title("contrast rises with dryness (rate-independent)")
    fig.suptitle("Morphology vs DRYNESS (matched-state): the demixing DEGREE "
                 "(contrast) is set by dryness, ~independent of rate", y=1.03,
                 fontsize=12)
    _save(fig, fname)


def fig_morphology(res, hist, fname):
    """Polymer field at the matched dryness levels, in BOTH the fixed
    computational frame and the PHYSICAL frame (vertical extent x h(t))."""
    rows = base_rate_keys(hist)
    phis_tgt = sorted(res["config"]["matched_phis"], reverse=True)
    nrow = len(rows)
    fig, axes = plt.subplots(nrow, 2 * len(phis_tgt),
                             figsize=(3.0 * len(phis_tgt), 3.0 * nrow),
                             dpi=140, squeeze=False)
    vmax = 0.6
    for i, (ke, pre) in enumerate(rows):
        rkey = f"ke{ke:.3f}_ds{res['config']['ds_base']:.3f}"
        Bi = res["runs"][rkey]["Bi"]
        for j, ph in enumerate(phis_tgt):
            gp = hist[pre + f"gp_{ph:.2f}"]
            h = float(hist[pre + f"h_{ph:.2f}"])
            axf = axes[i][2 * j]         # fixed computational frame (unit box)
            axf.imshow(gp.T, origin="lower", cmap="viridis", vmin=0, vmax=vmax,
                       extent=[0, 1, 0, 1], aspect="auto")
            axp = axes[i][2 * j + 1]     # physical frame: vertical extent x h(t)
            axp.imshow(gp.T, origin="lower", cmap="viridis", vmin=0, vmax=vmax,
                       extent=[0, 1, 0, h], aspect="auto")
            axp.set_ylim(0, 1.0)
            if i == 0:
                axf.set_title(fr"$\phi_s$={ph:g}" + "\nfixed", fontsize=9)
                axp.set_title(fr"$\phi_s$={ph:g}" + "\nphysical", fontsize=9)
            for a in (axf, axp):
                a.set_xticks([]); a.set_yticks([])
        axes[i][0].set_ylabel(f"$k_e$={ke:g}\nBi={Bi:g}", fontsize=10)
    fig.suptitle("Polymer field at matched dryness — fixed computational frame "
                 "vs physical (thinning) frame", fontsize=12, y=1.01)
    _save(fig, fname)


def fig_regime(res, fname):
    """Biot regime map: each (k_e, D_s) run on the k_e x D_s plane, coloured by
    matched-state wavelength, with Bi = k_e h0 / D_s iso-contours + the Bi=1
    drying/diffusion crossover."""
    runs = res["runs"]
    kes = sorted(set(r["k_e"] for r in runs.values()))
    dss = sorted(set(r["D_s"] for r in runs.values()))
    mps = sorted(res["config"]["matched_phis"])
    mid = mps[len(mps) // 2]
    Z = np.full((len(dss), len(kes)), np.nan)
    for r in runs.values():
        Z[dss.index(r["D_s"]), kes.index(r["k_e"])] = \
            r["matched"].get(f"{mid:.2f}", {}).get("wl_cells", np.nan)
    fig, ax = plt.subplots(figsize=(7.4, 5.4), dpi=150)
    if len(kes) > 1 and len(dss) > 1:
        im = ax.pcolormesh(kes, dss, Z, shading="nearest", cmap="viridis")
        fig.colorbar(im, ax=ax, label=fr"wavelength (cells) at $\phi_s$={mid:g}")
        KE, DS = np.meshgrid(np.linspace(min(kes), max(kes), 60),
                             np.linspace(min(dss), max(dss), 60))
        BI = KE * res["config"].get("h0", 1.0) / DS
        cs = ax.contour(KE, DS, BI, levels=[0.75, 1.0, 1.5, 3.0, 6.0],
                        colors="w", linewidths=1.0)
        ax.clabel(cs, fmt="Bi=%.2g", fontsize=8)
        c1 = ax.contour(KE, DS, BI, levels=[1.0], colors="r", linewidths=2.0)
        ax.clabel(c1, fmt="Bi=1", fontsize=9)
    for r in runs.values():
        ax.plot(r["k_e"], r["D_s"], "ko", ms=5)
        ax.annotate(f"Bi={r['Bi']:.2g}", (r["k_e"], r["D_s"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.set_xlabel("evaporation rate $k_e$")
    ax.set_ylabel("solute mobility $D_s$")
    ax.set_title("Biot regime map: morphology collapses onto "
                 r"Bi $=k_e h_0/D_s$" + "\n(drying-limited Bi>1, "
                 "diffusion-limited Bi<1)")
    _save(fig, fname)


def _save(fig, fname):
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


# ----------------------------------------------------------------------
def _mac(name, val):
    return rf"\newcommand{{\{name}}}{{{val}}}"


def _sci(x, sig=1):
    """LaTeX \\ensuremath scientific-notation body (no bare $ or \\times)."""
    if not np.isfinite(x):
        return r"\ensuremath{\mathrm{NaN}}"
    if x == 0 or abs(x) < 1e-13:
        return r"\ensuremath{<10^{-13}}"
    m, e = f"{x:.{sig}e}".split("e")
    return f"\\ensuremath{{{m}\\times10^{{{int(e)}}}}}"


def write_numbers(res, hist):
    ck = res["checks"]
    ds_base = res["config"]["ds_base"]
    slow_k, fast_k = ck["slow_ke"], ck["fast_ke"]
    slow = res["runs"][f"ke{slow_k:.3f}_ds{ds_base:.3f}"]
    fast = res["runs"][f"ke{fast_k:.3f}_ds{ds_base:.3f}"]

    def wlc(run, ph):
        return run["matched"].get(f"{ph:.2f}", {}).get("wl_cells", float("nan"))

    def ifc(run, ph):
        return run["matched"].get(f"{ph:.2f}", {}).get("iface", float("nan"))

    lines = ["% AUTO-GENERATED by gen_figures.py - do not edit.",
             _mac("PfiveKeSlow", f"{slow_k:g}"),
             _mac("PfiveKeFast", f"{fast_k:g}"),
             _mac("PfiveBiSlow", f"{slow['Bi']:.2g}"),
             _mac("PfiveBiFast", f"{fast['Bi']:.2g}"),
             _mac("PfiveBiMin", f"{ck['Bi_min']:.2g}"),
             _mac("PfiveBiMax", f"{ck['Bi_max']:.2g}"),
             _mac("PfiveTdrySlow", f"{slow['t_dry']:.2f}"),
             _mac("PfiveTdryFast", f"{fast['t_dry']:.2f}"),
             _mac("PfiveHslow", f"{slow['h_final']:.3f}"),
             _mac("PfiveHfast", f"{fast['h_final']:.3f}"),
             _mac("PfiveReasonSlow", slow["reason"].replace("_", r"\_")),
             _mac("PfiveReasonFast", fast["reason"].replace("_", r"\_")),
             _mac("PfiveWLslowThirty", f"{wlc(slow, 0.30):.1f}"),
             _mac("PfiveWLslowTwenty", f"{wlc(slow, 0.20):.1f}"),
             _mac("PfiveWLslowTen", f"{wlc(slow, 0.10):.1f}"),
             _mac("PfiveWLfastThirty", f"{wlc(fast, 0.30):.1f}"),
             _mac("PfiveWLfastTwenty", f"{wlc(fast, 0.20):.1f}"),
             _mac("PfiveWLfastTen", f"{wlc(fast, 0.10):.1f}"),
             _mac("PfiveIfaceSlowTwenty", f"{ifc(slow, 0.20):.2f}"),
             _mac("PfiveIfaceFastTwenty", f"{ifc(fast, 0.20):.2f}"),
             _mac("PfiveContrastThirty", f"{ck['contrast_slow_030']:.3f}"),
             _mac("PfiveContrastTwenty", f"{ck['contrast_slow_020']:.3f}"),
             _mac("PfiveContrastTen", f"{ck['contrast_slow_010']:.3f}"),
             _mac("PfiveContrastSpread",
                  f"{100*ck['contrast_rate_spread_020']:.1f}"),
             _mac("PfiveSolvRemoved",
                  f"{100*slow['budget']['solvent_removed_frac']:.0f}"),
             _mac("PfiveDhSlow", f"{slow['budget']['dh']:.3f}"),
             _mac("PfiveBalPoly", _sci(ck["balance_polymer_rel_max"])),
             _mac("PfiveBalFull", _sci(ck["balance_fullerene_rel_max"])),
             _mac("PfiveBudgetClosure", _sci(ck["budget_closure_max"])),
             _mac("PfiveCollapseSpread",
                  f"{100*ck['bi_collapse_spread']:.1f}"),
             _mac("PfiveNruns", f"{ck['n_runs']}")]
    os.makedirs(os.path.dirname(NUMTEX), exist_ok=True)
    with open(NUMTEX, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", NUMTEX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.join(HERE, "outputs", "p5"),
                    help="a completed harness run directory")
    args = ap.parse_args()
    res, hist = load_run(args.run_dir)
    os.makedirs(FIGDIR, exist_ok=True)
    fig_drying(res, hist, "p5_drying.png")
    fig_matched(res, hist, "p5_matched.png")
    fig_morphology(res, hist, "p5_morphology.png")
    fig_regime(res, "p5_regime.png")
    write_numbers(res, hist)


if __name__ == "__main__":
    main()
