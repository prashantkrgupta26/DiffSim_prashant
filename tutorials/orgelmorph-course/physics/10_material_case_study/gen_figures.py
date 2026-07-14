"""OrgElMorph P10 - render ALL figures + number macros from a SAVED harness run
directory (results.json + history.npz).  Nothing is re-simulated (spec Phase 1:
figures from saved data), so the document's figures and numbers ARE the checked
reference case.

    PYTHONPATH=<repo>/src python gen_figures.py --run-dir outputs/p10

Figures -> ../../latex/figures/p10_*.png:
  p10_provenance.png   the loaded-parameter provenance audit (which numbers
                       are measured / fitted / assumed, + their uncertainty)
  p10_ladder.png       drying curves phi_s(t), h(t) for the spin-speed ladder
  p10_matched.png      morphology metrics vs dryness (matched state)
  p10_morphology.png   polymer fields at matched dryness for each rung
  p10_convergence.png  mesh + time convergence of the matched-state metrics
  p10_sensitivity.png  chi_pf order-of-magnitude uncertainty sweep
Numbers -> ../../latex/numbers/p10.tex.
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
NUMTEX = os.path.join(HERE, "..", "..", "latex", "numbers", "p10.tex")
COLORS = ["C0", "C2", "C3", "C1", "C4", "C5"]


def load_run(run_dir):
    with open(os.path.join(run_dir, "results.json")) as fh:
        res = json.load(fh)
    hist = dict(np.load(os.path.join(run_dir, "history.npz")))
    return res, hist


def _ladder(res):
    """Rung dicts sorted by rpm (ascending drying rate) that were actually run."""
    labels = set(res["runs"])
    return [r for r in res["ladder"] if r["label"] in labels]


# ----------------------------------------------------------------------
def fig_provenance(res, fname):
    """The honest audit table: every loaded parameter, its status, source type,
    and uncertainty -- which numbers can the reader trust."""
    prov = res["provenance"]
    order = ["chi_polymer_fullerene", "chi_polymer_solvent",
             "chi_fullerene_solvent", "N", "biot", "diffusivity_ratio",
             "noise_b_reg"]
    order = [k for k in order if k in prov] + [k for k in prov if k not in order]
    rows, cell_colors = [], []
    cmap = {"production": "#bfe3b6", "accelerated_tutorial": "#ffd9a0"}
    smap = {"measured": "#bfe3b6", "fitted": "#ffd9a0", "literature": "#cfe0f5",
            "assumed": "#f5c6c6", "unknown": "#e0e0e0", None: "#e0e0e0"}
    for k in order:
        p = prov[k]
        unc = (f"{p['uncertainty_type']}"
               + (f" {p['uncertainty_value']}" if p["uncertainty_value"]
                  not in (None, "") else ""))
        known = "yes" if p["value_known"] else "NO (range only)"
        rows.append([k, p["status"], str(p["source_type"]), unc, known])
        cell_colors.append(["white", cmap.get(p["status"], "#e0e0e0"),
                            smap.get(p["source_type"], "#e0e0e0"),
                            "white", "white" if p["value_known"] else "#f5c6c6"])
    fig, ax = plt.subplots(figsize=(11.5, 0.5 * len(rows) + 1.4), dpi=150)
    ax.axis("off")
    tbl = ax.table(cellText=rows, cellColours=cell_colors,
                   colLabels=["parameter", "status", "source", "uncertainty",
                              "value known?"],
                   colColours=["#d0d0d0"] * 5, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.35)
    ax.set_title(f"P10 provenance audit — {res['system']} "
                 "(loaded via the validated materials loader)\n"
                 "every parameter is accelerated_tutorial / fitted "
                 "— representative, NOT measured table values",
                 fontsize=11)
    _save(fig, fname)


def fig_ladder(res, hist, fname):
    rungs = _ladder(res)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.2, 4.4), dpi=150)
    for r, c in zip(rungs, COLORS):
        pre = r["label"] + "_"
        t, phis, h = hist[pre + "t"], hist[pre + "phis"], hist[pre + "h"]
        run = res["runs"][r["label"]]
        ax0.plot(t, phis, c, lw=2,
                 label=f"{r['rpm']} rpm ($k_e$={r['k_e']:g}, Bi={run['Bi']:g})")
        ax0.plot(t[-1], phis[-1], c + "o", ms=6)
        ax0.annotate(run["reason"], (t[-1], phis[-1]),
                     textcoords="offset points", xytext=(4, 6), fontsize=8,
                     color=c)
        ax1.plot(t, h, c, lw=2, label=f"{r['rpm']} rpm")
    for ph in res["config"]["matched_phis"]:
        ax0.axhline(ph, color="0.7", ls=":", lw=1)
    ax0.set_xlabel("time $t$"); ax0.set_ylabel(r"mean solvent $\phi_s$")
    ax0.set_title("Drying curves (dots = march exit)")
    ax0.legend(fontsize=8); ax0.grid(alpha=0.3)
    ax1.set_xlabel("time $t$"); ax1.set_ylabel("film height $h(t)$")
    ax1.set_title(r"Film thins as solvent leaves")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    fig.suptitle(f"{res['system']}: spin speed -> drying rate "
                 "(recorded Biot ordering mapped to a resolvable window)",
                 fontsize=12, y=1.02)
    _save(fig, fname)


def fig_matched(res, hist, fname):
    rungs = _ladder(res)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), dpi=150)
    metrics = [("wl_cells", "lateral wavelength (cells)"),
               ("iface", "interfacial length"),
               ("contrast", "phase contrast (std)")]
    for r, c in zip(rungs, COLORS):
        pre = r["label"] + "_"
        phis = hist[pre + "phis"]
        run = res["runs"][r["label"]]
        for ax, (key, _) in zip(axes, metrics):
            ax.plot(phis, hist[pre + key], c, lw=2,
                    label=f"{r['rpm']} rpm (Bi={run['Bi']:g})")
    for ax, (key, ylab) in zip(axes, metrics):
        for ph in res["config"]["matched_phis"]:
            ax.axvline(ph, color="0.7", ls=":", lw=1)
        ax.set_xlabel(r"mean solvent $\phi_s$  (dryness $\rightarrow$)")
        ax.set_ylabel(ylab); ax.invert_xaxis(); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].set_title("wavelength: noisy, weakly rate-ordered")
    axes[1].set_title("interfacial length vs dryness")
    axes[2].set_title("contrast rises with dryness (rate-independent)")
    fig.suptitle("Morphology vs DRYNESS (matched state): demixing DEGREE is set "
                 "by dryness, ~independent of spin speed", y=1.03, fontsize=12)
    _save(fig, fname)


def fig_morphology(res, hist, fname):
    rungs = _ladder(res)
    phis_tgt = sorted(res["config"]["matched_phis"], reverse=True)
    nrow = len(rungs)
    fig, axes = plt.subplots(nrow, len(phis_tgt),
                             figsize=(3.0 * len(phis_tgt), 2.7 * nrow),
                             dpi=140, squeeze=False)
    for i, r in enumerate(rungs):
        pre = r["label"] + "_"
        run = res["runs"][r["label"]]
        for j, ph in enumerate(phis_tgt):
            key = pre + f"gp_{ph:.2f}"
            ax = axes[i][j]
            if key in hist:
                ax.imshow(hist[key].T, origin="lower", cmap="viridis",
                          vmin=0, vmax=0.6, extent=[0, 1, 0, 1], aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(fr"$\phi_s$={ph:g}", fontsize=10)
        axes[i][0].set_ylabel(f"{r['rpm']} rpm\nBi={run['Bi']:g}", fontsize=9)
    fig.suptitle("Polymer field at matched dryness (fixed computational frame)",
                 fontsize=12, y=1.01)
    _save(fig, fname)


def fig_convergence(res, fname):
    mc, tc = res["mesh_conv"], res["time_conv"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.0, 4.4), dpi=150)
    levels = sorted(int(k) for k in mc["levels"])
    nx = [2 ** lv for lv in levels]
    con = [mc["levels"][str(lv)]["contrast"] for lv in levels]
    wlf = [mc["levels"][str(lv)]["wl_frac"] for lv in levels]
    ax0.plot(nx, con, "o-", color="C3", label="phase contrast")
    ax0b = ax0.twinx()
    ax0b.plot(nx, wlf, "s--", color="C0", label="wavelength (box frac)")
    ax0.set_xlabel("lateral cells $n_x$"); ax0.set_ylabel("contrast", color="C3")
    ax0b.set_ylabel("wavelength / box", color="C0")
    ax0.set_title(f"Mesh convergence at $k_e$={mc['k_e']:g}, "
                  fr"$\phi_s$={mc['phi_s']:g}")
    ax0.grid(alpha=0.3)
    dts = sorted(tc["dts"], key=lambda s: float(s))
    dtv = [float(s) for s in dts]
    cont = [tc["dts"][s]["contrast"] for s in dts]
    ax1.plot(dtv, cont, "o-", color="C3")
    ax1.set_xlabel(r"$\Delta t_{\max}$"); ax1.set_ylabel("phase contrast")
    ax1.set_title(f"Time convergence at $k_e$={tc['k_e']:g}, "
                  fr"$\phi_s$={tc['phi_s']:g}")
    ax1.grid(alpha=0.3); ax1.invert_xaxis()
    fig.suptitle("The demixing-degree conclusion is mesh- and time-robust "
                 "(small relative change under refinement)", y=1.02, fontsize=12)
    _save(fig, fname)


def fig_sensitivity(res, fname):
    s = res["sensitivity"]
    prov = res["provenance"].get("chi_polymer_fullerene", {})
    chis = sorted(float(k) for k in s["runs"])
    con = [s["runs"][f"{c:g}"]["contrast"] for c in chis]
    wl = [s["runs"][f"{c:g}"]["wl_cells"] for c in chis]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.0, 4.4), dpi=150)
    ax0.plot(chis, con, "o-", color="C3")
    ax0.axvline(s["base"], color="0.5", ls=":",
                label=f"recorded $\\chi_{{pf}}$={s['base']:g}")
    ax0.set_xlabel(r"$\chi_{pf}$ (polymer-fullerene)")
    ax0.set_ylabel(f"phase contrast at $\\phi_s$={s['phi_s']:g}")
    ax0.set_title("demixing DEGREE tracks chi (robust, monotone)")
    ax0.legend(fontsize=9); ax0.grid(alpha=0.3)
    ax1.plot(chis, wl, "s-", color="C0")
    ax1.axvline(s["base"], color="0.5", ls=":")
    ax1.set_xlabel(r"$\chi_{pf}$"); ax1.set_ylabel("wavelength (cells)")
    ax1.set_title("wavelength: noisier, weaker chi dependence")
    ax1.grid(alpha=0.3)
    ut = prov.get("uncertainty_type"); uv = prov.get("uncertainty_value")
    fig.suptitle(r"chi$_{pf}$ uncertainty sweep "
                 f"({ut} {uv}): which conclusions survive an order-of-magnitude "
                 "change in chi", y=1.02, fontsize=12)
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
    if not np.isfinite(x):
        return r"\ensuremath{\mathrm{NaN}}"
    if x == 0 or abs(x) < 1e-13:
        return r"\ensuremath{<10^{-13}}"
    m, e = f"{x:.{sig}e}".split("e")
    return f"\\ensuremath{{{m}\\times10^{{{int(e)}}}}}"


def _yn(b):
    return "yes" if b else "no"


def write_numbers(res):
    ck = res["checks"]
    cfg = res["config"]
    prov = res["provenance"]["chi_polymer_fullerene"]
    lines = [
        "% AUTO-GENERATED by gen_figures.py - do not edit.",
        _mac("PtenSystem", res["system"].replace("_", ":")),
        _mac("PtenChiPF", f"{cfg['chi'][0]:g}"),
        _mac("PtenChiPS", f"{cfg['chi'][1]:g}"),
        _mac("PtenChiFS", f"{cfg['chi'][2]:g}"),
        _mac("PtenNpoly", f"{cfg['N'][0]:g}"),
        _mac("PtenNfull", f"{cfg['N'][1]:g}"),
        _mac("PtenNrungs", f"{ck['n_rungs']}"),
        _mac("PtenProvStatus",
             res["provenance"]["chi_polymer_fullerene"]["status"].replace(
                 "_", r"\_")),
        _mac("PtenChiPFunc", f"{prov['uncertainty_value']:g}"),
        _mac("PtenBiSlow", f"{ck['Bi_min']:g}"),
        _mac("PtenBiFast", f"{ck['Bi_max']:g}"),
        _mac("PtenSlowRpm", res["runs"][ck["slow_label"]]["rpm"].__str__()
             if isinstance(res["runs"][ck["slow_label"]].get("rpm"), str)
             else f"{res['runs'][ck['slow_label']]['rpm']}"),
        _mac("PtenFastRpm", f"{res['runs'][ck['fast_label']]['rpm']}"),
        _mac("PtenContrastSlowHi", f"{ck['contrast_slow_hi']:.3f}"),
        _mac("PtenContrastSlowMid", f"{ck['contrast_slow_mid']:.3f}"),
        _mac("PtenContrastSlowLo", f"{ck['contrast_slow_lo']:.3f}"),
        _mac("PtenContrastFastMid", f"{ck['contrast_fast_mid']:.3f}"),
        _mac("PtenWLslowMid", f"{ck['wl_cells_slow_mid']:.1f}"),
        _mac("PtenWLfastMid", f"{ck['wl_cells_fast_mid']:.1f}"),
        _mac("PtenFinerOrder", _yn(ck["finer_ordering_mid"])),
        _mac("PtenContrastRiseSlow", _yn(ck["contrast_rises_slow"])),
        _mac("PtenContrastRiseFast", _yn(ck["contrast_rises_fast"])),
        _mac("PtenContrastRateSpread",
             f"{100 * ck['contrast_rate_spread_mid']:.1f}"),
        _mac("PtenBiCollapseSpread",
             f"{100 * ck['bi_collapse_wl_spread_mid']:.1f}"),
        _mac("PtenMeshDContrast", f"{100 * ck['mesh_contrast_rel_change']:.1f}"),
        _mac("PtenMeshDWL", f"{100 * ck['mesh_wl_frac_rel_change']:.1f}"),
        _mac("PtenTimeDContrast", f"{100 * ck['time_contrast_rel_change']:.1f}"),
        _mac("PtenSensSpan", f"{100 * ck['sens_contrast_rel_span']:.1f}"),
        _mac("PtenSensMonotone", _yn(ck["sens_contrast_monotone_in_chi"])),
        _mac("PtenBalP", _sci(ck["balance_polymer_rel_max"])),
        _mac("PtenBalF", _sci(ck["balance_fullerene_rel_max"])),
    ]
    os.makedirs(os.path.dirname(NUMTEX), exist_ok=True)
    with open(NUMTEX, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", NUMTEX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.join(HERE, "outputs", "p10"))
    args = ap.parse_args()
    res, hist = load_run(args.run_dir)
    os.makedirs(FIGDIR, exist_ok=True)
    fig_provenance(res, "p10_provenance.png")
    fig_ladder(res, hist, "p10_ladder.png")
    fig_matched(res, hist, "p10_matched.png")
    fig_morphology(res, hist, "p10_morphology.png")
    fig_convergence(res, "p10_convergence.png")
    fig_sensitivity(res, "p10_sensitivity.png")
    write_numbers(res)


if __name__ == "__main__":
    main()
