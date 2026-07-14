"""OrgElMorph course - Physics P4: figures + numbers, from the SAVED run dir.

Renders every figure and emits latex/numbers/p4.tex purely from a harness
run directory (results.json + history.npz) -- no recomputation, so the
document's figures and numbers ARE the checked run.

    PYTHONPATH=<repo>/src python run_harness.py --config configs/p4.yaml \\
        --mode reference --output outputs/p4 --overwrite
    python gen_figures.py --run outputs/p4

Figures:
  p4_landscape.png  free-energy contours + simplex + spinodal (det=0) + IC
                    + predicted binodal + simulated GMM cloud endpoints
  p4_spinodal.png   Hessian min-eigenvalue sign map + spinodal curve + IC
  p4_gibbs.png      composition-cloud evolution on the Gibbs triangle
  p4_fields.png     the two final solute fields
  p4_nshift.png     symmetric vs asymmetric-N clouds + spinodal curves
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

from ternary import bary_to_xy

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, "..", "..", "latex", "figures")
NUMTEX = os.path.join(HERE, "..", "..", "latex", "numbers", "p4.tex")

# affine barycentric -> triangle-xy Jacobian (for transforming covariances)
_J = np.array([[1.0, 0.5], [0.0, np.sqrt(3.0) / 2.0]])


def _triangle(ax):
    v = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax.plot(v[:, 0], v[:, 1], "k-", lw=1.2)
    ax.text(-0.02, -0.03, "solvent", ha="right", va="top", fontsize=9)
    ax.text(1.02, -0.03, r"$\phi_1$ (solute 1)", ha="left", va="top",
            fontsize=9)
    ax.text(0.5, np.sqrt(3) / 2 + 0.02, r"$\phi_2$ (solute 2)",
            ha="center", va="bottom", fontsize=9)
    ax.set_aspect("equal"); ax.axis("off")


def _cov_ellipse(ax, mean_phi, cov_phi, **kw):
    """Draw a 1-sigma covariance ellipse (composition-space cov mapped to
    triangle-xy through the affine bary Jacobian)."""
    cov_xy = _J @ np.asarray(cov_phi) @ _J.T
    vals, vecs = np.linalg.eigh(cov_xy)
    vals = np.maximum(vals, 0.0)
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    x, y = bary_to_xy(mean_phi[0], mean_phi[1])
    e = Ellipse((float(x), float(y)), 2 * np.sqrt(vals[0]),
                2 * np.sqrt(vals[1]), angle=ang, fill=False, **kw)
    ax.add_patch(e)


def load_run(run_dir):
    with open(os.path.join(run_dir, "results.json")) as fh:
        res = json.load(fh)
    hist = np.load(os.path.join(run_dir, "history.npz"))
    return res, hist


# --------------------------------------------------------------------------
def fig_landscape(res, hist, fname):
    """The signature theory-vs-simulation figure."""
    bx, by = bary_to_xy(hist["grid_p1"], hist["grid_p2"])
    f = hist["grid_f"]; det = hist["grid_det"]
    fig, ax = plt.subplots(figsize=(6.6, 6.0), dpi=150)
    _triangle(ax)
    # free-energy contours
    lv = np.nanpercentile(f, np.linspace(2, 98, 18))
    ax.contour(bx, by, f, levels=lv, cmap="viridis", linewidths=0.6,
               alpha=0.7)
    # spinodal: det(H) = 0, shade the unstable det<0 region
    ax.contourf(bx, by, (det < 0).astype(float), levels=[0.5, 1.5],
                colors=["#d6483b"], alpha=0.12)
    ax.contour(bx, by, det, levels=[0.0], colors="#d6483b", linewidths=2.0)
    # simulated cloud endpoints (GMM means + 1-sigma cov ellipses)
    cl = res["clustering"]
    for lab, key, ckey, col in (("phase A (sim)", "phaseA_mean", "phaseA_cov",
                                 "#1f6feb"),
                                ("phase B (sim)", "phaseB_mean", "phaseB_cov",
                                 "#f59f00")):
        m = cl[key]; x, y = bary_to_xy(m[0], m[1])
        ax.scatter([x], [y], s=90, c=col, edgecolor="k", zorder=6, label=lab)
        _cov_ellipse(ax, m, cl[ckey], edgecolor=col, lw=1.4, zorder=6)
    # predicted binodal tie-line
    b = res.get("binodal")
    if b:
        ax_, ay_ = bary_to_xy(b["phaseA"][0], b["phaseA"][1])
        bx_, by_ = bary_to_xy(b["phaseB"][0], b["phaseB"][1])
        ax.plot([ax_, bx_], [ay_, by_], "k--o", lw=1.6, ms=5, zorder=5,
                label="predicted binodal (common tangent)")
    # spinodal legend proxy
    ax.plot([], [], color="#d6483b", lw=2.0, label="spinodal (det $H=0$)")
    # initial blend
    ix, iy = bary_to_xy(res["phi0"][0], res["phi0"][1])
    ax.scatter([ix], [iy], marker="*", s=260, c="w", edgecolor="k",
               zorder=7, label="initial blend")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title("Free-energy landscape, spinodal, predicted binodal\n"
                 "and the simulated coexisting phases", fontsize=11)
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def fig_spinodal(res, hist, fname):
    bx, by = bary_to_xy(hist["grid_p1"], hist["grid_p2"])
    me = hist["grid_min_eig"]; det = hist["grid_det"]
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=150)
    _triangle(ax)
    vmax = np.nanmax(np.abs(me))
    pcm = ax.pcolormesh(bx, by, me, cmap="RdBu", shading="auto",
                        vmin=-vmax, vmax=vmax)
    ax.contour(bx, by, det, levels=[0.0], colors="k", linewidths=2.0)
    ix, iy = bary_to_xy(res["phi0"][0], res["phi0"][1])
    ax.scatter([ix], [iy], marker="*", s=260, c="w", edgecolor="k", zorder=7)
    fig.colorbar(pcm, ax=ax, fraction=0.04, pad=0.02,
                 label=r"smaller Hessian eigenvalue $\lambda_{\min}$")
    ax.set_title(r"Hessian eigenvalue sign map: $\lambda_{\min}<0$ (red) is "
                 "spinodal-unstable" "\n(black = det $H=0$; star = initial "
                 "blend)", fontsize=10)
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def fig_gibbs(res, hist, fname):
    snaps = sorted(float(k.split("_")[-1]) for k in hist.files
                   if k.startswith("snap_p1_"))
    fig, axes = plt.subplots(1, len(snaps), figsize=(3.4 * len(snaps), 3.4),
                             dpi=150)
    axes = np.atleast_1d(axes)
    for ax, tt in zip(axes, snaps):
        _triangle(ax)
        p1 = hist[f"snap_p1_{tt:.4f}"]; p2 = hist[f"snap_p2_{tt:.4f}"]
        x, y = bary_to_xy(p1, p2)
        ax.scatter(x, y, s=2, c=p2, cmap="plasma", alpha=0.35, vmin=0, vmax=0.9)
        ax.set_title(f"t = {tt:.3g}", fontsize=11)
    fig.suptitle("Composition cloud on the Gibbs triangle: the blob opens "
                 "along a tie-line as the blend demixes", fontsize=12, y=1.02)
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def fig_fields(res, hist, fname):
    side = res["side"]
    p1 = hist["p1f"].reshape(side, side); p2 = hist["p2f"].reshape(side, side)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0), dpi=150)
    for ax, (fld, lab, cm) in zip(
            axes, [(p1, r"$\phi_1$ (solute 1)", "Blues"),
                   (p2, r"$\phi_2$ (solute 2)", "Oranges")]):
        im = ax.imshow(fld, origin="lower", cmap=cm, vmin=0, vmax=0.9)
        ax.set_title(lab, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Final ternary morphology (the two solute fields)",
                 fontsize=12, y=1.02)
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def fig_nshift(res, hist, fname):
    """Symmetric vs asymmetric-N: clouds + spinodal curves on the triangle."""
    from ternary import spinodal_maps
    ns = res["nshift"]
    bx, by = bary_to_xy(hist["grid_p1"], hist["grid_p2"])
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.4), dpi=150)
    panels = [("symmetric N = " + str(res["N"]), res["N"], "p1f", "p2f",
               res["clustering"]["phaseA_mean"],
               res["clustering"]["phaseB_mean"]),
              ("asymmetric N = " + str(ns["asym_N"]), ns["asym_N"],
               "asym_p1f", "asym_p2f",
               ns.get("asym_phaseA_mean"), ns.get("asym_phaseB_mean"))]
    for ax, (title, N, k1, k2, A, B) in zip(axes, panels):
        _triangle(ax)
        _, _, _, det, _ = spinodal_maps(res["chi"], N, n=200)
        gbx, gby = bary_to_xy(*np.meshgrid(
            np.linspace(1e-3, 1 - 1e-3, det.shape[0]),
            np.linspace(1e-3, 1 - 1e-3, det.shape[0])))
        ax.contourf(gbx, gby, (det < 0).astype(float), levels=[0.5, 1.5],
                    colors=["#d6483b"], alpha=0.12)
        ax.contour(gbx, gby, det, levels=[0.0], colors="#d6483b",
                   linewidths=1.8)
        if k1 in hist.files:
            x, y = bary_to_xy(hist[k1], hist[k2])
            ax.scatter(x, y, s=3, c="#333", alpha=0.3)
        for m, col in ((A, "#1f6feb"), (B, "#f59f00")):
            if m is not None:
                x, y = bary_to_xy(m[0], m[1])
                ax.scatter([x], [y], s=90, c=col, edgecolor="k", zorder=6)
        ix, iy = bary_to_xy(res["phi0"][0], res["phi0"][1])
        ax.scatter([ix], [iy], marker="*", s=220, c="w", edgecolor="k",
                   zorder=7)
        ax.set_title(title, fontsize=11)
    fig.suptitle(r"The $N_i$ shift: unequal degrees of polymerization enlarge "
                 r"the spinodal (critical $\chi_{12}$ drops "
                 f"{ns['sym_chi12_crit']:.2f}" r"$\to$"
                 f"{ns['asym_chi12_crit']:.2f})", fontsize=12, y=1.01)
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


# --------------------------------------------------------------------------
def _sci(x, sig=1):
    """LaTeX \\ensuremath a x 10^{b} for a float (machine-zero aware)."""
    if abs(x) < 1e-14:
        return r"\ensuremath{<10^{-14}}"
    m, e = f"{x:.{sig}e}".split("e")
    return rf"\ensuremath{{{m}\times10^{{{int(e)}}}}}"


def _pair(m):
    return rf"\ensuremath{{({m[0]:.3f},\,{m[1]:.3f})}}"


def write_numbers(res):
    def mac(name, val):
        return rf"\newcommand{{\{name}}}{{{val}}}"
    cl = res["clustering"]; sp = res["spinodal_ic"]; ns = res["nshift"]
    cons = res["conservation"]; lev = res["lever"]; b = res.get("binodal")
    lines = ["% AUTO-GENERATED by gen_figures.py - do not edit.",
             mac("PfourChiab", f"{res['chi'][0]:g}"),
             mac("PfourChias", f"{res['chi'][1]:g}"),
             mac("PfourChibs", f"{res['chi'][2]:g}"),
             mac("PfourNone", f"{res['N'][0]:g}"),
             mac("PfourNtwo", f"{res['N'][1]:g}"),
             mac("PfourNs", f"{res['N'][2]:g}"),
             mac("PfourPhione", f"{res['phi0'][0]:.2f}"),
             mac("PfourPhitwo", f"{res['phi0'][1]:.2f}"),
             mac("PfourSpreadStart", f"{res['spread_start']:.4f}"),
             mac("PfourSpreadEnd", f"{res['spread_end']:.4f}"),
             mac("PfourPhaseA", _pair(cl["phaseA_mean"])),
             mac("PfourPhaseB", _pair(cl["phaseB_mean"])),
             mac("PfourPopA", f"{cl['population'][0]:.2f}"),
             mac("PfourPopB", f"{cl['population'][1]:.2f}"),
             mac("PfourSens", f"{cl['endpoint_sensitivity']:.3f}"),
             mac("PfourPhaseAie", _pair(cl["phaseA_mean_iface_excl"])),
             mac("PfourPhaseBie", _pair(cl["phaseB_mean_iface_excl"])),
             mac("PfourSimplex", _sci(res["admissibility"]["simplex_max_abs"])),
             mac("PfourClipFrac",
                 f"{res['admissibility']['clipped_fraction']:.3g}"),
             mac("PfourPhiOneRange",
                 rf"\ensuremath{{[{res['admissibility']['phi1_min']:.3f},\,"
                 rf"{res['admissibility']['phi1_max']:.3f}]}}"),
             mac("PfourDriftOne", _sci(cons["content1_drift"])),
             mac("PfourDriftTwo", _sci(cons["content2_drift"])),
             mac("PfourLever", _sci(lev["residual"], sig=2)),
             mac("PfourDet", f"{sp['det']:.3f}"),
             mac("PfourEigMin", f"{sp['eig_min']:.3f}"),
             mac("PfourChiCrit", f"{sp['chi12_crit']:.3f}"),
             mac("PfourAsymN", f"{ns['asym_N'][0]:g}"),
             mac("PfourAsymDet", f"{ns['asym_det']:.3f}"),
             mac("PfourAsymChiCrit", f"{ns['asym_chi12_crit']:.3f}"),
             mac("PfourDeltaChiCrit", f"{ns['delta_chi12_crit']:.3f}")]
    if b:
        lines += [mac("PfourBinodalA", _pair(b["phaseA"])),
                  mac("PfourBinodalB", _pair(b["phaseB"]))]
    if "asym_phaseA_mean" in ns:
        lines += [mac("PfourAsymPhaseA", _pair(ns["asym_phaseA_mean"])),
                  mac("PfourAsymPhaseB", _pair(ns["asym_phaseB_mean"]))]
    os.makedirs(os.path.dirname(NUMTEX), exist_ok=True)
    with open(NUMTEX, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", NUMTEX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(HERE, "outputs", "p4"),
                    help="harness run directory (results.json + history.npz)")
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    res, hist = load_run(args.run)
    fig_landscape(res, hist, "p4_landscape.png")
    fig_spinodal(res, hist, "p4_spinodal.png")
    fig_gibbs(res, hist, "p4_gibbs.png")
    fig_fields(res, hist, "p4_fields.png")
    fig_nshift(res, hist, "p4_nshift.png")
    write_numbers(res)


if __name__ == "__main__":
    main()
