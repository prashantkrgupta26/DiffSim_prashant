"""OrgElMorph P11 - render ALL figures + number macros from a SAVED harness run
directory (results.json + history.npz).  Nothing is re-simulated (spec Phase 1),
so the document's figures and numbers ARE the checked verification run.

    PYTHONPATH=<repo>/src python gen_figures.py --run-dir outputs/p11

Figures -> ../../latex/figures/p11_*.png:
  p11_mechanism.png    the added mechanism: mobility M(phi)=M0 a(phi) + arrest a(phi)
  p11_dispersion.png   limiting-case CH dispersion sigma(k): discrete vs analytic
  p11_convergence.png  MMS spatial (2nd order) + temporal (1st order) convergence
  p11_coarsening.png   the scientific result: L(t) and final fields, free vs arrest
Numbers -> ../../latex/numbers/p11.tex.
"""
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, "..", "..", "latex", "figures")
NUMTEX = os.path.join(HERE, "..", "..", "latex", "numbers", "p11.tex")


def load_run(run_dir):
    with open(os.path.join(run_dir, "results.json")) as fh:
        res = json.load(fh)
    hist = dict(np.load(os.path.join(run_dir, "history.npz")))
    return res, hist


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------------------------
def fig_mechanism(res, fname):
    """The added term: composition-dependent mobility M(phi)=M0 a(phi) with the
    vitrification arrest a(phi)=sigmoid((phi_g-phi)/w)."""
    c = res["config"]
    phi_g, w, M0 = c["phi_g"], c["w"], c["M0"]
    phi = np.linspace(0.0, 1.0, 400)
    a = _sigmoid((phi_g - phi) / w)
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=150)
    ax.plot(phi, M0 * a, "C0", lw=2.5, label=r"mobility $M(\phi)=M_0\,a(\phi)$")
    ax.plot(phi, a, "C3--", lw=1.8, label=r"arrest factor $a(\phi)$")
    ax.axvline(phi_g, color="0.5", ls=":", lw=1.5)
    ax.annotate(fr"$\phi_g={phi_g:g}$", (phi_g, 0.5), xytext=(6, 0),
                textcoords="offset points", color="0.3")
    ax.axvspan(phi_g, 1.0, color="0.85", alpha=0.5)
    ax.text(0.5 * (phi_g + 1.0), 0.08, "vitrified\n(arrested)", ha="center",
            fontsize=9, color="0.3")
    ax.text(0.5 * phi_g, 0.9, "mobile", ha="center", fontsize=9, color="0.3")
    ax.set_xlabel(r"composition $\phi$")
    ax.set_ylabel("mobility / arrest")
    ax.set_title(r"Vitrification mechanism: mobility collapses above $\phi_g$"
                 f"  ($w={w:g}$)")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    _save(fig, fname)


def fig_dispersion(res, hist, fname):
    """Limiting case (arrest off -> constant mobility): the measured single-mode
    growth rate vs the analytic CH dispersion sigma(k)."""
    k = hist["disp_k"]; sa = hist["disp_sigma_analytic"]; sm = hist["disp_sigma_measured"]
    ck = res["checks"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    ax.plot(k, sa, "C0-", lw=2, label="analytic $\\sigma(k)$")
    ax.plot(k, sm, "C3o", ms=7, label="measured (one BE step)")
    ax.set_xlabel(r"wavenumber $k$")
    ax.set_ylabel(r"growth rate $\sigma(k)$")
    ax.set_title("Limiting-case CH dispersion (arrest off): "
                 f"fundamental rel.\\ err {ck['dispersion_rel_err']:.1e}")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    _save(fig, fname)


def fig_convergence(res, hist, fname):
    """MMS spatial convergence (2nd order) and temporal convergence (1st order),
    each with the theoretical reference slope."""
    ck = res["checks"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.0, 4.6), dpi=150)
    # -- MMS spatial (h^2) --
    hh = np.asarray(hist["mms_h"], float); ee = np.asarray(hist["mms_err"], float)
    ax0.loglog(hh, ee, "C0o-", lw=2, ms=7, label="MMS $L_2$ error")
    ref = ee[0] * (hh / hh[0]) ** 2
    ax0.loglog(hh, ref, "k--", lw=1.2, label=r"slope 2 ($h^2$)")
    ax0.set_xlabel("grid spacing $h$"); ax0.set_ylabel(r"$L_2$ error vs $\phi^*$")
    ax0.set_title(f"MMS spatial convergence (mean order "
                  f"{ck['mms_order_mean']:.2f})")
    ax0.legend(fontsize=9); ax0.grid(alpha=0.3, which="both")
    # -- temporal (dt^1) --
    dt = np.asarray(hist["temporal_dt"], float); te = np.asarray(hist["temporal_err"], float)
    ax1.loglog(dt, te, "C3s-", lw=2, ms=7, label="time error vs fine-$\\Delta t$")
    ref1 = te[0] * (dt / dt[0])
    ax1.loglog(dt, ref1, "k--", lw=1.2, label=r"slope 1 ($\Delta t$)")
    ax1.set_xlabel(r"time step $\Delta t$"); ax1.set_ylabel(r"$L_2$ error")
    ax1.set_title(f"Backward-Euler temporal order ({ck['temporal_order']:.2f})")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3, which="both")
    fig.suptitle("Verification: hand-derived Jacobian FD-error "
                 f"{ck['jac_fd_max_err']:.1e} (autograd match "
                 f"{ck['jac_autograd_max_err']:.0e})", y=1.02, fontsize=12)
    _save(fig, fname)


def fig_coarsening(res, hist, fname):
    """The scientific result: domain length L(t) grows without arrest but plateaus
    with the polymer-rich matrix vitrified; final fields show frozen morphology."""
    ck = res["checks"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.0, 4.6), dpi=150)
    tf = hist["coarsen_t_free"]; Lf = hist["coarsen_Lt_free"]
    ta = hist["coarsen_t_arrest"]; La = hist["coarsen_Lt_arrest"]
    ax0.plot(tf, Lf, "C0-o", ms=3, lw=2, label="no arrest (coarsens)")
    ax0.plot(ta, La, "C3-s", ms=3, lw=2, label="vitrified (arrested)")
    ax0.set_xlabel("time $t$"); ax0.set_ylabel(r"domain scale $L(t)$")
    ax0.set_title(f"Coarsening arrest: $L$ ratio {ck['arrest_ratio']:.2f}, "
                  f"plateau growth {100*ck['arrest_plateau_growth']:.1f}%")
    ax0.legend(fontsize=10); ax0.grid(alpha=0.3)
    x = hist["coarsen_x"]
    ax1.plot(x, hist["coarsen_phi0"], "0.6", lw=1, label="initial")
    ax1.plot(x, hist["coarsen_phi_final_free"], "C0", lw=2, label="final (free)")
    ax1.plot(x, hist["coarsen_phi_final_arrest"], "C3", lw=2,
             label="final (arrested)")
    ax1.axhline(res["config"]["phi_g"], color="0.5", ls=":", lw=1)
    ax1.set_xlabel("position $x$"); ax1.set_ylabel(r"composition $\phi$")
    ax1.set_title("Final morphology (arrested stays fine)")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
    fig.suptitle("Scientific result: vitrification freezes the drying-film "
                 "morphology (the P10 clock stops)", y=1.02, fontsize=12)
    _save(fig, fname)


def fig_profile(prof, fname):
    """GPU profile: per-Newton-step wall time vs grid size N (GPU vs CPU) and
    peak device memory. Rendered from the saved profile.json measurement."""
    N = np.asarray(prof["N"], float)
    gpu = np.asarray(prof["gpu_ms"], float)
    cpu = np.asarray(prof["cpu_ms"], float)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.0, 4.4), dpi=150)
    ax0.loglog(N, gpu, "C0o-", lw=2, ms=6, label=f"{prof['device']}")
    ax0.loglog(N, cpu, "C3s--", lw=2, ms=6, label="cpu")
    ax0.set_xlabel("grid size $N$")
    ax0.set_ylabel("per-Newton-step wall time (ms)")
    ax0.set_title("Newton step cost (residual + analytic Jacobian + dense solve)")
    ax0.legend(fontsize=9); ax0.grid(alpha=0.3, which="both")
    ax1.loglog(N, np.asarray(prof["gpu_peak_mb"], float), "C2^-", lw=2, ms=6)
    ax1.set_xlabel("grid size $N$"); ax1.set_ylabel("peak device memory (MB)")
    ax1.set_title(f"Peak GPU memory (dense $N\\times N$ Jacobian; "
                  f"{prof['speedup_maxN']:.1f}x vs CPU at $N$={int(N[-1])})")
    ax1.grid(alpha=0.3, which="both")
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


def write_numbers(res, prof=None):
    ck = res["checks"]; c = res["config"]
    lines = [
        "% AUTO-GENERATED by gen_figures.py - do not edit.",
        _mac("PelevenPhiG", f"{c['phi_g']:g}"),
        _mac("PelevenW", f"{c['w']:g}"),
        _mac("PelevenKappa", _sci(c["kappa"])),
        _mac("PelevenM", f"{c['M0']:g}"),
        _mac("PelevenN", f"{c['N']}"),
        _mac("PelevenNsteps", f"{c['nsteps']}"),
        _mac("PelevenJacFDerr", _sci(ck["jac_fd_max_err"])),
        _mac("PelevenJacAGerr", _sci(ck["jac_autograd_max_err"])),
        _mac("PelevenFprimeErr", _sci(ck["fprime_cd_err"])),
        _mac("PelevenFppErr", _sci(ck["fpp_cd_err"])),
        _mac("PelevenDispRelErr", _sci(ck["dispersion_rel_err"])),
        _mac("PelevenSigAna", f"{ck['dispersion_sigma_analytic']:.2f}"),
        _mac("PelevenSigMeas", f"{ck['dispersion_sigma_measured']:.2f}"),
        _mac("PelevenMMSorder", f"{ck['mms_order_mean']:.3f}"),
        _mac("PelevenMMSorderMin", f"{ck['mms_order_min']:.3f}"),
        _mac("PelevenMMSerrCoarse", _sci(ck["mms_err_coarse"])),
        _mac("PelevenMMSerrFine", _sci(ck["mms_err_fine"])),
        _mac("PelevenTempOrder", f"{ck['temporal_order']:.3f}"),
        _mac("PelevenLfree", f"{ck['L_final_free']:.3f}"),
        _mac("PelevenLarrest", f"{ck['L_final_arrest']:.3f}"),
        _mac("PelevenArrestRatio", f"{ck['arrest_ratio']:.2f}"),
        _mac("PelevenPlateauGrowth",
             f"{100 * ck['arrest_plateau_growth']:.1f}"),
        _mac("PelevenMassDrift", _sci(ck["coarsen_mass_drift_max"])),
        _mac("PelevenNewtonIters", f"{ck['coarsen_max_newton_iters']}"),
    ]
    if prof is not None:
        Nmax = int(prof["N"][-1])
        lines += [
            _mac("PelevenProfDevice", str(prof["device"]).replace("_", r"\_")),
            _mac("PelevenProfMaxN", f"{Nmax}"),
            _mac("PelevenProfMs", f"{prof['gpu_ms'][-1]:.2f}"),
            _mac("PelevenProfSpeedup", f"{prof['speedup_maxN']:.1f}"),
            _mac("PelevenProfPeakMB", f"{prof['gpu_peak_mb'][-1]:.1f}"),
        ]
    os.makedirs(os.path.dirname(NUMTEX), exist_ok=True)
    with open(NUMTEX, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", NUMTEX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.join(HERE, "outputs", "p11"))
    args = ap.parse_args()
    res, hist = load_run(args.run_dir)
    os.makedirs(FIGDIR, exist_ok=True)
    fig_mechanism(res, "p11_mechanism.png")
    fig_dispersion(res, hist, "p11_dispersion.png")
    fig_convergence(res, hist, "p11_convergence.png")
    fig_coarsening(res, hist, "p11_coarsening.png")
    prof_path = os.path.join(args.run_dir, "profile.json")
    prof = None
    if os.path.exists(prof_path):
        with open(prof_path) as fh:
            prof = json.load(fh)
        fig_profile(prof, "p11_profile.png")
    write_numbers(res, prof)


if __name__ == "__main__":
    main()
