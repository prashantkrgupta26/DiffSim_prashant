"""OrgElMorph course - Physics P1: render figures + numbers from SAVED DATA.

Every figure and every cited number is produced from one real harness run's
``history.npz`` + ``results.json`` (spec: figures generated from saved data,
not a re-computation and not hand-copied). Run the harness first, then::

    PYTHONPATH=<repo>/src python run_harness.py --config configs/p1.yaml \\
        --mode reference --output outputs/p1 --overwrite --solver splu
    python gen_figures.py --run-dir outputs/p1

It writes into ../../latex/figures/ (the swappable-image convention) and
emits ../../latex/numbers/p1.tex.
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
NUMTEX = os.path.join(HERE, "..", "..", "latex", "numbers", "p1.tex")


# ---------------------------------------------------------------------------
def load(run_dir):
    with open(os.path.join(run_dir, "results.json")) as fh:
        results = json.load(fh)
    hist = dict(np.load(os.path.join(run_dir, "history.npz")))
    return results, hist


# ---------------------------------------------------------------------------
def morphology_strip(hist, e, fname, cmap):
    steps = hist[f"{e}_snap_steps"].astype(int)
    ts = hist[f"{e}_snap_t"]
    fields = [hist[f"{e}_snap_{n}"] for n in steps]
    vlo = min(f.min() for f in fields[1:]) if len(fields) > 1 else fields[-1].min()
    vhi = max(f.max() for f in fields[1:]) if len(fields) > 1 else fields[-1].max()
    fig, axes = plt.subplots(1, len(steps), figsize=(3.1 * len(steps), 3.2),
                             dpi=150, squeeze=False)
    for ax, n, tt, f in zip(axes[0], steps, ts, fields):
        im = ax.imshow(f, origin="lower", cmap=cmap, vmin=vlo, vmax=vhi)
        ax.set_title(f"step {n}  (t = {tt:.2f})", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="c")
    fig.suptitle(f"Spinodal decomposition - {e.upper()} free energy",
                 fontsize=12, y=1.02)
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def energy_curves(hist, energies, fname):
    fig, axes = plt.subplots(1, len(energies), figsize=(6.2 * len(energies), 4.0),
                             dpi=150, squeeze=False)
    for ax, e in zip(axes[0], energies):
        t = hist[f"{e}_t"]
        ax.plot(t, hist[f"{e}_F_total"], "k-", lw=2, label="total $F$")
        ax.plot(t, hist[f"{e}_F_bulk"], "C0--", lw=1.5, label="bulk")
        ax.plot(t, hist[f"{e}_F_interface"], "C3-.", lw=1.5, label="interfacial")
        ax.set_xlabel("time $t$"); ax.set_ylabel("free energy")
        ax.set_title(f"{e.upper()} free energy")
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("Ginzburg-Landau energy decay (Lyapunov functional)",
                 fontsize=12, y=1.02)
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def dispersion_fig(results, hist, energies, fname):
    """Linear-stability dispersion: the analytic sigma(k) (line) against the
    MEASURED growth spectrum from a small-dt probe (points). The measured peak
    lands on the predicted fastest mode k* -> the selected wavelength lambda*."""
    fig, axes = plt.subplots(1, len(energies), figsize=(6.0 * len(energies), 4.0),
                             dpi=150, squeeze=False)
    for ax, e in zip(axes[0], energies):
        k = hist[f"{e}_disp_k"]; sigma = hist[f"{e}_disp_sigma"]
        s = results[e]
        ax.plot(k, sigma, "C0-", lw=2, label="analytic $\\sigma(k)$")
        if f"{e}_probe_q" in hist:
            q = hist[f"{e}_probe_q"]; sm = hist[f"{e}_probe_sigma_meas"]
            m = np.isfinite(sm)
            ax.plot(q[m], sm[m], "C2o", ms=4, label="measured (probe)")
        ax.axhline(0, color="0.6", lw=0.8)
        kstar = s["k_star"]
        ax.axvline(kstar, color="C3", ls="--", lw=1.5,
                   label=fr"$k^*={kstar:.1f}$, $\lambda^*={s['lambda_star']:.3f}$")
        km = s.get("probe_k_meas")
        if km and km == km:
            ax.axvline(km, color="C2", ls=":", lw=1.2,
                       label=fr"measured $k={km:.1f}$")
        ax.set_xlim(0, 1.4 * s["k_star"])
        ax.set_ylim(min(0, np.nanmin(sigma)) * 1.1, np.nanmax(sigma) * 1.3 + 1e-9)
        ax.set_xlabel(r"wavenumber $k$"); ax.set_ylabel(r"growth rate $\sigma(k)$")
        ax.set_title(f"{e.upper()}: $f''={s['fpp']:.2g}$")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Cahn-Hilliard linear stability: analytic vs measured growth spectrum",
                 fontsize=12, y=1.02)
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def coarsening_fig(results, hist, energies, fname):
    """L(t) on log-log with the fitted exponent, three length definitions.
    The interfacial-area length (headline) is the robust monotone measure."""
    fig, axes = plt.subplots(1, len(energies), figsize=(6.0 * len(energies), 4.0),
                             dpi=150, squeeze=False)
    for ax, e in zip(axes[0], energies):
        t = hist[f"{e}_Lt_t"]
        La = hist[f"{e}_Lt_area"]; Lp = hist[f"{e}_Lt_peak"]; Lf = hist[f"{e}_Lt_fm"]
        s = results[e]
        ax.loglog(t, La, "C3^-", ms=5, label="area $A/P$ (headline)")
        ax.loglog(t, Lp, "C0o-", ms=4, alpha=0.6, label="peak $S(q)$")
        ax.loglog(t, Lf, "C1s-", ms=4, alpha=0.6, label="first moment")
        n = s.get("coarsen_n_area")
        if n is not None and n == n:
            tmin = s.get("coarsen_t_min", t.min())
            tt = np.array([tmin, t.max()])
            L_at = La[-1] * (tt / t[-1]) ** n
            ax.loglog(tt, L_at, "k--", lw=1.4,
                      label=fr"$n={n:.2f}\pm{s['coarsen_sd_area']:.2f}$")
        ax.set_xlabel("time $t$"); ax.set_ylabel("domain length $L(t)$")
        ax.set_title(f"{e.upper()} coarsening")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.suptitle("Coarsening $L(t)\\sim t^{n}$ (three length definitions, fitted exponent)",
                 fontsize=12, y=1.02)
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


# ---------------------------------------------------------------------------
def write_numbers(results):
    """Emit latex/numbers/p1.tex — the document's numbers ARE this run's
    results.json (no hand-copied values). Number macros use \\ensuremath so
    they work in text and math mode alike."""
    p, f = results["poly"], results["fh"]

    def mac(name, val):
        return rf"\newcommand{{\{name}}}{{{val}}}"

    def crange(d):
        return (f"\\ensuremath{{[{d['c_min']:.2f},\\,"
                f"{d['c_max']:.2f}]}}")

    def mass(d):
        m = d["mass_drift"]
        if m < 1e-14:
            return r"\ensuremath{<10^{-14}} (machine)"
        mant, exp = f"{m:.1e}".split("e")
        return f"\\ensuremath{{{mant}\\times10^{{{int(exp)}}}}}"

    def sci(x, sig=2):
        mant, exp = f"{x:.{sig-1}e}".split("e")
        return f"\\ensuremath{{{mant}\\times10^{{{int(exp)}}}}}"

    lines = ["% AUTO-GENERATED by gen_figures.py from results.json - do not edit.",
             mac("FHA", f"{f['fh_A']:g}"),
             mac("FHB", f"{f['fh_B']:g}")]
    for pre, d in (("POLY", p), ("FH", f)):
        lines += [
            mac(f"{pre}Fstart", f"{d['F0']:.3g}"),
            mac(f"{pre}Fend", f"{d['Fend']:.3g}"),
            mac(f"{pre}Fintpeak", f"{d['Fint_peak']:.3g}"),
            mac(f"{pre}Fintend", f"{d['Fint_end']:.3g}"),
            mac(f"{pre}crange", crange(d)),
            mac(f"{pre}mass", mass(d)),
            # discrete energy-stability (largest positive stepwise increment)
            mac(f"{pre}largestinc", sci(max(d['largest_positive_increment'], 1e-300))
                if d['largest_positive_increment'] > 0 else r"\ensuremath{0}"),
            mac(f"{pre}startup", f"{d['startup_increment']:.3g}"),
            # interface width ell ~ sqrt(kappa/W): length and cells
            mac(f"{pre}ell", f"{d['interface_width']:.4f}"),
            mac(f"{pre}ellcells", f"{d['interface_cells']:.1f}"),
            # linear stability (probe-measured growth spectrum vs prediction)
            mac(f"{pre}fpp", f"{d['fpp']:.3g}"),
            mac(f"{pre}kstar", f"{d['k_star']:.1f}"),
            mac(f"{pre}lamstar", f"{d['lambda_star']:.3f}"),
            mac(f"{pre}lammeas", f"{d['probe_lambda_meas']:.3f}"),
            mac(f"{pre}lamratio", f"{d['probe_lambda_ratio']:.2f}"),
            # coarsening exponent + uncertainty. Interfacial-area length is the
            # robust headline; peak/first-moment reported alongside.
            mac(f"{pre}coarsenN", f"{d.get('coarsen_n_area', float('nan')):.2f}"),
            mac(f"{pre}coarsenSD", f"{d.get('coarsen_sd_area', float('nan')):.2f}"),
            mac(f"{pre}coarsenR", f"{d.get('coarsen_r2_area', float('nan')):.2f}"),
            mac(f"{pre}coarsenNfm", f"{d.get('coarsen_n_fm', float('nan')):.2f}"),
            mac(f"{pre}coarsenNpk", f"{d.get('coarsen_n_peak', float('nan')):.2f}"),
        ]
    # FH admissibility projection (the honest regularization report)
    lines += [
        mac("FHprojdofs", f"{f['proj_dofs']}"),
        mac("FHprojmax", f"{f['proj_max']:.2e}" if f['proj_max'] > 0 else "0"),
        mac("FHregeps", sci(f['fh_reg_eps']) if f['fh_reg_eps'] > 0 else "0"),
        mac("POLYprojdofs", f"{p['proj_dofs']}"),
    ]
    os.makedirs(os.path.dirname(NUMTEX), exist_ok=True)
    with open(NUMTEX, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", NUMTEX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.join(HERE, "outputs", "p1"),
                    help="harness output dir (results.json + history.npz)")
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    results, hist = load(args.run_dir)
    energies = [e for e in ("poly", "fh") if f"{e}_t" in hist]
    for e, cmap in (("poly", "RdBu_r"), ("fh", "viridis")):
        if e in energies:
            morphology_strip(hist, e, f"p1_morph_{e}.png", cmap)
    energy_curves(hist, energies, "p1_energy.png")
    dispersion_fig(results, hist, energies, "p1_dispersion.png")
    coarsening_fig(results, hist, energies, "p1_coarsening.png")
    write_numbers(results)


if __name__ == "__main__":
    main()
