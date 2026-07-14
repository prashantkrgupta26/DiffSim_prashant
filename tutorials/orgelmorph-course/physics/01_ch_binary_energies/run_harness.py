"""P1 — the harness-based entry point (the Phase-0 reference port).

Same physics as ``run.py`` (the student-facing driver), but driven through the
course's standard harness so it is a *repeatable workflow*: a YAML config
(``configs/p1.yaml``, the canonical record), a provenance ``metadata.json``, a
``results.json`` checked against ``baseline.yaml``, and the standard output
layout. It reproduces EXPECTED.md's numbers — the regression that guards the
Phase-0 foundation. ``run.py`` is intentionally left untouched (that per-chapter
port is a later phase); this file shows tutorials how to adopt the harness.

    python run_harness.py --config configs/p1.yaml --mode reference \\
        --output outputs/p1 --overwrite

Reuses the diagnostics library for the energy budget so the numbers come from
the shared, unit-tested metrics, not a bespoke re-computation.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                os.pardir, os.pardir)))
from common import config as cfgmod                      # noqa: E402
from common.run_base import build_parser, run_tutorial    # noqa: E402
from diffsim.diagnostics import energy as dgen            # noqa: E402
from diffsim.diagnostics import conservation as dcons     # noqa: E402
from diffsim.diagnostics import stochastic as dstoch      # noqa: E402

from spinodal import (run_spinodal, linear_dispersion,     # noqa: E402
                      interface_width, fit_coarsening_exponent,
                      _lengthscales, linear_probe)

SCHEMA = cfgmod.ConfigSchema(name="p1", fields={
    "energy": cfgmod.Field(str, default="both",
                           choices=("poly", "fh", "both")),
    "level": cfgmod.Field(int, default=6, min=2, max=9),
    "steps": cfgmod.Field(int, default=250, min=1),
    "dt": cfgmod.Field(float, default=0.02, min=0.0),
    "M": cfgmod.Field(float, default=1.0, min=0.0),
    "kappa": cfgmod.Field(float, default=5e-4, min=0.0),
    "fh_A": cfgmod.Field(float, default=1.0, min=0.0),
    "fh_B": cfgmod.Field(float, default=2.5, min=0.0),
    "amp": cfgmod.Field(float, default=0.05, min=0.0),
    "seed": cfgmod.Field(int, default=3),
    "n_seeds": cfgmod.Field(int, default=1, min=1, max=16),
    "measure_every": cfgmod.Field(int, default=10, min=0),
    "precision": cfgmod.Field(str, default="fp64"),
})


def _summarize(rec):
    """Scalar summary from a run_spinodal record, built from the shared
    diagnostics library. Covers the four SEPARATE stories the chapter must not
    conflate (spec P1): the energy budget + the three energy-stability notions,
    the admissibility projection, the linear-stability prediction vs the
    measured S(q), and the coarsening exponent with a real uncertainty."""
    F = rec["F_total"]
    Fint = rec["F_interface"]
    dx = float(rec["dx"])
    energy = rec["energy"]
    p = rec["params"]
    A, B, c_bar = p["fh_A"], p["fh_B"], float(rec["c_avg"])

    # --- energy budget + the three distinct stability notions --------------
    end = rec["snaps"][max(rec["snaps"])]
    out = {
        "F0": float(F[0]), "F1": float(F[1]), "Fend": float(F[-1]),
        "Fbulk_end": float(rec["F_bulk"][-1]),
        "Fint_end": float(Fint[-1]), "Fint_peak": float(np.max(Fint)),
        # c_min/c_max are the CONVERGED end-field bounds (the two coexisting
        # phases); c_traj_* are the trajectory extremes incl. transient
        # overshoot (poly overshoots [-1,1] on the quench; FH is held interior
        # by the box projection).
        "c_min": float(end.min()), "c_max": float(end.max()),
        "c_traj_min": float(rec["cmin"].min()), "c_traj_max": float(rec["cmax"].max()),
        "mass_drift": dcons.mass_drift(rec["mass"]),
        # (i) continuous Lyapunov holds by construction (dF/dt<=0). (ii) DISCRETE
        # energy-stability: are the stepwise increments <=0 (after start-up)?
        # (iii) THIS one monotone run. Report the honest measured numbers.
        "monotone_after_step1": dgen.is_monotone_decreasing(F, skip=1),
        "largest_positive_increment": dgen.largest_positive_increment(F[1:]),
        "startup_increment": float(F[1] - F[0]),
    }

    # echo the run parameters so the document's numbers come wholly from
    # results.json (no hand-copied constants in gen_figures).
    out["fh_A"] = float(A)
    out["fh_B"] = float(B)
    out["kappa"] = float(rec["kappa"])
    out["M"] = float(p["M"])

    # --- admissibility: FH regularization + box-projection -----------------
    out["proj_dofs"] = int(rec["proj_dofs"])
    out["proj_max"] = float(rec["proj_max"])
    out["fh_reg_eps"] = float(rec["fh_reg_eps"])
    # the overshoot the projection/regularization must cope with = trajectory
    # extreme (converged states stay in c_min/c_max above)
    out["phi_min"] = out["c_traj_min"]
    out["phi_max"] = out["c_traj_max"]

    # --- interface width ell ~ sqrt(kappa/W), reported in length AND cells --
    ell = interface_width(rec["kappa"], energy, c_bar, A, B)
    out["interface_width"] = ell
    out["interface_cells"] = ell / dx

    # --- linear stability: dispersion -> fastest k -> predicted wavelength --
    disp = linear_dispersion(c_bar, rec["kappa"], p["M"], energy, A, B)
    out["fpp"] = disp["fpp"]
    out["k_star"] = disp["k_star"]
    out["lambda_star"] = disp["lambda_star"]   # PREDICTED fastest wavelength
    # measured early-time wavelength (S(q) peak) — the spinodal's selected scale
    snaps = sorted(rec["snaps"])
    early = next((s for s in snaps if s > 0), snaps[-1])
    lp_e, lf_e, _la_e = _lengthscales(rec["snaps"][early], dx)
    out["lambda_measured_early"] = lp_e
    out["lambda_measured_early_fm"] = lf_e
    lp_end, lf_end, la_end = _lengthscales(end, dx)
    out["lambda_end_peak"] = lp_end
    out["lambda_end_fm"] = lf_end
    out["L_area_end"] = la_end

    # --- coarsening L(t) ~ t^n with 3 length defs + fitted exponent + CI -----
    # The interfacial-area length (L_area = area/interface) is the robust,
    # monotone coarsening measure and the headline; peak & first-moment are
    # reported alongside (peak is quantization-limited, first-moment tail-biased).
    # FIT WINDOW: the genuine power-law regime is bounded above by FINITE-SIZE
    # saturation -- on a unit box the domains stop growing once L reaches ~box/3
    # (they feel the periodic images), after which the exponent collapses to ~0.
    # We fit the pre-saturation window (skip the onset sample; cap at box/3;
    # require >=3 points, else fall back to the first 40% of samples) and report
    # the window so the reader sees exactly what was fit (spec P1).
    tt = np.asarray(rec["Lt_t"]); La = np.asarray(rec["Lt_area"])
    if tt.size >= 3:
        box = 1.0
        mask = np.zeros(tt.size, bool)
        mask[1:] = (La[1:] < box / 3.0)          # pre-saturation, skip onset
        if mask.sum() < 3:                        # fallback: first 40% of samples
            k = max(3, int(np.ceil(0.4 * tt.size)))
            mask = np.zeros(tt.size, bool); mask[1:k + 1] = True
        t_lo = float(tt[mask].min()); t_hi = float(tt[mask].max())
        n_ar, sd_ar, r2_ar = fit_coarsening_exponent(tt[mask], La[mask])
        n_pk, sd_pk, r2_pk = fit_coarsening_exponent(
            tt[mask], np.asarray(rec["Lt_peak"])[mask])
        n_fm, sd_fm, r2_fm = fit_coarsening_exponent(
            tt[mask], np.asarray(rec["Lt_fm"])[mask])
        out.update(coarsen_n_area=n_ar, coarsen_sd_area=sd_ar,
                   coarsen_r2_area=r2_ar,
                   coarsen_n_peak=n_pk, coarsen_sd_peak=sd_pk,
                   coarsen_r2_peak=r2_pk, coarsen_n_fm=n_fm,
                   coarsen_sd_fm=sd_fm, coarsen_r2_fm=r2_fm,
                   coarsen_t_lo=t_lo, coarsen_t_hi=t_hi,
                   coarsen_npts=int(mask.sum()), coarsen_t_min=t_lo)
    return out


def p1_run(cfg, ctx):
    which = ["poly", "fh"] if cfg["energy"] == "both" else [cfg["energy"]]
    ctx.provenance.update(
        mesh={"level": cfg["level"], "side": 2 ** cfg["level"] + 1,
              "dim": 2, "p": 1},
        time_integrator="BDF1",
        linear_tol=1e-10,
        nonlinear_tol=1e-8)
    results = {}
    for e in which:
        ctx.log(f"P1 {e}: level {cfg['level']} ({2**cfg['level']}^2), "
                f"{cfg['steps']} steps, solver={ctx.solver}")
        rec = run_spinodal(energy=e, level=cfg["level"], steps=cfg["steps"],
                           dt=cfg["dt"], M=cfg["M"], kappa=cfg["kappa"],
                           fh_A=cfg["fh_A"], fh_B=cfg["fh_B"], amp=cfg["amp"],
                           seed=cfg["seed"], device=ctx.device,
                           linsolver=ctx.solver,
                           measure_every=cfg.get("measure_every", 10))
        s = _summarize(rec)
        # linear-stability VERIFICATION: a small-dt probe measures the growth
        # spectrum sigma_meas(k); its peak must land on the predicted k* (the
        # production dt saturates the quench within one step, so the linear
        # window needs its own tiny-dt probe). Level 7 for spectral resolution.
        pr = linear_probe(energy=e, level=max(cfg["level"], 7), dt=2e-4,
                          n_steps=10, M=cfg["M"], kappa=cfg["kappa"],
                          fh_A=cfg["fh_A"], fh_B=cfg["fh_B"], amp=0.02,
                          seed=cfg["seed"], device=ctx.device, linsolver=ctx.solver)
        s["probe_k_meas"] = pr["k_meas"]
        s["probe_lambda_meas"] = pr["lambda_meas"]
        s["probe_lambda_ratio"] = (pr["lambda_meas"] / s["lambda_star"]
                                   if s["lambda_star"] > 0 else float("nan"))
        s["probe_tau"] = pr["tau"]
        results[e] = s
        ctx.history[f"{e}_probe_q"] = pr["q"]
        ctx.history[f"{e}_probe_sigma_meas"] = pr["sigma_meas"]
        ctx.history[f"{e}_probe_sigma_analytic"] = pr["sigma_analytic"]
        ctx.log(f"  F: {s['F0']:.5g} -> {s['Fend']:.5g}; "
                f"interface peak {s['Fint_peak']:.5g} -> {s['Fint_end']:.5g}; "
                f"phi in [{s['phi_min']:.3f}, {s['phi_max']:.3f}]; "
                f"|dm|={s['mass_drift']:.2e}; monotone="
                f"{s['monotone_after_step1']}; proj_dofs={s['proj_dofs']}")
        ctx.log(f"  linear: f''={s['fpp']:.3g}, lambda* predicted "
                f"{s['lambda_star']:.4g} vs probe-measured {s['probe_lambda_meas']:.4g} "
                f"(ratio {s['probe_lambda_ratio']:.3g}); interface {s['interface_cells']:.2f} cells")
        if "coarsen_n_area" in s:
            ctx.log(f"  coarsening n (area) = {s['coarsen_n_area']:.3f} "
                    f"+/- {s['coarsen_sd_area']:.3f} (R2={s['coarsen_r2_area']:.2f}); "
                    f"n (peak) = {s['coarsen_n_peak']:.3f}; "
                    f"n (first-moment) = {s['coarsen_n_fm']:.3f}")
        # store time series + a dispersion curve for figures (gen_figures.py
        # renders EVERY figure from this saved history + results.json, so the
        # document's figures ARE this run's data, never a re-computation).
        for k in ("t", "F_total", "F_bulk", "F_interface", "mass",
                  "Lt_t", "Lt_peak", "Lt_fm", "Lt_area"):
            ctx.history[f"{e}_{k}"] = np.asarray(rec[k])
        disp = linear_dispersion(float(rec["c_avg"]), rec["kappa"], rec["params"]["M"],
                                 e, rec["params"]["fh_A"], rec["params"]["fh_B"])
        ctx.history[f"{e}_disp_k"] = disp["k"]
        ctx.history[f"{e}_disp_sigma"] = disp["sigma"]
        ctx.history[f"{e}_lambda_star"] = np.array([disp["lambda_star"]])
        ctx.history[f"{e}_dx"] = np.array([rec["dx"]])
        # morphology snapshots (fields) for the strip figures
        snap_steps = sorted(rec["snaps"])
        ctx.history[f"{e}_snap_steps"] = np.array(snap_steps)
        ctx.history[f"{e}_snap_t"] = np.array([rec["t"][n] for n in snap_steps])
        for n in snap_steps:
            ctx.history[f"{e}_snap_{n}"] = np.asarray(rec["snaps"][n])

    # --- research ensemble: 5 seeds, mean +/- sd of seed-independent physics -
    n_seeds = int(cfg.get("n_seeds", 1))
    if n_seeds > 1:
        for e in which:
            fends, lends, nexps = [], [], []
            for sd in range(cfg["seed"], cfg["seed"] + n_seeds):
                rec = run_spinodal(energy=e, level=cfg["level"],
                                   steps=cfg["steps"], dt=cfg["dt"], M=cfg["M"],
                                   kappa=cfg["kappa"], fh_A=cfg["fh_A"],
                                   fh_B=cfg["fh_B"], amp=cfg["amp"], seed=sd,
                                   device=ctx.device, linsolver=ctx.solver,
                                   measure_every=cfg.get("measure_every", 10))
                fends.append(float(rec["F_total"][-1]))
                _lp, _lf, le = _lengthscales(rec["snaps"][max(rec["snaps"])], rec["dx"])
                lends.append(le)
                if rec["Lt_t"].size >= 3:
                    nexps.append(fit_coarsening_exponent(
                        rec["Lt_t"], rec["Lt_area"],
                        t_min=0.3 * float(rec["Lt_t"].max()))[0])
            agF = dstoch.ensemble_aggregate(np.array(fends))
            agL = dstoch.ensemble_aggregate(np.array(lends))
            ens = {"n_seeds": n_seeds,
                   "Fend_mean": float(agF["mean"]), "Fend_sd": float(agF["sd"]),
                   "Lend_mean": float(agL["mean"]), "Lend_sd": float(agL["sd"])}
            if nexps:
                agn = dstoch.ensemble_aggregate(np.array(nexps))
                ens["coarsen_n_mean"] = float(agn["mean"])
                ens["coarsen_n_sd"] = float(agn["sd"])
            results[e]["ensemble"] = ens
            ctx.log(f"  ensemble[{e}] {n_seeds} seeds: Fend {ens['Fend_mean']:.5g}"
                    f" +/- {ens['Fend_sd']:.2g}; Lend {ens['Lend_mean']:.4g}"
                    f" +/- {ens['Lend_sd']:.2g}")
    return results


def main():
    args = build_parser("OrgElMorph P1 (harness port)").parse_args()
    here = os.path.dirname(__file__)
    # baseline.yaml encodes the reference-mode EXPECTED numbers; the
    # quick/research tiers use different meshes/horizons, so only gate on it
    # in reference mode (the regression that guards the Phase-0 port).
    baseline = (os.path.join(here, "baseline.yaml")
                if args.mode == "reference" else None)
    # This chapter's (c, mu) system is small and INDEFINITE: scipy SuperLU's
    # partial pivoting is exact here, whereas cuDSS (no pivoting) diverges on
    # this tiny saddle-point block. splu is therefore P1's documented `auto`
    # solver and the one the EXPECTED numbers were generated with. cuDSS is the
    # measured default at scale (C4 / research runs), reachable via
    # --solver cudss.
    run_tutorial(p1_run, schema=SCHEMA, args=args,
                 default_output=os.path.join(here, "outputs", "p1"),
                 baseline=baseline, default_solver="splu")


if __name__ == "__main__":
    main()
