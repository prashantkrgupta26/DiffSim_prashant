"""P2 -- adaptive time stepping: the harness-based scientific workflow.

Same physics as P1 (conserved Model-B gradient flow of the Ginzburg--Landau
energy), but the QUESTION is how to choose the time step. This harness turns the
chapter into a repeatable *verify -> profile -> compare* workflow rather than a
demonstration:

1. TEMPORAL ORDER VERIFICATION on a smooth, deterministic single-mode problem
   (a low-k cosine that decays smoothly): refine dt and measure the observed
   order of BDF1 (~1) and BDF2 (~2) with ``diagnostics.convergence``. This
   proves the integrator is correct *before* it is used on the chaotic quench.

2. REFERENCE-BASED ACCURACY on the spinodal quench. We build a tight small-dt
   reference and compare every method by the RELATIVE L2 ERROR of c(T) plus the
   errors in free energy, structure-factor length scale, domain scale and phase
   fraction -- we do NOT equate "lower energy" with "more accurate".

3. REAL COST. Accepted AND rejected steps, full + half solves (step-doubling is
   1 full + 2 half per attempt), total Newton iterations / linear solves, wall
   time and peak device memory -- not the accepted-step count alone.

4. HONEST MATCHED-ACCURACY SPEED-UP. Instead of the misleading "a fixed dt must
   use the quench's smallest step for the whole horizon", we run a fixed-dt
   sweep, find the largest fixed dt whose c(T) error matches the adaptive run,
   and report the true cost ratio. (The naive worst-case bound is still recorded
   -- clearly labelled as a worst case.)

5. CONVERGENCE / KNEE. >=4 tolerances: error vs tol, wall vs error, solves vs
   tol; the conclusion is the accuracy-cost knee.

    python run_harness.py --config configs/p2.yaml --mode reference \\
        --output outputs/p2 --overwrite

Figures + the document's number macros are rendered by ``gen_figures.py`` purely
from the saved run directory (results.json + history.npz).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse.linalg as _spla

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                os.pardir, os.pardir)))
from common import config as cfgmod                        # noqa: E402
from common.run_base import build_parser, run_tutorial      # noqa: E402
from diffsim.physics.cahn_hilliard import (CahnHilliardStepper,   # noqa: E402
                                           adaptive_march)
from diffsim.diagnostics import morphology as dmorph        # noqa: E402
from diffsim.diagnostics import convergence as dconv        # noqa: E402
from diffsim.diagnostics import profiling as dprof          # noqa: E402

from adaptive import (build_mesh_dm, free_energy,            # noqa: E402
                      _domain_scale)


# --------------------------------------------------------------------------
# Cost instrumentation (no src edits): count linear solves == Newton iters by
# wrapping scipy's splu (the CH stepper's documented small-problem solver does
# one splu factor+solve per Newton iteration), and wrap stepper.step() to record
# per-call (t_before, dt, newton_iters) so we can reconstruct the accept/reject
# ladder of adaptive_march and the dt(t)/Newton(t) time series.
# --------------------------------------------------------------------------
_ORIG_SPLU = _spla.splu


class _LinCounter:
    n = 0


_LIN = _LinCounter()


def _counting_splu(*a, **k):
    _LIN.n += 1
    return _ORIG_SPLU(*a, **k)


_spla.splu = _counting_splu   # step() does `from scipy.sparse.linalg import splu`


def _instrument(st):
    """Wrap st.step to record (t_before, dt, newton_iters) for every call."""
    st._calls = []
    _orig = st.step

    def wrapped():
        n0 = _LIN.n
        tb, dtb = st.t, st.dt
        out = _orig()
        st._calls.append((tb, dtb, _LIN.n - n0))
        return out

    st.step = wrapped
    return st


def _reconstruct(calls, n_accepted):
    """Turn the per-call log into per-ATTEMPT records. adaptive_march does
    exactly 1 full + 2 half step() calls per attempt (accept or reject); an
    attempt is ACCEPTED iff the next attempt starts at a later t (t advanced) --
    a reject rewinds t and retries. Returns a dict of arrays."""
    m = len(calls) // 3
    t0 = np.array([calls[3 * i][0] for i in range(m)])
    dtf = np.array([calls[3 * i][1] for i in range(m)])
    newt = np.array([calls[3 * i][2] + calls[3 * i + 1][2]
                     + calls[3 * i + 2][2] for i in range(m)])
    accepted = np.zeros(m, dtype=bool)
    for i in range(m):
        if i == m - 1:
            accepted[i] = True           # march exits only after an accept
        else:
            accepted[i] = t0[i + 1] > t0[i] + 1e-14
    # sanity: reconstructed accepts should match adaptive_march's count
    assert int(accepted.sum()) == n_accepted, (int(accepted.sum()), n_accepted)
    return dict(t0=t0, dt=dtf, newton=newt, accepted=accepted,
                acc_t_end=t0[accepted] + dtf[accepted],
                acc_dt=dtf[accepted], acc_newton=newt[accepted])


def _field(st, side):
    return np.asarray(st.cons.T @ st.x[0::2]).reshape(side, side)


# --------------------------------------------------------------------------
# accuracy metrics vs the reference field
# --------------------------------------------------------------------------
def _ref_stats(fld, F):
    return dict(F=float(F), lam1=float(dmorph.first_moment_wavelength(fld)),
                dscale=float(_domain_scale(fld)),
                phase_hi=float(dmorph.phase_fractions(fld)[1]),
                norm=float(np.linalg.norm(fld)))


def _metrics(fld, F, ref):
    """Errors of one solution vs the reference (all robust, seed-of-step
    independent statistics plus the pixelwise relative L2)."""
    lam1 = float(dmorph.first_moment_wavelength(fld))
    dsc = float(_domain_scale(fld))
    phi = float(dmorph.phase_fractions(fld)[1])
    return dict(
        F=float(F),
        relL2=float(np.linalg.norm(fld - ref["_fld"]) / ref["norm"]),
        err_energy=float(abs(F - ref["F"])),
        lam1=lam1, err_lam1=float(abs(lam1 - ref["lam1"]) / ref["lam1"]),
        dscale=dsc, err_dscale=float(abs(dsc - ref["dscale"]) / ref["dscale"]),
        phase_hi=phi, err_phase=float(abs(phi - ref["phase_hi"])),
        c_min=float(fld.min()), c_max=float(fld.max()))


# --------------------------------------------------------------------------
# 1. temporal-order verification (smooth deterministic single mode)
# --------------------------------------------------------------------------
def order_study(cfg, ctx):
    """A low-k cosine c=amp*cos(2 pi x) is a Neumann eigenmode (k=2 pi); with
    kappa*k^2 > 1 it RELAXES smoothly (exponential single-mode decay), so the
    exact-in-space solution is smooth in time and the observed temporal order is
    clean. We refine dt against a fine-dt reference on the SAME mesh (spatial
    error cancels) and read the order with diagnostics.convergence."""
    dm, mesh, cons = build_mesh_dm(cfg["order_level"], device=ctx.device)
    side = int(round(np.sqrt(len(mesh.node_coords))))
    kap, M, amp, T = (cfg["order_kappa"], cfg["M"], cfg["order_amp"],
                      cfg["order_T"])
    k2 = (2.0 * np.pi) ** 2
    ic = lambda x: amp * np.cos(2.0 * np.pi * x[:, 0])

    def march(order, dt):
        st = CahnHilliardStepper(dm, M, kap, dt, order=order, energy="poly")
        st.set_initial(ic, mu_init="consistent")
        for _ in range(int(round(T / dt))):
            st.dt = dt
            st.step()
        return st.x[0::2].copy()

    ref = march(2, cfg["order_ref_dt"])
    rnorm = float(np.linalg.norm(ref))
    dts = list(cfg["order_dts"])
    out = {"T": T, "kappa": kap, "k2": float(k2),
           "decay_rate": float(-M * k2 * (kap * k2 - 1.0)),
           "ref_dt": cfg["order_ref_dt"], "dts": dts, "side": side}
    for order, tag in ((1, "bdf1"), (2, "bdf2")):
        errs = []
        for dt in dts:
            c = march(order, dt)
            errs.append(float(np.linalg.norm(c - ref) / rnorm))
        p = float(dconv.observed_order(dts, errs))
        pair = [float(v) for v in dconv.pairwise_orders(dts, errs)]
        out[tag] = {"errors": errs, "observed_order": p, "pairwise": pair}
        ctx.log(f"  order {tag.upper()}: observed order {p:.3f} "
                f"(pairwise {[round(v, 2) for v in pair]})")
    return out


# --------------------------------------------------------------------------
# 2/3. quench runs (reference, fixed, adaptive) with real cost
# --------------------------------------------------------------------------
def _new_quench(dm, dt, order, cfg, ctx):
    st = CahnHilliardStepper(dm, cfg["M"], cfg["kappa"], dt, order=order,
                             energy="poly")
    rng = np.random.default_rng(cfg["seed"])
    st.set_initial(lambda x: cfg["c_avg"] + cfg["amp"]
                   * rng.standard_normal(len(x)), mu_init="consistent")
    return st


def fixed_run(dm, mesh, side, dt, cfg, ctx, record_energy=False):
    st = _new_quench(dm, dt, 2, cfg, ctx)
    nsteps = int(round(cfg["t_end"] / dt))
    _LIN.n = 0
    dprof.reset_peak_memory()
    ts, Fs = [0.0], [free_energy(st, dm, mesh, st.hist[0])]
    with dprof.timer(sync=True) as tm:
        for i in range(nsteps):
            st.dt = dt
            st.step()
            if record_energy and (i % max(1, nsteps // 60) == 0
                                  or i == nsteps - 1):
                ts.append(st.t)
                Fs.append(free_energy(st, dm, mesh, st.x[0::2]))
    fld = _field(st, side)
    F = free_energy(st, dm, mesh, st.x[0::2])
    return dict(dt=dt, nsteps=nsteps, newton=int(_LIN.n),
                wall_s=float(tm["seconds"]),
                peak_mem_mb=_mem_mb(), field=fld, F=float(F),
                energy_t=np.array(ts), energy_F=np.array(Fs))


def adaptive_run(dm, mesh, side, tol, cfg, ctx, headline=False):
    st = _new_quench(dm, cfg["dt0"], 2, cfg, ctx)
    _instrument(st)
    _LIN.n = 0
    dprof.reset_peak_memory()
    ts, Fs, dts = [0.0], [free_energy(st, dm, mesh, st.hist[0])], []
    with dprof.timer(sync=True) as tm:
        if headline:
            checks = np.linspace(cfg["t_end"] / cfg["nchecks"], cfg["t_end"],
                                 cfg["nchecks"])
            for tc in checks:
                _, d = adaptive_march(st, tc, tol=tol, dt_max=cfg["dt_max"],
                                      dt_min=cfg["dt_min"])
                dts.extend(d)
                ts.append(st.t)
                Fs.append(free_energy(st, dm, mesh, st.x[0::2]))
        else:
            _, dts = adaptive_march(st, cfg["t_end"], tol=tol,
                                    dt_max=cfg["dt_max"], dt_min=cfg["dt_min"])
    wall = float(tm["seconds"])
    fld = _field(st, side)
    F = free_energy(st, dm, mesh, st.x[0::2])
    rec = _reconstruct(st._calls, len(dts))
    attempts = int(len(rec["accepted"]))
    accepted = int(rec["accepted"].sum())
    rejected = attempts - accepted
    out = dict(tol=tol, accepted=accepted, rejected=rejected,
               attempts=attempts, full_solves=attempts,
               half_solves=2 * attempts, newton=int(_LIN.n),
               linear_solves=int(_LIN.n), wall_s=wall, peak_mem_mb=_mem_mb(),
               dt_min=float(min(dts)), dt_max=float(max(dts)),
               field=fld, F=float(F))
    if headline:
        out["series"] = dict(
            acc_t=rec["acc_t_end"].tolist(), acc_dt=rec["acc_dt"].tolist(),
            acc_newton=rec["acc_newton"].tolist(),
            energy_t=np.array(ts).tolist(), energy_F=np.array(Fs).tolist())
    return out


def _mem_mb():
    b = dprof.peak_device_memory_bytes()
    return None if b is None else round(b / 1024 / 1024, 1)


# --------------------------------------------------------------------------
# main run_fn
# --------------------------------------------------------------------------
def p2_run(cfg, ctx):
    ctx.provenance.update(
        mesh={"level": cfg["level"], "side": 2 ** cfg["level"] + 1,
              "dim": 2, "p": 1},
        time_integrator="BDF2 (variable-coeff) + step-doubling LTE control",
        kappa=cfg["kappa"], mobility=cfg["M"], horizon=cfg["t_end"],
        nonlinear_tol=1e-10)

    # --- 1. temporal order (smooth) -----------------------------------
    ctx.log("temporal-order verification (smooth single mode)...")
    order = order_study(cfg, ctx)

    # --- quench mesh --------------------------------------------------
    dm, mesh, cons = build_mesh_dm(cfg["level"], device=ctx.device)
    side = int(round(np.sqrt(len(mesh.node_coords))))
    ctx.log(f"quench: {side}x{side}, kappa={cfg['kappa']:g}, "
            f"t_end={cfg['t_end']:g}, solver={ctx.solver}")

    # --- 2. tight reference (+ Richardson verification) ---------------
    ctx.log(f"reference: fixed dt={cfg['ref_dt']:g} "
            f"({int(round(cfg['t_end'] / cfg['ref_dt']))} steps)...")
    refrun = fixed_run(dm, mesh, side, cfg["ref_dt"], cfg, ctx,
                       record_energy=True)
    rstats = _ref_stats(refrun["field"], refrun["F"])
    rstats["_fld"] = refrun["field"]
    # verify the reference: a coarser 2x-dt reference + Richardson error bar
    ref2 = fixed_run(dm, mesh, side, 2.0 * cfg["ref_dt"], cfg, ctx)
    ref_relL2 = float(np.linalg.norm(ref2["field"] - refrun["field"])
                      / rstats["norm"])
    ref_rich_F = float(dconv.richardson_error_estimate(
        ref2["F"], refrun["F"], 2.0, 2.0))
    ctx.log(f"  reference self-check: |c(2dt)-c(dt)|/|c| = {ref_relL2:.2e}, "
            f"Richardson dF ~ {ref_rich_F:.2e}")

    # --- 3. fixed-dt convergence sweep --------------------------------
    ctx.log("fixed-dt convergence sweep...")
    fixed = []
    for dt in cfg["fixed_dts"]:
        fr = fixed_run(dm, mesh, side, dt, cfg, ctx,
                       record_energy=(dt == cfg["fixed_dt"]))
        m = _metrics(fr["field"], fr["F"], rstats)
        fr.update(m)
        fixed.append(fr)
        ctx.log(f"  fixed dt={dt:.1e}: {fr['nsteps']} steps, "
                f"{fr['newton']} solves, relL2={m['relL2']:.2e}, "
                f"|dF|={m['err_energy']:.2e}")
    fixed_dts = [f["dt"] for f in fixed]
    fixed_relL2 = [f["relL2"] for f in fixed]
    # the observed order is only meaningful in the CONVERGENT regime (below the
    # stiffness cliff, where the quench is resolved); coarser fixed steps blow
    # the morphology up and leave the asymptotic range.
    conv = [(f["dt"], f["relL2"]) for f in fixed if f["relL2"] < 0.1]
    fixed_order = (float(dconv.observed_order([c[0] for c in conv],
                                              [c[1] for c in conv]))
                   if len(conv) >= 2 else float("nan"))

    # --- 4. adaptive tolerance sweep ----------------------------------
    ctx.log("adaptive tolerance sweep...")
    adaptive = []
    for tol in cfg["tols"]:
        ar = adaptive_run(dm, mesh, side, tol, cfg, ctx,
                          headline=(tol == cfg["work_tol"]))
        m = _metrics(ar["field"], ar["F"], rstats)
        ar.update(m)
        adaptive.append(ar)
        ctx.log(f"  adaptive tol={tol:.1e}: {ar['accepted']} acc / "
                f"{ar['rejected']} rej, {ar['newton']} solves, "
                f"relL2={m['relL2']:.2e}, |dF|={m['err_energy']:.2e}, "
                f"dt in [{ar['dt_min']:.1e}, {ar['dt_max']:.1e}]")

    # headline runs
    fx = next(f for f in fixed if f["dt"] == cfg["fixed_dt"])
    ad = next(a for a in adaptive if a["tol"] == cfg["work_tol"])

    # --- 5. matched-accuracy speed-up + honest worst-case bound -------
    A = ad["relL2"]                        # adaptive working accuracy
    # interpolate the fixed dt that reaches the SAME c(T) error (log-log), using
    # ONLY the convergent fixed runs (a clean order-p line); the cliff points
    # would corrupt the fit.
    cfix = [f for f in fixed if f["relL2"] < 0.1]
    cdt = [f["dt"] for f in cfix]
    crel = [f["relL2"] for f in cfix]
    lp = np.polyfit(np.log(cdt), np.log(crel), 1)
    matched_dt = float(np.exp((np.log(A) - lp[1]) / lp[0]))
    matched_steps = int(round(cfg["t_end"] / matched_dt))
    mean_newton_per_step = float(np.mean([f["newton"] / f["nsteps"]
                                          for f in cfix]))
    matched_solves = int(round(matched_steps * mean_newton_per_step))
    wall_per_step = float(np.mean([f["wall_s"] / f["nsteps"] for f in cfix]))
    matched_wall = matched_steps * wall_per_step
    matched = dict(
        adaptive_relL2=A, adaptive_solves=ad["newton"],
        adaptive_accepted=ad["accepted"], adaptive_wall_s=ad["wall_s"],
        matched_fixed_dt=matched_dt, matched_fixed_steps=matched_steps,
        matched_fixed_solves=matched_solves, matched_fixed_wall_s=matched_wall,
        speedup_solves=float(matched_solves / max(ad["newton"], 1)),
        speedup_wall=float(matched_wall / max(ad["wall_s"], 1e-9)),
        speedup_steps=float(matched_steps / max(ad["accepted"], 1)),
        within_tested_range=bool(min(cdt) <= matched_dt <= max(cdt)))
    # naive worst-case: a fixed step pinned at the quench's smallest step
    worst_dt = ad["dt_min"]
    worst = dict(dt=float(worst_dt),
                 implied_steps=int(round(cfg["t_end"] / worst_dt)),
                 note="worst case: a fixed step pinned at the quench's "
                      "smallest adaptive step for the whole horizon")
    ctx.log(f"matched accuracy (relL2={A:.2e}): fixed dt~{matched_dt:.2e} "
            f"({matched_steps} steps, ~{matched_solves} solves) vs adaptive "
            f"{ad['newton']} solves -> speedup(solves)="
            f"{matched['speedup_solves']:.2f}x")

    # --- 6. knee: tightest tol that still meaningfully improves error --
    tols = [a["tol"] for a in adaptive]
    rel = [a["relL2"] for a in adaptive]
    knee_tol, knee_rel = _knee(tols, rel)

    # ---- store arrays for figures ------------------------------------
    h = ctx.history
    h["order_dts"] = np.array(order["dts"])
    h["order_bdf1_err"] = np.array(order["bdf1"]["errors"])
    h["order_bdf2_err"] = np.array(order["bdf2"]["errors"])
    h["fixed_dts"] = np.array(fixed_dts)
    h["fixed_relL2"] = np.array(fixed_relL2)
    h["fixed_err_energy"] = np.array([f["err_energy"] for f in fixed])
    h["fixed_newton"] = np.array([f["newton"] for f in fixed])
    h["fixed_wall"] = np.array([f["wall_s"] for f in fixed])
    h["sweep_tols"] = np.array(tols)
    h["sweep_relL2"] = np.array(rel)
    h["sweep_err_energy"] = np.array([a["err_energy"] for a in adaptive])
    h["sweep_newton"] = np.array([a["newton"] for a in adaptive])
    h["sweep_accepted"] = np.array([a["accepted"] for a in adaptive])
    h["sweep_rejected"] = np.array([a["rejected"] for a in adaptive])
    h["sweep_wall"] = np.array([a["wall_s"] for a in adaptive])
    ser = ad["series"]
    h["ad_acc_t"] = np.array(ser["acc_t"])
    h["ad_acc_dt"] = np.array(ser["acc_dt"])
    h["ad_acc_newton"] = np.array(ser["acc_newton"])
    h["ad_energy_t"] = np.array(ser["energy_t"])
    h["ad_energy_F"] = np.array(ser["energy_F"])
    h["fx_energy_t"] = fx["energy_t"]
    h["fx_energy_F"] = fx["energy_F"]
    h["ref_energy_t"] = refrun["energy_t"]
    h["ref_energy_F"] = refrun["energy_F"]
    h["field_ref"] = rstats["_fld"]
    h["field_fixed"] = fx["field"]
    h["field_adapt"] = ad["field"]

    def _clean(d):
        return {k: v for k, v in d.items()
                if k not in ("field", "series", "energy_t", "energy_F", "_fld")}

    results = {
        "order": order,
        "reference": {"dt": cfg["ref_dt"], "nsteps": refrun["nsteps"],
                      "newton": refrun["newton"], "wall_s": refrun["wall_s"],
                      "F": rstats["F"], "lam1": rstats["lam1"],
                      "dscale": rstats["dscale"], "phase_hi": rstats["phase_hi"],
                      "selfcheck_relL2": ref_relL2, "richardson_dF": ref_rich_F},
        "fixed_sweep": [_clean(f) for f in fixed],
        "fixed_order": fixed_order,
        "adaptive_sweep": [_clean(a) for a in adaptive],
        "fixed": _clean(fx),
        "adaptive": _clean(ad),
        "matched": matched,
        "worst_case": worst,
        "knee": {"tol": knee_tol, "relL2": knee_rel},
        "mesh": {"level": cfg["level"], "side": side, "kappa": cfg["kappa"],
                 "t_end": cfg["t_end"]},
    }
    return results


def _knee(tols, rel):
    """Accuracy-cost knee: the tolerance at which the error against the
    reference is smallest. Below it, tightening the tolerance does not reduce
    the pixelwise error (it sits on the trajectory-sensitivity floor, and here
    even rises) while the cost keeps climbing -- so this is the tol worth
    using, and tightening further is wasted work."""
    i = int(np.argmin(rel))
    return float(tols[i]), float(rel[i])


def main():
    args = build_parser("OrgElMorph P2 (adaptive time stepping)").parse_args()
    here = os.path.dirname(__file__)
    schema = _SCHEMA
    baseline = (os.path.join(here, "baseline.yaml")
                if args.mode == "reference" else None)
    # Same INDEFINITE (c, mu) saddle system as P1: scipy SuperLU's partial
    # pivoting is exact here where cuDSS (no pivoting) diverges, and the solve
    # count is our Newton/linear-solve cost counter. splu is the documented
    # small-problem `auto` choice.
    run_tutorial(p2_run, schema=schema, args=args,
                 default_output=os.path.join(here, "outputs", "p2"),
                 baseline=baseline, default_solver="splu")


_SCHEMA = cfgmod.ConfigSchema(name="p2", fields={
    # quench problem
    "level": cfgmod.Field(int, default=5, min=2, max=8),
    "kappa": cfgmod.Field(float, default=2.0e-3, min=0.0),
    "M": cfgmod.Field(float, default=1.0, min=0.0),
    "amp": cfgmod.Field(float, default=0.05, min=0.0),
    "c_avg": cfgmod.Field(float, default=0.0),
    "seed": cfgmod.Field(int, default=7),
    "t_end": cfgmod.Field(float, default=0.6, min=0.0),
    "ref_dt": cfgmod.Field(float, default=2.0e-4, min=0.0),
    "fixed_dt": cfgmod.Field(float, default=2.0e-3, min=0.0),
    "work_tol": cfgmod.Field(float, default=5.0e-4, min=0.0),
    "tols": cfgmod.Field(list, default=[2e-3, 1e-3, 5e-4, 2e-4]),
    "fixed_dts": cfgmod.Field(list,
                              default=[5e-4, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3]),
    "dt0": cfgmod.Field(float, default=2.0e-4, min=0.0),
    "dt_max": cfgmod.Field(float, default=0.01, min=0.0),
    "dt_min": cfgmod.Field(float, default=1.0e-7, min=0.0),
    "nchecks": cfgmod.Field(int, default=40, min=2),
    # smooth temporal-order study (separate small mesh)
    "order_level": cfgmod.Field(int, default=4, min=2, max=7),
    "order_kappa": cfgmod.Field(float, default=0.038, min=0.0),
    "order_amp": cfgmod.Field(float, default=0.02, min=0.0),
    "order_T": cfgmod.Field(float, default=0.05, min=0.0),
    "order_ref_dt": cfgmod.Field(float, default=2.5e-5, min=0.0),
    "order_dts": cfgmod.Field(list, default=[1e-3, 5e-4, 2.5e-4, 1.25e-4]),
    "precision": cfgmod.Field(str, default="fp64"),
})


if __name__ == "__main__":
    main()
