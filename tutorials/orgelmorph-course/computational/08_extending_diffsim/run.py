"""OrgElMorph course - Computational C8 driver (harness-based, verified).

Walks the FULL contributor workflow for the worked new-term example (a sixth-
order Landau term added to the Cahn-Hilliard free energy) and prints a
PASS/FAIL gate for each stage -- the same gates ``test_sextic_ch.py`` enforces,
plus the physics check and the profile:

  1. TERM derivatives      f', f'' of the new term vs finite difference
  2. ANALYTIC Jacobian     the whole-system tangent vs a residual FD (beta=0,0.5)
  3. LIMITING case         beta -> 0 reduces to the base double well
  4. MMS convergence       steady manufactured solution, order ~ p+1
  5. CONSERVATION          no-flux march conserves INT c dV
  6. ENERGY dissipation    F non-increasing (Lyapunov)
  7. PHYSICS effect        the term bounds deep quenches (max|c| smaller)
  8. PROFILE               assembly vs solve cost; the new term is ~free

    PYTHONPATH=<repo>/src <repo>/.venv/bin/python run.py --device cpu \\
        --solver splu --output outputs/c8 --overwrite --mode reference

Config is the canonical record; results.json is tolerance-checked against
baseline.yaml; provenance goes to metadata.json.  gen_figures.py renders the
document's figures + numbers/c8.tex from the saved run.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                os.pardir, os.pardir)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import config as cfgmod                          # noqa: E402
from common.run_base import build_parser, run_tutorial        # noqa: E402
from diffsim.diagnostics import profiling                     # noqa: E402

import sextic_ch as s                                         # noqa: E402

SCHEMA = cfgmod.ConfigSchema(name="c8", fields={
    "beta": cfgmod.Field(float, default=0.5, min=0.0),
    "mms_levels": cfgmod.Field(list, default=[3, 4, 5]),
    "march_level": cfgmod.Field(int, default=5, min=2, max=7),
    "march_dt": cfgmod.Field(float, default=2e-3, min=0.0),
    "march_steps": cfgmod.Field(int, default=15, min=2),
    "quench_dt": cfgmod.Field(float, default=5e-2, min=0.0),
    "precision": cfgmod.Field(str, default="fp64"),
})


def _fd_jac_rel(level, beta, eps=1e-6, seed=0):
    prob = s.Problem(level=level)
    rng = np.random.default_rng(seed)
    x = 0.4 * rng.standard_normal(2 * prob.n)
    c_old = 0.4 * rng.standard_normal(prob.n)
    v = rng.standard_normal(2 * prob.n)
    R0 = s.residual(prob, x, c_old, 0.1, beta)
    fd = (s.residual(prob, x + eps * v, c_old, 0.1, beta) - R0) / eps
    J = s.jacobian(prob, x, 0.1, beta)
    return float(np.linalg.norm(J @ v - fd) / (np.linalg.norm(fd) + 1e-30))


def c8_run(cfg, ctx):
    beta = cfg["beta"]
    dev = "cpu"                       # the worked reference is transparent NumPy
    ctx.provenance.update(
        mesh={"family": "uniform box 2D", "mms_levels": cfg["mms_levels"],
              "march_level": cfg["march_level"]},
        time_integrator="backward Euler (transparent mixed CH)",
        note="C8 extending DiffSim: sextic free-energy term", beta=beta)

    # 1. term derivatives --------------------------------------------------
    ctx.log("1. TERM derivatives (new term f', f'' vs finite difference)")
    cc = np.linspace(-1.3, 1.3, 41)
    eps = 1e-6
    t = s.sextic_term(cc, beta)
    fp_fd = (s.sextic_term(cc + eps, beta)["f"] - t["f"]) / eps
    fpp_fd = (s.sextic_term(cc + eps, beta)["fp"] - t["fp"]) / eps
    term_fp_err = float(np.max(np.abs(fp_fd - t["fp"])))
    term_fpp_err = float(np.max(np.abs(fpp_fd - t["fpp"])))
    ctx.log(f"   f' err {term_fp_err:.2e}, f'' err {term_fpp_err:.2e}")

    # 2. analytic Jacobian vs FD ------------------------------------------
    ctx.log("2. ANALYTIC Jacobian vs finite difference")
    jac_rel = _fd_jac_rel(4, beta)
    jac_rel_base = _fd_jac_rel(4, 0.0)
    ctx.log(f"   beta={beta}: rel {jac_rel:.2e}; beta=0: rel {jac_rel_base:.2e}")

    # 3. limiting case -----------------------------------------------------
    ctx.log("3. LIMITING case (beta -> 0 reduces to base double well)")
    t0 = s.sextic_term(cc, 0.0)
    limiting_zero = bool(np.allclose(t0["f"], 0.0) and np.allclose(t0["fp"], 0.0)
                         and np.allclose(t0["fpp"], 0.0)
                         and np.allclose(s.fprime(cc, 0.0), cc ** 3 - cc))
    ctx.log(f"   new-term contributions vanish at beta=0: {limiting_zero}")

    # 4. MMS convergence ---------------------------------------------------
    ctx.log("4. MMS convergence (steady manufactured, order ~ p+1)")
    errs, hs = [], []
    for lv in cfg["mms_levels"]:
        prob = s.Problem(level=lv)
        c, info = s.solve_mms(prob, beta)
        assert info["converged"], f"MMS Newton failed at level {lv}"
        errs.append(s.l2_error(prob, c, s.c_star))
        hs.append(1.0 / (1 << lv))
        ctx.log(f"   level {lv}: L2={errs[-1]:.3e} ({info['iters']} its)")
    mms_order = float(np.polyfit(np.log(hs), np.log(errs), 1)[0])
    ctx.log(f"   observed order {mms_order:.2f} (expect 2)")

    # 5 + 6. conservation + energy ----------------------------------------
    ctx.log("5+6. CONSERVATION + ENERGY (no-flux march)")
    probm = s.Problem(level=cfg["march_level"])
    rng = np.random.default_rng(0)
    c0 = 0.05 * rng.standard_normal(probm.n)
    out = s.march(probm, c0, beta, cfg["march_dt"], cfg["march_steps"])
    mass_drift = abs(out["mass"][-1] - out["mass"][0])
    rel_mass = mass_drift / (abs(out["mass"][0]) + 1e-30)
    energy_inc = float(np.diff(out["energy"]).max())
    ctx.log(f"   mass drift {mass_drift:.2e} (rel {rel_mass:.2e}); energy "
            f"{out['energy'][0]:.4f} -> {out['energy'][-1]:.4f}, "
            f"max +incr {max(0.0, energy_inc):.2e}")

    # 7. physics effect: the term bounds a deep quench ---------------------
    ctx.log("7. PHYSICS effect (sextic term bounds a deep quench)")
    probq = s.Problem(level=cfg["march_level"])
    cq0 = 0.6 * np.random.default_rng(3).standard_normal(probq.n)
    base = s.march(probq, cq0, 0.0, cfg["quench_dt"], 3)
    sext = s.march(probq, cq0, max(beta, 0.5), cfg["quench_dt"], 3)
    qmax_base = float(np.max(np.abs(base["c"])))
    qmax_sext = float(np.max(np.abs(sext["c"])))
    ctx.log(f"   deep quench max|c|: base {qmax_base:.3f} -> sextic "
            f"{qmax_sext:.3f} (bounded)")

    # 8. profile: assembly vs solve, and the new term is ~free ------------
    ctx.log("8. PROFILE (assembly vs solve; new-term overhead)")
    prof = s.Problem(level=cfg["march_level"])
    xr = np.zeros(2 * prof.n); xr[0::2] = c0
    sink = {}
    with profiling.timer("residual", sync=False, sink=sink):
        for _ in range(20):
            s.residual(prof, xr, c0, cfg["march_dt"], beta)
    with profiling.timer("jacobian", sync=False, sink=sink):
        for _ in range(20):
            s.jacobian(prof, xr, cfg["march_dt"], beta)
    with profiling.timer("jacobian_base", sync=False, sink=sink):
        for _ in range(20):
            s.jacobian(prof, xr, cfg["march_dt"], 0.0)
    asm_ms = (sink["residual"] + sink["jacobian"]) / 20 * 1e3
    newterm_overhead = (sink["jacobian"] - sink["jacobian_base"]) \
        / sink["jacobian_base"] * 100.0
    ctx.log(f"   assemble {asm_ms:.2f} ms/iter; new-term Jacobian overhead "
            f"{newterm_overhead:+.1f}%")

    # figures history ------------------------------------------------------
    ctx.history["mms_h"] = np.asarray(hs)
    ctx.history["mms_err"] = np.asarray(errs)
    ctx.history["mass"] = np.asarray(out["mass"])
    ctx.history["energy"] = np.asarray(out["energy"])
    ctx.history["well_c"] = cc
    ctx.history["well_base"] = s.f_bulk(cc, 0.0)
    ctx.history["well_sextic"] = s.f_bulk(cc, max(beta, 0.5))

    results = {
        "beta": beta,
        "term_fp_err": term_fp_err, "term_fpp_err": term_fpp_err,
        "jac_fd_rel": jac_rel, "jac_fd_rel_base": jac_rel_base,
        "limiting_zero": limiting_zero,
        "mms_order": mms_order,
        "mms_err_coarse": errs[0], "mms_err_fine": errs[-1],
        "mass_drift": mass_drift, "rel_mass_drift": rel_mass,
        "energy_start": out["energy"][0], "energy_end": out["energy"][-1],
        "energy_max_increment": max(0.0, energy_inc),
        "quench_max_base": qmax_base, "quench_max_sextic": qmax_sext,
        "assemble_ms": asm_ms, "newterm_overhead_pct": newterm_overhead,
    }

    gates = {
        "new-term derivatives match FD (f' <1e-4, f'' <1e-3)":
            term_fp_err < 1e-4 and term_fpp_err < 1e-3,
        "analytic Jacobian matches FD (<1e-5), both beta":
            jac_rel < 1e-5 and jac_rel_base < 1e-5,
        "limiting case beta->0 reduces to base model":
            limiting_zero,
        "MMS order in [1.7, 2.3]":
            1.7 < mms_order < 2.3,
        "mass conserved (rel drift < 1e-10)":
            rel_mass < 1e-10,
        "energy non-increasing (max +incr < 1e-9)":
            max(0.0, energy_inc) < 1e-9,
        "sextic term bounds the deep quench (max|c| smaller)":
            qmax_sext < qmax_base,
        "new-term assembly overhead is small (< 25%)":
            newterm_overhead < 25.0,
    }
    ctx.log("--- self-check gates ---")
    for name, ok in gates.items():
        ctx.log(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    ctx.log(f"ALL GATES: {'PASS' if all(gates.values()) else 'FAIL'}")
    results["all_gates_pass"] = bool(all(gates.values()))
    return results


def main():
    args = build_parser("OrgElMorph C8 (extending DiffSim)").parse_args()
    here = os.path.dirname(__file__)
    baseline = (os.path.join(here, "baseline.yaml")
                if args.mode == "reference" else None)
    run_tutorial(c8_run, schema=SCHEMA, args=args,
                 default_output=os.path.join(here, "outputs", "c8"),
                 baseline=baseline, default_solver="splu")


if __name__ == "__main__":
    main()
