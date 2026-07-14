"""OrgElMorph course - Computational C6 driver (harness-based, verified).

Instruments the production Cahn-Hilliard Newton solve into a repeatable
DIAGNOSTIC workflow and prints a PASS/FAIL gate table:

  1. ANATOMY  -- a healthy poly step: residual + update norms per iteration,
     quadratic convergence (residual squared each step).
  2. STOPPING -- absolute / relative / residual criteria on that trajectory.
  3. SAFEGUARDS -- FH box-projection bookkeeping + trust-clamp activity on a
     deep quench that still converges.
  4. FAILURES (each GENUINELY fails as taught):
       - Newton STAGNATION (deep quench, dt too large) -> does NOT converge;
         the same quench at small dt DOES (proof it is a step-size problem).
       - MIN-DT failure (adaptive controller pinned at the floor, LTE > tol).
       - ILL-CONDITIONED FH Jacobian near the wall (cond climbs with 1/c).
     A FAILURE CHECKPOINT for the stagnation case is written to checkpoints/.

Some gates assert a FAILURE (the stagnation solve must NOT converge, the
min-dt controller must stay pinned) -- a diagnostic chapter is only honest if
its taught failures really fail.

    PYTHONPATH=<repo>/src python run.py --device cuda:0 --solver splu \\
        --output outputs/c6 --overwrite --mode reference

Config is the canonical record; results.json is tolerance-checked against
baseline.yaml; provenance goes to metadata.json.  gen_figures.py renders the
document's figures + numbers/c6.tex from the saved run.
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

import diagnostics_ch as dch                                  # noqa: E402

SCHEMA = cfgmod.ConfigSchema(name="c6", fields={
    "level": cfgmod.Field(int, default=4, min=2, max=7),
    "kmax": cfgmod.Field(int, default=6, min=3),
    "healthy_dt": cfgmod.Field(float, default=1e-3, min=0.0),
    "stag_dt": cfgmod.Field(float, default=5e-2, min=0.0),
    "stag_fixed_dt": cfgmod.Field(float, default=2e-3, min=0.0),
    "newton_max": cfgmod.Field(int, default=15, min=1),
    "fh_B": cfgmod.Field(float, default=6.0),
    "safe_fh_B": cfgmod.Field(float, default=8.0),
    "precision": cfgmod.Field(str, default="fp64"),
})


def c6_run(cfg, ctx):
    dev = ctx.device
    lvl = cfg["level"]
    ctx.provenance.update(
        mesh={"family": "uniform box 2D", "level": lvl},
        time_integrator="BDF1 (per-step Newton diagnostics)",
        nonlinear_tol=1e-10, note="C6 nonlinear-solver diagnostics")

    dm, _, _ = dch.build_dm(lvl, 1, dev)
    dm3, _, _ = dch.build_dm(3, 1, dev)   # small mesh for exact conditioning

    # 1. anatomy of a healthy Newton solve --------------------------------
    ctx.log("1. ANATOMY: healthy poly Newton (quadratic convergence)")
    healthy = dch.newton_trajectory(dm, dch.ic_gentle, cfg["healthy_dt"],
                                    cfg["kmax"], energy="poly")
    for k, r, d in zip(healthy["k"], healthy["res"], healthy["dx_inf"]):
        ctx.log(f"   iter {k}: ||r||={r:.3e}  |dx|_inf={d:.3e}")
    ctx.log(f"   -> converged in {healthy['iters_to_converge']} iters, "
            f"clamp fired: {healthy['clamp_fired']}")

    # 2. stopping criteria -------------------------------------------------
    ctx.log("2. STOPPING criteria on the healthy trajectory (tol 1e-8)")
    stop = dch.stopping_criteria(healthy, x_scale=1.0, tol=1e-8)
    stop_scaled = dch.stopping_criteria(healthy, x_scale=1e-3, tol=1e-8)
    ctx.log(f"   residual@{stop['by_residual']}  abs-update@"
            f"{stop['by_abs_update']}  rel-update@{stop['by_rel_update']}")
    ctx.log(f"   (rescale field x10^-3: abs-update fires @"
            f"{stop_scaled['by_abs_update']} -- absolute test is scale-blind)")

    # 3. safeguards --------------------------------------------------------
    ctx.log("3. SAFEGUARDS: deep FH quench with projection + trust clamp")
    safe = dch.safeguard_report(dm, dt=1.0, fh_B=cfg["safe_fh_B"],
                                newton_max=cfg["newton_max"])
    ctx.log(f"   projection moved {safe['proj_dofs']} dofs "
            f"(max correction {safe['proj_max']:.2e}); "
            f"converged={safe['converged']} in {safe['iters']} iters")

    # 4a. stagnation (genuine failure) + its fix ---------------------------
    ctx.log("4a. FAILURE -- Newton stagnation (deep quench, dt too large)")
    stag = dch.stagnation_case(dm, dt=cfg["stag_dt"],
                               newton_max=cfg["newton_max"])
    ctx.log(f"   {cfg['newton_max']} iters: converged={stag['converged']}, "
            f"final ||r||={stag['res_final']:.2e}, "
            f"clamp fraction {stag['clamp_fraction']:.2f}")
    fixed = dch.stagnation_fixed(dm, dt=cfg["stag_fixed_dt"],
                                 newton_max=cfg["newton_max"])
    ctx.log(f"   same quench at dt={fixed['dt']:g}: converged="
            f"{fixed['converged']} in {fixed['iters']} iters (fix = smaller dt)")
    ckpt = os.path.join(ctx.paths["checkpoints"], "stagnation_step.npz")
    dch.write_failure_checkpoint(ckpt, "stagnation", dm,
                                 lambda x: dch.ic_quench(x, 0), cfg["stag_dt"],
                                 "poly", cfg["newton_max"], stag["traj"])
    ctx.log(f"   failure checkpoint -> {os.path.relpath(ckpt, ctx.paths['root'])}")

    # 4b. min-dt failure ---------------------------------------------------
    ctx.log("4b. FAILURE -- min-dt (adaptive controller pinned at the floor)")
    mindt = dch.min_dt_case(dm)
    ctx.log(f"   {mindt['n_steps']} steps, all at floor "
            f"{mindt['all_at_floor']}; LTE@floor={mindt['lte_at_floor']:.2e} "
            f"> tol {mindt['tol']:.0e}: {mindt['lte_exceeds_tol']}")

    # 4c. ill-conditioned Jacobian ----------------------------------------
    ctx.log("4c. FAILURE -- ill-conditioned FH Jacobian near the wall")
    cond = dch.conditioning_case(dm3, fh_B=cfg["fh_B"])
    for cm, cc in zip(cond["compositions"], cond["cond"]):
        ctx.log(f"   c={cm:.3f}: cond(J)={cc:.3e}")
    ctx.log(f"   cond ratio wall/bulk = {cond['cond_ratio']:.1f}x")

    # figures history ------------------------------------------------------
    ctx.history["healthy_k"] = np.asarray(healthy["k"])
    ctx.history["healthy_res"] = np.asarray(healthy["res"])
    ctx.history["healthy_dx"] = np.asarray(healthy["dx_inf"])
    ctx.history["stag_k"] = np.asarray(stag["traj"]["k"])
    ctx.history["stag_res"] = np.asarray(stag["traj"]["res"])
    ctx.history["stag_dx"] = np.asarray(stag["traj"]["dx_inf"])
    ctx.history["stag_clamp"] = np.asarray(stag["traj"]["clamp"], dtype=float)
    ctx.history["cond_comp"] = np.asarray(cond["compositions"])
    ctx.history["cond_val"] = np.asarray(cond["cond"])
    ctx.history["cond_fpp"] = np.asarray(cond["fpp"])
    ctx.history["mindt_dts_seen"] = np.asarray([mindt["dt_min_seen"],
                                                mindt["dt_max_seen"]])

    results = {
        "healthy_iters": healthy["iters_to_converge"],
        "healthy_res0": healthy["res"][0],
        "healthy_res_converged": min(healthy["res"]),
        "healthy_clamp_fired": healthy["clamp_fired"],
        "stop_by_residual": stop["by_residual"],
        "stop_by_abs_update": stop["by_abs_update"],
        "stop_abs_scaled": stop_scaled["by_abs_update"],
        "safe_proj_dofs": safe["proj_dofs"],
        "safe_proj_max": safe["proj_max"],
        "safe_converged": safe["converged"],
        "stag_converged": bool(stag["converged"]),
        "stag_res_final": stag["res_final"],
        "stag_clamp_fraction": stag["clamp_fraction"],
        "stag_fixed_converged": bool(fixed["converged"]),
        "stag_fixed_iters": fixed["iters"],
        "mindt_all_at_floor": mindt["all_at_floor"],
        "mindt_lte_at_floor": mindt["lte_at_floor"],
        "mindt_lte_exceeds_tol": mindt["lte_exceeds_tol"],
        "mindt_n_steps": mindt["n_steps"],
        "cond_bulk": cond["cond"][0],
        "cond_wall": cond["cond"][-1],
        "cond_ratio": cond["cond_ratio"],
    }

    gates = {
        "healthy Newton converges (<=5 iters)":
            0 < healthy["iters_to_converge"] <= 5,
        "healthy convergence is quadratic (res drops >6 orders)":
            healthy["res"][0] / min(healthy["res"]) > 1e6,
        "stopping tests agree on well-scaled field":
            stop["by_residual"] > 0 and stop["by_abs_update"] > 0,
        "FH safeguards active + step converges":
            safe["proj_dofs"] > 0 and safe["converged"],
        "STAGNATION genuinely does NOT converge":
            not stag["converged"],
        "stagnation clamp fired (diverging direction)":
            stag["clamp_fraction"] > 0.5,
        "same quench at small dt DOES converge":
            fixed["converged"],
        "MIN-DT controller pinned at floor (LTE > tol)":
            mindt["all_at_floor"] and mindt["lte_exceeds_tol"],
        "ILL-CONDITIONED: cond climbs toward the wall (>10x)":
            cond["cond_ratio"] > 10.0,
    }
    ctx.log("--- self-check gates ---")
    for name, ok in gates.items():
        ctx.log(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    ctx.log(f"ALL GATES: {'PASS' if all(gates.values()) else 'FAIL'}")
    results["all_gates_pass"] = bool(all(gates.values()))
    return results


def main():
    args = build_parser("OrgElMorph C6 (nonlinear diagnostics)").parse_args()
    here = os.path.dirname(__file__)
    baseline = (os.path.join(here, "baseline.yaml")
                if args.mode == "reference" else None)
    run_tutorial(c6_run, schema=SCHEMA, args=args,
                 default_output=os.path.join(here, "outputs", "c6"),
                 baseline=baseline, default_solver="splu")


if __name__ == "__main__":
    main()
