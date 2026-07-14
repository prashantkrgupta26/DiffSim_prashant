"""OrgElMorph course - Computational C0 driver (harness-based, verified).

Walks the scalar reaction-diffusion problem of ``scalar_reaction.py`` through
the full weak-form -> CUDA pipeline and VERIFIES it, as a repeatable workflow:

  1. SPATIAL MMS convergence, p=1 and p=2 -- the L2 error falls at the
     predicted h^(p+1) rate (proof the whole assembly is correct).
  2. NEWTON convergence -- with the correct analytic Jacobian, Newton is
     quadratic (a handful of iterations, residual squared each step).
  3. JACOBIAN verification -- the analytic Jacobian matches a finite
     difference of the residual to ~1e-9 (the completed brick), whereas the
     hands-on STARTER (reaction Jacobian omitted) is off by ~1e-3.
  4. PRODUCTION cross-check -- the transparent stiffness equals the
     production Warp-kernel assembly (``assemble_brick_csr``) to round-off.

    PYTHONPATH=<repo>/src python run.py --device cpu --solver splu \\
        --output outputs/c0 --overwrite --mode reference

Config is the canonical record; results.json is tolerance-checked against
baseline.yaml; provenance goes to metadata.json.  A PASS/FAIL gate table is
printed; compare with EXPECTED.md.  gen_figures.py renders the document's
figures + numbers/c0.tex from the saved run.
"""
from __future__ import annotations

import importlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                os.pardir, os.pardir)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import config as cfgmod                          # noqa: E402
from common.run_base import build_parser, run_tutorial        # noqa: E402

import scalar_reaction as sr                                  # noqa: E402

SCHEMA = cfgmod.ConfigSchema(name="c0", fields={
    "alpha": cfgmod.Field(float, default=sr.ALPHA_DEFAULT, min=0.0),
    "p1_levels": cfgmod.Field(list, default=[3, 4, 5, 6]),
    "p2_levels": cfgmod.Field(list, default=[2, 3, 4, 5]),
    "newton_tol": cfgmod.Field(float, default=1e-10, min=0.0),
    "precision": cfgmod.Field(str, default="fp64"),
})


def _spatial_study(alpha, p, levels, tol, device, solver, log):
    errs, iters, res_last = [], [], []
    for lv in levels:
        prob = sr.Problem(level=lv, p=p, alpha=alpha, device=device)
        u, info = sr.newton_solve(prob, tol=tol, solver=solver, device=device)
        e = sr.l2_error(prob, u)
        errs.append(e); iters.append(info["iters"])
        res_last.append(info["res_norm"])
        log(f"  p={p} level {lv}: L2={e:.3e}  Newton {info['iters']} its "
            f"(res {info['res_norm']:.1e}, converged={info['converged']})")
        assert info["converged"], f"Newton failed at p={p} level {lv}"
    logh, loge = np.log([2.0 ** lv for lv in levels]), np.log(errs)
    order = float(-np.polyfit(logh, loge, 1)[0])   # error ~ h^order = 2^{-lv*order}
    return {"levels": list(levels), "l2": errs, "order": order,
            "newton_iters": iters, "newton_res_last": res_last}


def c0_run(cfg, ctx):
    alpha = cfg["alpha"]
    tol = cfg["newton_tol"]
    dev, solver = ctx.device, ctx.solver
    ctx.provenance.update(
        mesh={"family": "uniform box 2D", "p1_levels": cfg["p1_levels"],
              "p2_levels": cfg["p2_levels"]},
        time_integrator="steady (nonlinear Newton)",
        nonlinear_tol=tol, alpha=alpha)

    ctx.log("1. SPATIAL MMS convergence (error ~ h^(p+1))")
    p1 = _spatial_study(alpha, 1, cfg["p1_levels"], tol, dev, solver, ctx.log)
    p2 = _spatial_study(alpha, 2, cfg["p2_levels"], tol, dev, solver, ctx.log)

    ctx.log("2/3. Jacobian verification (analytic vs finite difference)")
    # completed brick vs the hands-on starter -- the red->green contrast
    prob = sr.Problem(level=4, alpha=alpha, device=dev)
    rng = np.random.default_rng(0)
    u = 0.6 * rng.standard_normal(prob.n); u[prob.bnd] = prob.gvals[prob.bnd]
    v = rng.standard_normal(prob.n)
    eps = 1e-6
    fd = (sr.residual(prob, u + eps * v) - sr.residual(prob, u)) / eps

    def rel(mod):
        J = mod.jacobian(prob, u)
        return float(np.linalg.norm(J @ v - fd) / (np.linalg.norm(fd) + 1e-30))

    jac_ok = rel(sr)
    starter = importlib.import_module("scalar_reaction_starter")
    jac_starter = rel(starter)
    ctx.log(f"  analytic-vs-FD rel error: complete {jac_ok:.2e}, "
            f"starter (reaction Jacobian missing) {jac_starter:.2e}")

    ctx.log("4. Production cross-check (transparent stiffness == Warp kernel)")
    from diffsim.api.equation import assemble_brick_csr
    from diffsim.api import PoissonBrick
    K_t = sr.stiffness_csr(prob)
    K_p = assemble_brick_csr(prob.dm, PoissonBrick())
    d = abs(K_t - K_p)
    stiff_diff = float(d.max()) if d.nnz else 0.0
    ctx.log(f"  max |K_transparent - K_production| = {stiff_diff:.2e}")

    # Newton quadratic-convergence trace at a representative level (for figure)
    probq = sr.Problem(level=5, alpha=alpha, device=dev)
    _u, infoq = sr.newton_solve(probq, tol=tol, solver=solver, device=dev)
    ctx.history["newton_res_history"] = np.asarray(infoq["res_history"])
    ctx.history["p1_levels"] = np.asarray(p1["levels"])
    ctx.history["p1_l2"] = np.asarray(p1["l2"])
    ctx.history["p2_levels"] = np.asarray(p2["levels"])
    ctx.history["p2_l2"] = np.asarray(p2["l2"])

    results = {
        "alpha": alpha,
        "p1": p1, "p2": p2,
        "jac_fd_rel_error": jac_ok,
        "jac_fd_rel_error_starter": jac_starter,
        "stiffness_vs_production_max": stiff_diff,
        "newton_iters_level5": infoq["iters"],
        "newton_res0": float(infoq["res_history"][0]),
        "newton_res_final": float(infoq["res_history"][-1]),
    }

    gates = {
        "p1 L2 order in [1.7,2.3]": 1.7 < p1["order"] < 2.3,
        "p2 L2 order in [2.6,3.4]": 2.6 < p2["order"] < 3.4,
        "analytic Jacobian matches FD (<1e-6)": jac_ok < 1e-6,
        "starter Jacobian is wrong (>1e-4)": jac_starter > 1e-4,
        "stiffness == production kernel (<1e-10)": stiff_diff < 1e-10,
        "Newton converged (level5)": infoq["converged"],
    }
    ctx.log("--- self-check gates ---")
    for name, ok in gates.items():
        ctx.log(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    ctx.log(f"ALL GATES: {'PASS' if all(gates.values()) else 'FAIL'}")
    results["all_gates_pass"] = bool(all(gates.values()))
    results["p1_order"] = p1["order"]
    results["p2_order"] = p2["order"]
    return results


def main():
    args = build_parser("OrgElMorph C0 (weak form to CUDA)").parse_args()
    here = os.path.dirname(__file__)
    baseline = (os.path.join(here, "baseline.yaml")
                if args.mode == "reference" else None)
    run_tutorial(c0_run, schema=SCHEMA, args=args,
                 default_output=os.path.join(here, "outputs", "c0"),
                 baseline=baseline, default_solver="splu")


if __name__ == "__main__":
    main()
