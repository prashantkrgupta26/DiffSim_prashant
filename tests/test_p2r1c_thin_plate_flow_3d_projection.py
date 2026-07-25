"""Smoke gate: 3-D thin-plate PROJECTION + two-sided shell + gpu_cg PPE.

PHYSICS-HONEST gate. The 3-D lagged-pressure projection split has a DOCUMENTED
defect (p2-r2a-monolithic-pivot verdict; docs/dev/2026-07-23-projection-ladder-
verdict.md): with an open outflow it cannot build the driving stagnation
pressure from rest, so the 3-D Cd is FINITE but NOT physically faithful. This
gate therefore asserts ONLY what the wiring guarantees — it does NOT assert
Cd > 0 or projection ≈ monolithic (both are blocked by the projection-split
defect, a separate research track):

  (1) end-to-end: the projection + two-sided-shell + gpu_cg march RUNS and
      produces a FINITE Cd (np.isfinite);
  (2) the PPE provably ran on gpu_cg AND converged — verified by spying on
      diffsim.solvers.dist_cg.pcg (the gpu_cg backend) and asserting it was
      called and every call reported converged;
  (3) the two-sided shell coupling is LOAD-BEARING — the two-sided force
      differs materially from the one-sided (drop-Gamma~+) force.

The MONOLITHIC driver (tests/p2r1c_thin_plate_flow_3d.py) remains the correct
3-D engine; this test guards the scalable projection INFRASTRUCTURE only.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

pytestmark = pytest.mark.tier5


def test_projection_shell_runs_finite_cd(device):
    """(1) End-to-end: the projection + two-sided shell + gpu_cg march runs
    on a tiny 3-D mesh (L3 uniform, 3 steps) and produces a FINITE Cd.

    NOT asserted: Cd > 0 or physical magnitude — the 3-D projection split is
    a documented weak/wrong steady state (see module docstring). Finiteness
    is exactly the wiring guarantee.
    """
    from p2r1c_thin_plate_flow_3d_projection import run_flow_past_3d_projection

    res = run_flow_past_3d_projection(level=3, nsteps=3, dt=0.01, nu=0.1,
                                      U_inf=1.0, alpha=50.0,
                                      ppe_solver="gpu_cg", verbose=False)

    assert np.all(np.isfinite(res["cd"])), f"Cd not finite: {res['cd']}"
    assert np.all(np.isfinite(res["cl_y"])), f"Cl_y not finite: {res['cl_y']}"
    assert np.all(np.isfinite(res["cl_z"])), f"Cl_z not finite: {res['cl_z']}"
    # the shell must be active (plate cuts the mesh)
    assert res["n_excluded"] > 0, "no cells excluded — shell pipeline inactive"
    # the driver reports the PPE solver it was asked to use
    assert res["ppe_solver"] == "gpu_cg", (
        f"ppe_solver field was {res['ppe_solver']!r}, expected 'gpu_cg'")


def test_ppe_ran_on_gpu_cg_and_converged(device, monkeypatch):
    """(2) The PPE provably ran on gpu_cg and converged.

    Spies on ``diffsim.solvers.dist_cg.pcg`` (the CG kernel the "gpu_cg"
    solver backend calls). Asserts the backend was invoked (call count > 0)
    and that EVERY invocation reported converged=True. This proves the PPE
    Laplacian was actually solved by the scalable gpu_cg path — not silently
    routed to splu or skipped.

    INDEPENDENT of the force/physics: it inspects the linear-solve backend
    directly.
    """
    import diffsim.solvers.dist_cg as dist_cg

    calls = {"n": 0, "all_converged": True, "iters": []}
    orig_pcg = dist_cg.pcg

    def spy_pcg(*args, **kwargs):
        x, info = orig_pcg(*args, **kwargs)
        calls["n"] += 1
        calls["all_converged"] = calls["all_converged"] and bool(
            info.get("converged"))
        calls["iters"].append(info.get("iters"))
        return x, info

    # patch BOTH the module attr and the name imported into linsolve's scope.
    monkeypatch.setattr(dist_cg, "pcg", spy_pcg)

    from p2r1c_thin_plate_flow_3d_projection import run_flow_past_3d_projection
    res = run_flow_past_3d_projection(level=3, nsteps=3, dt=0.01, nu=0.1,
                                      U_inf=1.0, alpha=50.0,
                                      ppe_solver="gpu_cg", verbose=False)

    print(f"\n[gpu_cg spy] pcg calls={calls['n']}  iters={calls['iters']}")
    assert calls["n"] > 0, (
        "gpu_cg backend (dist_cg.pcg) was never called — the PPE did NOT "
        "run on gpu_cg")
    assert calls["all_converged"], (
        "a gpu_cg PPE solve did NOT converge (info.converged=False)")
    # sanity: the run itself stayed finite (the solve fed a usable field)
    assert np.all(np.isfinite(res["cd"])), f"Cd not finite: {res['cd']}"


def test_two_sided_coupling_load_bearing(device):
    """(3) The two-sided shell coupling is load-bearing.

    Compares the two-sided assembly (Gamma~+ AND Gamma~-) against the
    one-sided anti-vacuity variant (drop Gamma~+). If the + side were inert,
    the two forces would coincide. They must differ materially — the + side
    carries real Nitsche blockage.

    INDEPENDENT reference: the one-sided path is the same anti-vacuity lever
    the monolithic driver uses (run_flow_past_3d(_two_sided=False)).
    """
    from p2r1c_thin_plate_flow_3d_projection import (
        run_flow_past_3d_projection, run_flow_past_3d_projection_one_sided)

    kw = dict(level=3, nsteps=3, dt=0.01, nu=0.1, U_inf=1.0, alpha=50.0,
              ppe_solver="gpu_cg", verbose=False)
    res_two = run_flow_past_3d_projection(**kw)
    res_one = run_flow_past_3d_projection_one_sided(**kw)

    cd_two = float(res_two["cd"][-1])
    cd_one = float(res_one["cd"][-1])
    print(f"\n[load-bearing] two-sided Cd={cd_two:.4f}  "
          f"one-sided(no+) Cd={cd_one:.4f}")

    assert np.isfinite(cd_two) and np.isfinite(cd_one), (
        f"non-finite Cd: two={cd_two}, one={cd_one}")
    rel_diff = abs(cd_two - cd_one) / max(abs(cd_two), 1e-6)
    assert rel_diff > 0.05, (
        f"two-sided coupling not load-bearing: two-sided={cd_two:.4f}, "
        f"one-sided={cd_one:.4f}, rel_diff={rel_diff:.3f} (< 5%) — the "
        "Gamma~+ side is inert")
