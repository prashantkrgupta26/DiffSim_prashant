"""P2-R1a smoke gate: 2-D thin-plate PROJECTION path + two-sided shell.

The 2-D mirror of tests/test_p2r1c_thin_plate_flow_3d_projection.py.  It guards
the projection leg of the both-solver-vs-literature harness — the
LeraySBMShellStepper Helmholtz-Leray split with the merged outflow-BC
p'-scheme levers (whole-outflow-line p'=0 Dirichlet + consistent_projection +
inner_iterate + rotational_pin_wall).  On the Mac CPU the PPE solver is "splu"
(gpu_cg is GPU-only); the resolved Re=250 run swaps PPE_SOLVER=gpu_cg on
gpubox/GH200.

PHYSICS-HONEST gate (mirrors the 3-D gate; NO literature magnitude asserted at
this tiny level):
  (1) end-to-end: the projection + two-sided-shell march RUNS and produces a
      finite, POSITIVE, NON-DIVERGING Cd (all steps > 0, last <= first — a
      decaying, not diverging, startup transient), same sign/shape as the
      monolithic reference on the SAME mesh;
  (2) the PPE provably RAN — verified by spying on
      diffsim.solvers.linsolve.solve_linear and asserting it was called with
      sym=True (the SPD pressure-Poisson sub-solve) at least once per step;
  (3) the two-sided shell coupling is LOAD-BEARING — the two-sided force
      differs materially from the one-sided (drop-Gamma~+) force.

RESIDUAL GAP (NOT asserted, honest): the split's converged Cd is ~40% below
the monolithic on the same mesh (deeper p2-r2a-monolithic-pivot defect; the
plate-surface pressure jump is right-signed but too small).  This gate asserts
sign + non-divergence + shape-tracking, NOT magnitude — settling whether the
projection's lower Cd is a feature vs the LITERATURE is the resolved gpubox
run, not this smoke.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from p2r1a_thin_plate_flow import (
    run_flow_past_projection, run_flow_past_projection_one_sided,
    run_flow_past,
)

pytestmark = pytest.mark.tier5

# Tiny CPU-fast config (splu PPE — no GPU on the Mac).
_SMOKE = dict(level=3, nsteps=6, dt=0.01, nu=0.1, U_inf=1.0, alpha=50.0)


def test_projection_positive_nondiverging_cd():
    """(1) The 2-D projection + two-sided-shell march produces a finite,
    POSITIVE, NON-DIVERGING Cd whose startup transient tracks the monolithic
    reference in sign + shape (the outflow-BC p'-scheme levers).
    """
    res = run_flow_past_projection(ppe_solver="splu", verbose=False, **_SMOKE)
    mono = run_flow_past(**_SMOKE)

    cd = res["cd"]
    assert len(cd) == _SMOKE["nsteps"], f"cd length {len(cd)}"
    assert np.all(np.isfinite(cd)), f"Cd not finite: {cd}"
    assert np.all(np.isfinite(res["cl"])), f"Cl not finite: {res['cl']}"
    assert res["n_excluded"] > 0, "no cells excluded — shell pipeline inactive"
    assert res["ppe_solver"] == "splu", (
        f"ppe_solver field was {res['ppe_solver']!r}, expected 'splu'")

    # PHYSICS: positive + non-diverging (correct drag sign + decaying transient)
    assert np.all(cd > 0.0), f"Cd not all positive (drag wrong-signed): {cd}"
    assert cd[-1] <= cd[0], (
        f"Cd diverging (last {cd[-1]:.3f} > first {cd[0]:.3f}): {cd}")

    # SHAPE/SIGN agreement with the monolithic reference on the same mesh.
    cd_p, cd_m = float(cd[-1]), float(mono["cd"][-1])
    assert cd_m > 0.0, f"monolithic reference Cd not positive: {cd_m}"
    ratio = cd_p / cd_m
    assert 0.3 <= ratio <= 1.2, (
        f"projection Cd {cd_p:.3f} vs monolithic {cd_m:.3f} out of the "
        f"sign/shape band (ratio {ratio:.2f} not in [0.3, 1.2]); the ~40%-low "
        "magnitude gap is expected but a wrong sign / divergence is not")


def test_ppe_ran_symmetric_solve(monkeypatch):
    """(2) The PPE provably ran — solve_linear was called with sym=True (the
    SPD pressure-Poisson sub-solve) at least once.  On the Mac CPU the PPE
    backend is splu; the resolved run swaps in gpu_cg (also routed through
    solve_linear with sym=True).  This proves the PPE Laplacian was actually
    solved — not skipped.
    """
    import diffsim.solvers.linsolve as linsolve

    calls = {"sym_true": 0, "total": 0}
    orig = linsolve.solve_linear

    def spy(A, b, solver="splu", sym=False, **kwargs):
        calls["total"] += 1
        if sym:
            calls["sym_true"] += 1
        return orig(A, b, solver=solver, sym=sym, **kwargs)

    monkeypatch.setattr(linsolve, "solve_linear", spy)

    res = run_flow_past_projection(ppe_solver="splu", verbose=False, **_SMOKE)

    print(f"\n[ppe spy] solve_linear total={calls['total']}  "
          f"sym=True={calls['sym_true']}")
    assert calls["sym_true"] >= _SMOKE["nsteps"], (
        f"PPE symmetric solve ran {calls['sym_true']} times, expected "
        f">= {_SMOKE['nsteps']} (one SPD PPE solve per step) — the PPE did "
        "not run")
    assert np.all(np.isfinite(res["cd"])), f"Cd not finite: {res['cd']}"


def test_two_sided_coupling_load_bearing():
    """(3) The two-sided shell coupling is load-bearing.

    Compares the two-sided assembly (Gamma~+ AND Gamma~-) against the
    one-sided anti-vacuity variant (drop Gamma~+, fold sfm onto both sides).
    If the + side were inert the forces would coincide; they must differ
    materially.
    """
    kw = dict(ppe_solver="splu", verbose=False, **_SMOKE)
    res_two = run_flow_past_projection(**kw)
    res_one = run_flow_past_projection_one_sided(**kw)

    cd_two = float(res_two["cd"][-1])
    cd_one = float(res_one["cd"][-1])
    print(f"\n[load-bearing] two-sided Cd={cd_two:.4f}  "
          f"one-sided(no+) Cd={cd_one:.4f}")

    assert np.isfinite(cd_two), f"two-sided Cd not finite: {cd_two}"
    assert np.isfinite(cd_one), f"one-sided Cd not finite: {cd_one}"
    rel_diff = abs(cd_two - cd_one) / max(abs(cd_two), 1e-6)
    assert rel_diff > 0.05, (
        f"two-sided coupling not load-bearing: two-sided={cd_two:.4f}, "
        f"one-sided={cd_one:.4f}, rel_diff={rel_diff:.3f} (< 5%) — the "
        "Gamma~+ side is inert")
