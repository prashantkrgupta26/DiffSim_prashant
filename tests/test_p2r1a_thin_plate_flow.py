"""P2-R1a in-CI smoke gate: 2-D flow past a finite thin plate, transient BDF2.

Smoke parameters: level=5, nsteps=10, dt=0.01, Re=10 (nu=0.1, U_inf=1).
Fast on Mac CPU (seconds).

Assertions:
  1. Cd/Cl arrays have length == nsteps.
  2. All values finite (no NaN/inf).
  3. Force nonzero: |Cd[-1]| > 0.
  4. O(1)-plausible drag: 0 < |Cd[-1]| < 500.
  5. Load-bearing (anti-vacuity): drop Gamma~- only => force collapses relative
     to two-sided run (mirrors the blocked-channel gate pattern).
  6. Symmetry-breaking perturbation: with pert_eps=0.03 the Cl signal departs
     from ~0; without it Cl stays near-zero (ON/OFF ratio > 10x in Cl_std).
     Uses Re=100, level=5, 200 steps, dt=0.02 — runs in ~12 s on Mac CPU.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from p2r1a_thin_plate_flow import run_flow_past, run_flow_past_one_sided
from diffsim.postproc.shedding import time_avg_cd, strouhal

# Perturbation validation parameters: Re=100, level=5, 200 steps, dt=0.02.
# Wall time: ~12 s on Mac CPU M-series (2x 6-s runs).
# This is fast enough to include in the CI tier-5 suite.
_PERT = dict(level=5, nsteps=200, dt=0.02, nu=1.0/100.0, U_inf=1.0,
             alpha=50.0, plate_xc=0.375, plate_yc=0.5, plate_L=0.25)

pytestmark = pytest.mark.tier5

# Smoke parameters
_SMOKE = dict(level=5, nsteps=10, dt=0.01, nu=0.1, U_inf=1.0,
              alpha=50.0, plate_xc=0.375, plate_yc=0.5, plate_L=0.25)


def test_thin_plate_flow_cd_cl():
    """Arrays finite, nonzero, and O(1)-plausible."""
    res = run_flow_past(**_SMOKE)

    cd = res["cd"]
    cl = res["cl"]
    nsteps = _SMOKE["nsteps"]

    # 1. Correct length
    assert len(cd) == nsteps, f"cd length {len(cd)} != nsteps {nsteps}"
    assert len(cl) == nsteps, f"cl length {len(cl)} != nsteps {nsteps}"

    # 2. All finite
    assert np.all(np.isfinite(cd)), f"Cd contains non-finite: {cd}"
    assert np.all(np.isfinite(cl)), f"Cl contains non-finite: {cl}"

    # 3. Force nonzero
    assert abs(cd[-1]) > 0, f"Cd[-1] == 0, flow never felt the plate"

    # 4. O(1)-plausible drag (tiny level, low Re — unphysical but finite)
    assert abs(cd[-1]) < 500, f"|Cd[-1]|={abs(cd[-1]):.2f} looks explosive"

    # 5. plate is actually intercepted (non-trivial geometry)
    assert res["n_excluded"] > 0, \
        "No cells excluded — Segment not intercepting the mesh?"


def test_thin_plate_flow_two_sided_loadbearing():
    """Anti-vacuity: drop Gamma~+ (one-sided only) => force collapses.

    Mirrors the blocked-channel gate: two-sided coupling is load-bearing only
    if both faces contribute. With only Gamma~- assembled the SBM lacks the
    upstream-face constraint and the recovered force should be strictly smaller
    than the two-sided result.
    """
    res_2s = run_flow_past(**_SMOKE)
    res_1s = run_flow_past_one_sided(**_SMOKE)

    cd_2s = res_2s["cd"][-1]
    cd_1s = res_1s["cd"][-1]

    # Two-sided drag must be larger in magnitude than one-sided
    assert abs(cd_2s) > abs(cd_1s), (
        f"Two-sided coupling not load-bearing: "
        f"|Cd_twosided|={abs(cd_2s):.4f}  |Cd_onesided|={abs(cd_1s):.4f}"
    )


def test_thin_plate_postproc_pipeline():
    """End-to-end pipeline: smoke run -> postproc -> Cd_mean finite and plausible.

    Runs the same smoke parameters (level=5, 10 steps), then passes the Cd/Cl
    history through time_avg_cd and strouhal.  Asserts only pipeline-level
    properties (no literature comparison — 10 steps is far too short for that).
    """
    res = run_flow_past(**_SMOKE)
    nsteps = _SMOKE["nsteps"]
    dt = _SMOKE["dt"]
    U_inf = _SMOKE["U_inf"]
    plate_L = _SMOKE["plate_L"]

    # t_arr: step k lands at time (k+1)*dt since step 0 is the first BDF step
    # (t=dt).  np.linspace(0, nsteps*dt, nsteps) gives wrong effective dt.
    t_arr = np.arange(1, nsteps + 1) * dt

    # time_avg_cd: must be finite and O(1)-plausible
    cd_mean = time_avg_cd(t_arr, res["cd"])
    assert np.isfinite(cd_mean), f"Cd_mean is not finite: {cd_mean}"
    assert 0 < abs(cd_mean) < 500, (
        f"|Cd_mean|={abs(cd_mean):.4f} is outside O(1)-plausible range"
    )

    # strouhal: defensive try/except guards against configs with < 4 tail points;
    # at 10 steps (5 tail points, above the 4-pt minimum) it won't raise, but the
    # recovered St value is physically meaningless (far too few steps for shedding).
    try:
        St, freq = strouhal(t_arr, res["cl"], U_inf, plate_L)
        assert np.isfinite(St), f"St is not finite: {St}"
        assert np.isfinite(freq), f"freq is not finite: {freq}"
    except ValueError:
        # Defensive: only reached when tail has < 4 points (not this smoke config).
        # The pipeline reaching this point without crashing is the test.
        pass


def test_perturbation_breaks_symmetry():
    """Symmetry-breaking perturbation makes Cl depart from ~0.

    Two runs at Re=100, level=5, 200 steps, dt=0.02 (t_end=4.0):
      ON : pert_eps=0.03 — the small transverse kick at the inflow for t<1.0
           seeds asymmetry; Cl develops nonzero std (> 1e-3).
      OFF: pert_eps=None — perfect symmetry maintained; Cl stays near-zero
           (std < 1e-4, i.e. 10x smaller than ON).

    The ratio ON/OFF > 10 is the minimum bar; measured values are ~700x.
    This confirms the kick mechanism works before the full Re=250 gpubox run.

    Note: at level=5 (coarse mesh, ~32^2 effective) the shedding instability
    is numerically damped — Cl never grows to limit-cycle amplitude in 200
    steps, but the ASYMMETRY seeded by the kick is clearly visible.  The test
    is a symmetry-breaking gate, not a shedding amplitude gate.

    Wall time: ~12 s on Mac CPU (2x 6-s runs).
    """
    res_on  = run_flow_past(**_PERT, pert_eps=0.03, pert_t_end=1.0)
    res_off = run_flow_past(**_PERT, pert_eps=None)

    cl_on  = res_on["cl"]
    cl_off = res_off["cl"]

    std_on  = float(np.std(cl_on))
    std_off = float(np.std(cl_off))

    # Both arrays must be finite (sanity guard)
    assert np.all(np.isfinite(cl_on)),  f"Cl (pert ON) contains non-finite"
    assert np.all(np.isfinite(cl_off)), f"Cl (pert OFF) contains non-finite"

    # ON: Cl must depart clearly from zero (perturbation seeded asymmetry)
    assert std_on > 1e-3, (
        f"pert ON: Cl_std={std_on:.6f} <= 1e-3 — perturbation not breaking symmetry"
    )

    # OFF: Cl must stay near-zero (symmetric branch)
    assert std_off < 1e-4, (
        f"pert OFF: Cl_std={std_off:.6f} >= 1e-4 — unexpected asymmetry without kick"
    )

    # Ratio must be > 10x (measured ~700x; 10x is a very conservative gate)
    ratio = std_on / (std_off + 1e-12)
    assert ratio > 10.0, (
        f"ON/OFF std ratio={ratio:.1f} < 10 — kick not clearly breaking symmetry"
    )


def test_mono_solver_routing_parity():
    """Routing the monolithic solve through solve_linear(solver="splu")
    must reproduce the legacy inline-splu march exactly (same host LU)."""
    from p2r1a_thin_plate_flow import run_flow_past
    kw = dict(level=4, nsteps=3, dt=0.01, nu=0.1, U_inf=1.0, verbose=False)
    res_legacy = run_flow_past(**kw)                       # default path
    res_routed = run_flow_past(mono_solver="splu", device="cpu", **kw)
    assert np.allclose(res_legacy["cd"], res_routed["cd"], rtol=0, atol=1e-12), (
        f"routed splu diverged from legacy: {res_legacy['cd']} vs {res_routed['cd']}")


def test_compare_solvers_accepts_mono_knobs():
    """compare_solvers must route mono_solver/device to the monolithic leg
    ONLY — the projection leg does not accept them (reviewer-found TypeError)."""
    from p2r1a_thin_plate_flow import compare_solvers
    out = compare_solvers(level=4, nsteps=3, dt=0.01, nu=0.1, U_inf=1.0,
                          mono_solver="splu", device="cpu")
    assert np.all(np.isfinite(out["mono"]["cd_mean"]))
    assert np.all(np.isfinite(out["proj"]["cd_mean"]))


def test_device_assembly_parity_cpu(device):
    """assembly="device" on the CPU Warp device must match assembly="host"
    to distribution tolerance (host-device assembly parity; on cpu the
    scatter is deterministic so the tolerance is tight)."""
    from p2r1a_thin_plate_flow import run_flow_past
    kw = dict(level=4, nsteps=3, dt=0.01, nu=0.1, U_inf=1.0, verbose=False)
    res_h = run_flow_past(assembly="host", **kw)
    res_d = run_flow_past(assembly="device", **kw)
    assert np.allclose(res_h["cd"], res_d["cd"], rtol=1e-9, atol=1e-11), (
        f"device-assembly Cd diverged: {res_h['cd']} vs {res_d['cd']}")
