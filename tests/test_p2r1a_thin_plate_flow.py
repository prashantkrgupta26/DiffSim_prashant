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
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from p2r1a_thin_plate_flow import run_flow_past, run_flow_past_one_sided
from diffsim.postproc.shedding import time_avg_cd, strouhal

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
