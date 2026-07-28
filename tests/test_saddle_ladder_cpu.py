"""Task A2 CPU gate: tiny iteration-ladder wiring check.

Two tests:
  1. test_ladder_cpu_gate — runs run_ladder_point on a tiny 2-D level=4, 2-step
     point with fgmres_bdiag on cpu.  Asserts: converged=True, dofs>0,
     iters_per_step is a list of length 2, iters_mean > 0.

  2. test_solver_stats_default_parity — solver_stats=None (default) must be
     byte-identical to a run with solver_stats omitted entirely.  Mirrors the
     pattern from test_p2r1a_thin_plate_flow.py::test_on_step_default_parity.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

pytestmark = pytest.mark.tier4


# ---------------------------------------------------------------------------
# 1. CPU gate: tiny level=4, 2 steps, fgmres_bdiag
# ---------------------------------------------------------------------------

def test_ladder_cpu_gate():
    """run_ladder_point on tiny 2-D level=4, 2 steps, fgmres_bdiag on cpu."""
    from gpu_saddle_ladder import run_ladder_point

    result = run_ladder_point(
        tag="cpu_gate",
        solver="fgmres_bdiag",
        dim=2,
        nsteps=2,
        level=4,
        dt=0.01,
        nu=0.1,
        U_inf=1.0,
        device="cpu",
    )

    assert result["converged"], (
        f"run_ladder_point reported converged=False: {result}"
    )
    assert result["dofs"] > 0, f"dofs={result['dofs']} must be positive"

    iters = result["iters_per_step"]
    assert isinstance(iters, list), f"iters_per_step must be a list, got {type(iters)}"
    assert len(iters) == 2, f"iters_per_step length {len(iters)} != nsteps=2"

    assert result["iters_mean"] > 0, (
        f"iters_mean={result['iters_mean']} must be positive (fgmres_bdiag should iterate)"
    )
    assert np.isfinite(result["s_per_step"]), (
        f"s_per_step={result['s_per_step']} is not finite"
    )


# ---------------------------------------------------------------------------
# 2. Parity: solver_stats=None default is byte-identical to omitting it
# ---------------------------------------------------------------------------

def test_solver_stats_default_parity():
    """solver_stats=None must be byte-identical to the legacy march (no kwarg).

    Mirrors test_on_step_default_parity from test_p2r1a_thin_plate_flow.py.
    Runs the 2-D driver at tiny level=3, 2 steps so the test is fast.
    """
    from p2r1a_thin_plate_flow import run_flow_past

    kw = dict(level=3, nsteps=2, dt=0.01, nu=0.1, U_inf=1.0, verbose=False)

    # Legacy path: no solver_stats kwarg at all
    res_a = run_flow_past(**kw)

    # Explicit None: must be bit-for-bit identical
    res_b = run_flow_past(solver_stats=None, **kw)

    assert np.allclose(res_a["cd"], res_b["cd"], rtol=0, atol=1e-14), (
        f"solver_stats=None changed cd: {res_a['cd']} vs {res_b['cd']}"
    )

    # solver_stats key must NOT be present in the result
    assert "solver_stats" not in res_b, (
        "solver_stats key unexpectedly present in result when solver_stats=None"
    )


def test_solver_stats_3d_default_parity():
    """solver_stats=None must be byte-identical to the 3-D legacy march.

    Mirrors test_mono3d_solver_routing_parity from test_p2r1c_thin_plate_flow_3d.py.
    """
    from p2r1c_thin_plate_flow_3d import run_flow_past_3d

    kw = dict(level=3, nsteps=2, dt=0.01, nu=0.1, U_inf=1.0, verbose=False)

    res_a = run_flow_past_3d(**kw)
    res_b = run_flow_past_3d(solver_stats=None, **kw)

    assert np.allclose(res_a["cd"], res_b["cd"], rtol=0, atol=1e-14), (
        f"3-D solver_stats=None changed cd: {res_a['cd']} vs {res_b['cd']}"
    )
    assert "solver_stats" not in res_b, (
        "solver_stats key unexpectedly present in 3-D result when solver_stats=None"
    )


def test_solver_stats_2d_collects_iters():
    """solver_stats list collects per-step iterations when using fgmres_bdiag."""
    from p2r1a_thin_plate_flow import run_flow_past

    stats = []
    res = run_flow_past(
        level=4, nsteps=2, dt=0.01, nu=0.1, U_inf=1.0, verbose=False,
        mono_solver="fgmres_bdiag", device="cpu",
        solver_stats=stats,
    )

    assert len(stats) == 2, f"solver_stats length {len(stats)} != nsteps=2"
    assert all(isinstance(s, int) and s > 0 for s in stats), (
        f"solver_stats entries must be positive ints: {stats}"
    )
    assert np.all(np.isfinite(res["cd"])), "cd not finite with fgmres_bdiag"


# ---------------------------------------------------------------------------
# 3. T3: assembly="device" pass-through via run_ladder_point
# ---------------------------------------------------------------------------

def test_ladder_device_assembly_cpu():
    """run_ladder_point with assembly="device" on cpu device — device assembly
    stack runs on "cpu" Warp device — asserts finite result and iterations
    recorded.  The device-assembly parity is already proven by
    test_device_assembly.py; this test gates the LADDER PASS-THROUGH.

    Two sub-checks:
    (a) assembly kwarg is actually forwarded to the driver (wraps run_flow_past
        with unittest.mock.patch to inspect the call).
    (b) The full run converges with iters recorded.
    """
    import unittest.mock as mock
    import p2r1a_thin_plate_flow as _drv
    from gpu_saddle_ladder import run_ladder_point

    # (a) Verify the kwarg is forwarded ----------------------------------------
    _orig = _drv.run_flow_past
    received_assembly = []

    def _spy(**kw):
        received_assembly.append(kw.get("assembly", "<NOT PASSED>"))
        return _orig(**kw)

    with mock.patch.object(_drv, "run_flow_past", side_effect=_spy):
        run_ladder_point(
            tag="spy_check",
            solver="fgmres_bdiag",
            dim=2,
            nsteps=1,
            level=3,
            dt=0.01,
            nu=0.1,
            U_inf=1.0,
            device="cpu",
            assembly="device",
        )

    assert received_assembly == ["device"], (
        f"assembly='device' was not forwarded to run_flow_past; "
        f"received: {received_assembly}"
    )

    # (b) Full run: finite result + iterations recorded -------------------------
    result = run_ladder_point(
        tag="cpu_device_asm",
        solver="fgmres_bdiag",
        dim=2,
        nsteps=2,
        level=4,
        dt=0.01,
        nu=0.1,
        U_inf=1.0,
        device="cpu",
        assembly="device",
    )

    assert result["converged"], (
        f"device-assembly ladder point reported converged=False: {result}"
    )
    assert result["dofs"] > 0, f"dofs={result['dofs']} must be positive"

    iters = result["iters_per_step"]
    assert isinstance(iters, list), f"iters_per_step must be a list, got {type(iters)}"
    assert len(iters) == 2, f"iters_per_step length {len(iters)} != nsteps=2"
    assert result["iters_mean"] > 0, (
        f"iters_mean={result['iters_mean']} must be positive (fgmres_bdiag should iterate)"
    )
    assert np.isfinite(result["s_per_step"]), (
        f"s_per_step={result['s_per_step']} is not finite"
    )


def test_ladder_assembly_none_parity():
    """run_ladder_point with assembly=None (explicit) is byte-identical to
    omitting assembly entirely — the default-None contract: pass-through must
    not silently change the host path.

    Two sub-checks:
    (a) assembly kwarg is NOT forwarded to the driver when None (host default
        preserved exactly — no surprise kwarg injection).
    (b) cd_last and iters_per_step are byte-identical to the no-kwarg reference.

    Mirrors test_solver_stats_default_parity and test_on_step_default_parity.
    """
    import unittest.mock as mock
    import p2r1a_thin_plate_flow as _drv
    from gpu_saddle_ladder import run_ladder_point

    # (a) Verify assembly is NOT forwarded when None ----------------------------
    _orig = _drv.run_flow_past
    received_assembly = []

    def _spy(**kw):
        received_assembly.append(kw.get("assembly", "<NOT PASSED>"))
        return _orig(**kw)

    with mock.patch.object(_drv, "run_flow_past", side_effect=_spy):
        run_ladder_point(
            tag="spy_none",
            solver="fgmres_bdiag",
            dim=2,
            nsteps=1,
            level=3,
            dt=0.01,
            nu=0.1,
            U_inf=1.0,
            device="cpu",
            assembly=None,
        )

    assert received_assembly == ["<NOT PASSED>"], (
        f"assembly=None should not be forwarded, but driver received: {received_assembly}"
    )

    # (b) Result parity: no-kwarg vs assembly=None ----------------------------
    kw = dict(
        solver="fgmres_bdiag",
        dim=2,
        nsteps=2,
        level=3,
        dt=0.01,
        nu=0.1,
        U_inf=1.0,
        device="cpu",
    )

    # Reference: no assembly kwarg at all (today's behavior)
    res_ref = run_ladder_point(tag="parity_ref", **kw)

    # Explicit assembly=None: must produce identical cd trajectory
    res_none = run_ladder_point(tag="parity_none", assembly=None, **kw)

    assert np.allclose(res_ref["cd_last"], res_none["cd_last"], rtol=0, atol=1e-14), (
        f"assembly=None changed cd_last: {res_ref['cd_last']} vs {res_none['cd_last']}"
    )
    assert res_ref["iters_per_step"] == res_none["iters_per_step"], (
        f"assembly=None changed iters_per_step: "
        f"{res_ref['iters_per_step']} vs {res_none['iters_per_step']}"
    )
