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


# ---------------------------------------------------------------------------
# 3b. A3-parity: saddle_restart=None (default) and saddle_x0=None (default)
#     are byte-identical to omitting the kwarg entirely.
# ---------------------------------------------------------------------------

def test_ladder_saddle_restart_none_parity():
    """saddle_restart=None must produce the same results as omitting it entirely.

    Mirrors test_solver_stats_default_parity and test_ladder_assembly_none_parity:
    None means "don't pass the kwarg to the driver" — the driver's own default
    (restart=60) should be used in both cases, giving bit-for-bit identical
    iteration lists and cd_last.

    Runs tiny cpu ladder point (level=4, 2 steps, fgmres_bdiag) twice.
    """
    from gpu_saddle_ladder import run_ladder_point

    kw = dict(
        solver="fgmres_bdiag",
        dim=2,
        nsteps=2,
        level=4,
        dt=0.01,
        nu=0.1,
        U_inf=1.0,
        device="cpu",
    )

    # Reference: no saddle_restart kwarg at all
    res_ref = run_ladder_point(tag="restart_parity_ref", **kw)

    # Explicit None: must produce identical results
    res_none = run_ladder_point(tag="restart_parity_none", saddle_restart=None, **kw)

    assert np.allclose(res_ref["cd_last"], res_none["cd_last"], rtol=0, atol=1e-14), (
        f"saddle_restart=None changed cd_last: "
        f"{res_ref['cd_last']} vs {res_none['cd_last']}"
    )
    assert res_ref["iters_per_step"] == res_none["iters_per_step"], (
        f"saddle_restart=None changed iters_per_step: "
        f"{res_ref['iters_per_step']} vs {res_none['iters_per_step']}"
    )


def test_ladder_saddle_x0_none_parity():
    """saddle_x0=None must produce the same results as omitting it entirely.

    Mirrors test_ladder_saddle_restart_none_parity: None means cold-start
    (the driver's own default), so both paths must be byte-identical.

    Runs tiny cpu ladder point (level=4, 2 steps, fgmres_bdiag) twice.
    """
    from gpu_saddle_ladder import run_ladder_point

    kw = dict(
        solver="fgmres_bdiag",
        dim=2,
        nsteps=2,
        level=4,
        dt=0.01,
        nu=0.1,
        U_inf=1.0,
        device="cpu",
    )

    # Reference: no saddle_x0 kwarg at all
    res_ref = run_ladder_point(tag="x0_parity_ref", **kw)

    # Explicit None: must produce identical results
    res_none = run_ladder_point(tag="x0_parity_none", saddle_x0=None, **kw)

    assert np.allclose(res_ref["cd_last"], res_none["cd_last"], rtol=0, atol=1e-14), (
        f"saddle_x0=None changed cd_last: "
        f"{res_ref['cd_last']} vs {res_none['cd_last']}"
    )
    assert res_ref["iters_per_step"] == res_none["iters_per_step"], (
        f"saddle_x0=None changed iters_per_step: "
        f"{res_ref['iters_per_step']} vs {res_none['iters_per_step']}"
    )


# ---------------------------------------------------------------------------
# 4. T3: 3-D assembly pass-through spy-mock verification
# ---------------------------------------------------------------------------

def test_ladder_assembly_passthrough_3d():
    """run_ladder_point dim=3 with assembly="device" pass-through — mock-only
    spy verification that the assembly kwarg is forwarded to run_flow_past_3d
    (the 3-D driver), and that assembly=None is NOT forwarded.

    Mirrors test_ladder_device_assembly_cpu sub-check (a) but for 3-D,
    patching run_flow_past_3d at the ladder module's namespace.

    Two sub-checks:
    (a) assembly="device" is forwarded to run_flow_past_3d.
    (b) assembly omitted (no kwarg) means NO assembly key in driver call.
    """
    import unittest.mock as mock
    import p2r1c_thin_plate_flow_3d as _drv_3d
    from gpu_saddle_ladder import run_ladder_point

    # (a) Verify assembly="device" is forwarded to run_flow_past_3d -----------
    _orig_3d = _drv_3d.run_flow_past_3d
    received_calls_device = []

    def _spy_device(**kw):
        received_calls_device.append(kw.get("assembly", "<NOT PASSED>"))
        # Return minimal structure that run_ladder_point needs
        return {
            "cd": np.array([1.0, 2.0]),  # nsteps=1, so one value suffices; ladder only checks length
        }

    with mock.patch.object(_drv_3d, "run_flow_past_3d", side_effect=_spy_device):
        run_ladder_point(
            tag="t3d_device",
            solver="fgmres_bdiag",
            dim=3,
            nsteps=1,
            level=3,
            dt=0.01,
            nu=0.1,
            U_inf=1.0,
            device="cpu",
            assembly="device",
        )

    assert received_calls_device == ["device"], (
        f"assembly='device' was not forwarded to run_flow_past_3d (3-D); "
        f"received: {received_calls_device}"
    )

    # (b) Verify assembly is NOT forwarded when omitted (default) ---------------
    received_calls_none = []

    def _spy_none(**kw):
        received_calls_none.append(kw.get("assembly", "<NOT PASSED>"))
        return {
            "cd": np.array([1.0, 2.0]),
        }

    with mock.patch.object(_drv_3d, "run_flow_past_3d", side_effect=_spy_none):
        run_ladder_point(
            tag="t3d_none",
            solver="fgmres_bdiag",
            dim=3,
            nsteps=1,
            level=3,
            dt=0.01,
            nu=0.1,
            U_inf=1.0,
            device="cpu",
            # assembly omitted (default behavior)
        )

    assert received_calls_none == ["<NOT PASSED>"], (
        f"assembly should not be forwarded when omitted, but received: {received_calls_none}"
    )


# ---------------------------------------------------------------------------
# 5. T5: pcd_ap_inner pass-through spy-mock verification
# ---------------------------------------------------------------------------

def test_ladder_pcd_ap_inner_passthrough():
    """run_ladder_point with pcd_ap_inner="amgx" pass-through — mock-only spy
    verification that the kwarg is forwarded to run_flow_past (2-D driver),
    and that pcd_ap_inner=None is NOT forwarded.

    Mirrors test_ladder_device_assembly_cpu sub-check (a) exactly, but for
    the T5 pcd_ap_inner kwarg.
    """
    import unittest.mock as mock
    import p2r1a_thin_plate_flow as _drv
    from gpu_saddle_ladder import run_ladder_point

    # (a) Verify pcd_ap_inner="amgx" is forwarded to run_flow_past ------------
    received_ap = []

    def _spy_ap(**kw):
        received_ap.append(kw.get("pcd_ap_inner", "<NOT PASSED>"))
        return {"cd": np.array([1.0, 1.0])}

    with mock.patch.object(_drv, "run_flow_past", side_effect=_spy_ap):
        run_ladder_point(
            tag="spy_ap",
            solver="fgmres_pcd",
            dim=2,
            nsteps=1,
            level=3,
            dt=0.01,
            nu=0.1,
            U_inf=1.0,
            device="cpu",
            pcd_ap_inner="amgx",
        )

    assert received_ap == ["amgx"], (
        f"pcd_ap_inner='amgx' was not forwarded to run_flow_past; "
        f"received: {received_ap}"
    )

    # (b) Verify pcd_ap_inner is NOT forwarded when None (default) ------------
    received_none = []

    def _spy_none(**kw):
        received_none.append(kw.get("pcd_ap_inner", "<NOT PASSED>"))
        return {"cd": np.array([1.0, 1.0])}

    with mock.patch.object(_drv, "run_flow_past", side_effect=_spy_none):
        run_ladder_point(
            tag="spy_ap_none",
            solver="fgmres_pcd",
            dim=2,
            nsteps=1,
            level=3,
            dt=0.01,
            nu=0.1,
            U_inf=1.0,
            device="cpu",
            # pcd_ap_inner omitted (default behavior)
        )

    assert received_none == ["<NOT PASSED>"], (
        f"pcd_ap_inner should not be forwarded when omitted; "
        f"received: {received_none}"
    )


# ---------------------------------------------------------------------------
# 5b. W2c: SADDLE_DEVICE_CSR device-resident handoff parity
# ---------------------------------------------------------------------------

def test_device_csr_handoff_parity():
    """W2c: SADDLE_DEVICE_CSR=1 (device-resident CSR handoff) must be
    bit-for-bit identical to the default (host-CSR pull) on the device-assembly
    fgmres_bdiag path — same per-step iterations and same cd trajectory to
    1e-14.  The opt-in only changes WHERE the assembled values live (device vs
    host); the arithmetic (SpMV, scalar block-Jacobi diagonal) is identical.

    Runs the tiny 2-D level=4 / 2-step device-assembly point on the "cpu" Warp
    device twice, toggling only the SADDLE_DEVICE_CSR env var.
    """
    from p2r1a_thin_plate_flow import run_flow_past

    kw = dict(
        level=4, nsteps=2, dt=0.01, nu=0.1, U_inf=1.0, verbose=False,
        mono_solver="fgmres_bdiag", device="cpu", assembly="device",
    )

    _prev = os.environ.get("SADDLE_DEVICE_CSR")
    try:
        # Default: host-CSR pull path
        os.environ["SADDLE_DEVICE_CSR"] = "0"
        stats_ref = []
        res_ref = run_flow_past(solver_stats=stats_ref, **kw)

        # Opt-in: device-resident handoff
        os.environ["SADDLE_DEVICE_CSR"] = "1"
        stats_dev = []
        res_dev = run_flow_past(solver_stats=stats_dev, **kw)
    finally:
        if _prev is None:
            os.environ.pop("SADDLE_DEVICE_CSR", None)
        else:
            os.environ["SADDLE_DEVICE_CSR"] = _prev

    # Identical iteration counts (the preconditioner + operator are the same).
    assert stats_ref == stats_dev, (
        f"SADDLE_DEVICE_CSR changed per-step iterations: "
        f"{stats_ref} (host) vs {stats_dev} (device)"
    )
    # Identical drag trajectory to 1e-14.
    assert np.allclose(res_ref["cd"], res_dev["cd"], rtol=0, atol=1e-14), (
        f"SADDLE_DEVICE_CSR changed cd trajectory: "
        f"{res_ref['cd']} (host) vs {res_dev['cd']} (device)"
    )


# ---------------------------------------------------------------------------
# 6. 3d-L7 uniform ladder point (GH200 hold session: identity_T device-assembly
#    leg at ~8.6M DOF — separates device-assembly-at-scale from the
#    constrained-scatter Warp 2^31 blocker that hit 3d-L7r9)
# ---------------------------------------------------------------------------

def test_ladder_point_3d_l7_uniform():
    """ALL_POINTS must contain a uniform 3d-L7 point (level=7, refine_to=None).

    The point must sit between 3d-L6 and 3d-L7r9 in ladder order, share the
    Phase-2 3-D physics constants (nu=0.004, dt=0.005, U_inf=1.0), and be
    uniform (refine_to=None) so that identity_T holds and the device-assembly
    ChunkTable path is usable at this scale.
    """
    from gpu_saddle_ladder import ALL_POINTS

    tags = [p["tag"] for p in ALL_POINTS]
    assert "3d-L7" in tags, f"3d-L7 missing from ALL_POINTS tags {tags}"

    p = ALL_POINTS[tags.index("3d-L7")]
    assert p["dim"] == 3
    assert p["level"] == 7
    assert p["refine_to"] is None, "3d-L7 must be UNIFORM (identity_T path)"
    assert p["nsteps"] == 5
    assert p["dt"] == 0.005
    assert p["nu"] == 0.004
    assert p["U_inf"] == 1.0

    # Ladder ordering: monotone problem-size ordering L6 < L7 < L7r9
    assert tags.index("3d-L6") < tags.index("3d-L7") < tags.index("3d-L7r9"), (
        f"3d-L7 must sit between 3d-L6 and 3d-L7r9 in ALL_POINTS order: {tags}"
    )


def test_ladder_point_3d_l8_uniform_capacity():
    """ALL_POINTS must contain a uniform 3d-L8 capacity rung (level=8).

    ~68M-DOF capacity probe point (257^3 nodes x 4 dof/node) for the GH200
    uniform-mesh device-assembly route.  nsteps=2 (capacity probe semantics,
    not a 5-step ladder measurement).  Must be last in ladder order.
    """
    from gpu_saddle_ladder import ALL_POINTS

    tags = [p["tag"] for p in ALL_POINTS]
    assert "3d-L8" in tags, f"3d-L8 missing from ALL_POINTS tags {tags}"

    p = ALL_POINTS[tags.index("3d-L8")]
    assert p["dim"] == 3
    assert p["level"] == 8
    assert p["refine_to"] is None, "3d-L8 must be UNIFORM (identity_T path)"
    assert p["nsteps"] == 2, "capacity rung defaults to 2 steps"
    assert p["dt"] == 0.005
    assert p["nu"] == 0.004

    assert tags.index("3d-L8") == len(tags) - 1, (
        f"3d-L8 must be the last (largest) ladder point: {tags}"
    )
