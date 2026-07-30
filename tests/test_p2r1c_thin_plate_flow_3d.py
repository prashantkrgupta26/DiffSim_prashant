"""P2-R1c 3-D thin-plate flow: smoke gate.

Smoke parameters: level=4, nsteps=5, dt=0.01, Re=10 (nu=0.1, U_inf=1).
Targets warp-CPU (Mac) — should complete in < 3 min.

Assertions:
  1. Cd/Cl_y/Cl_z arrays have length == nsteps.
  2. All values finite (no NaN/inf).
  3. Force nonzero: |Cd[-1]| > 0 (plate is active).
  4. O(1)-plausible drag: 0 < |Cd[-1]| < 1000.
  5. Cells excluded: n_excluded > 0 (plate intercepts the mesh).
  6. Symmetry: transverse components |Cl_y[-1]| and |Cl_z[-1]| are < |Cd[-1]|
     (streamwise force dominates; transverse ~ 0 by symmetry at short time).
  7. Load-bearing (anti-vacuity): drop Gamma~- only (one-sided) => |Cd| collapses
     to < |Cd_two_sided| (two-sided coupling is load-bearing).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from p2r1c_thin_plate_flow_3d import run_flow_past_3d, run_flow_past_3d_one_sided

pytestmark = pytest.mark.tier5

# Smoke parameters: tiny grid, few steps, fast on warp-CPU
_SMOKE = dict(
    level=4,
    nsteps=5,
    dt=0.01,
    nu=0.1,
    U_inf=1.0,
    alpha=50.0,
    plate_xc=0.375,
    plate_yc=0.5,
    plate_zc=0.5,
    plate_half_y=0.125,
    plate_half_z=0.125,
)


def test_3d_thin_plate_flow_runs_and_finite():
    """3-D BDF2 march completes end-to-end; Cd/Cl_y/Cl_z are finite and nonzero.

    Asserts the 3-D dim/shape machinery is correct: 4-component DOF layout,
    3-component force vector, and the octree properly intercepts the plate.
    """
    res = run_flow_past_3d(**_SMOKE)

    cd   = res["cd"]
    cl_y = res["cl_y"]
    cl_z = res["cl_z"]
    nsteps = _SMOKE["nsteps"]

    # 1. Correct length
    assert len(cd)   == nsteps, f"cd length {len(cd)} != nsteps {nsteps}"
    assert len(cl_y) == nsteps, f"cl_y length {len(cl_y)} != nsteps {nsteps}"
    assert len(cl_z) == nsteps, f"cl_z length {len(cl_z)} != nsteps {nsteps}"

    # 2. All finite
    assert np.all(np.isfinite(cd)),   f"Cd contains non-finite: {cd}"
    assert np.all(np.isfinite(cl_y)), f"Cl_y contains non-finite: {cl_y}"
    assert np.all(np.isfinite(cl_z)), f"Cl_z contains non-finite: {cl_z}"

    # 3. Force nonzero
    assert abs(cd[-1]) > 0, "Cd[-1] == 0, flow never felt the plate"

    # 4. O(1)-plausible
    assert abs(cd[-1]) < 1000, f"|Cd[-1]|={abs(cd[-1]):.2f} looks explosive"

    # 5. Plate is intercepted
    assert res["n_excluded"] > 0, \
        "n_excluded == 0 — FiniteSheet not intercepting the 3-D mesh?"

    # 6. Streamwise dominates transverse (symmetry: Cl_y, Cl_z should be small)
    # After only 5 BDF1/BDF2 steps from rest, transverse should be < drag
    assert abs(cl_y[-1]) <= abs(cd[-1]) + 1e-8, (
        f"Cl_y[-1]={cl_y[-1]:+.4f} dominates Cd[-1]={cd[-1]:+.4f} — symmetry broken?"
    )
    assert abs(cl_z[-1]) <= abs(cd[-1]) + 1e-8, (
        f"Cl_z[-1]={cl_z[-1]:+.4f} dominates Cd[-1]={cd[-1]:+.4f} — symmetry broken?"
    )

    print(f"\n[3d-smoke] n_excluded={res['n_excluded']}  "
          f"Cd[-1]={cd[-1]:+.4f}  Cl_y[-1]={cl_y[-1]:+.6f}  Cl_z[-1]={cl_z[-1]:+.6f}")


def test_3d_thin_plate_two_sided_loadbearing():
    """Anti-vacuity: drop Gamma~+ (one-sided only) => force collapses.

    The two-sided SBM coupling (Gamma~+ and Gamma~-) must both contribute.
    With only Gamma~- assembled the recovered force must be strictly smaller
    in magnitude than the two-sided result. Mirrors the 2-D loadbearing gate.
    """
    res_2s = run_flow_past_3d(**_SMOKE)
    res_1s = run_flow_past_3d_one_sided(**_SMOKE)

    cd_2s = res_2s["cd"][-1]
    cd_1s = res_1s["cd"][-1]

    # Two-sided drag must exceed one-sided in magnitude
    assert abs(cd_2s) > abs(cd_1s), (
        f"3-D two-sided coupling not load-bearing: "
        f"|Cd_twosided|={abs(cd_2s):.4f}  |Cd_onesided|={abs(cd_1s):.4f}"
    )
    print(f"\n[3d-loadbearing] Cd_2s={cd_2s:+.4f}  Cd_1s={cd_1s:+.4f}")


def test_mono3d_solver_routing_parity():
    kw = dict(level=3, nsteps=2, dt=0.01, nu=0.1, U_inf=1.0, verbose=False)
    res_legacy = run_flow_past_3d(**kw)
    res_routed = run_flow_past_3d(mono_solver="splu", device="cpu", **kw)
    assert np.allclose(res_legacy["cd"], res_routed["cd"], rtol=0, atol=1e-12)


def test_device_assembly_parity_cpu_3d():
    """assembly="device" matches assembly="host" on the 3-D monolithic march.

    DeviceNSAssembler volume fill + two-sided SBM face system (Af_c) via
    extra_matrix/extra_rhs device slots + strong rows on device must produce
    bit-for-bit identical Cd to the host LIL-surgery path.
    """
    from p2r1c_thin_plate_flow_3d import run_flow_past_3d
    kw = dict(level=3, nsteps=2, dt=0.01, nu=0.1, verbose=False)
    res_h = run_flow_past_3d(assembly="host", **kw)
    res_d = run_flow_past_3d(assembly="device", **kw)
    assert np.allclose(res_h["cd"], res_d["cd"], rtol=1e-9, atol=1e-11), (
        f"device-assembly 3-D Cd diverged: {res_h['cd']} vs {res_d['cd']}")


def test_device_assembly_parity_cpu_3d_adaptive():
    """assembly="device" works on the adaptive 3-D mesh and matches assembly="host".

    Mirrors test_device_assembly_parity_cpu_adaptive (2-D): confirms the
    constraint-aware device pattern covers all Af_c entries on a hanging-node
    mesh.  Uses level=3, refine_to=4, nsteps=2 — runs in ~2s on Mac CPU.
    """
    from p2r1c_thin_plate_flow_3d import run_flow_past_3d
    kw = dict(level=3, refine_to=4, nsteps=2, dt=0.01, nu=0.1, verbose=False)
    res_h = run_flow_past_3d(assembly="host", **kw)
    res_d = run_flow_past_3d(assembly="device", **kw)
    assert np.allclose(res_h["cd"], res_d["cd"], rtol=1e-9, atol=1e-11), (
        f"device-assembly 3-D adaptive Cd diverged: {res_h['cd']} vs {res_d['cd']}")


def test_device_csr_handoff_parity_3d():
    """W2c: SADDLE_DEVICE_CSR=1 (device-resident CSR handoff) is bit-for-bit
    identical to the default host-CSR pull on the 3-D device-assembly
    fgmres_bdiag path.  The 3-D driver is the path that ships to the GH200; the
    handoff only relocates WHERE vals_d lives (device vs host round-trip), so
    the drag trajectory must match to 1e-14.
    """
    from p2r1c_thin_plate_flow_3d import run_flow_past_3d
    kw = dict(level=3, nsteps=2, dt=0.01, nu=0.1, verbose=False,
              mono_solver="fgmres_bdiag", device="cpu", assembly="device")
    _prev = os.environ.get("SADDLE_DEVICE_CSR")
    try:
        os.environ["SADDLE_DEVICE_CSR"] = "0"
        res_ref = run_flow_past_3d(**kw)
        os.environ["SADDLE_DEVICE_CSR"] = "1"
        res_dev = run_flow_past_3d(**kw)
    finally:
        if _prev is None:
            os.environ.pop("SADDLE_DEVICE_CSR", None)
        else:
            os.environ["SADDLE_DEVICE_CSR"] = _prev
    assert np.allclose(res_ref["cd"], res_dev["cd"], rtol=0, atol=1e-14), (
        f"SADDLE_DEVICE_CSR changed 3-D cd trajectory: "
        f"{res_ref['cd']} (host) vs {res_dev['cd']} (device)")
