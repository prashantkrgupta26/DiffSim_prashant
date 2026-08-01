"""CPU-scale gates for the TRUCK case (config loader + volumetric SBM driver).

Three test groups (per task TU2 brief):
  1. loader round-trip on the REAL config.txt (22 active geometries, 2 region
     boxes, unknown-keys warning list matches expectation).
  2. tiny truck-in-a-box (level 5, ONE tire STL as the body): mesh builds,
     slab carve exact (zero intercepted slab cells — dyadic boundaries), march
     3 steps finite with BOTH force observables recorded.
  3. BC masks: node counts on each dyadic plane > 0 and pairwise-disjoint.

All CPU, seconds-scale.  Run:  .venv/bin/pytest tests/test_truck_flow.py
"""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from diffsim.cases.truck_config import load_truck_config, TruckConfig

# Detect CUDA availability for device-parity legs (skip on this Mac dev loop).
try:
    import warp as _wp
    _wp.init()
    _HAS_CUDA = _wp.get_cuda_device_count() > 0
except Exception:
    _HAS_CUDA = False

_CFG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "local_code_old",
    "truck_4case_fresh_inputs", "NewRun-no-shell-slope0p25", "config.txt")


def _tiny_tire_mesh(cfg, half=0.05, center=(0.35, 0.0625, 0.0625)):
    """ONE tire STL, recentered + enlarged into a resolvable box in the channel
    slab (the tiny-gate body).  Native tire is sub-cell at CI levels, so it is
    normalized to [-1,1] and scaled to ``half`` about ``center`` (all in
    unit-cube coords)."""
    from diffsim.geometry.merged_trimesh import MergedTriMesh, _read_stl
    v, t = _read_stl(os.path.join(cfg.config_dir, "tire_1.stl"))
    v = v - v.mean(0)
    v = v / np.abs(v).max()
    v = v * half + np.asarray(center, np.float64)
    return MergedTriMesh(v, t)


# ---------------------------------------------------------------------------
# Group 1: loader round-trip on the REAL config
# ---------------------------------------------------------------------------

def test_loader_roundtrip_real_config():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = load_truck_config(_CFG_PATH)
    assert isinstance(cfg, TruckConfig)
    # 22 ACTIVE bodies (Truck.stl + covers + details are commented out)
    assert len(cfg.bodies) == 22
    names = [b.mesh_path for b in cfg.bodies]
    assert "truck-head-new.stl" in names
    assert "Truck.stl" not in names          # commented -> not active
    assert "cover_1.stl" not in names        # commented -> not active
    # every active body carries the config placement + per-object level
    for b in cfg.bodies:
        assert b.position == (0.0, -0.002, -7.0)
        assert b.refine_lvl == 12
    # region refine boxes
    assert len(cfg.region_refine) == 2
    assert cfg.region_refine[0].refine_region_lvl == 12
    assert cfg.region_refine[1].refine_region_lvl == 10
    # scalars
    assert cfg.sbm_geo == "TRUCK"
    assert cfg.slope_near_ground == 0.25
    assert cfg.domain_max == (16, 2, 2)
    assert abs(cfg.domain_scale - 1.0 / 16.0) < 1e-15
    assert cfg.cb_f == 20.0
    assert cfg.refine_lvl_base == 7
    assert cfg.do_re_solver_ramp is True
    # exactly one warning, listing the ignored keys
    assert len(w) == 1
    assert issubclass(w[0].category, UserWarning)


def test_loader_unknown_keys_expected():
    """The warn-and-ignore set is pinned so the driver's consumed set is a
    documented contract (regressions surface as a diff here)."""
    cfg = load_truck_config(_CFG_PATH)
    expected_unknown = {
        "AverageOutput", "AverageOutputInterval", "Cb_e", "Cb_f_bulk",
        "CheckpointInterval", "CheckpointNumbackup", "Ci_e", "G_dir",
        "Gr_V", "Gr_ramping", "HeatSolverType", "HeatTimestepper",
        "NavierStokesSolverType", "OutputStartTime", "Pe_V", "Pe_ramping",
        "PostProcessingInterval", "ReSolverRampMaxEffort", "ReSolverRampOutput",
        "ReSolverRampReduceOnStruggle", "ReSolverRampReduction",
        "ReSolverRampStableSteps", "RemoveInterceptedForShell_CompleteOctree",
        "SolveHT", "SolveNS", "blockTolerance", "ifInter",
        "initial_condition", "iterMaxBlock", "solver_options_ht",
        "solver_options_ns", "thetaTimeStepping",
    }
    assert set(cfg.unknown_keys) == expected_unknown


# ---------------------------------------------------------------------------
# Group 2: tiny truck-in-a-box — mesh build, dyadic slab carve, 3-step march
# ---------------------------------------------------------------------------

def test_slab_carve_dyadic_exact():
    """The channel slab bounds (y=z=1/8) are dyadic => NO cell is cut; the
    carve retains exactly the interior slab with zero intercepted cells."""
    from diffsim.cases.truck import _channel_box, slab_carve
    from diffsim.octree.build import build_uniform
    cfg = load_truck_config(_CFG_PATH)
    for lvl in (3, 4, 5):
        tree = build_uniform(lvl, dim=3)
        cbox = _channel_box(cfg.domain_min, cfg.domain_max, cfg.domain_scale)
        ret, n_cut = slab_carve(tree, cbox)
        assert n_cut == 0
        # slab is [0,1] x [0,1/8] x [0,1/8] => 1/64 of the full cube
        assert len(ret) == len(tree) // 64


def test_tiny_truck_march():
    """Tiny one-tire case: mesh builds, slab carve exact, 3 BDF steps finite,
    BOTH force observables recorded and nonzero."""
    from diffsim.cases.truck import run_truck
    cfg = load_truck_config(_CFG_PATH)
    merged = _tiny_tire_mesh(cfg)
    res = run_truck(cfg, nsteps=3, base_level=5, truck_band_to=6,
                    band_cells=2, merged=merged, region_refine=False,
                    nu=1.0 / 50.0, dt=0.02, verbose=False)
    # mesh built with a real carved body and surrogate faces
    assert res["n_slab_cut"] == 0
    assert res["n_excluded"] > 0
    assert res["sf_faces"] > 0
    assert res["n_rxn_nodes"] > 0
    # 3 steps, both observables present and finite
    for k in ("cd", "cd_surr", "cl_y", "cl_z", "cl_y_surr", "cl_z_surr"):
        assert res[k].shape == (3,)
        assert np.all(np.isfinite(res[k]))
    # forces nonzero and O(1e3)-plausible (nondim by the tiny frontal area)
    assert abs(res["cd"][-1]) > 0.0
    assert abs(res["cd_surr"][-1]) > 0.0
    assert abs(res["cd"][-1]) < 1e5
    # canonical reaction drag is positive (downstream) at the final step
    assert res["cd"][-1] > 0.0


def test_nu_schedule_re_ramp():
    """nu_schedule from Re_V/Re_ramping is a monotone-in-Re, positive nu(t)."""
    from diffsim.cases.truck import make_nu_schedule
    cfg = load_truck_config(_CFG_PATH)
    sched = make_nu_schedule(cfg, U_inf=1.0, L_ref=1.0)
    # Re_ramping=[0,50,51], Re_V=[1e3,5e3,1e4]; nu = U*L/Re
    nu0 = sched(0.0)
    nu_mid = sched(50.0)
    nu_end = sched(60.0)     # clamped to last Re=1e4
    assert nu0 > nu_mid > nu_end > 0.0
    assert abs(nu0 - 1.0 / 1e3) < 1e-9
    assert abs(nu_end - 1.0 / 1e4) < 1e-9


# ---------------------------------------------------------------------------
# Group 2b: device-resident CSR handoff parity (P1, T5)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("saddle_device_csr", ["0", "1"])
def test_truck_device_assembly_parity(saddle_device_csr, monkeypatch):
    """assembly='device' reproduces the host march's forces trajectory-tight,
    both with SADDLE_DEVICE_CSR off (host CSR pull) and on (device-resident
    handoff).  Gates the P1 device path + the device reaction arbiter
    (w^T(A_full x) - w^T(Af x) - w^T b + w^T bf) against the host
    w^T(A_vol x - b_vol).  splu solver -> the linear solve is deterministic;
    the only host/device delta is the atomic-add scatter order (~1e-12)."""
    from diffsim.cases.truck import run_truck
    monkeypatch.setenv("SADDLE_DEVICE_CSR", saddle_device_csr)
    cfg = load_truck_config(_CFG_PATH)
    merged = _tiny_tire_mesh(cfg)
    common = dict(nsteps=3, base_level=5, truck_band_to=6, band_cells=2,
                  merged=merged, region_refine=False, nu=1.0 / 50.0, dt=0.02,
                  mono_solver="splu", device="cpu", verbose=False)
    res_h = run_truck(cfg, assembly="host", **common)
    res_d = run_truck(cfg, assembly="device", **common)
    for k in ("cd", "cd_surr"):
        assert np.all(np.isfinite(res_d[k]))
        assert np.allclose(res_h[k], res_d[k], rtol=0, atol=1e-8), (
            f"{k}: host {res_h[k]} vs device {res_d[k]}")
    # reaction sign/magnitude preserved (the physics observable)
    assert res_d["cd"][-1] > 0.0


def test_truck_warm_start_reduces_iters(monkeypatch):
    """P2 (T5): saddle_x0='extrap' warm-start flows through the truck march and
    reduces the summed inner-iteration count vs a cold start (x0=0).  Gates the
    knob end-to-end on the device CSR path (fgmres_bdiag)."""
    from diffsim.cases.truck import run_truck
    from diffsim.solvers import linsolve
    monkeypatch.setenv("SADDLE_DEVICE_CSR", "1")
    cfg = load_truck_config(_CFG_PATH)
    merged = _tiny_tire_mesh(cfg)

    def _run(x0):
        iters = []
        orig = linsolve.solve_linear

        def _wrap(*a, **k):
            out = orig(*a, **k)
            if linsolve._LAST_ITERS[0] is not None:
                iters.append(int(linsolve._LAST_ITERS[0]))
            return out
        monkeypatch.setattr("diffsim.cases.truck.truck_march.solve_linear", _wrap)
        run_truck(cfg, nsteps=5, base_level=5, truck_band_to=6, band_cells=2,
                  merged=merged, region_refine=False, nu=1.0 / 50.0, dt=0.02,
                  mono_solver="fgmres_bdiag", saddle_x0=x0, device="cpu",
                  assembly="device", linsolve_tol=1e-7, verbose=False)
        monkeypatch.setattr("diffsim.cases.truck.truck_march.solve_linear", orig)
        return iters

    warm = _run("extrap")
    cold = _run(None)
    # warm-start only affects steps >= 2 (first two are cold by construction);
    # over the whole march the summed iterations must not exceed cold and
    # should be strictly fewer once extrapolation kicks in.
    assert sum(warm) < sum(cold), f"warm {warm} vs cold {cold}"


def test_truck_viz_hook_fires_in_march(tmp_path, monkeypatch):
    """P3 (T5): the viz hook fires during a (device) march and writes frames at
    viz_interval.  Runs >= 2 intervals; asserts the per-frame extracts + a .vtu
    checkpoint land on disk and the hook's written-log records >= 2 frames."""
    from diffsim.cases.truck import run_truck
    monkeypatch.setenv("SADDLE_DEVICE_CSR", "1")
    cfg = load_truck_config(_CFG_PATH)
    merged = _tiny_tire_mesh(cfg)
    res = run_truck(cfg, nsteps=4, base_level=5, truck_band_to=6, band_cells=2,
                    merged=merged, region_refine=False, nu=1.0 / 50.0, dt=0.02,
                    mono_solver="splu", device="cpu", assembly="device",
                    viz_interval=2, viz_dir=str(tmp_path),
                    viz_checkpoint_interval=4, verbose=False)
    hook = res["viz_hook"]
    assert hook is not None
    # steps 1 and 3 (0-indexed) satisfy (step+1)%2==0 -> 2 frame batches
    frame_steps = sorted({s for (_kind, s, _p) in hook.written
                          if _kind in ("q_iso", "centerline", "surface_cp")})
    assert len(frame_steps) >= 2, f"expected >=2 frame steps, got {frame_steps}"
    # a checkpoint at step 4 (meshio .vtu, no pyvista dependency)
    ckpts = list((tmp_path / "checkpoints").glob("*.vtu"))
    assert len(ckpts) >= 1, "no .vtu checkpoint written"
    # at least one per-frame extract landed on disk
    on_disk = (list((tmp_path / "frames" / "q_iso").glob("*.vtp"))
               + list((tmp_path / "frames" / "centerline").glob("*.vtp"))
               + list((tmp_path / "frames" / "surface_cp").glob("*.vtp")))
    assert len(on_disk) >= 1, "no per-frame extract written to disk"


# ---------------------------------------------------------------------------
# Group 3: BC masks — dyadic planes nonempty and pairwise disjoint (where they
# must be)
# ---------------------------------------------------------------------------

def test_bc_masks_nonempty_and_structure():
    """Each dyadic plane carries >0 free nodes; the streamwise (inflow/outlet)
    and the transverse (ground/ceiling, side_zlo/side_zhi) opposite pairs are
    disjoint; strong-BC rows/vals are consistent length."""
    from diffsim.cases.truck import (build_truck_mesh, truck_strong_bc, truck_bc_masks)
    cfg = load_truck_config(_CFG_PATH)
    merged = _tiny_tire_mesh(cfg)
    fx = build_truck_mesh(cfg, base_level=5, region_refine=False,
                          truck_band_to=6, band_cells=2, merged=merged)
    ndof, dim = 4, 3
    masks, coords = truck_bc_masks(
        fx["mesh"], fx["cons"], fx["scale"], cfg.slope_near_ground,
        cfg.domain_max)
    for name in ("inflow", "outlet", "ground", "ceiling", "side_zlo",
                 "side_zhi"):
        assert masks[name].sum() > 0, f"plane {name} empty"
    # opposite planes are disjoint (a node cannot be on both x=0 and x=1, etc.)
    assert not np.any(masks["inflow"] & masks["outlet"])
    assert not np.any(masks["ground"] & masks["ceiling"])
    assert not np.any(masks["side_zlo"] & masks["side_zhi"])

    rows, vals, masks2, coords2 = truck_strong_bc(
        fx["mesh"], fx["cons"], ndof, dim, fx["scale"],
        cfg.slope_near_ground, cfg.domain_max)
    assert len(rows) == len(vals)
    assert len(rows) == len(set(rows.tolist()))   # unique DOF rows
    # inflow ramp: u_x DOF values in [0,1], nonzero somewhere (ABL ramp)
    coords_free = fx["mesh"].node_coords[fx["cons"].free_nodes]
    infl = np.where(masks2["inflow"])[0]
    ux_rows = {int(i) * ndof + 0: k for k, i in enumerate(infl)}
    ux_vals = [vals[np.where(rows == r)[0][0]] for r in ux_rows
               if r in set(rows.tolist())]
    ux_vals = np.asarray(ux_vals)
    assert np.all(ux_vals >= -1e-12) and np.all(ux_vals <= 1.0 + 1e-12)
    assert ux_vals.max() > 0.0

    # ground no-slip: all 3 velocity components pinned to 0 on y=0
    grd = np.where(masks2["ground"])[0]
    rowset = set(rows.tolist())
    val_of = {int(r): float(v) for r, v in zip(rows, vals)}
    for i in grd:
        for c in range(3):
            r = int(i) * ndof + c
            assert r in rowset, f"ground node {i} comp {c} not constrained"
            assert val_of[r] == 0.0


# ---------------------------------------------------------------------------
# Group 4: soft-start inlet amplitude ramp (TU5R)
# ---------------------------------------------------------------------------

def test_soft_start_amp_math():
    """soft_start_amp is a clamped linear ramp: 0->1 over soft_start*dt, then 1;
    off (None or <=0) is exactly 1.0."""
    from diffsim.cases.truck import soft_start_amp
    dt = 0.02
    N = 30.0
    # off
    assert soft_start_amp(0.0, None, dt) == 1.0
    assert soft_start_amp(5.0, 0, dt) == 1.0
    assert soft_start_amp(5.0, -1, dt) == 1.0
    # ramp: at t = k*dt the amplitude is k/N (for k<N)
    assert abs(soft_start_amp(1 * dt, N, dt) - 1.0 / N) < 1e-12
    assert abs(soft_start_amp(15 * dt, N, dt) - 15.0 / N) < 1e-12
    # exactly at the ramp end -> 1.0; beyond -> clamped 1.0
    assert abs(soft_start_amp(N * dt, N, dt) - 1.0) < 1e-12
    assert soft_start_amp(100 * dt, N, dt) == 1.0
    # monotone non-decreasing
    ts = [i * dt for i in range(0, 60)]
    amps = [soft_start_amp(t, N, dt) for t in ts]
    assert all(b >= a - 1e-15 for a, b in zip(amps, amps[1:]))


def test_soft_start_off_is_byte_identical():
    """soft_start=None reproduces the default march EXACTLY (the knob is a pure
    opt-in; default path must be untouched)."""
    from diffsim.cases.truck import run_truck
    cfg = load_truck_config(_CFG_PATH)
    merged = _tiny_tire_mesh(cfg)
    common = dict(nsteps=3, base_level=5, truck_band_to=6, band_cells=2,
                  merged=merged, region_refine=False, nu=1.0 / 50.0, dt=0.02,
                  device="cpu", assembly="host", mono_solver="splu",
                  verbose=False)
    res_default = run_truck(cfg, **common)
    res_off = run_truck(cfg, soft_start=None, **common)
    for k in ("cd", "cd_surr"):
        np.testing.assert_array_equal(res_default[k], res_off[k])


def test_soft_start_reduces_startup_response():
    """A soft-start ramp shrinks the impulsive step-0 inlet drive: the early
    reaction-drag magnitude is strictly smaller than the impulsive (full-
    amplitude) start, because the inlet is only a fraction of U at step 0."""
    from diffsim.cases.truck import run_truck
    cfg = load_truck_config(_CFG_PATH)
    merged = _tiny_tire_mesh(cfg)
    common = dict(nsteps=3, base_level=5, truck_band_to=6, band_cells=2,
                  merged=merged, region_refine=False, nu=1.0 / 50.0, dt=0.02,
                  device="cpu", assembly="host", mono_solver="splu",
                  verbose=False)
    res_impulse = run_truck(cfg, **common)                 # full amplitude
    res_soft = run_truck(cfg, soft_start=30.0, **common)   # ramp over 30 dt
    # at step 0, t=dt -> amp = 1/30, so the inlet-driven cd_surr magnitude is
    # much smaller under soft-start than the impulsive start.
    assert abs(res_soft["cd_surr"][0]) < abs(res_impulse["cd_surr"][0]), (
        f"soft {res_soft['cd_surr'][0]} vs impulse "
        f"{res_impulse['cd_surr'][0]}")
    # results still finite everywhere
    assert np.all(np.isfinite(res_soft["cd"]))
    assert np.all(np.isfinite(res_soft["cd_surr"]))


# ---------------------------------------------------------------------------
# Final-review I-8: regression tests for the late-campaign march machinery.
# All tiny-tire + splu (deterministic) so bit-exactness assertions are valid.
# ---------------------------------------------------------------------------

def _tiny_kw():
    return dict(base_level=5, truck_band_to=6, band_cells=2,
                region_refine=False, nu=1.0 / 50.0, dt=0.02, verbose=False)


def test_checkpoint_resume_bit_exact(tmp_path):
    """N-step splu march == k-step + checkpoint + resume, bit-for-bit; and
    a mesh-knob change is refused loudly (I-2 guard)."""
    from test_truck_viz import _tiny_tire_mesh, _CFG_PATH
    from diffsim.cases.truck_config import load_truck_config
    from diffsim.cases.truck import run_truck
    cfg = load_truck_config(_CFG_PATH)
    kw = _tiny_kw()
    ref = run_truck(cfg, nsteps=6, merged=_tiny_tire_mesh(cfg), **kw)
    run_truck(cfg, nsteps=4, merged=_tiny_tire_mesh(cfg),
              checkpoint_interval=2, checkpoint_dir=tmp_path, **kw)
    res = run_truck(cfg, nsteps=6, merged=_tiny_tire_mesh(cfg),
                    checkpoint_interval=2, checkpoint_dir=tmp_path,
                    resume=True, **kw)
    np.testing.assert_array_equal(ref["cd"], res["cd"])
    # I-2: refusing a mesh mismatch (band 6 -> 7 changes nfree)
    kw7 = dict(kw, truck_band_to=7)
    with pytest.raises(ValueError, match="mesh mismatch"):
        run_truck(cfg, nsteps=6, merged=_tiny_tire_mesh(cfg),
                  checkpoint_interval=2, checkpoint_dir=tmp_path,
                  resume=True, **kw7)


def test_interpolate_checkpoint_identity(tmp_path):
    """A->A transfer is exact: the interpolated-resume trajectory equals the
    straight-through splu march (validates the trilinear corner ordering)."""
    from test_truck_viz import _tiny_tire_mesh, _CFG_PATH
    from diffsim.cases.truck_config import load_truck_config
    from diffsim.cases.truck import run_truck, build_truck_mesh, interpolate_checkpoint
    import pathlib
    cfg = load_truck_config(_CFG_PATH)
    kw = _tiny_kw()
    ref = run_truck(cfg, nsteps=6, merged=_tiny_tire_mesh(cfg), **kw)
    da, db = tmp_path / "a", tmp_path / "b"
    run_truck(cfg, nsteps=4, merged=_tiny_tire_mesh(cfg),
              checkpoint_interval=2, checkpoint_dir=da, **kw)
    fxA = build_truck_mesh(cfg, base_level=5, truck_band_to=6, band_cells=2,
                           region_refine=False, device="cpu",
                           merged=_tiny_tire_mesh(cfg))
    ck = sorted(pathlib.Path(da).glob("march_ckpt_*.npz"),
                key=lambda q: q.stat().st_mtime)[-1]
    interpolate_checkpoint(ck, fxA, fxA, db)
    res = run_truck(cfg, nsteps=6, merged=_tiny_tire_mesh(cfg),
                    checkpoint_interval=99, checkpoint_dir=db, resume=True,
                    **kw)
    np.testing.assert_array_equal(res["cd"], ref["cd"])


# ---------------------------------------------------------------------------
# solver-escalation Task 2: fgmres_pcd as a first-class primary solver
# ---------------------------------------------------------------------------

def test_pcd_primary_fallback_guard():
    """mono_solver='fgmres_pcd' with saddle_fallback='pcd' raises ValueError
    (PCD cannot be its own fallback).  This guard fires before any solve, so
    it does not require pyamgx to be installed."""
    from test_truck_viz import _tiny_tire_mesh, _CFG_PATH
    from diffsim.cases.truck_config import load_truck_config
    from diffsim.cases.truck import run_truck
    cfg = load_truck_config(_CFG_PATH)
    common = dict(nsteps=3, base_level=5, truck_band_to=6, band_cells=2,
                  region_refine=False, nu=1.0 / 50.0, dt=0.02, verbose=False)
    with pytest.raises(ValueError, match="fallback"):
        run_truck(cfg, merged=_tiny_tire_mesh(cfg),
                  mono_solver="fgmres_pcd", saddle_fallback="pcd", **common)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("pyamgx") is None,
    reason="pyamgx not installed — fgmres_pcd primary requires AMGX "
           "inners to converge in reasonable time on 3-D BDF2 steps")
def test_pcd_primary_march():
    """mono_solver='fgmres_pcd' marches and tracks the splu trajectory."""
    from test_truck_viz import _tiny_tire_mesh, _CFG_PATH
    from diffsim.cases.truck_config import load_truck_config
    from diffsim.cases.truck import run_truck
    cfg = load_truck_config(_CFG_PATH)
    # linsolve_tol default is 1e-10 (< 1e-8) so the 1e-5 atol assert is valid
    # without an explicit tol override.
    # pcd_f_inner/pcd_ap_inner default to "amgx" (the driver default); requires
    # pyamgx (GPU box) — this test is skipped on CPU-only / no-pyamgx machines
    # (jacobi-CG inner diverges the 3-D BDF2 saddle without AMG strength).
    common = dict(nsteps=3, base_level=5, truck_band_to=6, band_cells=2,
                  region_refine=False, nu=1.0 / 50.0, dt=0.02, verbose=False)
    ref = run_truck(cfg, merged=_tiny_tire_mesh(cfg),
                    mono_solver="splu", **common)
    res = run_truck(cfg, merged=_tiny_tire_mesh(cfg),
                    mono_solver="fgmres_pcd", **common)
    for k in ("cd", "cd_surr"):
        assert np.all(np.isfinite(res[k]))
        np.testing.assert_allclose(res[k], ref[k], rtol=0, atol=1e-5)
    # PCD-as-primary cannot also be the fallback (moved to separate guard test
    # that runs unconditionally without amgx)


def test_dt_schedule_identity_and_variable_table():
    """(a) a constant dt_schedule equals dt_schedule=None bit-for-bit;
    (b) a mid-run dt switch (variable BDF2 table) marches finite/bounded,
    and bdf_coeffs matches the analytic variable-step BDF2 table exactly."""
    from test_truck_viz import _tiny_tire_mesh, _CFG_PATH
    from diffsim.cases.truck_config import load_truck_config
    from diffsim.cases.truck import run_truck
    from diffsim.solvers.timestepping import bdf_coeffs
    cfg = load_truck_config(_CFG_PATH)
    kw = _tiny_kw()
    ref = run_truck(cfg, nsteps=5, merged=_tiny_tire_mesh(cfg), **kw)
    same = run_truck(cfg, nsteps=5, merged=_tiny_tire_mesh(cfg),
                     dt_schedule=lambda s: 0.02, **kw)
    np.testing.assert_array_equal(ref["cd"], same["cd"])
    sw = run_truck(cfg, nsteps=5, merged=_tiny_tire_mesh(cfg),
                   dt_schedule=lambda s: 0.005 if s < 2 else 0.02, **kw)
    assert np.all(np.isfinite(sw["cd"]))
    assert np.all(np.abs(sw["cd"]) < 1e4)

    # (b) Direct BDF2 table check.
    # bdf_coeffs returns (b0, b1, b2) signed as the production convention:
    # BDF2 constant-step = (1.5, -2.0, 0.5); variable-step uses
    # r = dt/dt_prev: b0=(2r+1)/(r+1), b1=-(r+1), b2=r^2/(r+1).
    # Constant-step path: both dt_prev=None and dt_prev==dt give the literal
    # (1.5, -2.0, 0.5).
    b0, b1, b2 = bdf_coeffs(2, 0.02)
    np.testing.assert_allclose((b0, b1, b2), (1.5, -2.0, 0.5), rtol=1e-15)
    b0, b1, b2 = bdf_coeffs(2, 0.02, dt_prev=0.02)
    np.testing.assert_allclose((b0, b1, b2), (1.5, -2.0, 0.5), rtol=1e-15)

    # Variable-step: check r=0.5 and r=2.0 against the analytic formula
    # derived from the same double arithmetic as the code.
    for dt, dt_prev in ((0.005, 0.010), (0.020, 0.010)):
        r = dt / dt_prev
        c0_ref = (1.0 + 2.0 * r) / (1.0 + r)
        c1_ref = -(1.0 + r)
        c2_ref = r ** 2 / (1.0 + r)
        b0, b1, b2 = bdf_coeffs(2, dt, dt_prev=dt_prev)
        np.testing.assert_allclose((b0, b1, b2), (c0_ref, c1_ref, c2_ref),
                                   rtol=1e-15,
                                   err_msg=f"r={r}: code ({b0},{b1},{b2}) vs "
                                           f"analytic ({c0_ref},{c1_ref},{c2_ref})")


def test_tau_knobs_identity_and_engagement():
    """tau_dt matching the marching dt and tau_m_scale=1.0 are bit-identical
    to the defaults; tau_m_scale=0.1 changes the trajectory (knob engages)."""
    from test_truck_viz import _tiny_tire_mesh, _CFG_PATH
    from diffsim.cases.truck_config import load_truck_config
    from diffsim.cases.truck import run_truck
    cfg = load_truck_config(_CFG_PATH)
    kw = _tiny_kw()
    ref = run_truck(cfg, nsteps=4, merged=_tiny_tire_mesh(cfg), **kw)
    same = run_truck(cfg, nsteps=4, merged=_tiny_tire_mesh(cfg),
                     tau_dt=0.02, tau_m_scale=1.0, **kw)
    np.testing.assert_array_equal(ref["cd"], same["cd"])
    scaled = run_truck(cfg, nsteps=4, merged=_tiny_tire_mesh(cfg),
                       tau_m_scale=0.1, **kw)
    assert np.all(np.isfinite(scaled["cd"]))
    assert np.max(np.abs(np.asarray(scaled["cd"])
                         - np.asarray(ref["cd"]))) > 1e-3


@pytest.mark.skipif(not _HAS_CUDA,
                    reason="tau device-parity leg requires a CUDA device")
def test_tau_scale_device_parity():
    """tau_m_scale=0.1 host-vs-device forces agree.
    Mirrors test_truck_device_assembly_parity (atol=1e-8; splu deterministic).
    Skips on this Mac dev loop (no CUDA); passes on gpubox."""
    from test_truck_viz import _tiny_tire_mesh, _CFG_PATH
    from diffsim.cases.truck_config import load_truck_config
    from diffsim.cases.truck import run_truck
    cfg = load_truck_config(_CFG_PATH)
    merged = _tiny_tire_mesh(cfg)
    common = dict(nsteps=3, base_level=5, truck_band_to=6, band_cells=2,
                  merged=merged, region_refine=False, nu=1.0 / 50.0, dt=0.02,
                  mono_solver="splu", device="cuda:0", tau_m_scale=0.1,
                  verbose=False)
    res_h = run_truck(cfg, assembly="host", **common)
    res_d = run_truck(cfg, assembly="device", **common)
    for k in ("cd", "cd_surr"):
        assert np.all(np.isfinite(res_d[k]))
        assert np.allclose(res_h[k], res_d[k], rtol=0, atol=1e-8), (
            f"tau device parity {k}: host {res_h[k]} vs device {res_d[k]}")


# ---------------------------------------------------------------------------
# solver-escalation Task 1: dump_system knob — portable saddle snapshots
# ---------------------------------------------------------------------------

def test_dump_system_snapshot(tmp_path):
    """dump_system_steps writes a self-contained solvable saddle snapshot."""
    import json
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu as _splu
    from test_truck_viz import _tiny_tire_mesh, _CFG_PATH
    from diffsim.cases.truck_config import load_truck_config
    from diffsim.cases.truck import run_truck
    cfg = load_truck_config(_CFG_PATH)
    res = run_truck(cfg, nsteps=3, base_level=5, truck_band_to=6,
                    band_cells=2, merged=_tiny_tire_mesh(cfg),
                    region_refine=False, nu=1.0 / 50.0, dt=0.02,
                    verbose=False,
                    dump_system_steps=(1,), dump_system_dir=str(tmp_path))
    f = tmp_path / "sys_step0001.npz"
    assert f.exists()
    d = np.load(f)
    A = sp.csr_matrix((d["A_data"], d["A_indices"], d["A_indptr"]),
                      shape=tuple(d["A_shape"]))
    n = int(d["nfree"]) * int(d["ndof"])
    assert A.shape == (n, n)
    assert np.all(np.isfinite(d["b"])) and d["b"].shape == (n,)
    # snapshot must be solvable standalone
    x = _splu(A.tocsc()).solve(d["b"])
    assert np.all(np.isfinite(x))
    # PCD operators present and square in the pressure space (nfree x nfree)
    Ap = sp.csr_matrix((d["Ap_data"], d["Ap_indices"], d["Ap_indptr"]),
                       shape=tuple(d["Ap_shape"]))
    assert Ap.shape == (int(d["nfree"]), int(d["nfree"]))
    assert int(d["p_pin"]) >= 0
    json.loads(str(d["bd_json"]))          # knob record parses
    # knob OFF byte-identity: default run unaffected
    ref = run_truck(cfg, nsteps=3, base_level=5, truck_band_to=6,
                    band_cells=2, merged=_tiny_tire_mesh(cfg),
                    region_refine=False, nu=1.0 / 50.0, dt=0.02,
                    verbose=False)
    np.testing.assert_array_equal(res["cd"], ref["cd"])


# ---------------------------------------------------------------------------
# solver-escalation Task 3: offline solver lab on dumped snapshots
# ---------------------------------------------------------------------------

def test_solver_lab_on_snapshot(tmp_path):
    """The lab reproduces a converged solve on a dumped tiny system for
    the bdiag control and the pcd-jacobi candidate.

    Budget rule (task-3 review): pcd-jacobi with Jacobi-CG inners is
    inherently slow on 3-D BDF2 steps (measured ~490 ms/outer-iter on CPU,
    ~347+ outers to convergence — campaign-relevant data motivating the
    amgx inners).  The pcd-jacobi leg is therefore a bounded honest-report
    check (maxiter=60: path exercised, sentinel plumbing verified, outcome
    reported truthfully) rather than a convergence assert; bdiag stays a
    strict tol=1e-8 convergence assert.  Convergence of the pcd path
    itself is covered at BDF1 scale by test_pcd_primary_bdf1_wiring.
    """
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..",
                                      "cluster"))
    from test_truck_viz import _tiny_tire_mesh, _CFG_PATH
    from diffsim.cases.truck_config import load_truck_config
    from diffsim.cases.truck import run_truck
    cfg = load_truck_config(_CFG_PATH)
    run_truck(cfg, nsteps=2, base_level=5, truck_band_to=6, band_cells=2,
              merged=_tiny_tire_mesh(cfg), region_refine=False,
              nu=1.0 / 50.0, dt=0.02, verbose=False,
              dump_system_steps=(1,), dump_system_dir=str(tmp_path))
    from solver_lab import run_config
    snap = str(tmp_path / "sys_step0001.npz")

    # bdiag control: must converge to tight tol (not relaxed)
    row = run_config(snap, "bdiag", device="cpu", tol=1e-8)
    assert row["converged"], row
    assert row["relres"] < 1e-7
    assert row["n"] > 0 and row["wall_s"] > 0

    # pcd-jacobi: bounded honest-report leg (see docstring budget rule)
    row = run_config(snap, "pcd-jacobi", device="cpu", tol=1e-4, maxiter=60)
    assert row["n"] > 0 and row["wall_s"] > 0
    # F6: key renamed "iters" (total inner iterations); "outer" is gone.
    assert row["iters"] > 0 or not row["converged"]   # sentinel plumbing
    if row["converged"]:
        assert row["relres"] < 1e-3

    # F2: warm-start keys always present in snapshot (empty sentinels for a
    # fresh march where no prior step has populated the bdiag cache).
    import numpy as _np2
    d = _np2.load(snap)
    assert "x_prev" in d and "x_prev2" in d   # keys exist


def test_pcd_primary_bdf1_wiring():
    """Always-on CPU coverage of the fgmres_pcd PRIMARY wiring (meta build
    + sigma/nu refresh + cache consumption): one BDF1 bootstrap step with
    jacobi inners (~27 outers, seconds).  Trajectory quality is covered by
    the pyamgx-gated test; this one guards the plumbing (task-2 review)."""
    from test_truck_viz import _tiny_tire_mesh, _CFG_PATH
    from diffsim.cases.truck_config import load_truck_config
    from diffsim.cases.truck import run_truck
    cfg = load_truck_config(_CFG_PATH)
    res = run_truck(cfg, nsteps=1, base_level=5, truck_band_to=6,
                    band_cells=2, merged=_tiny_tire_mesh(cfg),
                    region_refine=False, nu=1.0 / 50.0, dt=0.02,
                    verbose=False, mono_solver="fgmres_pcd",
                    pcd_f_inner="jacobi", pcd_ap_inner="jacobi")
    assert np.all(np.isfinite(res["cd"]))
    assert np.all(np.isfinite(res["cd_surr"]))
