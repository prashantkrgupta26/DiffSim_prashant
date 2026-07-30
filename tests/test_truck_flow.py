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
        "solver_options_ns", "tauM_scale", "thetaTimeStepping",
    }
    assert set(cfg.unknown_keys) == expected_unknown


# ---------------------------------------------------------------------------
# Group 2: tiny truck-in-a-box — mesh build, dyadic slab carve, 3-step march
# ---------------------------------------------------------------------------

def test_slab_carve_dyadic_exact():
    """The channel slab bounds (y=z=1/8) are dyadic => NO cell is cut; the
    carve retains exactly the interior slab with zero intercepted cells."""
    from truck_flow import _channel_box, slab_carve
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
    from truck_flow import run_truck
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
    from truck_flow import make_nu_schedule
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
    from truck_flow import run_truck
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
    from truck_flow import run_truck
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
        monkeypatch.setattr("truck_flow.solve_linear", _wrap)
        run_truck(cfg, nsteps=5, base_level=5, truck_band_to=6, band_cells=2,
                  merged=merged, region_refine=False, nu=1.0 / 50.0, dt=0.02,
                  mono_solver="fgmres_bdiag", saddle_x0=x0, device="cpu",
                  assembly="device", linsolve_tol=1e-7, verbose=False)
        monkeypatch.setattr("truck_flow.solve_linear", orig)
        return iters

    warm = _run("extrap")
    cold = _run(None)
    # warm-start only affects steps >= 2 (first two are cold by construction);
    # over the whole march the summed iterations must not exceed cold and
    # should be strictly fewer once extrapolation kicks in.
    assert sum(warm) < sum(cold), f"warm {warm} vs cold {cold}"


# ---------------------------------------------------------------------------
# Group 3: BC masks — dyadic planes nonempty and pairwise disjoint (where they
# must be)
# ---------------------------------------------------------------------------

def test_bc_masks_nonempty_and_structure():
    """Each dyadic plane carries >0 free nodes; the streamwise (inflow/outlet)
    and the transverse (ground/ceiling, side_zlo/side_zhi) opposite pairs are
    disjoint; strong-BC rows/vals are consistent length."""
    from truck_flow import (build_truck_mesh, truck_strong_bc, truck_bc_masks)
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
