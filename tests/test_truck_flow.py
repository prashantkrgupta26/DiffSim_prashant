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
