"""GPU smoke: both solvers x both dims solve ON DEVICE. Run on a CUDA box:

    LD_LIBRARY_PATH=/usr/lib/wsl/lib .venv/bin/python tests/gpu_smoke_thinshell.py

Tiny cases, ~1-2 min total. Asserts finite Cd everywhere and prints SMOKE-OK.
NOT a pytest file (no test_ prefix): Mac CI never collects it."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

import warp as wp
wp.init()
assert wp.is_cuda_available(), (
    "no CUDA device — on gpubox set LD_LIBRARY_PATH=/usr/lib/wsl/lib")
DEV = "cuda:0"

from p2r1a_thin_plate_flow import run_flow_past, run_flow_past_projection
from p2r1c_thin_plate_flow_3d import run_flow_past_3d

ok = []
# 2-D monolithic on cuDSS (direct GPU)
r = run_flow_past(level=4, nsteps=3, dt=0.01, nu=0.1,
                  mono_solver="cudss", device=DEV, verbose=False)
assert np.all(np.isfinite(r["cd"])), r["cd"]; ok.append("2d-mono")
# 2-D projection: predictor fused (GPU BiCGStab), PPE gpu_cg (GPU CG)
r = run_flow_past_projection(level=4, nsteps=3, dt=0.01, nu=0.1,
                             predictor_solver="fused", ppe_solver="gpu_cg",
                             device=DEV, verbose=False)
assert np.all(np.isfinite(r["cd"])), r["cd"]; ok.append("2d-proj")
# 3-D monolithic on cuDSS
r = run_flow_past_3d(level=3, nsteps=2, dt=0.01, nu=0.1,
                     mono_solver="cudss", device=DEV, verbose=False)
assert np.all(np.isfinite(r["cd"])), r["cd"]; ok.append("3d-mono")
# 3-D projection fused + gpu_cg
from p2r1c_thin_plate_flow_3d_projection import run_flow_past_3d_projection
r = run_flow_past_3d_projection(level=3, nsteps=2, ppe_solver="gpu_cg",
                                predictor_solver="fused", device=DEV,
                                verbose=False)
assert np.all(np.isfinite(r["cd"])), r["cd"]; ok.append("3d-proj")
print("SMOKE-OK", *ok)
