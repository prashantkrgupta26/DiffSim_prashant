"""Re=250 GPU preflight: per-leg timing for the both-solver production run.

Runs the monolithic leg (cudss + device assembly) and the projection leg
(cudss predictor + gpu_cg PPE + device assembly incl. the 6b device
predictor) for NSTEPS steps each on the Re=250 adaptive mesh, printing a
per-leg timing table and projected wall-time for 1M steps.  Run on a CUDA
box (gpubox-run.sh injects the WSL lib path):

    bash scripts/remote/gpubox-run.sh \
        ".venv/bin/python tests/gpu_preflight_re250.py" re250-preflight2

Env: NSTEPS (default 100), DEVICE (default cuda:0).
Not a pytest file (no test_ prefix): never collected on the Mac.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import warp as wp

wp.init()
assert wp.is_cuda_available(), (
    "no CUDA device — on gpubox set LD_LIBRARY_PATH=/usr/lib/wsl/lib")

from p2r1a_thin_plate_flow import run_flow_past, run_flow_past_projection

NSTEPS = int(os.environ.get("NSTEPS", "100"))
DEV = os.environ.get("DEVICE", "cuda:0")

RE250 = dict(
    level=7, refine_to=9, wake_refine=9,
    nsteps=NSTEPS, dt=5e-5, nu=1.0 / 250.0, U_inf=1.0,
    plate_xc=5.0 / 36.0, plate_yc=0.5, plate_L=1.0 / 16.0,
    pert_eps=0.03, pert_t_end=1.0, verbose=False,
)

rows = []

t0 = time.time()
r_m = run_flow_past(mono_solver="cudss", assembly="device", device=DEV,
                    **RE250)
el_m = time.time() - t0
assert np.all(np.isfinite(r_m["cd"])), f"mono Cd not finite: {r_m['cd'][-5:]}"
rows.append(("monolithic cudss+dev-asm", el_m, r_m["cd"][-1]))

t0 = time.time()
r_p = run_flow_past_projection(predictor_solver="cudss", ppe_solver="gpu_cg",
                               device=DEV, device_assembly=True, **RE250)
el_p = time.time() - t0
assert np.all(np.isfinite(r_p["cd"])), f"proj Cd not finite: {r_p['cd'][-5:]}"
rows.append(("projection cudss-pred+gpu_cg+dev-asm", el_p, r_p["cd"][-1]))

print(f"\n=== RE250 PREFLIGHT ({NSTEPS} steps, {DEV}) ===")
print(f"{'leg':40s} {'wall(s)':>9s} {'s/step':>8s} {'1M-step(h)':>11s} "
      f"{'Cd[-1]':>10s}")
for name, el, cd in rows:
    print(f"{name:40s} {el:9.1f} {el / NSTEPS:8.3f} "
          f"{el / NSTEPS * 1e6 / 3600.0:11.1f} {cd:+10.4f}")
print("PREFLIGHT-OK")
