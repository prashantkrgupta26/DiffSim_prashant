"""Diagnostic: does the Re=250 projection leg diverge WITHOUT device assembly?

Discriminates 6b-device-predictor bug vs projection-scheme behavior at the
Re=250/L9/dt=5e-5 regime.  Same config as gpu_preflight_re250.py's projection
leg but device_assembly=False (host assembly; cudss predictor + gpu_cg PPE
still on GPU).  Env: NSTEPS (default 100), DEVICE (default cuda:0),
DEV_ASM=1 to re-enable device assembly.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import warp as wp

wp.init()
assert wp.is_cuda_available()

from p2r1a_thin_plate_flow import run_flow_past_projection

NSTEPS = int(os.environ.get("NSTEPS", "100"))
DEV = os.environ.get("DEVICE", "cuda:0")
DEV_ASM = os.environ.get("DEV_ASM", "0") == "1"

t0 = time.time()
r = run_flow_past_projection(
    level=7, refine_to=9, wake_refine=9,
    nsteps=NSTEPS, dt=5e-5, nu=1.0 / 250.0, U_inf=1.0,
    plate_xc=5.0 / 36.0, plate_yc=0.5, plate_L=1.0 / 16.0,
    pert_eps=0.03, pert_t_end=1.0,
    predictor_solver="cudss", ppe_solver="gpu_cg",
    device=DEV, device_assembly=DEV_ASM, verbose=False)
el = time.time() - t0
cd = np.asarray(r["cd"])
print(f"DIAG dev_asm={DEV_ASM} nsteps={NSTEPS} elapsed={el:.1f}s "
      f"s/step={el / NSTEPS:.3f}")
print("cd head:", np.array2string(cd[:5], precision=4))
print("cd tail:", np.array2string(cd[-5:], precision=4))
print("max|cd|:", float(np.max(np.abs(cd))))
print("DIAG-DONE", "DIVERGED" if np.max(np.abs(cd)) > 1e4 else "BOUNDED")
