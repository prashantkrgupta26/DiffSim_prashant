"""Re=250 thin-plate run in CONSISTENT OCTREE UNITS (config-bug fix probe).

The committed RE250_CONFIG mixes unit systems: geometry in octree units
(plate L=1/16) but nu and t_end in paper-physical units (L=1).  Effective
plate Reynolds of that config: U*L/nu = 0.0625/0.004 = 15.6 -> steady flow,
no shedding (observed in the 2026-07-25 production run).

This runner uses consistent octree units throughout:
    L      = 1/16                (plate length, octree)
    nu     = U*L/Re = 0.0625/Re  (Re=250 -> 2.5e-4)
    x_c    = 5/16 = 0.3125       (5 plate-lengths upstream, as in the paper)
    t_end  = T_LU * L/U          (T_LU convective times; default 56)
    St     = f * L / U           (all octree units — dimensionless anyway)

Env: RE (250), NSTEPS (8000), DT (5e-4), T_START_LU (20 convective times),
DEVICE (cuda:0), MONO_SOLVER (cudss), ASSEMBLY (device), PERT_EPS (0.03).
Saves full histories to results/re{RE}_corrected_hist.npz.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import warp as wp

wp.init()

from p2r1a_thin_plate_flow import run_flow_past
from diffsim.postproc.shedding import time_avg_cd, strouhal

RE = float(os.environ.get("RE", "250"))
DT = float(os.environ.get("DT", "5e-4"))
NSTEPS = int(os.environ.get("NSTEPS", "8000"))
T_START_LU = float(os.environ.get("T_START_LU", "20"))
DEV = os.environ.get("DEVICE", "cuda:0")
REFINE = int(os.environ.get("REFINE", "9"))    # plate-band target level
WAKE = int(os.environ.get("WAKE", "9"))        # wake-band target level

L = 1.0 / 16.0
U = 1.0
NU = U * L / RE
X_C = 5.0 / 16.0
t_start = T_START_LU * L / U   # averaging start, octree time

print(f"[re-corrected] Re={RE:.0f} nu={NU:.3e} L={L} x_c={X_C} "
      f"dt={DT:.0e} nsteps={NSTEPS} t_end={NSTEPS * DT:.3f} "
      f"({NSTEPS * DT / (L / U):.0f} L/U)  t_start={t_start:.3f}", flush=True)

t0 = time.time()
r = run_flow_past(
    level=7, refine_to=REFINE, wake_refine=WAKE,
    nsteps=NSTEPS, dt=DT, nu=NU, U_inf=U,
    plate_xc=X_C, plate_yc=0.5, plate_L=L,
    pert_eps=float(os.environ.get("PERT_EPS", "0.03")), pert_t_end=0.5,
    mono_solver=os.environ.get("MONO_SOLVER", "cudss"),
    assembly=os.environ.get("ASSEMBLY", "device"), device=DEV,
    verbose=False)
el = time.time() - t0

cd = np.asarray(r["cd"])
cl = np.asarray(r["cl"])
t = np.arange(1, NSTEPS + 1) * DT
assert np.all(np.isfinite(cd)), f"Cd not finite: {cd[-5:]}"

out = f"results/re{RE:.0f}_r{REFINE}_corrected_hist.npz"
os.makedirs("results", exist_ok=True)
np.savez(out, t=t, cd=cd, cl=cl, nu=NU, dt=DT, L=L)

mask = t >= t_start
cl_std_late = float(np.std(cl[mask]))
cd_mean = float(time_avg_cd(t, cd, t_start=t_start))
try:
    # St in consistent octree units: strouhal(t, cl, U, L) -> f*L/U
    St, freq = strouhal(t, cl, U, L)
    st_str = f"St={St:.4f} freq={freq:.3f}"
except Exception as exc:  # too short / no peak
    st_str = f"St=N/A ({exc})"

print(f"[re-corrected] done in {el:.0f}s ({el / NSTEPS:.3f} s/step)")
print(f"[re-corrected] Cd_mean(t>={t_start:.2f})={cd_mean:.4f}  {st_str}")
print(f"[re-corrected] Cl_std(late)={cl_std_late:.3e}  "
      f"SHEDDING={'YES' if cl_std_late > 1e-3 else 'NO/NOT-YET'}")
print(f"[re-corrected] history -> {out}")
print("RECORRECTED-OK")
