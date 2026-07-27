"""Re=250 dt-ladder probe: can BDF2 take 4-20x larger steps than the
paper-faithful dt=5e-5?

Marches the MONOLITHIC production config (cudss + device assembly) to a
common physical horizon T_PROBE at dt in {5e-5, 2e-4, 5e-4, 1e-3} and
compares the Cd(t) startup transients against the fine-dt reference
(interpolated to common times).  BDF2 (order=2, BDF1 bootstrap) throughout.

    bash scripts/remote/gpubox-run.sh \
        ".venv/bin/python tests/gpu_dtladder_re250.py" re250-dtladder

Env: T_PROBE (default 0.1), DEVICE (default cuda:0).
Not a pytest file: never collected on the Mac.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import warp as wp

wp.init()
assert wp.is_cuda_available()

from p2r1a_thin_plate_flow import run_flow_past

T_PROBE = float(os.environ.get("T_PROBE", "0.1"))
DEV = os.environ.get("DEVICE", "cuda:0")
DTS = [5e-5, 2e-4, 5e-4, 1e-3]

BASE = dict(
    level=7, refine_to=9, wake_refine=9,
    nu=1.0 / 250.0, U_inf=1.0,
    plate_xc=5.0 / 36.0, plate_yc=0.5, plate_L=1.0 / 16.0,
    pert_eps=0.03, pert_t_end=1.0, verbose=False,
    mono_solver="cudss", assembly="device", device=DEV,
)

runs = []
for dt in DTS:
    nsteps = int(round(T_PROBE / dt))
    t0 = time.time()
    r = run_flow_past(nsteps=nsteps, dt=dt, **BASE)
    el = time.time() - t0
    cd = np.asarray(r["cd"])
    t = np.arange(1, nsteps + 1) * dt
    ok = bool(np.all(np.isfinite(cd)))
    runs.append(dict(dt=dt, nsteps=nsteps, el=el, t=t, cd=cd, ok=ok))
    print(f"[dt-ladder] dt={dt:.0e} nsteps={nsteps:5d} wall={el:7.1f}s "
          f"s/step={el / nsteps:.3f} Cd(T)={cd[-1]:+.4f} finite={ok}",
          flush=True)

ref = runs[0]
print(f"\n=== DT-LADDER vs dt={ref['dt']:.0e} reference "
      f"(common window t in [{5 * DTS[-1]:.3f}, {T_PROBE}]) ===")
print(f"{'dt':>8s} {'CFL':>6s} {'steps/period':>13s} {'maxrel dCd':>11s} "
      f"{'Cd(T) rel-err':>14s} {'speedup':>8s}")
h_plate = 2.0 ** -9
period = (1.0 / 16.0) / 0.15  # plate_L / St in code units
for r in runs:
    # compare on the common window, skipping each rung's BDF1 bootstrap
    t_lo = max(5 * r["dt"], 5 * ref["dt"])
    mask = (r["t"] >= t_lo) & (r["t"] <= T_PROBE)
    cd_ref_i = np.interp(r["t"][mask], ref["t"], ref["cd"])
    denom = np.maximum(np.abs(cd_ref_i), 1e-3)
    maxrel = float(np.max(np.abs(r["cd"][mask] - cd_ref_i) / denom)) \
        if r is not ref else 0.0
    cdT_err = abs(r["cd"][-1] - ref["cd"][-1]) / max(abs(ref["cd"][-1]), 1e-3)
    print(f"{r['dt']:8.0e} {r['dt'] / h_plate:6.3f} "
          f"{period / r['dt']:13.0f} {maxrel:11.4f} {cdT_err:14.4f} "
          f"{ref['el'] / r['el']:8.1f}x")
print("DTLADDER-OK")
