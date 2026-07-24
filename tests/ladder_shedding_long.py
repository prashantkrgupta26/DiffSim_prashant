"""Vortex-shedding on an ELONGATED domain — genuine von Karman street + St.

Sibling of `ladder_shedding_square.py`. The prior study proved the scheme does
NOT shed on the unit-box mesh, but for a DOMAIN reason: the outflow sat only
~4*D downstream of the body, and the von Karman global instability needs
~10-20*D to develop (the same-mesh MONOLITHIC oracle also went steady, ruling
out a scheme defect). The fix is a longer streamwise wake.

This driver uses `build_channel_long` (`ladder_fixtures.py`): the octree is the
unit box, but the obstacle side D is small and the body sits NEAR THE INLET, so
there are ~(1-cx)/D obstacle-diameters of clear wake before the outflow. The
stepper numerics are UNCHANGED — same consistent-projection + weak-Nitsche
config (`consistent_projection=True`, `rotational_pin_wall=True`, gamma=50
grad-div, backflow beta=0.5) that the 2-D/3-D ladder validated faithful. The
symmetry-breaking trigger (transient transverse inflow tilt + optional
permanent `y_offset` seed) is reused verbatim.

Run (gpubox GPU1):
    CUDA_VISIBLE_DEVICES=1 PROJ_SOLVER=splu PYTHONPATH=src:tests \
      LEVEL=8 HALF=0.03125 CX=0.125 NSTEPS=20000 DT=0.005 YOFF_CELLS=0.5 \
      .venv/bin/python tests/ladder_shedding_long.py 100
"""
import json
import os
import sys
import numpy as np

from ladder_fixtures import build_channel_long, U_IN
from ladder_shedding_square import shed_projection, shed_monolithic, analyze


def main():
    Re = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
    level = int(os.environ.get("LEVEL", "8"))
    half = float(os.environ.get("HALF", "0.03125"))     # 8/256 => D=0.0625
    cell = 1.0 / (2 ** level)
    cx = float(os.environ.get("CX", str(max(4.0 * half, 0.1))))
    cy = float(os.environ.get("CY", "0.5"))
    dt = float(os.environ.get("DT", "0.005"))
    nsteps = int(os.environ.get("NSTEPS", "20000"))
    trigger_steps = int(os.environ.get("TRIGGER_STEPS", "400"))
    ramp_steps = int(os.environ.get("RAMP_STEPS", "100"))
    log_every = int(os.environ.get("LOG_EVERY", "500"))
    eps = float(os.environ.get("EPS", "0.1"))
    solver = os.environ.get("PROJ_SOLVER", "splu")
    device = os.environ.get("DEVICE", "cpu" if solver == "splu" else "cuda:0")
    do_mono = os.environ.get("MONO", "1") == "1"
    # ONLY_MONO: skip the projection march and run ONLY the monolithic oracle
    # (the slow same-mesh unsteady oracle, run standalone so it does not block
    # the projection run). Forces do_mono on.
    only_mono = os.environ.get("ONLY_MONO", "0") == "1"
    if only_mono:
        do_mono = True
    y_off_cells = float(os.environ.get("YOFF_CELLS", "0.5"))
    y_offset = y_off_cells * cell
    D = 2.0 * half
    downstream_D = (1.0 - cx) / D

    print(f"{'='*74}\n ELONGATED SHEDDING  level={level}  Re={Re}  half={half} "
          f"D={D:.4f}  blockage(D/Ly)={D:.3f}  cx={cx} cy={cy}  "
          f"downstream={downstream_D:.1f}*D\n dt={dt} nsteps={nsteps} "
          f"trigger={trigger_steps} eps={eps}  y_off={y_off_cells}cell="
          f"{y_offset:.6f}  solver={solver} dev={device}\n{'='*74}", flush=True)

    fx = build_channel_long(level, Re, half=half, offset=0, device=device,
                            cx=cx, cy=cy, y_offset=y_offset)
    if y_offset == 0.0:
        assert fx["dmax"] == 0, f"expected body-fitted; got dmax={fx['dmax']}"
    print(f" mesh: n_fluid_cells={fx['n_fluid_cells']}  "
          f"n_obstacle_nodes={int(fx['obstacle_node_mask'].sum())}  "
          f"nu={fx['nu']:.6f}  dmax={fx['dmax']}  center={fx['center']}",
          flush=True)

    if only_mono:
        pr = dict(cl_hist=[], cd_hist=[], mu_hist=[], div=float("nan"),
                  blew_up=False, steps=0)
        pra = None
        print("\n--- (projection skipped; ONLY_MONO) ---", flush=True)
    else:
        print("\n--- PROJECTION (consistent + weak Nitsche) ---", flush=True)
        pr = shed_projection(fx, dt, nsteps, trigger_steps=trigger_steps,
                             eps=eps, solver=solver, log_every=log_every,
                             ramp_steps=ramp_steps)
        pra = analyze(pr["cl_hist"], pr["cd_hist"], dt, D, label="proj")

    moa = None
    mo = None
    if do_mono:
        print("\n--- MONOLITHIC (same elongated mesh, same trigger) ---",
              flush=True)
        mo = shed_monolithic(fx, dt, nsteps, trigger_steps=trigger_steps,
                             eps=eps, log_every=log_every,
                             ramp_steps=ramp_steps)
        moa = analyze(mo["cl_hist"], mo["cd_hist"], dt, D, label="mono")

    out = dict(level=level, Re=Re, half=half, D=D, cx=cx, cy=cy,
               downstream_D=downstream_D, dt=dt, nsteps=nsteps,
               trigger_steps=trigger_steps, eps=eps, y_off_cells=y_off_cells,
               n_fluid_cells=fx["n_fluid_cells"], dmax=fx["dmax"],
               proj=pra, proj_div=pr["div"], proj_blew=pr["blew_up"],
               mono=moa)
    outdir = os.environ.get("OUTDIR", ".")
    tag = f"long_L{level}_Re{int(Re)}_D{D:.4f}"
    np.savez(os.path.join(outdir, f"shed_{tag}.npz"),
             proj_cl=pr["cl_hist"], proj_cd=pr["cd_hist"], proj_mu=pr["mu_hist"],
             mono_cl=(mo["cl_hist"] if do_mono else []),
             mono_cd=(mo["cd_hist"] if do_mono else []), dt=dt, D=D)
    print("\n[json]", json.dumps(out, default=lambda o: None), flush=True)
    return out


if __name__ == "__main__":
    main()
