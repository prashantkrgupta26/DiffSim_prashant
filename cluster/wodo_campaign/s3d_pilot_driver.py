"""S3-3D pilot driver (Nova A100-80): ONE measure-first case of the
recorded hero candidate — the 64 x 64 x 32 film slab (811,776 dofs)
that exceeds the 48 GB workstation ceiling (dev note
docs/dev/2026-07-13-m5-device-assembly.md Sec 5).

PURPOSE: measure, not campaign.  cuDSS 3-D at this size is expected
SOLVER-BOUND (minutes/step class); this pilot records the real A100-80
setup wall, s/step, and memory so the hero campaign can be sized after
the blockch-(M,K) route lands.  Reuses the D3 bench constructors
verbatim (single source of truth for the hero config; the bench CASES
dicts define the geometry, the bench make_stepper the physics).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "..", "benchmarks",
    "performance"))
from m5_device_assembly import CASES, build_case, make_stepper  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="3d_slab64z32",
                    choices=sorted(CASES))
    ap.add_argument("--assembly", default="device",
                    choices=["host", "device"])
    ap.add_argument("--noise", type=float, default=5e-3,
                    help="FDT psi noise (the production quench)")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--wall-cap", type=float, default=6 * 3600.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--outdir", default="s3d_pilot_out")
    ap.add_argument("--snap-every", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    t0 = time.time()
    c = CASES[args.case]
    dm = build_case(c, args.device)
    st = make_stepper(dm, args.assembly, dt=c.get("dt", 1e-4),
                      noise=args.noise)
    print(json.dumps(dict(event="setup", case=args.case,
                          assembly=args.assembly,
                          nfree=int(st.nfree),
                          ndof_total=int(st.nfree * st.ndof),
                          wall=round(time.time() - t0, 1))),
          flush=True)

    Tc = st.Tc
    full = lambda v: np.asarray(Tc @ v)

    def snap(k):
        np.savez(f"{args.outdir}/snap_{k:04d}.npz",
                 t=st.t, h=st.h_curr,
                 phi_f=full(st.phi(0)), phi_p=full(st.phi(1)),
                 psi=full(st.psi(0)), theta=full(st.theta(0)))

    snap(0)
    k = 0
    while k < args.steps and time.time() - t0 < args.wall_cap:
        ts = time.time()
        r = st.march(t_end=st.t + 1e9, dt_max=0.02, max_steps=1,
                     dt_min=1e-11, grow_iters=45, h_min=0.14,
                     phis_stop=0.02)
        k += 1
        phis = 1.0 - sum(float(np.mean(full(st.phi(i))))
                         for i in range(st.M))
        print(json.dumps(dict(event="step", k=k, t=round(st.t, 6),
                              h=round(st.h_curr, 6),
                              phis=round(phis, 6),
                              dt=st.dt, rejects=st.n_reject,
                              step_wall=round(time.time() - ts, 2),
                              total_wall=round(time.time() - t0, 1),
                              reason=r)), flush=True)
        if k % args.snap_every == 0:
            snap(k)
        if r in ("h_min", "phis_stop", "dt_underflow"):
            break
    snap(k)
    print(json.dumps(dict(event="done", steps=k,
                          wall=round(time.time() - t0, 1))),
          flush=True)


if __name__ == "__main__":
    main()
