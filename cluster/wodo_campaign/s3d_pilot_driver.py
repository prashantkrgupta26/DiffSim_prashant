"""S3-3D hero driver (Nova A100-80): the M5 evaporation-quench hero
case, now on the blockch-(M,K) preconditioner instead of cuDSS.

WHAT CHANGED (2026-07-14): the pilot was built to MEASURE cuDSS 3-D at
811k dofs, which was expected/measured SOLVER-BOUND (cuDSS CEILINGed at
that size: >16 min factorization crawl, 48.2 GB — dev note
docs/dev/2026-07-13-blockch-mpf.md Sec 3).  blockch_dev is now merged
(src/diffsim/solvers/linsolve.py; MultiPhaseStepper linsolver=
"blockch_dev") and MEASURED 8.4x faster than cuDSS at slab64
(2.10 vs 17.7 s/call) and marches the 811k cuDSS-CEILING case at
28.6 s/step.  So this driver defaults to linsolver="blockch_dev"
(assembly="device"): the solver-bound pilot becomes a real 80 GB
campaign that clears the cuDSS 3-D ceiling.

MEASURED walls (RTX 6000 Ada 48 GB, dev-note tables — NOT invented):
  * 3d_slab64z32  (811,008 dofs, M2/K1): 28.6 s/step, 2.16 s/call,
    ~22 GB GPU (blockch_dev; cuDSS CEILINGed here).      [Sec 3]
  * 3d_film128    (128x128x64, 6.39M dofs, M2/K1): 146.6 s/step,
    9.46 s/call solve, 22.0 GB GPU / 14.6 GB host (superset).  [Sec 5]
  * 3d_slab128z64_mk32 (128x128x64, M3/K2, 10.6M dofs): 160.8 s/step,
    9.06 s/call, 23.9 GB GPU / 18.9 GB host (block-masked, REQUIRED —
    the superset overflows int32 at this (M,K)).          [Sec 6]
A100-80 (Nova) is HBM2e ~2x GDDR6 bandwidth on the spmv-bound solve =>
~75-80 s/step estimate at the 6.4M-dof film128 rung, with 80 GB
clearing 3.6x the measured 22 GB footprint (dev note Sec 5, honest
extrapolation — NOT a measurement).

Reuses the D3 bench constructors verbatim (single source of truth for
the hero config; the bench CASES dicts define the geometry, the bench
make_stepper the S3b production energetics).  mk32 cases run the
(M=3, K=2) family with the REQUIRED block-masked pattern.
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
    ap.add_argument("--linsolver", default="blockch_dev",
                    choices=["blockch_dev", "blockch", "cudss", "splu"],
                    help="default blockch_dev — clears the cuDSS 3-D "
                         "ceiling (dev note 2026-07-13-blockch-mpf.md)")
    ap.add_argument("--block-sparse", action="store_true",
                    help="kron(G, blockmask) device pattern; REQUIRED "
                         "for the mk32 128-class rungs (int32-safe nnz)")
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
    mk = (3, 2) if args.case.endswith("_mk32") else (2, 1)
    dm = build_case(c, args.device)
    st = make_stepper(dm, args.assembly, dt=c.get("dt", 1e-4),
                      noise=args.noise, linsolver=args.linsolver,
                      mk=mk, block_sparse=args.block_sparse)
    print(json.dumps(dict(event="setup", case=args.case,
                          assembly=args.assembly,
                          linsolver=args.linsolver,
                          block_sparse=bool(args.block_sparse),
                          M=int(st.M), K=int(st.K),
                          nfree=int(st.nfree),
                          ndof_total=int(st.nfree * st.ndof),
                          wall=round(time.time() - t0, 1))),
          flush=True)

    Tc = st.Tc
    full = lambda v: np.asarray(Tc @ v)

    def snap(k):
        fields = dict(t=st.t, h=st.h_curr)
        for i in range(st.M):
            fields[f"phi_{i}"] = full(st.phi(i))
        for j in range(st.K):
            fields[f"psi_{j}"] = full(st.psi(j))
            fields[f"theta_{j}"] = full(st.theta(j))
        # back-compat aliases for the 2-D analysis tooling (M2/K1 names)
        fields["phi_f"] = fields["phi_0"]
        fields["phi_p"] = fields["phi_1"]
        fields["psi"] = fields["psi_0"]
        fields["theta"] = fields["theta_0"]
        np.savez(f"{args.outdir}/snap_{k:04d}.npz", **fields)

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
