"""Lid-driven cavity vs Ghia et al. (1982) — full benchmark driver.

Usage:
    python benchmarks/cavity_ghia.py [--level 6] [--re 100] [--stepper both]

Prints the centerline u(x=0.5, y) and v(x, y=0.5) tables against the Ghia
columns and the max deviations. Re = 100 references are embedded; for 400 /
1000 supply the Ghia columns yourself (1982 paper, Tables I-II) — the driver
prints our profiles either way.
"""

import os as _bos, sys as _bsys  # noqa: E402  (benchmark import bootstrap)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import _bench_bootstrap  # noqa: E402,F401
import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests"))
from test_cavity import GHIA_Y, GHIA_U, GHIA_X, GHIA_V, _lid_g  # noqa: E402

from diffsim import default_device                              # noqa: E402
from diffsim.octree.build import build_uniform                   # noqa: E402
from diffsim.mesh.nodes import build_mesh                        # noqa: E402
from diffsim.mesh.constraints import build_constraints           # noqa: E402
from diffsim.mesh.basis import basis_tables                      # noqa: E402
from diffsim.assembly.operators import DeviceMesh                # noqa: E402
from diffsim.mesh.pointeval import point_eval_weights            # noqa: E402


def run(level, re, stepper_name, device=None, max_steps=2000, dt=0.05,
        solver="splu"):
    device = default_device() if device is None else device
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    if stepper_name == "monolithic":
        from diffsim.steppers.linearized import LinearizedMonolithicStepper
        st = LinearizedMonolithicStepper(
            dm, 1.0 / re, dt, f_fn=lambda x, t: np.zeros((len(x), 2)),
            g_fn=_lid_g, order=1, solver=solver)
    else:
        from diffsim.steppers.leray import LerayProjectionStepper
        st = LerayProjectionStepper(
            dm, 1.0 / re, dt, f_fn=lambda x, t: np.zeros((len(x), 2)),
            g_fn=_lid_g, order=1, picard_iters=1, solver=solver)
    st.set_initial(lambda x: np.zeros((len(x), 2)))
    prev, steps = None, 0
    for steps in range(1, max_steps + 1):
        out = st.step()
        u = out[0][:, :2] if isinstance(out, tuple) else out[:, :2]
        if prev is not None and np.abs(u - prev).max() / dt < 1e-4:
            break
        prev = u.copy()
    T = dm.constraints.T.tocsr()
    W_u = point_eval_weights(mesh, np.stack(
        [np.full_like(GHIA_Y, 0.5), GHIA_Y], axis=1))
    W_v = point_eval_weights(mesh, np.stack(
        [GHIA_X, np.full_like(GHIA_X, 0.5)], axis=1))
    u_c = np.asarray(W_u @ np.asarray(T @ u[:, 0]))
    v_c = np.asarray(W_v @ np.asarray(T @ u[:, 1]))
    return u_c, v_c, steps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=6)
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--stepper", default="both",
                    choices=["monolithic", "leray", "both"])
    ap.add_argument("--solver", default="splu",
                    choices=["splu", "fused", "amgx", "cudss"])
    args = ap.parse_args()
    steppers = (["monolithic", "leray"] if args.stepper == "both"
                else [args.stepper])
    have_ref = args.re == 100.0
    for name in steppers:
        u_c, v_c, steps = run(args.level, args.re, name, solver=args.solver)
        print(f"\n=== {name} | level {args.level} | Re {args.re:g} | "
              f"{steps} steps ===")
        print(f"{'y':>8} {'u(0.5,y)':>10}" + ("  {:>10}".format("Ghia")
                                              if have_ref else ""))
        for i, y in enumerate(GHIA_Y):
            row = f"{y:>8.4f} {u_c[i]:>10.4f}"
            if have_ref:
                row += f" {GHIA_U[i]:>10.4f}"
            print(row)
        if have_ref:
            print(f"max |du| = {np.abs(u_c - GHIA_U).max():.4f}   "
                  f"max |dv| = {np.abs(v_c - GHIA_V).max():.4f}")
