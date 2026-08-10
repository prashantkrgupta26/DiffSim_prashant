#!/usr/bin/env python
r"""benchmarks/chns/run_dam3d.py — SP-0 Task 12: 3-D dam-break GPU smoke.

The first 3-D exercise of the monolithic CHNS stack (make_chns_newton(dim=3),
CHNSMonolithicStepper).  Builds a uniform 3-D unit-cube mesh at the requested
level, marches the DAM_BREAK_3D case a fixed number of BDF1 steps on the
requested device, and writes:

  - a metrics JSON  (dam3d_metrics.json): steps, mass series + drift, per-step
    wall, newton iters, device, GPU-mem-if-CUDA, compile time, no-NaN flag.
  - one VTU snapshot (dam3d_final.vtu) of the final phi/u/p on the octree mesh.

Smoke gate (plan): level 5, 20 steps, no NaN, mass flat, one VTU written.

CLI
---
  run_dam3d.py [--level N] [--steps K] [--device cpu|cuda:0|cuda:1]
               [--out DIR] [--dt DT]

Device parity note: the assembly Warp kernel runs on --device; the COO triplets
return to the host and the linear solve stays scipy splu (device-kernel gate).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_DIR = os.path.normpath(os.path.join(_HERE, ".."))
_REPO_DIR = os.path.normpath(os.path.join(_HERE, "..", ".."))
for _p in (_BENCH_DIR, _REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chns.cases import DAM_BREAK_3D           # noqa: E402


def _dambreak_ic_3d(coords, Cn, x_iface=0.5):
    # heavy column left of x_iface (phi=+1), light right (phi=-1); plane in x.
    return -np.tanh((coords[:, 0] - x_iface) / (Cn * np.sqrt(2.0)))


def make_dm(level, device, dim=3):
    import warp as wp
    wp.init()
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh
    tree = build_uniform(level, dim=dim)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim), device)
    return dm, mesh, cons


def _gpu_mem_mib(device):
    if not str(device).startswith("cuda"):
        return None
    try:
        import warp as wp
        idx = int(device.split(":")[1]) if ":" in device else 0
        free_b, total_b = wp.get_device(device).total_memory, None
        # Warp exposes free/total via the CUDA context; fall back to nvidia-smi.
    except Exception:
        pass
    try:
        import subprocess
        idx = int(device.split(":")[1]) if ":" in device else 0
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", "-i", str(idx)],
            text=True).strip()
        return float(out.splitlines()[0])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--linsolver", type=str, default="splu",
                    choices=["splu", "cudss"])
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--out", type=str,
                    default=os.path.join(_HERE, "results", "dam3d"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from diffsim.steppers.chns import CHNSMonolithicStepper
    from diffsim.viz.export import export_vtu

    case = DAM_BREAK_3D
    dt = float(args.dt) if args.dt is not None else float(case.dt0)

    t_build0 = time.perf_counter()
    dm, mesh, cons = make_dm(args.level, args.device, dim=3)
    build_wall = time.perf_counter() - t_build0
    coords = mesh.node_coords
    n_nodes = len(coords)
    blk = case.dim + 3
    ndof = blk * n_nodes

    # Compile timing: first stepper ctor triggers make_chns_newton(dim=3).
    t_comp0 = time.perf_counter()
    st = CHNSMonolithicStepper(dm, case, dt=dt, Cn_override="2h",
                               gravity=True, device=args.device,
                               linsolver=args.linsolver)
    compile_wall = time.perf_counter() - t_comp0

    phi0 = _dambreak_ic_3d(coords, st.Cn)
    st.set_initial(phi0)
    mass0 = st.mass_phi()

    mass_series, wall_series, nit_series, clamp_series = [], [], [], []
    died, death = False, None
    t0 = time.perf_counter()
    for k in range(args.steps):
        try:
            st.step()
        except RuntimeError as e:
            died, death = True, f"{type(e).__name__}: {e}"
            break
        if not np.isfinite(st.phi).all() or not np.isfinite(st.u).all():
            died, death = True, f"NaN at step {k + 1}"
            break
        mass_series.append(st.mass_phi())
        wall_series.append(float(st.last_wall))
        nit_series.append(int(st.last_newton_iters))
        clamp_series.append(int(st.last_clamped))
        print(f"[dam3d] step {k + 1}/{args.steps} t={st.t:.5f} "
              f"wall={st.last_wall:.1f}s newton={st.last_newton_iters} "
              f"mass={mass_series[-1]:.6e} clamped={clamp_series[-1]}",
              flush=True)
    total_wall = time.perf_counter() - t0
    steps_run = len(wall_series)

    mass_drift = (float(np.max(np.abs(np.asarray(mass_series) - mass0)))
                  if mass_series else float("nan"))
    rel_drift = mass_drift / (abs(mass0) or 1.0)

    # one VTU of the final state
    vtu_path = os.path.join(args.out, "dam3d_final.vtu")
    try:
        export_vtu(mesh, vtu_path,
                   fields={"phi": st.phi,
                           "u": np.column_stack([st.u,
                                                 np.zeros(len(st.u))])[:, :3]
                           if st.u.shape[1] == 2 else st.u,
                           "p": st.p})
        vtu_written = os.path.exists(vtu_path)
        vtu_mb = os.path.getsize(vtu_path) / 1e6 if vtu_written else None
    except Exception as e:
        vtu_written, vtu_mb = False, None
        death = death or f"VTU export failed: {type(e).__name__}: {e}"

    metrics = dict(
        case=case.name, dim=case.dim, level=args.level, device=args.device,
        linsolver=args.linsolver,
        dt=dt, n_nodes=int(n_nodes), n_dof=int(ndof), blk=int(blk),
        steps_requested=args.steps, steps_run=steps_run,
        build_wall_s=build_wall, compile_wall_s=compile_wall,
        total_march_wall_s=total_wall,
        wall_per_step_s=(float(np.mean(wall_series)) if wall_series
                         else None),
        newton_iters=nit_series, clamp_series=clamp_series,
        mass0=mass0, mass_series=mass_series, mass_drift=mass_drift,
        rel_mass_drift=rel_drift, died=died, death_reason=death,
        no_nan=(not died), vtu_written=vtu_written, vtu_mb=vtu_mb,
        gpu_mem_used_mib=_gpu_mem_mib(args.device),
    )
    jpath = os.path.join(args.out, "dam3d_metrics.json")
    with open(jpath, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ("mass_series", "newton_iters",
                                   "clamp_series")}, indent=2))
    print(f"metrics -> {jpath}")
    print(f"vtu     -> {vtu_path} ({vtu_mb} MB)" if vtu_written
          else "vtu     -> NOT WRITTEN")
    return 0 if (not died and vtu_written) else 1


if __name__ == "__main__":
    sys.exit(main())
