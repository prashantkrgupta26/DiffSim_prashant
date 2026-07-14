"""OrgElMorph course - C7: the workload Nsight Systems profiles.

A DELIBERATELY MINIMAL program: build the mesh, JIT-warm the kernels with one
throwaway step, then profile EXACTLY ONE complete Cahn-Hilliard time step
wrapped in an NVTX range.  Keeping the profiled region to a single step is the
whole point -- a profile of thousands of steps is unreadable; a profile of one
step tells you where a step's time goes.

Run it directly for a plain timed step, or under Nsight Systems to capture the
device timeline::

    nsys profile --trace=cuda,nvtx --sample=none --force-overwrite=true \\
        -o outputs/c7/step_profile \\
        <repo>/.venv/bin/python nsys_workload.py --device cuda:0 --level 6
    nsys stats outputs/c7/step_profile.nsys-rep

NVTX ranges (visible in the Nsight timeline and in ``nsys stats``) mark
``warmup`` and the profiled ``ch_step``.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from diffsim.octree.build import build_uniform          # noqa: E402
from diffsim.mesh.nodes import build_mesh               # noqa: E402
from diffsim.mesh.constraints import build_constraints  # noqa: E402
from diffsim.mesh.basis import basis_tables             # noqa: E402
from diffsim.assembly.operators import DeviceMesh       # noqa: E402
from diffsim.physics.cahn_hilliard import CahnHilliardStepper  # noqa: E402


def _nvtx_range(name):
    """A no-op-safe NVTX range context (torch.cuda.nvtx if available)."""
    try:
        import torch
        return torch.cuda.nvtx.range(name)
    except Exception:
        import contextlib
        return contextlib.nullcontext()


def build_stepper(level, device):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    st = CahnHilliardStepper(dm, 1.0, 5e-4, 5e-3, order=1, newton_tol=1e-10,
                             linsolver="splu")
    rng = np.random.default_rng(0)
    st.set_initial(lambda x: 0.1 * np.cos(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
                   + 0.02 * rng.standard_normal(len(x)), mu_init="consistent")
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--level", type=int, default=6)
    args = ap.parse_args()

    st = build_stepper(args.level, args.device)

    # JIT-warm: the FIRST step compiles the Warp Newton kernel; exclude it.
    with _nvtx_range("warmup"):
        st.step()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        torch = None

    # the ONE profiled step
    with _nvtx_range("ch_step"):
        st.step()

    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    print("profiled one CH step at level", args.level)


if __name__ == "__main__":
    main()
