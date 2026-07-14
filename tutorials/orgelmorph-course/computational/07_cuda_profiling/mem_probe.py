"""OrgElMorph course - C7: isolated device-memory footprint probe.

Warp's device mempool caches freed allocations, so an IN-PROCESS memory delta
(after other benchmarks have grown the pool) reads ~0.  This tiny script runs
in a FRESH process: it records the driver-level used memory right after the
CUDA context + Warp are ready, builds one Cahn-Hilliard problem and takes one
step, and reports the memory the CH problem actually added.  Emits one JSON
line so ``run.py`` can read it back.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def used_bytes():
    import torch
    free, total = torch.cuda.mem_get_info()
    return int(total - free), int(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--level", type=int, default=6)
    args = ap.parse_args()

    import torch  # noqa: F401  (context init)
    import warp as wp
    wp.init()
    # touch the device so the CUDA context + a minimal Warp pool exist
    _ = wp.zeros(1, dtype=wp.float64, device=args.device)
    wp.synchronize()
    base, total = used_bytes()

    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh
    from diffsim.physics.cahn_hilliard import CahnHilliardStepper

    tree = build_uniform(args.level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), args.device)
    st = CahnHilliardStepper(dm, 1.0, 5e-4, 5e-3, order=1, newton_tol=1e-10,
                             linsolver="splu")
    rng = np.random.default_rng(0)
    st.set_initial(lambda x: 0.1 * np.cos(np.pi * x[:, 0]) + 0.02
                   * rng.standard_normal(len(x)), mu_init="consistent")
    st.step()
    wp.synchronize()
    peak, _ = used_bytes()

    print(json.dumps({
        "level": args.level,
        "base_mb": base / 1e6,
        "peak_mb": peak / 1e6,
        "footprint_mb": (peak - base) / 1e6,
        "total_mb": total / 1e6,
    }))


if __name__ == "__main__":
    main()
