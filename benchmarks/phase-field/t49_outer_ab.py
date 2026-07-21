"""Task #49 G3: clean A/B of the device-resident OUTER FGMRES vs the host
scipy-lgmres outer, on the rung-b film config (128x128x48, blockch +
device assembly).  NO cProfile / launch-audit overhead — synchronized
wall timing only, so the number is the honest production s/step.  Both
arms run back-to-back in ONE process (same seeds, same Newton ladder),
env-toggled: the DEV_OUTER value is read at stepper construction.

Run (box):
  LD_LIBRARY_PATH=/usr/lib/wsl/lib .venv/bin/python \
      benchmarks/phase-field/t49_outer_ab.py --nx 128 --nz 48 \
      --device cuda:1 --warmup 2 --steps 5
"""
import os as _bos, sys as _bsys  # noqa: E402
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import _bench_bootstrap  # noqa: E402,F401
import argparse
import time

import numpy as np
import warp as wp

from diffsim.octree.build import build_uniform, Octree
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.wodo_film import WodoFilmStepper

import wodo_fig67 as f67

NS = f67.NS
LAT_PHYS = 3.3
PHI_S0 = 0.66


def _build(nx, ny, nz, device):
    level = int(np.ceil(np.log2(max(nx, ny, nz))))
    hc = 2.0 ** -level
    tree0 = build_uniform(level, dim=3)
    cen = tree0.centers()
    keep = ((cen[:, 0] < nx * hc) & (cen[:, 1] < ny * hc)
            & (cen[:, 2] < nz * hc))
    tree = Octree(tree0.keys[keep], tree0.levels[keep], dim=3,
                  periodic=tree0.periodic)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=3), device)
    return dm, mesh, hc


def _stepper(dm, mesh, hc, nx, dt):
    pp0 = pf0 = (1.0 - PHI_S0) / 2.0
    D0 = PHI_S0 * 1.0 + (1.0 - PHI_S0) * 1e-3
    M11 = D0 / (1.0 / (5.0 * pp0) + 1.0 / (NS * PHI_S0))
    M22 = D0 / (1.0 / (5.0 * pf0) + 1.0 / (NS * PHI_S0))
    kap = f67.kappa_of(5)
    lat_scale = LAT_PHYS / (nx * hc)
    st = WodoFilmStepper(
        dm, chi=(1.0, 0.3, 0.3), N=(5.0, 5.0, NS),
        M=(M11, 0.0, M22), kappa=(kap, kap), k_e=0.3, dt=dt,
        lat_scale=lat_scale, linsolver="blockch", noise=1e-3,
        var_mob=True, b_reg=1e-3, noise_seed=17, use_device_assembly=True)
    rng = np.random.default_rng(17)
    st.set_initial(
        lambda x: pp0 + 0.01 * rng.standard_normal(len(x)),
        lambda x: pf0 + 0.01 * rng.standard_normal(len(x)))
    return st


def _time_march(st, device, warmup, steps, dt):
    steps_rec = []

    def cb(s, K, d, iters):
        steps_rec.append((time.time(), d, iters))

    st.march(h_min=0.30, phis_stop=0.05, max_steps=warmup, callback=cb)
    wp.synchronize_device(device)
    n0 = len(steps_rec)
    t0 = time.time()
    st.march(h_min=0.30, phis_stop=0.05, max_steps=steps, callback=cb)
    wp.synchronize_device(device)
    wall = time.time() - t0
    nst = len(steps_rec) - n0
    iters = [s[2] for s in steps_rec[n0:]]
    return wall, nst, iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=None)
    ap.add_argument("--nz", type=int, default=48)
    ap.add_argument("--device", type=str, default="cuda:1")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--dt", type=float, default=1e-4)
    args = ap.parse_args()
    ny = args.ny if args.ny is not None else args.nx
    device = args.device
    wp.init()

    dm, mesh, hc = _build(args.nx, ny, args.nz, device)
    print(f"AB mesh: {4 * len(mesh.node_coords)} dofs, device={device}",
          flush=True)

    results = {}
    for mode, val in (("host_lgmres", "0"), ("dev_outer", "1")):
        _bos.environ["DIFFSIM_PRECOND_DEV_OUTER"] = val
        st = _stepper(dm, mesh, hc, args.nx, args.dt)
        assert st.precond_dev_outer == (val == "1")
        wall, nst, iters = _time_march(st, device, args.warmup, args.steps,
                                       args.dt)
        peak = wp.get_mempool_used_mem_high(device) / 2 ** 30
        results[mode] = (wall, nst, iters, peak)
        print(f"[{mode:11s}] wall {wall:.1f}s / {nst} steps = "
              f"{wall / max(nst, 1):.2f} s/step  newton {iters}  "
              f"gpu peak {peak:.1f} GB", flush=True)

    wh = results["host_lgmres"][0] / max(results["host_lgmres"][1], 1)
    wd = results["dev_outer"][0] / max(results["dev_outer"][1], 1)
    print(f"\n== T49 G3 A/B ==")
    print(f"  host lgmres outer : {wh:.2f} s/step")
    print(f"  device FGMRES outer: {wd:.2f} s/step")
    print(f"  delta: {wd - wh:+.2f} s/step ({100 * (wd - wh) / wh:+.1f}%)")


if __name__ == "__main__":
    main()
