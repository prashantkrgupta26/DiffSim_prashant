"""G5 measured ladder runner: 3-D Wodo film (wodo_nova fig3d/stretch
physics) at arbitrary nx x ny x nz ELEMENT resolution through the
device-resident blockch preconditioner + node-graph device assembly.

Rungs a/b (96x96x32, 128x128x48) are committed in ff6f3c1; this runner
exists for rung c — the paper's FULL-RES mesh 230x230x70 (231x231x71
nodes = 15,154,524 dofs) on one 48 GB card — and for the host-memory
gate sizes (144x144x48 = 4.12M dofs, 160x160x48 = 5.08M dofs) that the
old dof-COO pattern build exit-137'd at 62 GB host.

Physics (rung b's exactly): Np = Nf = 5, chi = (1, .3, .3), Bi = 0.3,
phi_s0 = 0.66, Lx = Ly = 3.3, kappa = kappa_of(5) = 2e-4, CHC noise
1e-3, var_mob (D_ratio 1e-3), b_reg = 1e-3, dt0 = 1e-4, noise_seed 17.

Run:  python benchmarks/wodo_g5_rung.py --nx 230 --nz 70 \
          --device cuda:1 --wall-cap 2400
"""

import os as _bos, sys as _bsys  # noqa: E402
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import _bench_bootstrap  # noqa: E402,F401
import argparse
import resource
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
LAT_PHYS = 3.3          # physical Lx = Ly (wodo_nova fig3d)
PHI_S0 = 0.66           # the paper's 3-D value


def host_peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


class TimedFilm(WodoFilmStepper):
    """Instrumented: wall split of the device blockch solve vs the rest
    (assembly + GP fields + flux adds) inside each attempt."""
    t_solve = 0.0
    n_solve = 0

    def _solve_device_blockch(self, asm):
        t0 = time.time()
        r = super()._solve_device_blockch(asm)
        self.t_solve += time.time() - t0
        self.n_solve += 1
        return r


class RecCache(dict):
    def __init__(self):
        super().__init__()
        self.iters = []

    def __setitem__(self, k, v):
        if isinstance(k, tuple) and k[0] == "blockch_iters":
            self.iters.append(v)
        super().__setitem__(k, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--ny", type=int, default=None)
    ap.add_argument("--nz", type=int, required=True)
    ap.add_argument("--device", type=str, default="cuda:1")
    ap.add_argument("--max-steps", type=int, default=200000)
    ap.add_argument("--wall-cap", type=float, default=2400.0)
    ap.add_argument("--dt", type=float, default=1e-4)
    args = ap.parse_args()
    ny = args.ny if args.ny is not None else args.nx
    nx, nz = args.nx, args.nz
    device = args.device
    wp.init()

    level = int(np.ceil(np.log2(max(nx, ny, nz))))
    hc = 2.0 ** -level
    t0 = time.time()
    tree0 = build_uniform(level, dim=3)
    cen = tree0.centers()
    keep = ((cen[:, 0] < nx * hc) & (cen[:, 1] < ny * hc)
            & (cen[:, 2] < nz * hc))
    tree = Octree(tree0.keys[keep], tree0.levels[keep], dim=3,
                  periodic=tree0.periodic)
    del tree0, cen, keep
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    ndofs = 4 * len(mesh.node_coords)
    print(f"g5 rung: {len(tree)} elements ({nx}x{ny}x{nz}), "
          f"{len(mesh.node_coords)} nodes, {ndofs} dofs, level={level}, "
          f"device={device}  [mesh {time.time() - t0:.0f}s, "
          f"host {host_peak_gb():.1f} GB]", flush=True)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=3), device)

    pp0 = pf0 = (1.0 - PHI_S0) / 2.0
    D0 = PHI_S0 * 1.0 + (1.0 - PHI_S0) * 1e-3
    M11 = D0 / (1.0 / (5.0 * pp0) + 1.0 / (NS * PHI_S0))
    M22 = D0 / (1.0 / (5.0 * pf0) + 1.0 / (NS * PHI_S0))
    kap = f67.kappa_of(5)
    lat_scale = LAT_PHYS / (nx * hc)
    st = TimedFilm(
        dm, chi=(1.0, 0.3, 0.3), N=(5.0, 5.0, NS),
        M=(M11, 0.0, M22), kappa=(kap, kap), k_e=0.3, dt=args.dt,
        lat_scale=lat_scale, linsolver="blockch", noise=1e-3,
        var_mob=True, b_reg=1e-3, noise_seed=17,
        use_device_assembly=True)
    st._solver_cache = RecCache()
    rng = np.random.default_rng(17)
    st.set_initial(
        lambda x: pp0 + 0.01 * rng.standard_normal(len(x)),
        lambda x: pf0 + 0.01 * rng.standard_normal(len(x)))
    print(f"stepper up: kappa={kap:.2e} M=({M11:.3f},{M22:.3f}) "
          f"lat_scale={lat_scale:.4f}  [host {host_peak_gb():.1f} GB]",
          flush=True)

    t0 = time.time()
    st._init_device_assembly()
    t_asm_setup = time.time() - t0
    gpu_hi = wp.get_mempool_used_mem_high(device)
    print(f"assembler setup {t_asm_setup:.0f}s: nnz={st._asm.nnz} "
          f"({st._asm.nnz / 2 ** 31 * 100:.0f}% of int32 range)  "
          f"[host {host_peak_gb():.1f} GB, gpu hi "
          f"{gpu_hi / 2 ** 30:.1f} GB]", flush=True)

    steps = []
    t_march0 = time.time()

    def cb(s, K, dt, iters):
        now = time.time()
        t_prev = steps[-1][0] if steps else t_march0
        steps.append((now, dt, iters))
        p1 = np.asarray(s.Tc @ s.x[0::4])
        p2 = np.asarray(s.Tc @ s.x[2::4])
        ps = 1.0 - p1 - p2
        print(f"  step {len(steps):4d} t={s.t:.5f} h={s.h_curr:.4f} "
              f"phi_s={ps.mean():.4f} dt={dt:.2e} newton={iters} "
              f"[{now - t_prev:.0f}s step, solve {s.t_solve:.0f}s cum, "
              f"host {host_peak_gb():.1f} GB, gpu hi "
              f"{wp.get_mempool_used_mem_high(device) / 2 ** 30:.1f} GB]",
              flush=True)

    reason = st.march(h_min=0.30, phis_stop=0.05,
                      max_steps=args.max_steps, callback=cb,
                      wall_cap=args.wall_cap)
    wall = time.time() - t_march0
    its = st._solver_cache.iters
    outer = [i[0] % 1000 for i in its]
    fb = sum(1 for i in its if i[0] >= 1000)
    n = len(steps)
    print(f"\nRESULT rung {nx}x{ny}x{nz}: dofs={ndofs} nnz={st._asm.nnz} "
          f"steps={n} rejects={st.n_reject} reason={reason} "
          f"wall={wall:.0f}s s/step={wall / max(n, 1):.1f} "
          f"(steady={np.median(np.diff([t_march0] + [s[0] for s in steps])):.1f}) "
          f"solve={st.t_solve:.0f}s/{st.n_solve} solves "
          f"setup={t_asm_setup:.0f}s outer_max={max(outer) if outer else 0} "
          f"outer={outer[:20]}{'...' if len(outer) > 20 else ''} "
          f"fallbacks={fb} host_peak={host_peak_gb():.1f}GB "
          f"gpu_peak={wp.get_mempool_used_mem_high(device) / 2 ** 30:.1f}GB",
          flush=True)


if __name__ == "__main__":
    main()
