"""Task #41 G3: GP-buffer residency + per-step cost, persistent vs
batch_local, at a rung where the difference is visible.

Measures, for the SAME 3-D film config in each gp_residency mode:
  - GP-buffer residency = sum of nbytes over the packed GP buffers
    (_vals_dev + _grads_dev + _histv_dev + _hist_dev + _hist2v_dev +
    _q_dev), i.e. the residents #37 flagged as +6.3 GB extrapolated at
    rung-c.  This is the number the fallback trades against.
  - GPU mempool high-water (warp) over the timed steps.
  - wall s/step (warmup excluded) -> the per-step recompute cost of the
    batch-local re-eval.

Run (box):
  LD_LIBRARY_PATH=/usr/lib/wsl/lib .venv/bin/python \
      benchmarks/phase-field/wodo_gp_residency_mem.py \
      --nx 96 --nz 40 --device cuda:1 --warmup 2 --steps 5
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


def _gp_buf_bytes(st):
    """Bytes held in the packed GP residents (the buffers #41 shrinks)."""
    def nb(a):                       # bytes held by a wp.array (float64)
        return int(np.prod(a.shape)) * 8
    tot = 0
    for pv in st.dm.bins:
        tot += nb(st._vals_dev[pv]) + nb(st._grads_dev[pv])
        tot += nb(st._histv_dev[pv]) + nb(st._hist_dev[pv])
        tot += nb(st._q_dev[pv][0]) + nb(st._q_dev[pv][1])
        if st._hist2v_dev is not None and pv in st._hist2v_dev:
            tot += nb(st._hist2v_dev[pv])
    return tot


def _build(mesh, cons, device, nx, hc, residency):
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=3), device)
    pp0 = pf0 = (1.0 - PHI_S0) / 2.0
    D0 = PHI_S0 * 1.0 + (1.0 - PHI_S0) * 1e-3
    M11 = D0 / (1.0 / (5.0 * pp0) + 1.0 / (NS * PHI_S0))
    M22 = D0 / (1.0 / (5.0 * pf0) + 1.0 / (NS * PHI_S0))
    kap = f67.kappa_of(5)
    lat_scale = LAT_PHYS / (nx * hc)
    st = WodoFilmStepper(
        dm, chi=(1.0, 0.3, 0.3), N=(5.0, 5.0, NS), M=(M11, 0.0, M22),
        kappa=(kap, kap), k_e=0.3, dt=1e-4, lat_scale=lat_scale,
        linsolver="blockch", noise=1e-3, var_mob=True, b_reg=1e-3,
        noise_seed=17, use_device_assembly=True, gp_residency=residency)
    rng = np.random.default_rng(17)
    st.set_initial(
        lambda x: pp0 + 0.01 * rng.standard_normal(len(x)),
        lambda x: pf0 + 0.01 * rng.standard_normal(len(x)))
    return st


def _force_nbatch(st, nbatch):
    """Force ~nbatch element batches by capping the Ae/be batch size
    (mirrors how a much larger rung-c mesh naturally splits under the
    fixed ~2 GB Ae budget); grow the batch_local GP buffers to match."""
    d = st.dm.device
    st._init_device_assembly()
    for k_bin, (pv, b, ne, nbf, _g) in enumerate(st._asm._bins):
        cap = max(1, (ne + nbatch - 1) // nbatch)
        nl = 4 * nbf
        st._batch_bufs[k_bin] = (
            cap, wp.zeros((cap, nl, nl), dtype=wp.float64, device=d),
            wp.zeros((cap, nl), dtype=wp.float64, device=d))
        if st._gp_batch_local:
            st._grow_gp_batch_bufs(pv, cap * b["nqp"])


def _run(mesh, cons, device, nx, hc, residency, warmup, steps, nbatch):
    st = _build(mesh, cons, device, nx, hc, residency)
    if nbatch:
        _force_nbatch(st, nbatch)
    steps_log = []

    def cb(s, K, dt, iters):
        steps_log.append(time.time())

    st.march(h_min=0.30, phis_stop=0.05, max_steps=warmup, callback=cb)
    # GP-buffer residency is fixed after _init_device_fields (persistent)
    # or bounded to one batch (batch_local): report it post-warmup.
    gp_gb = _gp_buf_bytes(st) / 2 ** 30
    n0 = len(steps_log)
    wp.synchronize_device(device)
    t0 = time.time()
    st.march(h_min=0.30, phis_stop=0.05, max_steps=steps, callback=cb)
    wp.synchronize_device(device)
    wall = time.time() - t0
    nst = len(steps_log) - n0
    peak_gb = wp.get_mempool_used_mem_high(device) / 2 ** 30
    return dict(gp_gb=gp_gb, peak_gb=peak_gb,
                sstep=wall / max(nst, 1), nst=nst,
                nbatch=_nbatch(st))


def _nbatch(st):
    nb = {}
    for k_bin, (pv, b, ne, nbf, _g) in enumerate(st._asm._bins):
        cap = (st._batch_bufs[k_bin][0] if k_bin in st._batch_bufs
               else st._nb_cap(ne, nbf))
        nb[pv] = (ne + cap - 1) // cap
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=96)
    ap.add_argument("--ny", type=int, default=None)
    ap.add_argument("--nz", type=int, default=40)
    ap.add_argument("--device", type=str, default="cuda:1")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--nbatch", type=int, default=0,
                    help="force ~N element batches (0 = the ~2GB Ae "
                         "budget default); larger N mirrors rung-c's "
                         "natural batch count so the ~1/nbatch residency "
                         "scaling is visible on a smaller card")
    args = ap.parse_args()
    ny = args.ny if args.ny is not None else args.nx
    nx, nz, device = args.nx, args.nz, args.device
    wp.init()

    level = int(np.ceil(np.log2(max(nx, ny, nz))))
    hc = 2.0 ** -level
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
    print(f"mem bench mesh: {len(tree)} elements ({nx}x{ny}x{nz}), "
          f"{ndofs} dofs, device={device}", flush=True)

    res = {}
    for mode in ("persistent", "batch_local"):
        r = _run(mesh, cons, device, nx, hc, mode, args.warmup,
                 args.steps, args.nbatch)
        res[mode] = r
        print(f"\n[{mode}] nbatch={r['nbatch']}  "
              f"GP-buffer residency {r['gp_gb']:.3f} GB  "
              f"mempool peak {r['peak_gb']:.2f} GB  "
              f"{r['sstep']:.2f} s/step ({r['nst']} steps)", flush=True)

    p, b = res["persistent"], res["batch_local"]
    saved = p["gp_gb"] - b["gp_gb"]
    ratio = b["gp_gb"] / p["gp_gb"] if p["gp_gb"] else float("nan")
    dcost = b["sstep"] - p["sstep"]
    print(f"\n== #41 G3 SUMMARY ==")
    print(f"  GP-buffer residency: persistent {p['gp_gb']:.3f} GB "
          f"-> batch_local {b['gp_gb']:.3f} GB  "
          f"(saved {saved:.3f} GB; batch_local = {ratio:.3f}x = "
          f"~1/{1 / ratio:.1f} of persistent)")
    print(f"  mempool peak: persistent {p['peak_gb']:.2f} GB "
          f"-> batch_local {b['peak_gb']:.2f} GB")
    print(f"  per-step cost: persistent {p['sstep']:.2f} "
          f"-> batch_local {b['sstep']:.2f} s/step  "
          f"(delta {dcost:+.2f} s/step, "
          f"{100 * dcost / max(p['sstep'], 1e-9):+.1f}%)")


if __name__ == "__main__":
    main()
