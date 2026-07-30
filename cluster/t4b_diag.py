"""T4b diagnostic probe: assemble the truck saddle at step 0, dump diagonal
statistics (min/max/percentiles split by dof type u vs p and by node level),
then run ONE fgmres_bdiag solve capturing the relres-vs-iteration curve.

Purpose: confirm/refute the diagonal-range hypothesis (Cb_f=20 Nitsche
penalties on level-12 surrogate faces vs level-7 bulk => many-order diagonal
span => scalar-Jacobi floor) BEFORE writing the equilibration fix.

Run (inside srun --overlap on the hold node):
    PYTHONPATH=src:tests python cluster/t4b_diag.py
"""
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

os.environ.setdefault("SADDLE_DEVICE_CSR", "0")   # host CSR: we need A.diagonal()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CONF = os.environ.get("TRUCK_CONFIG", os.path.join(
    os.path.dirname(__file__), "..", "local_code_old",
    "truck_4case_fresh_inputs", "NewRun-no-shell-slope0p25", "config.txt"))
BASE_LEVEL = int(os.environ.get("BASE_LEVEL", "7"))
BAND_TO = int(os.environ.get("BAND_TO", "12"))
DEVICE = os.environ.get("DIAG_DEVICE", "cuda")

from diffsim.cases.truck_config import load_truck_config
from diffsim.octree.build import build_uniform
from truck_flow import (build_truck_mesh, truck_strong_bc, _pressure_pin,
                        make_nu_schedule)
from diffsim.sbm.vector import sbm_vector_dirichlet
from diffsim.api.ns_bricks import assemble_linear_ns
from diffsim.solvers.timestepping import bdf_coeffs


def pct(a, ps):
    a = np.asarray(a)
    if a.size == 0:
        return {p: float("nan") for p in ps}
    return {p: float(np.percentile(a, p)) for p in ps}


def main():
    t0 = time.time()
    cfg = load_truck_config(CONF)
    dim = 3
    ndof = dim + 1
    nu_sched = make_nu_schedule(cfg, U_inf=1.0, L_ref=1.0)
    nu0 = float(nu_sched(0.0))
    alpha = cfg.cb_f
    dt = float(cfg.dt_v[1])
    print(f"[diag] cfg bodies={len(cfg.bodies)} Cb_f={alpha} nu0={nu0:.4f} "
          f"Re0={1.0/nu0:.0f} dt={dt}", flush=True)

    fx = build_truck_mesh(cfg, BASE_LEVEL, region_refine=True,
                          truck_band_to=BAND_TO, band_cells=3, device=DEVICE)
    dm, mesh, cons, sf, geo = (fx["dm"], fx["mesh"], fx["cons"],
                               fx["sf"], fx["geo"])
    scale = fx["scale"]
    print(f"[diag] mesh cells={fx['n_cells']} carved={fx['n_excluded']} "
          f"slab_cut={fx['n_slab_cut']} sf_faces={sf.elem.size} "
          f"({time.time()-t0:.0f}s)", flush=True)

    T = cons.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    nfree = T.shape[1]
    N = nfree * ndof
    print(f"[diag] free_nodes={nfree}  N(free DOFs)={N}", flush=True)

    # per-free-node finest incident cell level (max over incident cells) ->
    # DOUBLED to per-dof via node-major interleave.  conn_of maps p-bin
    # elements to FULL node ids; free_nodes selects the free subset.
    lev_cell = mesh.tree.levels.astype(np.int64)
    node_lev_full = np.zeros(len(mesh.node_coords), np.int64)
    for pv in dm.bins:
        eids = dm.bins[pv]["eids"]
        conn = mesh.conn_of[pv]              # [nb, npe] full-node ids
        el = lev_cell[eids]
        np.maximum.at(node_lev_full, conn.ravel(),
                      np.repeat(el, conn.shape[1]))
    node_lev = node_lev_full[cons.free_nodes]     # [nfree]

    # BCs / SBM at step 0 (order-1 BDF, u_pre=0)
    bc_rows, bc_vals, masks, coords = truck_strong_bc(
        mesh, cons, ndof, dim, scale, cfg.slope_near_ground, cfg.domain_max)
    p_pin = _pressure_pin(coords, ndof, dim)
    noslip = lambda y: np.zeros((len(y), dim))
    Af_raw, bf_raw = sbm_vector_dirichlet(dm, sf, geo, noslip, nu0, ndof,
                                          alpha=alpha)
    Af_c = (T_vec.T @ Af_raw @ T_vec).tocsr()
    bf_c = np.asarray(T_vec.T @ bf_raw)

    b0, b1, b2 = bdf_coeffs(1, dt)
    sigma = b0 / dt
    aq = {}; dq = {}
    for pv in dm.bins:
        tb = dm.tables_by_p[pv]
        z = np.zeros((mesh.conn_of[pv].shape[0], tb.N.shape[0], dim))
        aq[pv] = z.reshape(-1, dim)
        dq[pv] = np.zeros(z.shape[0] * z.shape[1])
    fq_raw = {}
    for pv in dm.bins:
        tb = dm.tables_by_p[pv]
        fq_raw[pv] = np.zeros((mesh.conn_of[pv].shape[0], tb.N.shape[0], dim)
                              ).reshape(-1, dim)

    A, b = assemble_linear_ns(dm, aq, dq, fq_raw, nu0, sigma=sigma)
    A = (A + Af_c).tolil()
    b = b + bf_c
    for r, v in zip(bc_rows, bc_vals):
        A.rows[r] = [int(r)]; A.data[r] = [1.0]; b[r] = v
    A.rows[p_pin] = [p_pin]; A.data[p_pin] = [1.0]; b[p_pin] = 0.0
    Acsr = A.tocsr()
    print(f"[diag] assembled A: shape={Acsr.shape} nnz={Acsr.nnz} "
          f"({time.time()-t0:.0f}s)", flush=True)

    d = np.asarray(Acsr.diagonal()).copy()
    ad = np.abs(d)
    p_mask = np.zeros(N, dtype=bool)
    p_mask[np.arange(nfree) * ndof + dim] = True
    u_mask = ~p_mask
    dof_lev = np.repeat(node_lev, ndof)          # [N] per-dof level

    print("=" * 72, flush=True)
    print("[diag] DIAGONAL STATISTICS  |diag(A)|", flush=True)
    ps = [0, 1, 50, 99, 100]
    for name, m in (("ALL", np.ones(N, bool)), ("u", u_mask), ("p", p_mask)):
        am = ad[m]
        nz = am[am > 0]
        print(f"  [{name:3s}] n={m.sum():>9d}  min={am.min():.3e} "
              f"max={am.max():.3e}  span={am.max()/max(am.min(),1e-300):.3e}",
              flush=True)
        pc = pct(am, ps)
        print(f"        pct {ps} = "
              f"{[f'{pc[p]:.3e}' for p in ps]}", flush=True)
        print(f"        n(|d|==0)={int((am==0).sum())}  "
              f"nz-min={nz.min() if nz.size else float('nan'):.3e}",
              flush=True)

    print("[diag] |diag(A)| BY NODE LEVEL (u dofs):", flush=True)
    for lv in sorted(np.unique(dof_lev)):
        sel = u_mask & (dof_lev == lv)
        if sel.sum() == 0:
            continue
        am = ad[sel]
        print(f"  lvl={lv:>2d}  n_u={sel.sum():>9d}  min={am.min():.3e} "
              f"max={am.max():.3e}  med={np.median(am):.3e}", flush=True)
    print("[diag] |diag(A)| BY NODE LEVEL (p dofs):", flush=True)
    for lv in sorted(np.unique(dof_lev)):
        sel = p_mask & (dof_lev == lv)
        if sel.sum() == 0:
            continue
        am = ad[sel]
        print(f"  lvl={lv:>2d}  n_p={sel.sum():>9d}  min={am.min():.3e} "
              f"max={am.max():.3e}  med={np.median(am):.3e}", flush=True)

    # surrogate-face node subset: nodes incident to a surrogate face carry
    # the Cb_f Nitsche penalty -> the suspected large-diagonal rows.
    print("=" * 72, flush=True)
    print("[diag] RELRES-VS-ITERATION CURVE (fgmres_bdiag, current path):",
          flush=True)
    from diffsim.assembly.operators import CSROperator
    from diffsim.solvers.fgmres_dev import fgmres_dev
    from diffsim.solvers.saddle_precond import make_bdiag_apply
    import warp as wp
    op = CSROperator(Acsr, DEVICE)
    apply_dev = make_bdiag_apply(Acsr, ndof, DEVICE, block="scalar")
    b_dev = wp.array(np.ascontiguousarray(b, np.float64), dtype=wp.float64,
                     device=DEVICE)
    # capture the curve: run with restart=60, a handful of cycles, print each
    # cycle's entry relres by calling fgmres_dev with maxiter=1 repeatedly and
    # warm-starting.  Cheaper: single call, rely on info; but we want the
    # curve, so step cycles.
    x0 = None
    for cyc in range(1, 9):
        xd, info = fgmres_dev(op.matvec, b_dev, apply_dev, N, DEVICE,
                              tol=1e-8, atol=1e-13, restart=60, maxiter=cyc,
                              x0_dev=x0)
        print(f"  cycles={cyc:>2d}  inner={info['inner']:>4d}  "
              f"relres={info['relres']:.6e}  converged={info['converged']}",
              flush=True)
        x0 = xd
        if info["converged"]:
            break
    print(f"[diag] DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
