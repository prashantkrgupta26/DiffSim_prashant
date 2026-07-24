"""M2-A3 composition (WIP — advective Nitsche faces pending): heated
immersed cylinder. STATUS 2 (advective upwind faces IN): T [0,1.16], Q = 0.032 — the
advective term lifted Q 30x but it remains ~10x under the correlation
estimate (Nu_D ~ 2 -> Q ~ Nu*pi*kappa ~ 0.3). NEXT SUSPECT: flux
EXTRACTION — direct gradient at surrogate faces is the naive method;
implement CONSISTENT-FLUX (residual-based) extraction (Q from the
discrete residual paired with the boundary indicator function — the
standard Nitsche/immersed-flux recovery, typically +1 order) and
compare; then the L5/L6/L7 flux-convergence ladder decides.

Original doc: heated immersed cylinder (SBM Dirichlet T=1) in a
frozen uniform stream — the SBM+thermal smoke, plus the Nu surface-flux
functional (area-corrected). The full coupled Nu(Re) correlation gate
runs this with the NS velocity field.
"""

import os as _bos, sys as _bsys  # noqa: E402  (benchmark import bootstrap)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import _bench_bootstrap  # noqa: E402,F401
import sys
import numpy as np
from scipy.sparse.linalg import splu

from diffsim import default_device
from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.mesh.faces import face_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.geometry.csg import Sphere
from diffsim.sbm.surrogate import (classify_lambda, extract_surrogate,
                                   GeometryData)
from diffsim.sbm.poisson import SBMPoisson
from diffsim.physics.scalar_transport import assemble_scalar_ad
from diffsim.physics.poisson import gauss_points

R, CTR = 0.15, (0.34, 0.5)          # non-dyadic (findings 4b(h) rule)


def heat_flux_functional(dm, sf, geo, kappa):
    """LINEAR heat-flux QoI (mirrors traction_functional): Q = w . T_all
    with Q = integral over the true boundary of -kappa dT/dn (true
    normal, area-corrected surrogate measure). Linear => adjoint-ready:
    dQ/dT = w."""
    from diffsim.mesh.faces import face_tables as _ft
    dim = dm.dim
    mesh = dm.mesh
    pv = int(np.unique(np.asarray(mesh.p_elem)[sf.elem])[0])
    ftab = _ft(pv, dim)
    nqf, nbf = ftab.nqf, ftab.nbf
    conn = mesh.conn_of[pv][np.searchsorted(mesh.bins[pv], sf.elem)]
    h = mesh.tree.h()[sf.elem]
    jacS = (h / 2.0) ** (dim - 1)
    dscale = 2.0 / h
    w_vec = np.zeros(dm.n_nodes)
    for fi in range(len(sf.elem)):
        f = int(sf.face[fi])
        dN = ftab.dN[f] * dscale[fi]
        for q in range(nqf):
            wq = ftab.w[q] * jacS[fi] * geo.corr[fi * nqf + q]
            n = geo.n[fi * nqf + q]
            dn_n = dN[q] @ n
            for b in range(nbf):
                w_vec[conn[fi, b]] += -kappa * wq * dn_n[b]
    return w_vec


def advective_faces(dm, sf, geo, a_fn, g_fn):
    """Weak-Dirichlet ADVECTIVE term on surrogate faces (the diagnosed
    A3 gap; Bazilevs-style upwind): -(w, (a.n)^- (T - g)) with
    (a.n)^- = min(a.n, 0), n = surrogate outward normal. Returns
    (A_adv csr constrained, b_adv)."""
    import scipy.sparse as sp
    from diffsim.mesh.faces import face_tables as _ft
    dim = dm.dim
    mesh = dm.mesh
    pv = int(np.unique(np.asarray(mesh.p_elem)[sf.elem])[0])
    ftab = _ft(pv, dim)
    nqf, nbf = ftab.nqf, ftab.nbf
    conn = mesh.conn_of[pv][np.searchsorted(mesh.bins[pv], sf.elem)]
    h = mesh.tree.h()[sf.elem]
    jacS = (h / 2.0) ** (dim - 1)
    rows, cols, vals = [], [], []
    b_full = np.zeros(dm.n_nodes)
    for fi in range(len(sf.elem)):
        f = int(sf.face[fi])
        ax, side = f // 2, f % 2
        n_s = np.zeros(dim)
        n_s[ax] = -1.0 if side == 0 else 1.0
        N = ftab.N[f]
        for q in range(nqf):
            gp = fi * nqf + q
            xq_ = geo.xq[gp] if hasattr(geo, "xq") else None
            a_q = a_fn(gp)
            an = float(a_q @ n_s)
            an_m = min(an, 0.0)
            if an_m == 0.0:
                continue
            wq = ftab.w[q] * jacS[fi]
            g_q = g_fn(gp)
            for a_ in range(nbf):
                for b_ in range(nbf):
                    rows.append(conn[fi, a_]); cols.append(conn[fi, b_])
                    vals.append(-an_m * N[q, a_] * N[q, b_] * wq)
                b_full[conn[fi, a_]] += -an_m * N[q, a_] * g_q * wq
    K = sp.coo_matrix((vals, (rows, cols)),
                      shape=(dm.n_nodes, dm.n_nodes)).tocsr()
    T = dm.constraints.T.tocsr()
    return (T.T @ K @ T).tocsr(), np.asarray(T.T @ b_full)


def run(level=6, kappa=0.05, device=None):
    device = default_device() if device is None else device
    tree = build_uniform(level, dim=2)
    oracle = Sphere(CTR, R)
    ret, _ = classify_lambda(tree, oracle, 1.0, domain="outside")
    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, 2), domain="outside")

    # frozen uniform stream (the coupled version swaps in the NS field)
    xq = gauss_points(mesh, dm.tables_by_p)
    aq = {pv: np.tile([1.0, 0.0], (len(xq[pv]), 1)) for pv in xq}
    fq = {pv: np.zeros(len(xq[pv])) for pv in xq}
    A_v, b_v = assemble_scalar_ad(dm, aq, fq, kappa, sigma=0.0)

    prob = SBMPoisson(dm, geo=geo, sf=sf,
                      g_fn=lambda x: np.ones(len(x)), kappa=kappa,
                      # Peclet-aware Nitsche penalty (L7 isolation,
                      # 2026-07-07: alpha=20 blew up at L7 WITHOUT the
                      # advective faces — under-penalization once
                      # advection consumes coercivity margin; 50 is
                      # bounded at L5-L7): alpha ~ 20*(1 + Pe_face)
                      alpha=20.0 * (1.0 + 1.0 * 2 * R / kappa / 4.0))
    A_f, b_f = prob.face_system()
    A_a, b_a = advective_faces(dm, sf, geo,
                               a_fn=lambda gp: np.array([1.0, 0.0]),
                               g_fn=lambda gp: 1.0)
    A = (A_v + A_f + A_a).tolil()
    b = b_v + b_f + b_a
    coords = mesh.node_coords[cons.free_nodes]
    inflow = np.abs(coords[:, 0]) < 1e-12
    for i in np.where(inflow)[0]:
        A.rows[i] = [int(i)]
        A.data[i] = [1.0]
        b[i] = 0.0                                   # cold inflow
    T_free = splu(A.tocsr().tocsc()).solve(b)
    T_all = np.asarray(cons.T @ T_free)
    w_q = heat_flux_functional(dm, sf, geo, kappa)
    nu_val = float(w_q @ T_all)
    # CONSISTENT-FLUX extraction: pair the RAW interior residual (volume
    # operator only, no face blocks/strong rows) with the boundary
    # indicator chi (1 at surrogate-face nodes). For the exact solution
    # the interior residual vanishes on interior test functions; against
    # chi it equals the TOTAL boundary flux (diffusive + advective) —
    # the standard Nitsche/immersed flux recovery (typically +1 order
    # vs direct gradients). NOTE: with the frozen penetrating stream the
    # advective part is nonzero; in the COUPLED (no-slip) run
    # Q_consistent ~ the diffusive Nu.
    fnodes = np.unique(dm.mesh.conn_of[1][
        np.searchsorted(dm.mesh.bins[1], sf.elem)].ravel())
    chi_full = np.zeros(dm.n_nodes)
    chi_full[fnodes] = 1.0
    chi = np.asarray(cons.T.T @ chi_full)
    q_cons = float(chi @ (A_v @ T_free - b_v))
    print(f"L{level}: T [{T_all.min():.3f},{T_all.max():.3f}] "
          f"Q_direct={nu_val:.4f} Q_consistent={q_cons:.4f} "
          f"(corr est ~{2.0 * np.pi * kappa:.2f})", flush=True)
    return nu_val, q_cons


if __name__ == "__main__":
    lv = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    run(lv)
