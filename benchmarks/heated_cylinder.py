"""M2-A3 composition (WIP — advective Nitsche faces pending): heated
immersed cylinder. STATUS: field solves (T in [0,1.22]); flux functional
on the traction convention; measured Q ~ 1e-3 vs ~1 expected with
per-face gradient wiggle -> DIAGNOSIS: the reused POISSON Nitsche faces
lack the advective boundary term ((a.n)-weighted inflow treatment; the
vector SBM has the backflow analogue). That term is the remaining A3
composition physics — see the overnight runbook.

Original doc: heated immersed cylinder (SBM Dirichlet T=1) in a
frozen uniform stream — the SBM+thermal smoke, plus the Nu surface-flux
functional (area-corrected). The full coupled Nu(Re) correlation gate
runs this with the NS velocity field.
"""
import sys
import numpy as np
from scipy.sparse.linalg import splu

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


def run(level=6, kappa=0.05, device="cuda:0"):
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
                      alpha=20.0)
    A_f, b_f = prob.face_system()
    A = (A_v + A_f).tolil()
    b = b_v + b_f
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
    print(f"L{level}: T range [{T_all.min():.3f}, {T_all.max():.3f}], "
          f"surface heat flux Q = {nu_val:.4f} "
          f"(Nu = {nu_val / (kappa * 1.0):.2f} vs 2D conduction-limit "
          f"~O(1-10 at Pe={1.0 * 2 * R / kappa:.0f}))", flush=True)
    return nu_val


if __name__ == "__main__":
    lv = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    run(lv)
