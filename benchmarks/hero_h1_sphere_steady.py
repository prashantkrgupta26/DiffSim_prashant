"""Hero H1: recover a hidden GENIE edit of the sphere INR from steady
flow probes (plan 2026-07-06-m1c-hero). 3-D, L4 warmup -> L5, Re~100
confined box flow, steady Picard, probe-mismatch QoI, adjoint gradients
in alpha-space.

Run: python benchmarks/hero_h1_sphere_steady.py [level] [n_epochs]
"""
import sys

import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse.linalg import splu

sys.path.insert(0, "tests")
from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.mesh.faces import face_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.geometry.provided_inr import ProvidedINROracle, extract_modes
from diffsim.sbm.surrogate import (classify_lambda, extract_surrogate,
                                   GeometryData)
from diffsim.api.ns_bricks import assemble_linear_ns
from diffsim.sbm.vector import sbm_vector_dirichlet
from diffsim.sbm.ns_shape import face_dbar_sweep, _gp_field_transpose
from diffsim.sbm.ns_adjoint import ns_volume_cotangents, ns_load_cotangents
from diffsim.sbm.adjoint import distance_torch
from diffsim.physics.poisson import gauss_points
from diffsim.mesh.pointeval import point_eval_weights
import warp as wp

SPHERE_PT = "SDF examples/model_single_head0.pt"
U_IN, NU, ALPHA_F = 1.0, 0.01, 10.0
K = 4
# probes: downstream of the (windowed, shifted) sphere at c=(0.47,...)
_py, _pz = np.meshgrid(np.linspace(0.30, 0.64, 4),
                       np.linspace(0.30, 0.64, 4))
PROBES = np.column_stack([np.full(16, 0.78), _py.ravel(), _pz.ravel()])


def make_oracle(alpha_np, V):
    o = ProvidedINROracle.from_state_dict(SPHERE_PT, n_layers=7, w0=1.0,
                                          window_half=0.5,
                                          window_center=0.03)
    o._V = torch.tensor(V)
    o.alpha = torch.zeros(K, dtype=torch.float64)
    with torch.no_grad():
        o.alpha += torch.tensor(alpha_np)
    return o


_EPOCH = {}


def build_epoch(V, level):
    """FROZEN CLASSIFICATION EPOCH (the H1-v1 lesson: independent carves
    per alpha make the landscape jumpy — J fell 63% while alpha went
    nowhere. Within-epoch, the landscape is smooth and the machinery is
    exactly what the 1e-8 gates validate; edits stay within O(h) of the
    frozen surrogate, legitimate SBM usage)."""
    o = make_oracle(np.zeros(K), V)
    tree = build_uniform(level, dim=3)
    ret, _ = classify_lambda(tree, o, 1.0, domain="outside")
    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=3), "cuda:0")
    _EPOCH.update(ret=ret, sf=sf, mesh=mesh, cons=cons, dm=dm, tree=tree)


def steady(alpha_np, V, level, picard=60, tol=1e-11):
    """Steady confined flow past the alpha-edited sphere INR on the
    FROZEN epoch. Returns the pieces the adjoint needs."""
    dim, ndof = 3, 4
    if not _EPOCH:
        build_epoch(V, level)
    o = make_oracle(alpha_np, V)
    ret, sf, mesh, cons, dm = (_EPOCH["ret"], _EPOCH["sf"], _EPOCH["mesh"],
                               _EPOCH["cons"], _EPOCH["dm"])
    geo = GeometryData.evaluate(o, ret, sf, face_tables(1, 3),
                                domain="outside")
    T = cons.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    nfree = T.shape[1]
    coords = mesh.node_coords[cons.free_nodes]
    xq = gauss_points(mesh, dm.tables_by_p)
    on = lambda v, c: np.abs(coords[:, c] - v) < 1e-12
    strong = np.where(on(0.0, 0) | on(0.0, 1) | on(1.0, 1)
                      | on(0.0, 2) | on(1.0, 2))[0]
    g_strong = np.zeros((len(strong), 3))
    g_strong[np.abs(coords[strong, 0]) < 1e-12, 0] = U_IN
    pin = int(np.argmax(coords.sum(1)))
    Af, bf = sbm_vector_dirichlet(
        dm, sf, geo, lambda y: np.zeros((len(y), 3)), NU, ndof,
        alpha=ALPHA_F)
    Af_c = (T_vec.T @ Af @ T_vec).tocsr()
    bf_c = np.asarray(T_vec.T @ bf)

    def gp_field(node_vec):
        full = np.asarray(T @ node_vec)
        aq, dq = {}, {}
        for pv in dm.bins:
            tb = dm.tables_by_p[pv]
            conn = mesh.conn_of[pv]
            vals = full[conn]
            aq[pv] = np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dim)
            h = mesh.tree.h()[mesh.bins[pv]]
            dq[pv] = (np.einsum("qad,ead->eq", tb.dN, vals)
                      * (2.0 / h)[:, None]).reshape(-1)
        return aq, dq

    x = np.zeros(nfree * ndof)
    prev = None
    for it in range(picard):
        u_node = x.reshape(nfree, ndof)[:, :dim]
        aq, dq = gp_field(u_node)
        fq = {pv: np.zeros_like(aq[pv]) for pv in xq}
        A, b = assemble_linear_ns(dm, aq, dq, fq, NU, sigma=0.0)
        A = (A + Af_c).tolil()
        b = b + bf_c
        for k_, i in enumerate(strong):
            for c in range(dim):
                r_ = i * ndof + c
                A.rows[r_] = [int(r_)]
                A.data[r_] = [1.0]
                b[r_] = g_strong[k_, c]
        rp = pin * ndof + dim
        A.rows[rp] = [rp]
        A.data[rp] = [1.0]
        b[rp] = 0.0
        A = A.tocsr()
        x = splu(A.tocsc()).solve(b)
        u_new = x.reshape(nfree, ndof)[:, :dim]
        if prev is not None and np.abs(u_new - prev).max() < tol:
            break
        prev = u_new.copy()
    strong_rows = np.concatenate(
        [np.array([i * ndof + c for i in strong for c in range(3)],
                  np.int64), np.array([pin * ndof + 3], np.int64)])
    # probe operator on velocity components (u_x, u_y, u_z stacked)
    W = point_eval_weights(mesh, PROBES)          # [np, n_nodes]
    return dict(dm=dm, sf=sf, geo=geo, oracle=o, A=A, x=x, T_vec=T_vec,
                aq=aq, dq=dq, fq=fq, strong_rows=strong_rows, W=W,
                cons=cons, ret=ret, nfree=nfree)


def probe_values(st):
    full = np.asarray(st["T_vec"] @ st["x"]).reshape(-1, 4)
    return np.stack([st["W"] @ full[:, c] for c in range(3)], axis=1)


def alpha_gradient(st, u_target):
    """dJ/dalpha for J = 0.5 sum |u(probes) - u*|^2 via the adjoint-
    Picard fixed-point loop (steady state) + face cotangents."""
    dm = st["dm"]
    ndof, dim = 4, 3
    up = probe_values(st)
    resid = up - u_target                          # [np, 3]
    # dJ/dx (free layout)
    full_bar = np.zeros((dm.n_nodes, 4))
    for c in range(3):
        full_bar[:, c] = st["W"].T @ resid[:, c]
    dJdx = np.asarray(st["T_vec"].T @ full_bar.reshape(-1))
    At = splu(st["A"].tocsc().T)
    lam = At.solve(dJdx)
    lam[st["strong_rows"]] = 0.0
    # adjoint-Picard: aq(x) fixed-point coupling
    for _ in range(80):
        lam_full = np.asarray(st["T_vec"] @ lam)
        aqb_v, _ = ns_volume_cotangents(
            dm, st["aq"], st["dq"], NU, 0.0, 0.5,
            np.asarray(st["T_vec"] @ st["x"]), lam_full,
            timestab=False, tau_frozen=False)
        aqb_l, _, _ = ns_load_cotangents(
            dm, st["aq"], st["fq"], NU, 0.0, lam_full, timestab=False)
        G = _gp_field_transpose(
            dm, {pv: aqb_v[pv][0] + aqb_l[pv] for pv in dm.bins},
            {pv: aqb_v[pv][1] for pv in dm.bins}, ndof)
        lam_new = At.solve(dJdx + np.asarray(st["T_vec"].T @ G))
        lam_new[st["strong_rows"]] = 0.0
        dl = np.abs(lam_new - lam).max()
        lam = lam_new
        if dl < 1e-11 * max(1.0, np.abs(lam).max()):
            break
    lam_full = np.asarray(st["T_vec"] @ lam)
    lam_d = wp.array(lam_full, dtype=wp.float64, device=dm.device)
    x_d = wp.array(np.ascontiguousarray(np.asarray(st["T_vec"] @ st["x"])),
                   dtype=wp.float64, device=dm.device)
    dbar = face_dbar_sweep(dm, st["sf"], st["geo"], x_d, lam_d, NU,
                           ALPHA_F, ndof)
    o = st["oracle"]
    for p_ in o.params:
        p_.requires_grad_(True)
        if p_.grad is not None:
            p_.grad = None
    d_t, n_t, ok = distance_torch(o, st["geo"].xq)
    loss = -(d_t * torch.tensor(dbar, dtype=torch.float64)).sum()
    loss.backward()
    J = 0.5 * float((resid ** 2).sum())
    return J, o.alpha.grad.numpy().copy()


def main(level=4, n_epochs=25):
    # modes from the base oracle
    o0 = ProvidedINROracle.from_state_dict(SPHERE_PT, n_layers=7, w0=1.0,
                                           window_half=0.5,
                                           window_center=0.03)
    rng = np.random.default_rng(11)
    band_dirs = rng.standard_normal((800, 3))
    band_dirs /= np.linalg.norm(band_dirs, axis=1, keepdims=True)
    band = 0.47 + (0.262 + rng.uniform(-0.05, 0.05, 800))[:, None] \
        * band_dirs
    V, evals, stab = extract_modes(o0, band, k=K, check_pts01=band[::-1].copy())
    print(f"modes: stability={stab:.3f}")

    # hidden edit + target probes
    alpha_star = np.array([0.015, -0.010, 0.008, -0.012])
    st_star = steady(alpha_star, V, level)
    u_target = probe_values(st_star)
    print(f"hidden alpha* = {alpha_star}")

    # sanity: gradient at alpha=0 vs FD (one mode)
    st0 = steady(np.zeros(K), V, level)
    J0, g0 = alpha_gradient(st0, u_target)
    eps = 1e-6
    ap = np.zeros(K); ap[0] = eps
    am = np.zeros(K); am[0] = -eps
    stp = steady(ap, V, level)
    stm = steady(am, V, level)
    Jp = 0.5 * float(((probe_values(stp) - u_target) ** 2).sum())
    Jm = 0.5 * float(((probe_values(stm) - u_target) ** 2).sum())
    fd = (Jp - Jm) / (2 * eps)
    print(f"gradient check m0: adj={g0[0]:+.5e} fd={fd:+.5e} "
          f"rel={abs(g0[0]-fd)/max(abs(fd),1e-14):.2e}")

    # recovery: GAUSS-NEWTON on the probe residual (4 params, 48
    # residuals). Plain gradient descent stalled in the ill-conditioned
    # valley (measured: J 3.08 -> 0.956 then nine line-search rejections
    # at 1e-6 steps — mode sensitivities differ by orders of magnitude).
    # The ADJOINT gradient is still computed each epoch as the
    # differentiability cross-check (and the scalable path at large k).
    alpha = np.zeros(K)
    print(f"{'ep':>3} {'J':>12} {'|alpha-alpha*|':>15} {'|adj-GN cos|':>12}")
    for ep in range(n_epochs):
        st = steady(alpha, V, level)
        r0 = (probe_values(st) - u_target).reshape(-1)
        J = 0.5 * float(r0 @ r0)
        _, g_adj = alpha_gradient(st, u_target)
        # forward-FD Jacobian (k columns; each = one steady solve)
        eps_j = 1e-5
        Jac = np.zeros((len(r0), K))
        for k_ in range(K):
            ap = alpha.copy(); ap[k_] += eps_j
            rp = (probe_values(steady(ap, V, level))
                  - u_target).reshape(-1)
            Jac[:, k_] = (rp - r0) / eps_j
        # adjoint-vs-Jacobian consistency (J^T r == adjoint gradient)
        g_jac = Jac.T @ r0
        cos = float(g_adj @ g_jac /
                    max(np.linalg.norm(g_adj) * np.linalg.norm(g_jac),
                        1e-30))
        err = np.linalg.norm(alpha - alpha_star)
        print(f"{ep:>3} {J:12.6e} {err:15.6e} {cos:12.6f}", flush=True)
        if J < 1e-12:
            break
        # damped GN step (Levenberg lambda tiny; residual near-linear)
        lamb = 1e-8 * np.trace(Jac.T @ Jac) / K
        step = np.linalg.solve(Jac.T @ Jac + lamb * np.eye(K),
                               -(Jac.T @ r0))
        alpha = alpha + step
    print(f"final alpha  = {np.round(alpha, 5)}")
    print(f"hidden alpha*= {np.round(alpha_star, 5)}")
    print(f"recovery err = {np.linalg.norm(alpha - alpha_star):.3e}")


if __name__ == "__main__":
    lv = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    ne = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    main(lv, ne)
