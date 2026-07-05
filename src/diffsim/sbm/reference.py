"""Torch dense reference twin of the SBM Dirichlet Poisson solve (spec S16.2
"fast path + clear path, tested equal"). End-to-end autograd: theta ->
(d, gbar via the IFT closest-point projection) -> dense A, b -> solve -> QoI.
This is simultaneously the readable weak-form documentation and the
"unrolled tape" leg of the Tier-AD three-way gradient check.

TINY MESHES ONLY (dense (n_free)^2): <= ~1500 free nodes.
Scalar kappa, uniform p only (the twin exists for gradient verification, not
generality)."""
import numpy as np
import torch

from ..geometry.project import distance_torch
from ..mesh.faces import face_tables


def solve_dense_torch(dm, sf, oracle, g_fn_torch, f_fn, kappa=1.0,
                      alpha=10.0, g_outer_fn=None):
    """Returns (u_free torch [n_free], mesh arrays dict) — u_free carries the
    full autograd graph to oracle.params (and g's parameters if any)."""
    mesh = dm.mesh
    dim = mesh.dim
    assert len(mesh.bins) == 1, "twin supports uniform-p meshes only"
    pv = int(mesh.p)
    tb = dm.tables_by_p[pv]
    ftab = face_tables(pv, dim)
    tree = mesh.tree
    h = torch.tensor(tree.h(), dtype=torch.float64)
    conn = torch.tensor(mesh.conn_of[pv].astype(np.int64))
    Nn = len(mesh.node_coords)

    # ---- volume stiffness + load (torch einsum over elements) ----------
    Ntab = torch.tensor(tb.N, dtype=torch.float64)
    dNtab = torch.tensor(tb.dN, dtype=torch.float64)
    wtab = torch.tensor(tb.w, dtype=torch.float64)
    jac = (h / 2.0) ** dim                                # [Ne]
    dscale = 2.0 / h
    # Ke[e,a,b] = sum_q kappa (dN_a . dN_b) dscale^2 w_q jac_e
    Ke = torch.einsum("qad,qbd,q->ab", dNtab, dNtab, wtab)
    Ke = kappa * Ke.unsqueeze(0) * (dscale ** 2 * jac).view(-1, 1, 1)
    from ..physics.poisson import gauss_points
    xq = gauss_points(mesh, dm.tables_by_p)[pv]
    fq = torch.tensor(f_fn(xq).reshape(len(tree), tb.nqp),
                      dtype=torch.float64)
    be = torch.einsum("qa,eq,q->ea", Ntab, fq, wtab) * jac.view(-1, 1)

    A = torch.zeros(Nn, Nn, dtype=torch.float64)
    bvec = torch.zeros(Nn, dtype=torch.float64)
    A.index_put_((conn.unsqueeze(2).expand(-1, -1, conn.shape[1]),
                  conn.unsqueeze(1).expand(-1, conn.shape[1], -1)),
                 Ke, accumulate=True)
    bvec.index_put_((conn,), be, accumulate=True)

    # ---- SBM Dirichlet face terms (the weak form, readable) ------------
    from .surrogate import face_gauss_points
    xqf = face_gauss_points(tree, sf, ftab)               # [Nf*nqf, dim]
    d_t, n_t, ok = distance_torch(oracle, xqf)            # theta chain
    gbar = g_fn_torch(torch.tensor(xqf, dtype=torch.float64) + d_t)
    nqf, nbf = ftab.nqf, ftab.nbf
    Nf = torch.tensor(ftab.N, dtype=torch.float64)
    dNf = torch.tensor(ftab.dN, dtype=torch.float64)
    d2Nf = torch.tensor(ftab.d2N, dtype=torch.float64)
    wff = torch.tensor(ftab.w, dtype=torch.float64)
    for i in range(len(sf.elem)):
        e = int(sf.elem[i])
        f = int(sf.face[i])
        ax, side = f // 2, f % 2
        sgn = -1.0 if side == 0 else 1.0
        he = h[e]
        dsc = 2.0 / he
        jacS = (he / 2.0) ** (dim - 1)
        dloc = d_t[i * nqf:(i + 1) * nqf]                 # [nqf, dim]
        gloc = gbar[i * nqf:(i + 1) * nqf]                # [nqf]
        # shifted basis S N_a = N_a + grad.d (+ 1/2 d^T H d at p2)
        SN = Nf[f] + torch.einsum("qad,qd->qa", dNf[f] * dsc, dloc)
        if pv == 2:
            SN = SN + 0.5 * torch.einsum(
                "qaij,qi,qj->qa", d2Nf[f] * dsc ** 2, dloc, dloc)
        gN = sgn * dNf[f, :, :, ax] * dsc                 # grad N . n_tilde
        dS = wff * jacS
        # consistency + adjoint-consistency + penalty (Main & Scovazzi)
        Ae = kappa * (
            -torch.einsum("qa,qb,q->ab", Nf[f], gN, dS)
            - torch.einsum("qa,qb,q->ab", gN, SN, dS)
            + (alpha / he) * torch.einsum("qa,qb,q->ab", SN, SN, dS))
        beF = kappa * (
            -torch.einsum("qa,q,q->a", gN, gloc, dS)
            + (alpha / he) * torch.einsum("qa,q,q->a", SN, gloc, dS))
        idx = conn[e]
        A.index_put_((idx.unsqueeze(1).expand(-1, nbf),
                      idx.unsqueeze(0).expand(nbf, -1)), Ae, accumulate=True)
        bvec.index_put_((idx,), beF, accumulate=True)

    # ---- constraints + outer Dirichlet + dense solve --------------------
    T = torch.tensor(dm.constraints.T.toarray(), dtype=torch.float64)
    Ac = T.T @ A @ T
    bc = T.T @ bvec
    if g_outer_fn is not None:
        dirf = dm.mesh.boundary_nodes[dm.constraints.free_nodes]
        idx = np.where(dirf)[0]
        if len(idx):
            mask = torch.ones(Ac.shape[0], dtype=torch.float64)
            mask[idx] = 0.0
            Ac = Ac * mask.unsqueeze(1)
            Ac[idx, idx] = 1.0
            coords = dm.mesh.node_coords[dm.constraints.free_nodes][idx]
            bc = bc.clone()
            bc[idx] = torch.tensor(g_outer_fn(coords), dtype=torch.float64)
    u_free = torch.linalg.solve(Ac, bc)
    return u_free, {"T": T}
