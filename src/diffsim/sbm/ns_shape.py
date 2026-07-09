"""M1c: shape gradient of surface-force QoIs for the immersed NS problem
(linearized s=1/2 stepper) — d(drag)/d(obstacle center).

Epoch pattern (m1a): classification FROZEN (trust region), the center c
enters through (i) the SBM face terms' distance vectors d (implicit, via
the state) and (ii) the traction QoI's true-geometry factors n, corr
(explicit). With no-slip (g == 0) the gbar chain vanishes and the face RHS
is identically zero, so

    dJ/dc = dF/dc|_x  (torch, explicit)  -  lam^T dR_face/dd . dd/dc,
    A^T lam = dJ/dx   (transposed solve; the backflow term carries no
                       d-dependence — evaluated at surrogate GPs).

The vector face residual REUSES the m1a scalar taped kernel shape verbatim,
launched once per velocity component with strided (node-major, ndof)
indexing — findings 4c rules: plain locals, depth-1 accumulators, no
structs."""
import numpy as np
import torch
import warp as wp

from ..assembly.operators import _kernel_cache
from ..mesh.faces import face_tables
from .poisson import _shift_fn_for
from .adjoint import distance_torch


def make_sbm_vector_dirichlet_residual(nbf: int, nqf: int, dim: int,
                                       comp: int, ndof: int):
    """Momentum-component face residual (Nitsche Dirichlet, kappa=nu),
    differentiable in dvec; identical math to the m1a scalar kernel with
    node-major strided indexing for component `comp`."""
    key = ("sbm_vec_dir_res", nbf, nqf, dim, comp, ndof)
    if key in _kernel_cache:
        return _kernel_cache[key]

    shift_fn = _shift_fn_for(nbf, dim)

    @wp.kernel(module="unique", enable_backward=True)
    def sbm_vdir_res(felem: wp.array(dtype=wp.int32),
                     fface: wp.array(dtype=wp.int32),
                     conn: wp.array2d(dtype=wp.int32),
                     h: wp.array(dtype=wp.float64),
                     Nf: wp.array3d(dtype=wp.float64),
                     dNf: wp.array4d(dtype=wp.float64),
                     d2Nf: wp.array4d(dtype=wp.float64),
                     wf: wp.array(dtype=wp.float64),
                     dvec: wp.array2d(dtype=wp.float64),   # DIFF
                     u: wp.array(dtype=wp.float64),        # frozen state
                     alpha: wp.float64, kappa: wp.float64,
                     r: wp.array(dtype=wp.float64)):
        fi = wp.tid()
        e = felem[fi]
        f = fface[fi]
        he = h[e]
        jacS = wp.pow(he * wp.float64(0.5), wp.float64(dim - 1))
        dscale = wp.float64(2.0) / he
        ax = f / 2
        sgn = wp.float64(1.0)
        if f % 2 == 0:
            sgn = wp.float64(-1.0)
        for q in range(nqf):
            gp = fi * nqf + q
            dS = wf[q] * jacS
            gradun = wp.float64(0.0)
            Su = wp.float64(0.0)
            for b in range(nbf):
                ub = u[conn[e, b] * ndof + comp]
                gradun += sgn * dNf[f, q, b, ax] * dscale * ub
                Su += shift_fn(Nf, dNf, d2Nf, f, q, b, gp, dvec,
                               dscale, dim) * ub
            # no-slip: gbar == 0, mis = Su
            for a in range(nbf):
                Na = Nf[f, q, a]
                gna = sgn * dNf[f, q, a, ax] * dscale
                Sa = shift_fn(Nf, dNf, d2Nf, f, q, a, gp, dvec, dscale, dim)
                wp.atomic_add(r, conn[e, a] * ndof + comp,
                              kappa * (-Na * gradun - gna * Su
                                       + alpha / he * Sa * Su) * dS)

    _kernel_cache[key] = sbm_vdir_res
    return sbm_vdir_res


def traction_functional(dm, sf, geo, nu, ndof, direction=0):
    """The traction QoI is LINEAR in the state: F_dir = w . x_full.
    Returns w (len n_nodes*ndof), mirroring surrogate_traction exactly."""
    dim = dm.dim
    mesh = dm.mesh
    pv = int(np.unique(np.asarray(mesh.p_elem)[sf.elem])[0])
    ftab = face_tables(pv, dim)
    nqf, nbf = ftab.nqf, ftab.nbf
    conn = mesh.conn_of[pv][np.searchsorted(mesh.bins[pv], sf.elem)]
    h = mesh.tree.h()[sf.elem]
    jacS = (h / 2.0) ** (dim - 1)
    dscale = 2.0 / h
    w_vec = np.zeros(dm.n_nodes * ndof)
    i = direction
    for fi in range(len(sf.elem)):
        f = int(sf.face[fi])
        N = ftab.N[f]
        dN = ftab.dN[f] * dscale[fi]
        for q in range(nqf):
            wq = ftab.w[q] * jacS[fi] * geo.corr[fi * nqf + q]
            n = geo.n[fi * nqf + q]
            dn_n = dN[q] @ n                       # [nbf]: sum_d dN[b,d] n_d
            for b in range(nbf):
                w_vec[conn[fi, b] * ndof + dim] += wq * N[q, b] * n[i]
                w_vec[conn[fi, b] * ndof + i] += -wq * nu * dn_n[b]
    return w_vec


def traction_torch(dm, sf, geo, x_all, nu, ndof, n_t, corr_t, direction=0):
    """F_dir with FROZEN state x but torch-differentiable n, corr — the
    QoI's explicit geometry chain."""
    dim = dm.dim
    mesh = dm.mesh
    pv = int(np.unique(np.asarray(mesh.p_elem)[sf.elem])[0])
    ftab = face_tables(pv, dim)
    nqf = ftab.nqf
    conn = mesh.conn_of[pv][np.searchsorted(mesh.bins[pv], sf.elem)]
    xv = x_all.reshape(dm.n_nodes, ndof)
    h = mesh.tree.h()[sf.elem]
    jacS = (h / 2.0) ** (dim - 1)
    dscale = 2.0 / h
    F = torch.zeros((), dtype=torch.float64)
    i = direction
    for fi in range(len(sf.elem)):
        f = int(sf.face[fi])
        un = torch.tensor(xv[conn[fi], :dim], dtype=torch.float64)
        pn = torch.tensor(xv[conn[fi], dim], dtype=torch.float64)
        N = torch.tensor(ftab.N[f], dtype=torch.float64)
        dN = torch.tensor(ftab.dN[f] * dscale[fi], dtype=torch.float64)
        for q in range(nqf):
            gp = fi * nqf + q
            wq = ftab.w[q] * jacS[fi] * corr_t[gp]
            n = n_t[gp]
            gradu = dN[q].T @ un
            pq = N[q] @ pn
            F = F + wq * (pq * n[i] - nu * (gradu.T @ n)[i])
    return F


def _gp_field_transpose(dm, aq_bar_bins, dq_bar_bins, ndof):
    """Transpose of the advecting-field interpolation: GP cotangents ->
    full node-major vector (velocity components)."""
    dim = dm.dim
    mesh = dm.mesh
    out = np.zeros(dm.n_nodes * ndof)
    ov = out.reshape(dm.n_nodes, ndof)
    for pv in dm.bins:
        tb = dm.tables_by_p[pv]
        conn = mesh.conn_of[pv]
        ne, nqp = len(conn), tb.N.shape[0]
        ab = aq_bar_bins[pv].reshape(ne, nqp, dim)
        db = dq_bar_bins[pv].reshape(ne, nqp)
        h = mesh.tree.h()[mesh.bins[pv]]
        contrib = (np.einsum("qa,eqd->ead", tb.N, ab)
                   + np.einsum("qad,eq->ead", tb.dN, db)
                   * (2.0 / h)[:, None, None])
        np.add.at(ov[:, :dim], conn.reshape(-1),
                  contrib.reshape(-1, dim))
    return out


def face_dbar_sweep(dm, sf, geo, x_d, lam_d, nu, alpha, ndof):
    """lam^T dR_face/dd for the momentum SBM Dirichlet block (no-slip):
    per-component taped sweeps, cotangents add. x_d/lam_d: FULL node-major
    wp arrays."""
    from .poisson import _FaceSet
    dim = dm.dim
    d = dm.device
    fs = _FaceSet(dm, sf, geo)
    b = dm.bins[fs.pv]
    dbar = np.zeros_like(geo.d)
    for comp in range(dim):
        k = make_sbm_vector_dirichlet_residual(fs.ftab.nbf, fs.ftab.nqf,
                                               dim, comp, ndof)
        tape = wp.Tape()
        dvec = wp.array(np.ascontiguousarray(geo.d), dtype=wp.float64,
                        device=d, requires_grad=True)
        r = wp.zeros(dm.n_nodes * ndof, dtype=wp.float64, device=d,
                     requires_grad=True)
        with tape:
            wp.launch(k, dim=len(sf.elem),
                      inputs=[fs.felem_d, fs.fface_d, b["conn"], b["h"],
                              fs.Nf_d, fs.dNf_d, fs.d2Nf_d, fs.wf_d,
                              dvec, x_d, wp.float64(alpha),
                              wp.float64(nu), r],
                      device=d)
        tape.backward(grads={r: lam_d})
        dbar += tape.gradients[dvec].numpy()
    return dbar


def drag_shape_gradient(dm, sf, geo, oracle, A, x_full, nu, alpha,
                        strong_rows, ndof, direction=0,
                        fixed_point=False, sigma=0.0, s_skew=0.5,
                        max_picard=40, tol=1e-12):
    """dF_dir/d(center): adjoint + torch composition. Returns the gradient
    w.r.t. oracle.params (accumulated into .grad, m1a convention) and F."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    dim = dm.dim
    d = dm.device
    # ---- QoI + adjoint solve -----------------------------------------
    w_vec = traction_functional(dm, sf, geo, nu, ndof, direction)
    T = dm.constraints.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    dJdx_free = np.asarray(T_vec.T @ w_vec)
    At = spla.splu(A.tocsc().T)
    lam = At.solve(dJdx_free)
    lam[strong_rows] = 0.0            # identity rows are c-independent
    if fixed_point:
        # adjoint-Picard: (A + C)^T lam = dJdx with C = dR/daq . daq/dx
        # (requires sigma=0 steady: b is aq-independent so the taped
        # volume kernel covers dR/daq exactly; timestab off)
        from .ns_adjoint import ns_volume_cotangents
        from ..physics.poisson import gauss_points
        xq_gp = gauss_points(dm.mesh, dm.tables_by_p)
        # advecting field = the converged state itself
        aq_bins, dq_bins = {}, {}
        xv = x_full.reshape(dm.n_nodes, ndof)
        for pv in dm.bins:
            tb = dm.tables_by_p[pv]
            conn = dm.mesh.conn_of[pv]
            vals = xv[conn][:, :, :dim].reshape(len(conn), -1, dim) \
                if False else xv[conn.reshape(-1), :dim].reshape(
                    len(conn), -1, dim)
            aq_bins[pv] = np.einsum("qa,ead->eqd", tb.N,
                                    vals).reshape(-1, dim)
            h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
            dq_bins[pv] = (np.einsum("qad,ead->eq", tb.dN, vals)
                           * (2.0 / h)[:, None]).reshape(-1)
        for it in range(max_picard):
            lam_full_it = np.asarray(T_vec @ lam)
            bar = ns_volume_cotangents(dm, aq_bins, dq_bins, nu, sigma,
                                       s_skew, x_full, lam_full_it,
                                       timestab=False)
            aq_bar = {pv: -bar[0][pv][0] for pv in dm.bins}   # +lam^T dR/daq
            dq_bar = {pv: -bar[0][pv][1] for pv in dm.bins}
            Ct_lam_full = _gp_field_transpose(dm, aq_bar, dq_bar, ndof)
            Ct_lam = np.asarray(T_vec.T @ Ct_lam_full)
            lam_new = At.solve(dJdx_free - Ct_lam)
            lam_new[strong_rows] = 0.0
            dl = np.abs(lam_new - lam).max()
            lam = lam_new
            if dl < tol * max(1.0, np.abs(lam).max()):
                break
    lam_full = np.asarray(T_vec @ lam)
    lam_d = wp.array(lam_full, dtype=wp.float64, device=d)
    x_d = wp.array(np.ascontiguousarray(x_full), dtype=wp.float64, device=d)

    dbar = face_dbar_sweep(dm, sf, geo, x_d, lam_d, nu, alpha, ndof)

    # ---- torch geometry chain ------------------------------------------
    for p in oracle.params:
        p.requires_grad_(True)
        if p.grad is not None:
            p.grad = None
    own = getattr(type(oracle), "distance_torch", None)
    d_t, n_t, ok = (oracle.distance_torch(geo.xq) if own
                    else distance_torch(oracle, geo.xq))
    n_t = -n_t     # raw gradient normal -> domain-signed (measured: geo.n
    #                = -n_t uniformly for domain="outside"; d_t == geo.d)
    # corr = n . n_tilde with the FIXED surrogate face normal n_tilde;
    # geo arrays hold nqf quadrature points per surrogate face
    nqf = geo.n.shape[0] // len(sf.elem)
    ntil = np.zeros_like(geo.n)
    for fi in range(len(sf.elem)):
        f = int(sf.face[fi])
        ntil[fi * nqf:(fi + 1) * nqf, f // 2] = -1.0 if f % 2 == 0 else 1.0
    corr_t = (n_t * torch.tensor(ntil, dtype=torch.float64)).sum(1)
    corr_chk = corr_t.detach().numpy()
    assert np.allclose(corr_chk, geo.corr, atol=1e-12), (
        "corr reconstruction mismatch — n_tilde convention drifted")
    F_t = traction_torch(dm, sf, geo, x_full, nu, ndof, n_t, corr_t,
                         direction)
    loss = F_t - (d_t * torch.tensor(dbar, dtype=torch.float64)).sum()
    loss.backward()
    return float(F_t.detach())
