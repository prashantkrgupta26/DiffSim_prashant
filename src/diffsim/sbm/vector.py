"""Vector SBM Dirichlet for NS velocity fields (M1b Task 7).

The production NS SBM Dirichlet is the LAPLACIAN (component-diagonal)
Nitsche form — each velocity component gets exactly the M1a scalar terms
(consistency + adjoint-consistency on shifted test + penalty, second-order
shift p2-gated), with kappa -> nu. This module REUSES the verified M1a
scalar face kernels and places their blocks at node-major (node*ndof + c),
c < dim: no new warp code, M1a patch exactness carries over verbatim.

Backflow (inflow) stabilization on OUTFLOW faces (production
`inflow_g < 0` branch): + beta_bf * -(a.n)_- (w . u) on the face, assembled
host-side from a face-GP advecting field (face counts are small — the M1a
reuse-map keeps SBM face terms on the CSR path). Zero field => zero term.
"""
import numpy as np
import scipy.sparse as sp
import warp as wp

from .poisson import (make_sbm_dirichlet_Ae, make_sbm_dirichlet_be, _FaceSet,
                      make_sbm_penalty_Ae, make_sbm_penalty_be)


def sbm_vector_dirichlet(dm, sf, geo, g_fn, nu, ndof, alpha=10.0,
                         a_face=None, beta_backflow=1.0):
    """(A_face, b_face) over FULL node-major vector DOFs (unconstrained).
    g_fn(y) -> [Ngp, dim] velocity data at mapped points; a_face optional
    [Ngp, dim] advecting field at face GPs for backflow."""
    dim = dm.dim
    d = dm.device
    fs = _FaceSet(dm, sf, geo)
    ne_f = len(sf.elem)
    nbf, nqf = fs.ftab.nbf, fs.ftab.nqf
    b_ = dm.bins[fs.pv]
    gbar = np.ascontiguousarray(g_fn(geo.xq + geo.d), np.float64)

    Ae = wp.zeros((ne_f, nbf, nbf), dtype=wp.float64, device=d)
    kA = make_sbm_dirichlet_Ae(nbf, nqf, dim)
    wp.launch(kA, dim=ne_f,
              inputs=[fs.felem_d, fs.fface_d, b_["h"], fs.Nf_d, fs.dNf_d,
                      fs.d2Nf_d, fs.wf_d, fs.dvec_d, wp.float64(alpha),
                      wp.float64(nu), Ae], device=d)
    Aeh = Ae.numpy()

    kb = make_sbm_dirichlet_be(nbf, nqf, dim)
    b_comp = []
    for c in range(dim):
        gc = wp.array(np.ascontiguousarray(gbar[:, c]), dtype=wp.float64,
                      device=d)
        bfull = wp.zeros(dm.n_nodes, dtype=wp.float64, device=d)
        wp.launch(kb, dim=ne_f,
                  inputs=[fs.felem_d, fs.fface_d, b_["conn"], b_["h"],
                          fs.Nf_d, fs.dNf_d, fs.d2Nf_d, fs.wf_d, fs.dvec_d,
                          gc, wp.float64(alpha), wp.float64(nu), bfull],
                  device=d)
        b_comp.append(bfull.numpy())

    # backflow: + beta * (-(a.n)_-) N_a N_b per component (surrogate face)
    Ab = np.zeros_like(Aeh)
    if a_face is not None:
        an = np.einsum("gd,gd->g", np.asarray(a_face), geo.n)
        an_neg = np.minimum(an, 0.0).reshape(ne_f, nqf)
        h = dm.mesh.tree.h()[sf.elem]
        jacS = (h / 2.0) ** (dim - 1)
        Nf = fs.ftab.N[sf.face]                      # [ne_f, nqf, nbf]
        Ab = -beta_backflow * np.einsum(
            "eqa,eqb,eq,q,e->eab", Nf, Nf, an_neg, fs.ftab.w, jacS)

    conn = fs.scatter_conn(dm)                       # bin-local -> global?
    conn_glob = dm.mesh.conn_of[fs.pv][fs.felem_rows].astype(np.int64)
    Nn = dm.n_nodes * ndof
    rows, cols, vals = [], [], []
    brhs = np.zeros(Nn)
    for c in range(dim):
        gdof = conn_glob * ndof + c
        rows.append(np.repeat(gdof, nbf, axis=1).ravel())
        cols.append(np.tile(gdof, (1, nbf)).ravel())
        vals.append((Aeh + Ab).ravel())
        brhs.reshape(dm.n_nodes, ndof)[:, c] += b_comp[c]
    A = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(Nn, Nn)).tocsr()
    return A, brhs


def sbm_vector_penalty(dm, sf, geo, g_fn, nu, ndof, alpha=10.0):
    """(A_face, b_face) of the PENALTY-ONLY viscous Nitsche block over FULL
    node-major vector DOFs (unconstrained): the third Dirichlet term alone,
        A[a i, b i] += (alpha nu / h) S(N_a) S(N_b) dS      (per component i)
        b[a i]      += (alpha nu / h) S(N_a) g_i dS
    with NO consistency / adjoint-consistency terms. This is the FN1 fix's
    velocity-update re-pin: added to the correction mass system it weakly
    re-imposes u_new -> g on the surrogate wall AFTER the pressure correction
    (the weak analog of the strong-node overwrite, leray.py line 307).

    Component-diagonal (each velocity component gets the SAME scalar penalty,
    exactly like sbm_vector_dirichlet), so it mirrors that block's layout.
    g_fn(y) -> [Ngp, dim] the wall velocity data at mapped points (no-slip
    => zeros, in which case b_face is zero)."""
    dim = dm.dim
    d = dm.device
    fs = _FaceSet(dm, sf, geo)
    ne_f = len(sf.elem)
    nbf, nqf = fs.ftab.nbf, fs.ftab.nqf
    b_ = dm.bins[fs.pv]
    gbar = np.ascontiguousarray(g_fn(geo.xq + geo.d), np.float64)

    Ae = wp.zeros((ne_f, nbf, nbf), dtype=wp.float64, device=d)
    kA = make_sbm_penalty_Ae(nbf, nqf, dim)
    wp.launch(kA, dim=ne_f,
              inputs=[fs.felem_d, fs.fface_d, b_["h"], fs.Nf_d, fs.dNf_d,
                      fs.d2Nf_d, fs.wf_d, fs.dvec_d, wp.float64(alpha),
                      wp.float64(nu), Ae], device=d)
    Aeh = Ae.numpy()

    kb = make_sbm_penalty_be(nbf, nqf, dim)
    b_comp = []
    for c in range(dim):
        gc = wp.array(np.ascontiguousarray(gbar[:, c]), dtype=wp.float64,
                      device=d)
        bfull = wp.zeros(dm.n_nodes, dtype=wp.float64, device=d)
        wp.launch(kb, dim=ne_f,
                  inputs=[fs.felem_d, fs.fface_d, b_["conn"], b_["h"],
                          fs.Nf_d, fs.dNf_d, fs.d2Nf_d, fs.wf_d, fs.dvec_d,
                          gc, wp.float64(alpha), wp.float64(nu), bfull],
                  device=d)
        b_comp.append(bfull.numpy())

    conn_glob = dm.mesh.conn_of[fs.pv][fs.felem_rows].astype(np.int64)
    Nn = dm.n_nodes * ndof
    rows, cols, vals = [], [], []
    brhs = np.zeros(Nn)
    for c in range(dim):
        gdof = conn_glob * ndof + c
        rows.append(np.repeat(gdof, nbf, axis=1).ravel())
        cols.append(np.tile(gdof, (1, nbf)).ravel())
        vals.append(Aeh.ravel())
        brhs.reshape(dm.n_nodes, ndof)[:, c] += b_comp[c]
    A = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(Nn, Nn)).tocsr()
    return A, brhs


def sbm_wall_pressure_traction(dm, sf, geo, p_node, ndof):
    """Lagged wall pressure-traction RHS on the surrogate faces (FN1 fix term
    (2), Dokken et al. eq. 4.16): the boundary force

        b[a i] += <p* n_i, N_a>_Gamma~ = int p*(x_q) n_i N_a dS~

    with n = geo.n (the DOMAIN-outward surrogate normal) and p* the LAGGED
    pressure at the face GPs (interpolated, NOT shifted — matches the collocated
    pressure the monolithic carries at the wall). Returned as a FULL node-major
    (dm.n_nodes*ndof) RHS vector (velocity components only; pressure rows zero),
    to be ADDED to the momentum predictor RHS.

    Sign contract: the monolithic momentum weak form integrates grad p by parts
    to (p, div v) - <p n, v>_Gamma; the split predictor keeps grad p* in the
    interior (p pinned in the coupled block) but DROPS the wall boundary term.
    This restores it, LAGGED at p*, on the surrogate wall — the missing wall
    pressure force. (geo.corr area-correction folded in for the shifted face.)"""
    dim = dm.dim
    mesh = dm.mesh
    p_face = np.unique(np.asarray(mesh.p_elem)[sf.elem])
    if len(p_face) != 1:
        from ..errors import ConfigError
        raise ConfigError(
            f"SBM face helper assumes uniform p on the face, got "
            f"orders {p_face.tolist()}")
    pv = int(p_face[0])
    from ..mesh.faces import face_tables
    ftab = face_tables(pv, dim)
    nqf, nbf = ftab.nqf, ftab.nbf
    conn = mesh.conn_of[pv][np.searchsorted(mesh.bins[pv], sf.elem)]
    p_full = np.asarray(dm.constraints.T @ p_node) if p_node.shape[0] == \
        dm.constraints.T.shape[1] else np.asarray(p_node)
    h = mesh.tree.h()[sf.elem]
    jacS = (h / 2.0) ** (dim - 1)
    Nn = dm.n_nodes * ndof
    brhs = np.zeros(Nn)
    bview = brhs.reshape(dm.n_nodes, ndof)
    for fi in range(len(sf.elem)):
        f = int(sf.face[fi])
        pn = p_full[conn[fi]]                          # [nbf]
        N = ftab.N[f]                                  # [nqf, nbf]
        for q in range(nqf):
            gp = fi * nqf + q
            w = ftab.w[q] * jacS[fi] * geo.corr[gp]
            n = geo.n[gp]
            pq = N[q] @ pn
            # b[a, i] += p*_q n_i N_a w  (the <p* n, N_a> wall traction)
            contrib = w * pq * np.outer(N[q], n)       # [nbf, dim]
            for i in range(dim):
                np.add.at(bview[:, i], conn[fi], contrib[:, i])
    return brhs


def surrogate_traction(dm, sf, geo, x_all, nu, ndof):
    """Force of the fluid ON the immersed obstacle: F = oint sigma . n_hat
    dS with n_hat the OBSTACLE-outward normal. ORIENTATION CONTRACT
    (measured on the Re=20 cylinder: Cd = -2.85 before the flip, +2.85
    after — drag must point downstream): geo.n is the DOMAIN-outward normal
    = INTO the obstacle for exterior flow, so n_hat = -geo.n and the
    integrand is +p*geo.n - nu (S grad u).geo.n. Production SurfaceLoop
    Force = (-p n_j + viscous) pattern, area-corrected (Surrogate2True).
    x_all: FULL node-major (u, p) vector. Host-side observable (spec S14
    layer kernelizes later)."""
    dim = dm.dim
    mesh = dm.mesh
    p_face = np.unique(np.asarray(mesh.p_elem)[sf.elem])
    if len(p_face) != 1:
        from ..errors import ConfigError
        raise ConfigError(
            f"SBM face helper assumes uniform p on the face, got "
            f"orders {p_face.tolist()}")
    pv = int(p_face[0])
    from ..mesh.faces import face_tables
    ftab = face_tables(pv, dim)
    nqf, nbf = ftab.nqf, ftab.nbf
    conn = mesh.conn_of[pv][
        np.searchsorted(mesh.bins[pv], sf.elem)]        # [Nf, nbf]
    xv = x_all.reshape(dm.n_nodes, ndof)
    h = mesh.tree.h()[sf.elem]
    jacS = (h / 2.0) ** (dim - 1)
    dscale = 2.0 / h
    F = np.zeros(dim)
    for fi in range(len(sf.elem)):
        f = int(sf.face[fi])
        un = xv[conn[fi], :dim]                          # [nbf, dim]
        pn = xv[conn[fi], dim]                           # [nbf]
        N = ftab.N[f]                                    # [nqf, nbf]
        dN = ftab.dN[f] * dscale[fi]                     # [nqf, nbf, dim]
        for q in range(nqf):
            w = ftab.w[q] * jacS[fi] * geo.corr[fi * nqf + q]
            n = geo.n[fi * nqf + q]
            # p1 velocity gradient is elementwise-constant: S grad u = grad u
            # (the p2 Hessian-shift variant lands with the p2 NS bricks)
            gradu = dN[q].T @ un
            pq = N[q] @ pn
            F += w * (pq * n - nu * (gradu.T @ n))   # n_hat = -geo.n
    return F
