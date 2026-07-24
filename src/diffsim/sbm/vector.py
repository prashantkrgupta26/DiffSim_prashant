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


def sbm_vector_dirichlet_twosided(dm, sf_plus, geo_plus, sf_minus, geo_minus,
                                  g_fn, nu, ndof, alpha=10.0,
                                  a_face_plus=None, a_face_minus=None,
                                  beta_backflow=1.0):
    """Two-sided co-dim-1 shell Nitsche assembly (ThinShell §2.1 convention:
    <a,b>_Gamma~ := <a,b>_Gamma~+ + <a,b>_Gamma~-). The zero-thickness rigid
    shell carries a weak Dirichlet (no-slip, g_fn) on BOTH sides of Gamma; the
    two-sided surrogate has DISTINCT outward normals on Gamma~+ and Gamma~-, so
    the Nitsche blocks are the SUM of the single-sided blocks over each side,
    each assembled with its OWN (sf, geo) — reusing the verified single-sided
    kernel ``sbm_vector_dirichlet`` verbatim. Summing the two opposing-normal
    sides is exactly what makes the shell BLOCK the flow (the opposing normal
    contributions add rather than cancel) and admits a two-sided pressure jump.

    Returns (A_face, b_face) over the FULL node-major vector DOFs, the sum of
    the two sides. Either side may be empty (handled by the single-sided
    assembler returning zero blocks). ``a_face_plus`` / ``a_face_minus`` are
    the optional per-side advecting fields for backflow stabilization."""
    Ap, bp = sbm_vector_dirichlet(dm, sf_plus, geo_plus, g_fn, nu, ndof,
                                  alpha=alpha, a_face=a_face_plus,
                                  beta_backflow=beta_backflow)
    Am, bm = sbm_vector_dirichlet(dm, sf_minus, geo_minus, g_fn, nu, ndof,
                                  alpha=alpha, a_face=a_face_minus,
                                  beta_backflow=beta_backflow)
    return (Ap + Am).tocsr(), bp + bm


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


def sbm_wall_pressure_neumann(dm, sf, geo, u_node, pstar_node, nu, ndof,
                              f_node=None, include_conv=False,
                              include_visc=True, include_poff=True,
                              scale=1.0, return_parts=False):
    r"""DIAGNOSTIC (KIO-style) Neumann wall-pressure PPE source on the
    surrogate faces — the FN4 investigation record (2026-07-23). NOT the
    rung-B drag fix: that is the stepper's ``rotational_pin_wall`` (leray.py).
    Kept, default OFF everywhere, because its term-level switches DOCUMENT
    the measured defect structure. Verdict: docs/dev/2026-07-23-projection-
    ladder-verdict.md. NOT dead code (P2harden prune check, 2026-07-23): still
    imported+called by tests/ladder_rungB_square_nitsche.py and
    tests/ladder_rungB_seed_probe.py, which feed the green test_ladder_rungB.py
    regression — do NOT remove without first dropping those references.

    Karniadakis-Israeli-Orszag (JCP 1991) consistent pressure Neumann BC from
    the normal momentum balance at the immersed no-slip wall, rotational
    (curl-curl) viscous form:

        dp/dn = n . [ -du/dt - (a.grad)u - nu grad x (grad x u) + f ]   (KIO)

    with n = geo.n the DOMAIN-outward surrogate normal (== the true wall
    normal at d=0), applied to the INCREMENT phi = p_hat - p* (main.pdf p')
    as  dphi/dn = dp_consistent/dn - dp*/dn : the boundary source

        rhs[a] += scale * oint_{Gamma~} ( g_KIO - grad(p*).n ) N_a dGamma~

    added to the weak PPE RHS. Returns a FULL node-major (dm.n_nodes) scalar
    (added BEFORE the constraint reduction; mirrors bvs_ppe_source). 2-D only
    (curl u scalar).

    MEASURED FN4 VERDICT (why this is diagnostic-only): the face assembly
    (w * (h/2)^{dim-1} * geo.corr — identical to sbm_wall_pressure_traction /
    bvs_ppe_source) carries NO h-power bug, and the datum is dimensionally a
    correct dp/dn — yet EVERY variant of this raw ``oint g q`` source is a
    mesh-DEPENDENT overcorrector on rung B (Re=40, from rest, scale=1.0):

        poff only:               Cd +2.01 (L4) / +8.92 (L5)
        poff+visc (KIO no-slip): Cd +2.08 (L4) / +9.08 (L5)
        poff+visc+conv:          Cd +2.20 (L4) / +9.18 (L5)
        same-mesh mono target:      +1.39 (L4) / +2.02 (L5)

    because the projection's wall-pressure defect is NOT a missing Neumann
    datum: the seeded-monolithic step-1 decomposition shows the predictor is
    an EXACT fixed point and the PPE reproduces the monolithic Cd to 0.1%
    under the homogeneous-Neumann wall — the error is written by the
    ROTATIONAL update's -nu*q term at the weak wall (O(0.4-0.5) on the wall
    nodes, GROWING under refinement). A boundary source can only counter-
    fight that bias, which is why the Opus sweep needed the mesh-dependent
    damping (~0.07 at L5). With ``rotational_pin_wall`` the homogeneous-
    Neumann wall (Suresh Remark 3.9) is CORRECT as-is and needs no source.

    Term evaluation at each surrogate GP (P1: grad u elementwise-constant):
      * viscous rotational (include_visc, the KIO no-slip datum): in 2-D
        curl u = omega (scalar); n.(grad x omega) = -d(omega)/d(tau) is a
        TANGENTIAL derivative along the wall, so it integrates by parts along
        the CLOSED surrogate loop to the P1-exact weak form
            rhs_a += -nu int_Gamma (grad N_a x n) . omega dGamma
        (needs only FIRST derivatives; same rotational form as the outflow
        bvs_ppe_source, but weighted by the physical nu — it is a BC, not a
        tau_m-weighted stabilization).
      * increment offset -grad(p*).n (include_poff): collocated from the
        lagged pressure.
      * convective -(u_h.grad)u_h.n (include_conv, DIAGNOSTIC ONLY): the
        discrete-trace term discussed above; + f at the GP when f_node given.

    u_node / pstar_node: CONSTRAINED free-node fields (n_free[,dim]) OR full
    node-major (auto-detected by length). f_node optional full/free body force
    at GPs (None => 0). ``return_parts=True`` returns the dict
    {"conv": ..., "visc": ..., "poff": ...} of UNSCALED per-term node-major
    vectors instead of the combined source (diagnostics)."""
    dim = dm.dim
    if dim != 2:
        from ..errors import ConfigError
        raise ConfigError("consistent wall-pressure Neumann is implemented "
                          "for 2-D (curl u scalar); got dim=%d" % dim)
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
    Tc = dm.constraints.T

    def _to_full(field, k):
        arr = np.asarray(field)
        if arr.shape[0] == Tc.shape[1]:                # constrained free-node
            return np.asarray(Tc @ arr)
        return arr                                     # already full node-major

    u_full = _to_full(u_node, dim)                     # [n_nodes, dim]
    p_full = _to_full(pstar_node, 1)                   # [n_nodes]
    h = mesh.tree.h()[sf.elem]
    jacS = (h / 2.0) ** (dim - 1)
    dscale = 2.0 / h
    parts = {"conv": np.zeros(dm.n_nodes),
             "visc": np.zeros(dm.n_nodes),
             "poff": np.zeros(dm.n_nodes)}
    for fi in range(len(sf.elem)):
        f = int(sf.face[fi])
        un = u_full[conn[fi], :dim]                    # [nbf, dim]
        pn = p_full[conn[fi]]                          # [nbf]
        N = ftab.N[f]                                  # [nqf, nbf]
        dN = ftab.dN[f] * dscale[fi]                   # [nqf, nbf, dim] physical
        for q in range(nqf):
            gp = fi * nqf + q
            w = ftab.w[q] * jacS[fi] * geo.corr[gp]
            n = geo.n[gp]                              # domain-outward normal
            gradu = dN[q].T @ un                       # gradu[coord, comp]
            uq = N[q] @ un                             # [dim] discrete trace
            gradp = dN[q].T @ pn                       # [dim] grad p*
            # DIAGNOSTIC convective part: -(u_h.grad)u_h.n at the discrete
            # trace (+ f). Mesh-divergent (see docstring); off by default.
            conv = -(uq @ gradu)                       # [dim]
            if f_node is not None:
                conv = conv + np.asarray(f_node)[gp]
            np.add.at(parts["conv"], conn[fi], w * (n @ conv) * N[q])
            # increment offset: -grad(p*).n, collocated.
            np.add.at(parts["poff"], conn[fi], -w * (gradp @ n) * N[q])
            # viscous rotational (weak, P1-exact, the KIO no-slip datum):
            #   -nu (grad N_a x n) . curl u
            omega = (dN[q][:, 0] @ un[:, 1]            # du_y/dx
                     - dN[q][:, 1] @ un[:, 0])         # - du_x/dy  (scalar)
            cross = dN[q][:, 0] * n[1] - dN[q][:, 1] * n[0]   # [nbf]
            np.add.at(parts["visc"], conn[fi], -nu * w * cross * omega)
    if return_parts:
        return parts
    rhs = np.zeros(dm.n_nodes)
    if include_conv:
        rhs += parts["conv"]
    if include_visc:
        rhs += parts["visc"]
    if include_poff:
        rhs += parts["poff"]
    return scale * rhs


def sbm_wall_pressure_kio_vms(dm, sf, geo, uhat, u1, u2, bdf, dt, nu, ndof,
                              timestab=True, scale=1.0):
    r"""REJECTED FN4 experiment (2026-07-23, kept as the investigation
    record; default OFF everywhere): the KIO wall-pressure source weighted by
    the PPE's OWN fine-scale channel. NOT the rung-B drag fix — that is the
    stepper's ``rotational_pin_wall`` (leray.py). Verdict: docs/dev/2026-07-23-
    projection-ladder-verdict.md. NOT dead code (P2harden prune check,
    2026-07-23): still imported+called by tests/ladder_rungB_square_nitsche.py
    and tests/ladder_rungB_seed_probe.py (feeding green test_ladder_rungB.py) —
    do NOT remove without first dropping those references.

    THE IDEA: the consistent-projection PPE assembles

        (grad phi, grad q) = -sigma (div u_hat, q) - sigma (tau_m r_m, grad q)

    whose fine-scale term carries the natural wall flux
    dphi/dn = -sigma tau_m r_m . n  (r_m the strong momentum residual), so a
    raw ``oint g q`` Neumann source adds data at WEIGHT 1 against a channel
    of weight sigma*tau_m (h-dependent). Imposing the KIO balance THROUGH the
    same weight,  s = sigma tau_m [ r_m.n + g_KIO - grad(p*).n ],  makes the
    convective and grad(p*) parts cancel ALGEBRAICALLY (same discrete
    evaluations), leaving

        s = sigma tau_m [ (du/dt|_BDF).n  -  nu (curl curl u).n ]

    — the BDF trace acceleration (collocated) + the rotational viscous term
    (weak, P1-exact: -sigma tau_m nu (grad N_a x n).omega).

    MEASURED VERDICT — REJECTED: the sigma^2 tau_m (u_hat - u_hist).n
    acceleration term is a DERIVATIVE feedback on the wall trace that
    amplifies per-step predictor drift (sigma^2 tau_m ~ 20), and the split
    lands far from the monolithic: from rest Cd = +10.9 (L4) / +11.7 (L5),
    seeded-monolithic driven to +15/+11, with or without the rotational wall
    pin (target +1.39/+2.02). The premise was also wrong: the wall-pressure
    defect is the rotational -nu*q update at the weak wall (see
    sbm_wall_pressure_neumann's verdict note), not a missing PPE wall datum —
    with ``rotational_pin_wall`` the homogeneous-Neumann wall needs no
    source.

    Args: ``uhat`` the predicted velocity (free-node [n_free,dim] or full
    node-major), ``u1``/``u2`` the BDF history velocities (same layout; u2
    may be None), ``bdf=(b0,b1,b2)`` the CURRENT BDF coefficients (b2=0 on
    startup), ``dt`` the step, ``timestab`` matches the stepper's tau_m
    transient term (tau uses dt when True — mirrors the PPE's taum_fs).
    Returns a FULL node-major (dm.n_nodes) scalar to ADD to the PPE RHS
    before the constraint reduction. 2-D only (curl u scalar)."""
    dim = dm.dim
    if dim != 2:
        from ..errors import ConfigError
        raise ConfigError("KIO VMS wall-pressure source is implemented for "
                          "2-D (curl u scalar); got dim=%d" % dim)
    from ..physics.vms import tau_hbased_host
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
    nqf = ftab.nqf
    conn = mesh.conn_of[pv][np.searchsorted(mesh.bins[pv], sf.elem)]
    Tc = dm.constraints.T

    def _to_full(field):
        if field is None:
            return None
        arr = np.asarray(field)
        if arr.shape[0] == Tc.shape[1]:                # constrained free-node
            return np.asarray(Tc @ arr)
        return arr                                     # already full node-major

    b0, b1, b2 = bdf
    sigma = b0 / dt
    u_full = _to_full(uhat)                            # [n_nodes, dim]
    u1_full = _to_full(u1)
    u2_full = _to_full(u2)
    h = mesh.tree.h()[sf.elem]
    jacS = (h / 2.0) ** (dim - 1)
    dscale = 2.0 / h
    rhs = np.zeros(dm.n_nodes)
    for fi in range(len(sf.elem)):
        f = int(sf.face[fi])
        un = u_full[conn[fi], :dim]                    # [nbf, dim]
        u1n = u1_full[conn[fi], :dim]
        u2n = (u2_full[conn[fi], :dim] if u2_full is not None else None)
        N = ftab.N[f]                                  # [nqf, nbf]
        dN = ftab.dN[f] * dscale[fi]                   # [nqf, nbf, dim] physical
        for q in range(nqf):
            gp = fi * nqf + q
            w = ftab.w[q] * jacS[fi] * geo.corr[gp]
            n = geo.n[gp]                              # domain-outward normal
            uq = N[q] @ un                             # [dim] trace of u_hat
            # tau_m at the face GP — mirrors the PPE's taum_fs (h-based tau at
            # the fluid element's h, transient term at dt when timestab).
            taum = float(tau_hbased_host(
                np.array([np.linalg.norm(uq)]), np.array([h[fi]]), nu,
                dt=(dt if timestab else None), dim=dim)[0])
            st_w = sigma * taum
            # BDF trace acceleration (du/dt|_BDF).n, collocated: the exact
            # time content of r_m at the face GP (sigma u_hat - history).
            dudt = (b0 * uq - b1 * (N[q] @ u1n)
                    - (b2 * (N[q] @ u2n) if u2n is not None else 0.0)) / dt
            np.add.at(rhs, conn[fi], st_w * w * (n @ dudt) * N[q])
            # rotational viscous KIO term, weak (P1-exact):
            #   -sigma tau_m nu (grad N_a x n) . omega
            omega = (dN[q][:, 0] @ un[:, 1]            # du_y/dx
                     - dN[q][:, 1] @ un[:, 0])         # - du_x/dy  (scalar)
            cross = dN[q][:, 0] * n[1] - dN[q][:, 1] * n[0]
            np.add.at(rhs, conn[fi], -st_w * nu * w * cross * omega)
    return scale * rhs


def sbm_consistent_flux(dm, sf, geo, x_all, nu, ndof, alpha=10.0,
                        g_fn=None, symmetric_grad=False, include_penalty=False):
    """CONSISTENT-FLUX boundary force on a weakly-imposed (Nitsche/SBM) wall,
    per the group's NSHT_SBM reference.

    NSHT_SBM's PRODUCTION Cd path is `include/BoundaryCalc_ExtraInter.h`
    (`imgaTraversalOperation` -> `computeForce` -> `forceCalcIBM`). It computes
    ONLY

        Force_pressure(j) = p n_j  ,   Force_viscous(j) = -(grad(u).n)_j * nu

    where p and grad(u) are LINEARLY EXTRAPOLATED from two interior points
    (`fe`, `fe2`) out to the TRUE boundary GP (the SBM "consistent" shift), and
    the surface measure is the TRUE-boundary area `TrueGPAreaGlobal`. The
    Nitsche `Force_penalty` term IS DEFINED but is NEVER accumulated in the
    active traversal (it is commented out in NSHTIBMPost.h / SumGPIBMPost.h and
    absent from BoundaryCalc_ExtraInter.h) — so the production drag = pressure +
    viscous of the field extrapolated to the true wall, NO penalty.

    For a BODY-FITTED wall (d == 0, surrogate == true boundary, geo.corr == 1)
    the extrapolation is the identity and this reduces EXACTLY to
    `surrogate_traction`. The SBM shift enters ONLY as the field extrapolation
    grad(u).d (Taylor) applied before taking the traction; it is 0 here.

    This helper reproduces that: per direction j at each surrogate GP,

        f_j = (p + grad(p).d?) n_j                         (pressure, shifted)
              - nu (grad(u).n [+ grad(u)^T.n])_j            (viscous, shifted)
              [+ (alpha nu / h)(u_j + (grad(u).d)_j - g_j)] (penalty, OPT-IN)

    with the field values Taylor-shifted to the true boundary via grad(u).d
    (pressure shift needs grad(p), omitted here — negligible for d small; exact
    at d=0). `include_penalty=True` adds the full Nitsche penalty REACTION (the
    `weakBCpenaltyParameter = Cb_f nu / h` term, Cb_f == alpha) — this is the
    reaction functional's penalty flux, but note it is NOT part of the NSHT_SBM
    production drag and on a leaky weak wall (u.n != 0) it is large and should
    NOT be added for Cd. Default OFF to match the reference.

    `symmetric_grad`: our own Nitsche assembly
    (poisson.make_sbm_dirichlet_Ae, line 326 `-Na*gnb`) is the
    component-diagonal LAPLACIAN form (one-sided grad(u).n), so
    `symmetric_grad=False` (default) is consistent with OUR discretization AND
    with the reference `calculateDiff` (which also uses the one-sided
    grad(u).n). `symmetric_grad=True` uses the full stress grad(u)+grad(u)^T.

    Orientation matches `surrogate_traction` (n = geo.n domain-outward; drag
    downstream). For a STRONG node (u == g, d == 0) this reduces to the raw
    traction. `g_fn(y) -> [Ngp, dim]` is the wall velocity data at the mapped
    (true) points; None => no-slip g=0.

    x_all: FULL node-major (u, p) vector. Host-side observable."""
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
    xv = x_all.reshape(dm.n_nodes, ndof)
    h = mesh.tree.h()[sf.elem]
    jacS = (h / 2.0) ** (dim - 1)
    dscale = 2.0 / h
    # wall velocity data at the mapped (true) surrogate points y = xq + d
    gbar = (np.zeros((geo.n.shape[0], dim)) if g_fn is None
            else np.ascontiguousarray(g_fn(geo.xq + geo.d), np.float64))
    # SBM shift vector d at each face GP: [ne_f*nqf, dim]
    dvec = geo.d.reshape(-1, dim) if geo.d.ndim == 2 else \
        geo.d.reshape(len(sf.elem) * nqf, dim)
    F = np.zeros(dim)
    for fi in range(len(sf.elem)):
        f = int(sf.face[fi])
        un = xv[conn[fi], :dim]                          # [nbf, dim]
        pn = xv[conn[fi], dim]                           # [nbf]
        N = ftab.N[f]                                    # [nqf, nbf]
        dN = ftab.dN[f] * dscale[fi]                     # [nqf, nbf, dim]
        for q in range(nqf):
            gp = fi * nqf + q
            w = ftab.w[q] * jacS[fi] * geo.corr[gp]
            n = geo.n[gp]
            gradu = dN[q].T @ un                         # gradu[coord, comp]
            pq = N[q] @ pn
            uq = N[q] @ un                               # [dim] wall velocity
            # SBM field extrapolation to the TRUE boundary (grad(u).d Taylor
            # shift; d==0 for body-fitted => identity, matches reference).
            # (grad(u).d)_comp = sum_coord d_coord * du_comp/d(coord)
            shift = dvec[gp] @ gradu                     # [comp]
            # viscous flux (per component): one-sided (grad(u).n)_comp =
            # sum_coord n_coord du_comp/d(coord) = (gradu.T @ n)_comp.
            # Matches surrogate_traction AND our Laplacian assembly / reference
            # calculateDiff; symmetric_grad adds grad(u)^T.n = (gradu @ n).
            visc = gradu.T @ n
            if symmetric_grad:
                visc = visc + gradu @ n
            F += w * (pq * n - nu * visc)
            if include_penalty:
                # Nitsche penalty REACTION (alpha nu / h)(u + grad(u).d - g);
                # NOT part of the NSHT_SBM production drag (see docstring).
                pen_coef = alpha * nu * dscale[fi]       # alpha nu / h
                F += w * pen_coef * (uq + shift - gbar[gp])
    return F


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
