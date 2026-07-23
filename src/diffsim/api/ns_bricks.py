r"""NS bricks for M1b Task 5 — the linearized monolithic (u, p) momentum block.

See src/diffsim/api/example_bricks.py for the Poisson strong -> weak -> code
walk-through; this is the same recipe for a coupled, stabilized, nonlinear
system, so it is necessarily denser.

-------------------------------------------------------------------------------
The incompressible Navier-Stokes weak form
-------------------------------------------------------------------------------

STRONG FORM.  Velocity u and pressure p on Omega (density 1, viscosity nu):

        u_t + (u . grad) u = -grad p + nu div(grad u) + f      (momentum)
                   div u   = 0                                  (continuity)

WEAK FORM.  Test momentum with w, continuity with q. Integrate the viscous and
pressure terms by parts (the "do-nothing" outflow drops their surface terms):

    Int w . u_t + Int w . (u.grad)u + nu Int grad w : grad u
        - Int (div w) p          = Int w . f                    (momentum)
    Int q (div u)                = 0                            (continuity)

Equal-order (u, p) is inf-sup UNSTABLE, so we add residual-based VMS/SUPG-PSPG
stabilization on the momentum strong residual res_M = u_t + (u.grad)u + grad p
- nu lap u - f: a SUPG term tau_M (a.grad w).res_M, a PSPG term tau_M (grad q).
res_M (this is what lets equal-order work), and a grad-div term tau_C (div w)
(div u). See physics/vms.py for tau_M, tau_C.

The rest of this header documents how the NONLINEAR convection is linearized
into the monolithic block this brick assembles:

- ndof = dim+1, node-major (u_1..u_dim, p);

- ndof = dim+1, node-major (u_1..u_dim, p);
- advecting velocity `a` is a PRECOMPUTED Gauss-point field (Stokes: a = 0;
  Oseen: extrapolated fine-scale-corrected velocity — the stepper's job);
- convection in the s = 1/2 skew form: a.grad u + (1/2)(div a) u
  (energy-stable; the operator's (div a) uses the DISCRETE advecting field);
- tau frozen at the advecting velocity (calc_tau at a), metric form,
  sigma = b0/dt into both the mass term and tau's transient part;
- stabilization on the LINEARIZED strong residual (retrofit G4,
  2026-07-13: the COMPLETE form)
      res_M = sigma u + a.grad u + grad p - nu lap u_h - f
  The -nu lap u_h term rides the lapN tables: identically ZERO at p1
  (Q1 diagonal second derivatives vanish — p1 assembly bit-identical to
  the pre-G4 code) and REQUIRED at p2, where omitting it caps the
  velocity L2 order at 2 (measured 2.15/1.85 pre-fix — the same
  incomplete-residual mechanism measured on the scalar brick);
- SUPG test (a.grad w) tau res_M, PSPG (grad q) tau res_M, grad-div
  tauC (div w)(div u); NO tauM^2 Reynolds terms (linearized operator drops
  them — conventions item 4).

The brick writes the FULL (dim+1)x(dim+1) node blocks; strong velocity
Dirichlet rows and the pressure pin are applied by the caller (stepper).
"""
import numpy as np
import scipy.sparse as sp
import warp as wp

from ..assembly.femelm import FEMElm, fe_N, fe_dN_s, fe_detJxW_s
from ..assembly.operators import _kernel_cache
from ..physics.vms import tau_m_metric, tau_c_metric


# --------------------------------------------------------------------------
# Outflow backflow stabilization (change #6 of the consistent-projection fix)
# --------------------------------------------------------------------------
def outflow_faces(mesh, axis=0, side=1, coord=1.0, tol=1e-9):
    r"""Enumerate the OUTER-BOUNDARY faces on the outflow plane ``x_axis=coord``.

    Returns ``(elem, face, ntilde)``: ``elem`` [Nf] element ids (into the tree),
    ``face`` [Nf] local face ids ``2*axis+side``, and ``ntilde`` [dim] the constant
    domain-outward unit normal on that plane (e.g. ``(+1,0)`` for the x=1 face).

    A face qualifies when it is a mesh boundary face (no neighbour across it) of
    the requested orientation AND its face centre lies on the plane. For the unit
    cube channel the default (``axis=0, side=1, coord=1.0``) is the free x=1
    outflow. Host-side; the outflow-face count is O(mesh side), so a small loop is
    fine (mirrors the SBM surrogate-face path)."""
    from ..octree.lookup import face_neighbors, face_offsets
    from ..octree import morton
    dim = mesh.dim
    tree = mesh.tree
    f = 2 * axis + side
    nbr = face_neighbors(tree)[f]                       # -1 == boundary face
    scale = 2.0 ** -morton.lmax(dim)
    lo = tree.anchors() * scale
    h = tree.h()
    face_coord = lo[:, axis] + (h if side == 1 else 0.0)
    keep = np.where((nbr < 0) & (np.abs(face_coord - coord) < tol))[0]
    ntilde = face_offsets(dim).astype(np.float64)[f]    # unit outward normal
    return keep.astype(np.int64), np.full(len(keep), f, np.int8), ntilde


def assemble_backflow_block(dm, u_free, beta, ndof, faces=None,
                            axis=0, side=1, coord=1.0, rho=1.0):
    r"""Velocity-based backflow stabilization on the outflow face (Bazilevs et al.
    CMAME 2009; Esmaily-Moghadam et al. Comput. Mech. 2011; directional-do-nothing,
    Braack-Mucha). Adds the weak-form term

        - beta * rho * \int_{Gamma_out} (u . n)_-  (u . v) dGamma ,
          (u . n)_- = min(u . n, 0)

    to the momentum block, ACTIVE ONLY where there is reverse flow (u.n < 0). The
    open "do-nothing" outflow leaves the convective energy flux unbounded when the
    wake pushes fluid back in through the outlet; this term restores a dissipative
    (negative-definite) contribution exactly there, and is ~0 (benign) with no
    backflow. Picard-linearized: (u.n)_- is frozen at the supplied advecting field
    ``u_free`` and the trial/test pair is the remaining (u . v).

    ``u_free`` [n_free, dim] is the advecting velocity in CONSTRAINED free-node
    space (the current Picard iterate for the predictor, the current Newton state
    for the monolithic). ``beta`` = 0 returns a zero matrix (bit-for-bit OFF).
    Returns a CSR of shape (n_free*ndof, n_free*ndof), constrained & node-major —
    add it to the assembled momentum block BEFORE the strong-row overwrite."""
    dim = dm.dim
    mesh = dm.mesh
    if beta == 0.0:
        Nn = dm.n_nodes * ndof
        Z = sp.csr_matrix((Nn, Nn))
        T = dm.constraints.T.tocsr()
        T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
        return (T_vec.T @ Z @ T_vec).tocsr()
    if faces is None:
        elem, face, ntilde = outflow_faces(mesh, axis=axis, side=side,
                                            coord=coord)
    else:
        elem, face, ntilde = faces
    # uniform p on the outflow face (channel is p1; matches surrogate_traction).
    from ..mesh.faces import face_tables
    p_face = np.unique(np.asarray(mesh.p_elem)[elem]) if len(elem) else \
        np.array([1])
    if len(p_face) != 1:
        from ..errors import ConfigError
        raise ConfigError(
            f"outflow backflow assumes uniform p on the face, got "
            f"orders {p_face.tolist()}")
    pv = int(p_face[0])
    ftab = face_tables(pv, dim)
    nqf, nbf = ftab.nqf, ftab.nbf
    conn = mesh.conn_of[pv][np.searchsorted(mesh.bins[pv], elem)]  # [Nf, nbf]
    u_full = np.asarray(dm.constraints.T @ u_free)                # node-major
    h = mesh.tree.h()[elem]
    jacS = (h / 2.0) ** (dim - 1)
    Nn = dm.n_nodes * ndof
    rows, cols, vals = [], [], []
    for fi in range(len(elem)):
        f = int(face[fi])
        N = ftab.N[f]                                             # [nqf, nbf]
        un = u_full[conn[fi], :dim]                               # [nbf, dim]
        # face-GP advecting velocity and its outward-normal component
        uq = N @ un                                               # [nqf, dim]
        un_dot_n = uq @ ntilde                                    # [nqf]
        un_neg = np.minimum(un_dot_n, 0.0)                        # (u.n)_-
        # Ab[a,b] = -beta*rho * int (u.n)_- N_a N_b dGamma  (per component)
        Ab = -beta * rho * np.einsum(
            "qa,qb,q,q->ab", N, N, un_neg, ftab.w) * jacS[fi]
        gnodes = conn[fi]
        for c in range(dim):
            gdof = gnodes * ndof + c
            rows.append(np.repeat(gdof, nbf))
            cols.append(np.tile(gdof, nbf))
            vals.append(Ab.ravel())
    if rows:
        A = sp.coo_matrix((np.concatenate(vals),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(Nn, Nn)).tocsr()
    else:
        A = sp.csr_matrix((Nn, Nn))
    T = dm.constraints.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    return (T_vec.T @ A @ T_vec).tocsr()


def make_linear_ns_Ae(nbf: int, nqp: int, dim: int):
    """Element-matrix kernel: inputs aq [ne*nqp, dim] advecting field at GPs,
    div_aq [ne*nqp], scalars nu, sigma, sig2tau; writes Ae node-major."""
    key = ("lin_ns_Ae", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]
    ndof = dim + 1
    dim_pow = float(dim)
    dim_f = float(dim)

    # findings 6: the fully-unrolled dim-3 ndof=4 element kernel was a
    # >79-min one-time nvrtc compile; rolled loops (max_unroll=0) trade a
    # few %% runtime for a compile measured in minutes (finding-1b pattern).
    @wp.kernel(module="unique", enable_backward=False,
               module_options=({"max_unroll": 0}
                               if (dim >= 3 or nbf > 4) else {}))
    def lin_ns_Ae(conn: wp.array2d(dtype=wp.int32),
                  h: wp.array(dtype=wp.float64),
                  Ntab: wp.array2d(dtype=wp.float64),
                  dNtab: wp.array3d(dtype=wp.float64),
                  lapNtab: wp.array2d(dtype=wp.float64),
                  wtab: wp.array(dtype=wp.float64),
                  aq: wp.array2d(dtype=wp.float64),
                  div_aq: wp.array(dtype=wp.float64),
                  gaq: wp.array2d(dtype=wp.float64),
                  nu: wp.float64, sigma: wp.float64, sig2tau: wp.float64,
                  s_skew: wp.float64, newton: wp.int32,
                  Ae: wp.array3d(dtype=wp.float64)):
        e = wp.tid()
        fe = FEMElm()
        fe.e = e
        fe.he = h[e]
        jac = wp.pow(fe.he * wp.float64(0.5), wp.float64(dim_pow))
        dscale = wp.float64(2.0) / fe.he
        for q in range(nqp):
            fe.q = q
            dJxW = fe_detJxW_s(wtab, fe, jac)
            gp = e * nqp + q
            amag = wp.float64(0.0)
            for d in range(dim):
                amag += aq[gp, d] * aq[gp, d]
            amag = wp.sqrt(amag)
            tauM = tau_m_metric(amag, fe.he, nu, sig2tau, wp.float64(dim_f))
            tauC = tau_c_metric(tauM, fe.he, wp.float64(dim_f))
            diva = div_aq[gp]
            for a in range(nbf):
                Na = fe_N(Ntab, fe, a)
                # a.grad(w_a) for SUPG test
                agw = wp.float64(0.0)
                for d in range(dim):
                    agw += aq[gp, d] * fe_dN_s(dNtab, fe, a, d, dscale)
                for b in range(nbf):
                    Nb = fe_N(Ntab, fe, b)
                    agu = wp.float64(0.0)   # a.grad(N_b)
                    lap = wp.float64(0.0)   # grad(w_a).grad(N_b)
                    for d in range(dim):
                        agu += aq[gp, d] * fe_dN_s(dNtab, fe, b, d, dscale)
                        lap += fe_dN_s(dNtab, fe, a, d, dscale) \
                            * fe_dN_s(dNtab, fe, b, d, dscale)
                    # generalized M_{a,s} convection (s runtime; default 1/2)
                    conv = agu + s_skew * diva * Nb
                    # linearized momentum strong residual factor on u_b —
                    # COMPLETE (G4): sigma u + a.grad u - nu lap u.  The
                    # lapN term is exactly 0 at p1 (bit-identical) and
                    # restores order 3 at p2 (scalar-brick mechanism).
                    resu = sigma * Nb + conv \
                        - nu * lapNtab[fe.q, b] * dscale * dscale
                    diag = (sigma * Na * Nb + Na * conv + nu * lap
                            + tauM * agw * resu) * dJxW
                    for i in range(dim):
                        wp.atomic_add(Ae, e, ndof * a + i, ndof * b + i, diag)
                        # grad-div: tauC (dw_i/dx_i)(du_j/dx_j)
                        for j in range(dim):
                            wp.atomic_add(
                                Ae, e, ndof * a + i, ndof * b + j,
                                tauC * fe_dN_s(dNtab, fe, a, i, dscale)
                                * fe_dN_s(dNtab, fe, b, j, dscale) * dJxW)
                        # pressure gradient: -(div w) p  (IBP form)
                        wp.atomic_add(
                            Ae, e, ndof * a + i, ndof * b + dim,
                            (-fe_dN_s(dNtab, fe, a, i, dscale) * Nb
                             + tauM * agw
                             * fe_dN_s(dNtab, fe, b, i, dscale)) * dJxW)
                        # continuity: q (du_i/dx_i)
                        wp.atomic_add(
                            Ae, e, ndof * a + dim, ndof * b + i,
                            (Na * fe_dN_s(dNtab, fe, b, i, dscale)
                             + tauM * fe_dN_s(dNtab, fe, a, i, dscale)
                             * resu) * dJxW)
                    # NEWTON cross-term (du.grad)a — component-coupling
                    # block Na (grad a)_{ij} Nb; Galerkin only (SUPG stays
                    # Picard-level, standard practice — documented delta
                    # from full Newton). RHS partner (a.grad)a is folded
                    # into f_eff by the caller, so it inherits SUPG/PSPG
                    # consistency for free.
                    if newton == 1:
                        for i in range(dim):
                            for j in range(dim):
                                wp.atomic_add(
                                    Ae, e, ndof * a + i, ndof * b + j,
                                    Na * Nb * gaq[gp, i * dim + j] * dJxW)
                    # PSPG pressure-pressure: tauM grad q . grad p
                    wp.atomic_add(Ae, e, ndof * a + dim, ndof * b + dim,
                                  tauM * lap * dJxW)

    _kernel_cache[key] = lin_ns_Ae
    return lin_ns_Ae


def make_linear_ns_be(nbf: int, nqp: int, dim: int):
    """RHS kernel: f at GPs [ne*nqp, dim] (body force + BDF history term),
    with SUPG/PSPG consistency on the test side."""
    key = ("lin_ns_be", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]
    ndof = dim + 1
    dim_pow = float(dim)
    dim_f = float(dim)

    # findings 6: the fully-unrolled dim-3 ndof=4 element kernel was a
    # >79-min one-time nvrtc compile; rolled loops (max_unroll=0) trade a
    # few %% runtime for a compile measured in minutes (finding-1b pattern).
    @wp.kernel(module="unique", enable_backward=False,
               module_options=({"max_unroll": 0}
                               if (dim >= 3 or nbf > 4) else {}))
    def lin_ns_be(conn: wp.array2d(dtype=wp.int32),
                  h: wp.array(dtype=wp.float64),
                  Ntab: wp.array2d(dtype=wp.float64),
                  dNtab: wp.array3d(dtype=wp.float64),
                  wtab: wp.array(dtype=wp.float64),
                  aq: wp.array2d(dtype=wp.float64),
                  fq: wp.array2d(dtype=wp.float64),
                  nu: wp.float64, sig2tau: wp.float64,
                  be: wp.array2d(dtype=wp.float64)):
        e = wp.tid()
        fe = FEMElm()
        fe.e = e
        fe.he = h[e]
        jac = wp.pow(fe.he * wp.float64(0.5), wp.float64(dim_pow))
        dscale = wp.float64(2.0) / fe.he
        for q in range(nqp):
            fe.q = q
            dJxW = fe_detJxW_s(wtab, fe, jac)
            gp = e * nqp + q
            amag = wp.float64(0.0)
            for d in range(dim):
                amag += aq[gp, d] * aq[gp, d]
            amag = wp.sqrt(amag)
            tauM = tau_m_metric(amag, fe.he, nu, sig2tau, wp.float64(dim_f))
            for a in range(nbf):
                Na = fe_N(Ntab, fe, a)
                agw = wp.float64(0.0)
                for d in range(dim):
                    agw += aq[gp, d] * fe_dN_s(dNtab, fe, a, d, dscale)
                for i in range(dim):
                    wp.atomic_add(be, e, ndof * a + i,
                                  (Na + tauM * agw) * fq[gp, i] * dJxW)
                    # PSPG on the RHS: tauM grad q . f
                    wp.atomic_add(be, e, ndof * a + dim,
                                  tauM * fe_dN_s(dNtab, fe, a, i, dscale)
                                  * fq[gp, i] * dJxW)

    _kernel_cache[key] = lin_ns_be
    return lin_ns_be


def assemble_linear_ns(dm, aq_by_bin, div_aq_by_bin, fq_by_bin, nu,
                       sigma=0.0, sig2tau=None, s_skew=0.5,
                       gaq_by_bin=None, newton=False):
    """(A, b) constrained, node-major ndof=dim+1. aq/div_aq/fq: per-bin GP
    arrays (host numpy). sig2tau defaults to (2 sigma)^2."""
    dim = dm.dim
    ndof = dim + 1
    if sig2tau is None:
        sig2tau = (2.0 * sigma) ** 2
    d = dm.device
    rows, cols, vals = [], [], []
    F_full = np.zeros(dm.n_nodes * ndof)
    for pv, b in dm.bins.items():
        ne = len(b["eids"])
        nbf, nqp = b["nbf"], b["nqp"]
        aq = wp.array(np.ascontiguousarray(aq_by_bin[pv]), dtype=wp.float64,
                      device=d)
        dq = wp.array(np.ascontiguousarray(div_aq_by_bin[pv]),
                      dtype=wp.float64, device=d)
        ga_np = (np.zeros((len(aq_by_bin[pv]), dim * dim))
                 if gaq_by_bin is None else
                 np.ascontiguousarray(
                     gaq_by_bin[pv].reshape(-1, dim * dim)))
        gaq = wp.array(ga_np, dtype=wp.float64, device=d)
        fq = wp.array(np.ascontiguousarray(fq_by_bin[pv]), dtype=wp.float64,
                      device=d)
        Ae = wp.zeros((ne, nbf * ndof, nbf * ndof), dtype=wp.float64,
                      device=d)
        be = wp.zeros((ne, nbf * ndof), dtype=wp.float64, device=d)
        kA = make_linear_ns_Ae(nbf, nqp, dim)
        kb = make_linear_ns_be(nbf, nqp, dim)
        wp.launch(kA, dim=ne, inputs=[b["conn"], b["h"], b["N"], b["dN"],
                                      b["lapN"],       # G4: complete resu
                                      b["w"], aq, dq, gaq, wp.float64(nu),
                                      wp.float64(sigma), wp.float64(sig2tau),
                                      wp.float64(s_skew),
                                      wp.int32(1 if newton else 0), Ae],
                  device=d)
        wp.launch(kb, dim=ne, inputs=[b["conn"], b["h"], b["N"], b["dN"],
                                      b["w"], aq, fq, wp.float64(nu),
                                      wp.float64(sig2tau), be], device=d)
        Aeh, beh = Ae.numpy(), be.numpy()
        conn = dm.mesh.conn_of[pv].astype(np.int64)
        gdof = (conn[:, :, None] * ndof
                + np.arange(ndof)[None, None, :]).reshape(ne, nbf * ndof)
        rows.append(np.repeat(gdof, nbf * ndof, axis=1).ravel())
        cols.append(np.tile(gdof, (1, nbf * ndof)).ravel())
        vals.append(Aeh.ravel())
        np.add.at(F_full, gdof.ravel(), beh.ravel())
    Nfull = dm.n_nodes * ndof
    K = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(Nfull, Nfull)).tocsr()
    T = dm.constraints.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    return (T_vec.T @ K @ T_vec).tocsr(), np.asarray(T_vec.T @ F_full)
