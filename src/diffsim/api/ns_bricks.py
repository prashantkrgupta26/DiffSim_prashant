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
