"""M2-A1: scalar advection-diffusion brick (temperature / species).

See src/diffsim/api/example_bricks.py for the Poisson strong -> weak -> code
walk-through. Poisson is the pure-diffusion (kappa grad T . grad w) core of this
brick; here we add transport by a velocity field a and the stabilization that
makes it robust at high Peclet number.

STRONG FORM.  For scalar T advected by a with diffusivity kappa and source f:

        sigma T + a . grad T - kappa div(grad T) = f

(sigma = reaction / BDF time coefficient, 0 for steady). Testing with w and
integrating the diffusion term by parts gives the Galerkin weak form
sigma (T, w) + (a.grad T, w) + kappa (grad T, grad w) = (f, w). Pure Galerkin
oscillates once advection dominates diffusion, so we add SUPG/VMS stabilization
on the strong residual:

The one-dof member of the NS kernel family: for scalar T with advecting
field a (frozen at GPs, same contract as the linearized NS step),

    sigma (T, w) + (a.grad T, w) + kappa (grad T, grad w)
        + tau_M (a.grad w, sigma T + a.grad T - f)         [SUPG]
        = (f, w) + tau_M (a.grad w, f)

sigma = reaction / BDF time coefficient (0 for steady), tau_M = the NS
metric tau with nu -> kappa (stabilization consistency across the
coupled system). Boundary machinery is REUSED, not duplicated: strong
Dirichlet rows at the caller (stepper convention), SBM Dirichlet/Neumann
face blocks from SBMPoisson (they are already kappa-parameterized and
add linearly to the volume block).

Brick genericity (A2): a species brick is THIS brick with a different
coefficient set — no new kernels.
"""
import numpy as np
import scipy.sparse as sp
import warp as wp

from ..assembly.operators import _kernel_cache
from ..api.ns_bricks import tau_m_metric
from .poisson import gauss_points  # noqa: F401  (re-export for callers)


def make_scalar_ad_Ae(nbf: int, nqp: int, dim: int):
    key = ("scalar_ad_Ae", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]
    dim_pow = float(dim)
    dim_f = float(dim)

    @wp.kernel(module="unique", enable_backward=False,
               module_options=({"max_unroll": 0} if dim >= 3 else {}))
    def scalar_ad_Ae(conn: wp.array2d(dtype=wp.int32),
                     h: wp.array(dtype=wp.float64),
                     Ntab: wp.array2d(dtype=wp.float64),
                     dNtab: wp.array3d(dtype=wp.float64),
                     lapNtab: wp.array2d(dtype=wp.float64),
                     wtab: wp.array(dtype=wp.float64),
                     aq: wp.array2d(dtype=wp.float64),
                     kq: wp.array(dtype=wp.float64),
                     sigma: wp.float64, sig2tau: wp.float64,
                     supg: wp.float64,
                     Ae: wp.array3d(dtype=wp.float64)):
        e = wp.tid()
        he = h[e]
        jac = wp.pow(he * wp.float64(0.5), wp.float64(dim_pow))
        dscale = wp.float64(2.0) / he
        for q in range(nqp):
            dJxW = wtab[q] * jac
            gp = e * nqp + q
            kap = kq[gp]
            amag = wp.float64(0.0)
            for d in range(dim):
                amag += aq[gp, d] * aq[gp, d]
            amag = wp.sqrt(amag)
            tauM = supg * tau_m_metric(amag, he, kap, sig2tau,
                                       wp.float64(dim_f))
            for a in range(nbf):
                Na = Ntab[q, a]
                agw = wp.float64(0.0)
                for d in range(dim):
                    agw += aq[gp, d] * dNtab[q, a, d] * dscale
                for b in range(nbf):
                    Nb = Ntab[q, b]
                    agu = wp.float64(0.0)
                    lap = wp.float64(0.0)
                    for d in range(dim):
                        agu += aq[gp, d] * dNtab[q, b, d] * dscale
                        lap += dNtab[q, a, d] * dNtab[q, b, d] \
                            * dscale * dscale
                    # COMPLETE strong residual on T_b (VMS): the
                    # -kappa*lap term vanishes at p1 (Q1 Laplacians are
                    # zero) and is REQUIRED at p2 — omitting it caps L2
                    # at order 2 (measured 2.11/2.03 vs 3.00 complete)
                    resu = (sigma * Nb + agu
                            - kap * lapNtab[q, b] * dscale * dscale)
                    wp.atomic_add(
                        Ae, e, a, b,
                        (sigma * Na * Nb + Na * agu + kap * lap
                         + tauM * agw * resu) * dJxW)

    _kernel_cache[key] = scalar_ad_Ae
    return scalar_ad_Ae


def make_scalar_ad_be(nbf: int, nqp: int, dim: int):
    key = ("scalar_ad_be", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]
    dim_pow = float(dim)
    dim_f = float(dim)

    @wp.kernel(module="unique", enable_backward=False,
               module_options=({"max_unroll": 0} if dim >= 3 else {}))
    def scalar_ad_be(conn: wp.array2d(dtype=wp.int32),
                     h: wp.array(dtype=wp.float64),
                     Ntab: wp.array2d(dtype=wp.float64),
                     dNtab: wp.array3d(dtype=wp.float64),
                     wtab: wp.array(dtype=wp.float64),
                     aq: wp.array2d(dtype=wp.float64),
                     kq: wp.array(dtype=wp.float64),
                     fq: wp.array(dtype=wp.float64),
                     sig2tau: wp.float64, supg: wp.float64,
                     be: wp.array2d(dtype=wp.float64)):
        e = wp.tid()
        he = h[e]
        jac = wp.pow(he * wp.float64(0.5), wp.float64(dim_pow))
        dscale = wp.float64(2.0) / he
        for q in range(nqp):
            dJxW = wtab[q] * jac
            gp = e * nqp + q
            kap = kq[gp]
            amag = wp.float64(0.0)
            for d in range(dim):
                amag += aq[gp, d] * aq[gp, d]
            amag = wp.sqrt(amag)
            tauM = supg * tau_m_metric(amag, he, kap, sig2tau,
                                       wp.float64(dim_f))
            for a in range(nbf):
                agw = wp.float64(0.0)
                for d in range(dim):
                    agw += aq[gp, d] * dNtab[q, a, d] * dscale
                wp.atomic_add(be, e, a,
                              (Ntab[q, a] + tauM * agw) * fq[gp] * dJxW)

    _kernel_cache[key] = scalar_ad_be
    return scalar_ad_be


def assemble_scalar_ad(dm, aq_by_bin, fq_by_bin, kappa, sigma=0.0,
                       sig2tau=None, supg=None):
    """Constrained scalar system (T^T K T, T^T F). kappa: scalar or
    callable kappa(x)->[M] per GP (the B1 field pathway). aq: advecting
    field at GPs per bin [ngp, dim] (zeros -> pure diffusion + reaction:
    the Poisson limit, gate-checked)."""
    from .poisson import gauss_points as _gp
    if sig2tau is None:
        sig2tau = (2.0 * sigma) ** 2
    # VMS-complete residual (Baskar 2026-07-07: the formulation extends
    # to p2 naturally — the earlier order cap was the INCOMPLETE
    # residual, not SUPG): lapN tables carry -kappa*lap(T_b); SUPG-type
    # stabilization is ON at every p by default.
    supg_by_p = {1: 1.0 if supg is None else float(supg),
                 2: 1.0 if supg is None else float(supg)}
    d = dm.device
    xq = _gp(dm.mesh, dm.tables_by_p)
    rows, cols, vals = [], [], []
    F_full = np.zeros(dm.n_nodes)
    for pv, b in dm.bins.items():
        conn = dm.mesh.conn_of[pv].astype(np.int64)
        ne, nbf = conn.shape
        nqp = b["nqp"]
        kq_np = (kappa(xq[pv]) if callable(kappa)
                 else np.full(len(xq[pv]), float(kappa)))
        aq = wp.array(np.ascontiguousarray(aq_by_bin[pv]),
                      dtype=wp.float64, device=d)
        kq = wp.array(np.ascontiguousarray(kq_np), dtype=wp.float64,
                      device=d)
        fq = wp.array(np.ascontiguousarray(fq_by_bin[pv]),
                      dtype=wp.float64, device=d)
        Ae = wp.zeros((ne, nbf, nbf), dtype=wp.float64, device=d)
        be = wp.zeros((ne, nbf), dtype=wp.float64, device=d)
        kA = make_scalar_ad_Ae(nbf, nqp, dim=dm.dim)
        kb = make_scalar_ad_be(nbf, nqp, dim=dm.dim)
        sg = wp.float64(supg_by_p.get(pv, 1.0))
        wp.launch(kA, dim=ne, inputs=[b["conn"], b["h"], b["N"],
                                      b["dN"], b["lapN"], b["w"], aq, kq,
                                      wp.float64(sigma),
                                      wp.float64(sig2tau), sg, Ae],
                  device=d)
        wp.launch(kb, dim=ne, inputs=[b["conn"], b["h"], b["N"],
                                      b["dN"], b["w"], aq, kq, fq,
                                      wp.float64(sig2tau), sg, be],
                  device=d)
        Aeh, beh = Ae.numpy(), be.numpy()
        rows.append(np.repeat(conn, nbf, axis=1).ravel())
        cols.append(np.tile(conn, (1, nbf)).ravel())
        vals.append(Aeh.ravel())
        np.add.at(F_full, conn.ravel(), beh.ravel())
    K = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(dm.n_nodes, dm.n_nodes)).tocsr()
    T = dm.constraints.T.tocsr()
    return (T.T @ K @ T).tocsr(), np.asarray(T.T @ F_full)

# PROVENANCE (CMAME_NSPNP_WeakBC.pdf, Def. 3 Eq. 28 + Remark 4, p.7 —
# the group's published scalar form): ONE stabilization term,
# +sum_K (tau (a.grad q), Res)_K — SUPG-type test, NO plain-w term;
# residual truncated to dc/dt + a.grad c at p1 (second-order terms
# dropped by Remark 4's H1 argument; body force joins the residual in
# the NS analogue Eq. 30). THIS module implements that form PLUS the
# -kappa*lapN residual term — identical at p1 (lapN=0), and the
# measured requirement for order 3 at p2 (Remark 4's truncation is a
# p1 statement, not a formulation limit).
# FORMULATION NOTE (Baskar, 2026-07-07): the full VMS fine-scale
# substitution yields three terms: (1) +tau(a.grad w, Res) [SUPG-type —
# CRUCIAL, implemented, with the COMPLETE residual incl. -kappa*lapN];
# (2) +tau((div a) w, Res) [plain-w; zero for solenoidal a, small for
# discrete NS velocities — omitted, minor for convergence];
# (3) +kappa*tau(lap w, Res) [adjoint-diffusion; zero at p1 — omitted,
# minor; lapN tables make it ~3 lines when p2 VMS-exactness is wanted].
