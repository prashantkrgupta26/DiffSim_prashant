"""M2-A1: scalar advection-diffusion brick (temperature / species).

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
                    resu = sigma * Nb + agu       # strong residual on T_b
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
    # SUPG is a P1 DEVICE: its strong residual here omits -kappa*lap(T)
    # (exact at p1 where element Laplacians vanish; a CONSISTENCY error
    # at p2 that caps L2 order at 2 — measured 2.11/2.03 vs Galerkin's
    # 3). Default: SUPG on p1 bins, GALERKIN on p2 bins. p2-SUPG with
    # the Hessian-completed residual arrives with M2-C's basis-Hessian
    # tables.
    supg_by_p = ({1: 1.0, 2: 0.0} if supg is None
                 else {1: float(supg), 2: float(supg)})
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
                                      b["dN"], b["w"], aq, kq,
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
