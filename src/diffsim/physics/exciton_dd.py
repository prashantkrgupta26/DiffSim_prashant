"""SP-1 B1: XDD Poisson brick — λ²-form quasi-static electrostatics.

WEAK FORMS (SP-1 brick family — keep in sync as B2/B4 add terms)
-----------------------------------------------------------------
B1  Poisson (this brick):
    Strong (nondim): −∇·(λ² ε̂(x) ∇φ̂) − (p̂ − n̂) = 0
    Weak:   λ² (ε̂ ∇φ̂, ∇w) = (p̂ − n̂, w)
    NO stabilization — operator is elliptic; SUPG is a deliberate drop vs the
    CPU code (recorded deviation: CPU SUPG-on-Poisson removed per B1 spec).

    λ² from XDDParams.scales().lambda2  (= φ₀ ε_m / (x₀² C₀ q), Debye² scale)
    ε̂(x) = ε_r(x) / max(ε_A, ε_D)  — supplied as a GP field by the caller
    (same GP-field contract as kappa in scalar_transport.py).

KERNEL FACTORIES (house idiom — see scalar_transport.py)
---------------------------------------------------------
  make_xdd_poisson_Ae(nbf, nqp, dim)   →  warp kernel (Ke stiffness)
  make_xdd_poisson_be(nbf, nqp, dim)   →  warp kernel (fe load, rho source)

ASSEMBLY
--------
  assemble_xdd_poisson(dm, eps_gp, rho_gp, *, lam2=None, params=None,
                       f_src_gp=None)
      → (K_constrained: csr_matrix, F_constrained: ndarray)
  Caller passes lam2 explicitly OR XDDParams (params); at least one must be
  supplied.  eps_gp and rho_gp are dicts {p: np.ndarray[ngp]} keyed by poly
  degree (same as tables_by_p).  Optional f_src_gp adds an extra body load
  (for future B4 use; default None = zeros).

B2/B4 consumers: factories and assembly signature are stable.  Add B2 carrier
bricks (make_xdd_ndd_Ae etc.) in this file following the same pattern.
"""
import numpy as np
import scipy.sparse as sp
import warp as wp

from ..assembly.femelm import FEMElm, fe_dN_s, fe_detJxW_s, fe_N
from ..assembly.operators import _kernel_cache

wp.set_module_options({"enable_backward": False})


# ──────────────────────────────────────────────────────────────────────────────
# Stiffness kernel: Ke[e,a,b] += λ² ε̂_q (∇Na · ∇Nb) dJxW
# ──────────────────────────────────────────────────────────────────────────────

def make_xdd_poisson_Ae(nbf: int, nqp: int, dim: int):
    """Element stiffness for the XDD Poisson operator  λ² (ε̂ ∇φ, ∇w).

    Kernel signature (matches house idiom in scalar_transport / operators):
        conn   [ne, nbf]   int32   — local→global DOF map
        h      [ne]        f64     — element size
        dNtab  [nqp,nbf,dim] f64  — reference basis gradients
        wtab   [nqp]       f64     — quadrature weights
        eps_gp [ne*nqp]    f64     — ε̂ at every Gauss point
        lam2               f64     — λ² scalar
        Ae     [ne,nbf,nbf] f64   — output (pre-zeroed by caller)

    dim is a Python compile-time constant closed over from the factory;
    jac = (he/2)^dim via a power loop (house idiom).
    """
    key = ("xdd_poisson_Ae", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    dim_pow = float(dim)

    @wp.kernel(module="unique", enable_backward=False)
    def xdd_poisson_Ae(
        conn:   wp.array2d(dtype=wp.int32),
        h:      wp.array(dtype=wp.float64),
        dNtab:  wp.array3d(dtype=wp.float64),
        wtab:   wp.array(dtype=wp.float64),
        eps_gp: wp.array(dtype=wp.float64),   # [ne*nqp]
        lam2:   wp.float64,
        Ae:     wp.array3d(dtype=wp.float64),  # [ne, nbf, nbf]
    ):
        e = wp.tid()
        he = h[e]
        jac = wp.pow(he * wp.float64(0.5), wp.float64(dim_pow))
        dscale = wp.float64(2.0) / he
        fe = FEMElm()
        fe.e = e
        fe.he = he
        for q in range(nqp):
            fe.q = q
            dJxW = wtab[q] * jac
            gp = e * nqp + q
            eps_q = eps_gp[gp]
            coeff = lam2 * eps_q * dJxW
            for a in range(nbf):
                for b in range(nbf):
                    v = wp.float64(0.0)
                    for d in range(dim):
                        v = v + (fe_dN_s(dNtab, fe, a, d, dscale)
                                 * fe_dN_s(dNtab, fe, b, d, dscale))
                    wp.atomic_add(Ae, e, a, b, v * coeff)

    _kernel_cache[key] = xdd_poisson_Ae
    return xdd_poisson_Ae


# ──────────────────────────────────────────────────────────────────────────────
# Load kernel: be[e,a] += (ρ_q + f_q) N_a dJxW
# ──────────────────────────────────────────────────────────────────────────────

def make_xdd_poisson_be(nbf: int, nqp: int, dim: int):
    """Element load for the XDD Poisson RHS  (p̂ − n̂, w) + (f_src, w).

    Kernel signature:
        conn   [ne, nbf]   int32
        h      [ne]        f64
        Ntab   [nqp,nbf]   f64   — basis values
        wtab   [nqp]       f64   — quadrature weights
        rho_gp [ne*nqp]    f64   — charge density p̂−n̂ at GPs
        f_src  [ne*nqp]    f64   — extra body load (zeros if not used)
        be     [ne, nbf]   f64   — output (pre-zeroed by caller)
    """
    key = ("xdd_poisson_be", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    dim_pow = float(dim)

    @wp.kernel(module="unique", enable_backward=False)
    def xdd_poisson_be(
        conn:   wp.array2d(dtype=wp.int32),
        h:      wp.array(dtype=wp.float64),
        Ntab:   wp.array2d(dtype=wp.float64),
        wtab:   wp.array(dtype=wp.float64),
        rho_gp: wp.array(dtype=wp.float64),   # [ne*nqp]
        f_src:  wp.array(dtype=wp.float64),   # [ne*nqp]
        be:     wp.array2d(dtype=wp.float64),  # [ne, nbf]
    ):
        e = wp.tid()
        he = h[e]
        jac = wp.pow(he * wp.float64(0.5), wp.float64(dim_pow))
        fe = FEMElm()
        fe.e = e
        fe.he = he
        for q in range(nqp):
            fe.q = q
            dJxW = wtab[q] * jac
            gp = e * nqp + q
            fv = rho_gp[gp] + f_src[gp]
            for a in range(nbf):
                wp.atomic_add(be, e, a, fe_N(Ntab, fe, a) * fv * dJxW)

    _kernel_cache[key] = xdd_poisson_be
    return xdd_poisson_be


# ──────────────────────────────────────────────────────────────────────────────
# Assembly helper
# ──────────────────────────────────────────────────────────────────────────────

def assemble_xdd_poisson(dm, eps_gp, rho_gp, *, lam2=None, params=None,
                          f_src_gp=None):
    """Assemble the constrained XDD Poisson system T^T K T, T^T F.

    Parameters
    ----------
    dm : DeviceMesh
    eps_gp : dict {p: np.ndarray[ne_p * nqp_p]}
        Normalised permittivity field ε̂(x) at every Gauss point, per p-bin.
    rho_gp : dict {p: np.ndarray[ne_p * nqp_p]}
        Charge density (p̂ − n̂) at every Gauss point, per p-bin.
    lam2 : float, optional
        λ² Debye-squared scale.  Exactly one of lam2 or params must be given.
    params : XDDParams, optional
        If supplied, lam2 = params.scales().lambda2 is used.
    f_src_gp : dict {p: ndarray} or None
        Extra body load (B4 coupling hook); None → zeros everywhere.

    Returns
    -------
    K : scipy.sparse.csr_matrix  (constrained, shape n_free × n_free)
    F : np.ndarray               (constrained, length n_free)
    """
    if lam2 is None and params is None:
        raise ValueError("supply lam2 or params")
    if params is not None:
        lam2 = params.scales().lambda2
    lam2 = float(lam2)

    d = dm.device
    rows, cols, vals = [], [], []
    F_full = np.zeros(dm.n_nodes)

    for pv, b in dm.bins.items():
        conn_np = dm.mesh.conn_of[pv].astype(np.int64)
        ne, nbf = conn_np.shape
        nqp = b["nqp"]
        ngp = ne * nqp

        eps_np = np.ascontiguousarray(eps_gp[pv], dtype=np.float64)
        rho_np = np.ascontiguousarray(rho_gp[pv], dtype=np.float64)
        fsrc_np = (np.ascontiguousarray(f_src_gp[pv], dtype=np.float64)
                   if f_src_gp is not None else np.zeros(ngp))

        eps_d  = wp.array(eps_np,  dtype=wp.float64, device=d)
        rho_d  = wp.array(rho_np,  dtype=wp.float64, device=d)
        fsrc_d = wp.array(fsrc_np, dtype=wp.float64, device=d)

        Ae = wp.zeros((ne, nbf, nbf), dtype=wp.float64, device=d)
        be = wp.zeros((ne, nbf),      dtype=wp.float64, device=d)

        kA = make_xdd_poisson_Ae(nbf, nqp, dm.dim)
        kb = make_xdd_poisson_be(nbf, nqp, dm.dim)

        wp.launch(kA, dim=ne,
                  inputs=[b["conn"], b["h"], b["dN"], b["w"],
                          eps_d, wp.float64(lam2), Ae],
                  device=d)
        wp.launch(kb, dim=ne,
                  inputs=[b["conn"], b["h"], b["N"], b["w"],
                          rho_d, fsrc_d, be],
                  device=d)

        Aeh = Ae.numpy()
        beh = be.numpy()

        rows.append(np.repeat(conn_np, nbf, axis=1).ravel())
        cols.append(np.tile(conn_np, (1, nbf)).ravel())
        vals.append(Aeh.ravel())
        np.add.at(F_full, conn_np.ravel(), beh.ravel())

    K = sp.coo_matrix(
        (np.concatenate(vals),
         (np.concatenate(rows), np.concatenate(cols))),
        shape=(dm.n_nodes, dm.n_nodes),
    ).tocsr()
    T = dm.constraints.T.tocsr()
    return (T.T @ K @ T).tocsr(), np.asarray(T.T @ F_full)
