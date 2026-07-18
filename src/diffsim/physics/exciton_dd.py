"""SP-1 B1+B2+B3: XDD Poisson brick, carrier drift-diffusion bricks, and
exciton diffusion-reaction bricks.

WEAK FORMS (SP-1 brick family — keep in sync as B4 adds terms)
-----------------------------------------------------------------
B1  Poisson (this brick):
    Strong (nondim): −∇·(λ² ε̂(x) ∇φ̂) − (p̂ − n̂) = 0
    Weak:   λ² (ε̂ ∇φ̂, ∇w) = (p̂ − n̂, w)
    NO stabilization — operator is elliptic; SUPG is a deliberate drop vs the
    CPU code (recorded deviation: CPU SUPG-on-Poisson removed per B1 spec).

    λ² from XDDParams.scales().lambda2  (= φ₀ ε_m / (x₀² C₀ q), Debye² scale)
    ε̂(x) = ε_r(x) / max(ε_A, ε_D)  — supplied as a GP field by the caller
    (same GP-field contract as kappa in scalar_transport.py).

B2  Carrier drift-diffusion (n̂ electrons / p̂ holes) — CONSERVATIVE signed form
    (CPU DDEquation.h parity):
    Strong (nondim): ∂_t ĉ + ∇·(sign·μ̂(x) ĉ ∇φ̂) − μ̂∇²ĉ = f
    CPU weak residuals (the parity target — quote verbatim):
      be_n += μ̂_n(−n̂·(∇φ̂·∇w) + ∇n̂·∇w)   // electrons: eqm n̂ ∝ e^{+φ̂}
      be_p += μ̂_p(+p̂·(∇φ̂·∇w) + ∇p̂·∇w)   // holes:     eqm p̂ ∝ e^{−φ̂}

    FORM CHOICE — CONSERVATIVE  (drift term carries the DENSITY, weighted by
    ∇φ̂·∇w against the TEST function — NOT the advective a·∇ĉ).  This is the
    corrected form: the previous nonconservative a·∇ĉ (sign=∓μ̂∇φ̂) gave the
    WRONG equilibrium (n̂∝e^{−φ̂}) and DROPPED the ±μ̂n̂Δφ̂ reaction term (LARGE
    in devices: Δφ̂ = −(p̂−n̂)/(λ²ε̂), λ² small).  Block C physics gate caught it.

    aq_gp CONVENTION (unchanged literals, reinterpreted role): the caller passes
        aq_gp = sign · μ̂(x) · ∇φ̂     (sign=−1 electrons, +1 holes)
    which is the SIGNED DRIFT WEIGHT vector.  The kernel forms the drift residual
    as (aq·∇w) ĉ  — gradient on the TEST function N_a, density on the TRIAL N_b:
      electrons: aq=−μ̂_n∇φ̂ → drift residual −μ̂_n n̂(∇φ̂·∇w)   ✓ CPU A22 sign
      holes:     aq=+μ̂_p∇φ̂ → drift residual +μ̂_p p̂(∇φ̂·∇w)   ✓ CPU A33 sign

    Galerkin weak form (σ = BDF coefficient, f includes history term):
      σ(ĉ, w) + (aq·∇w) ĉ + μ̂(∇ĉ, ∇w) + SUPG = (f, w) + SUPG_rhs

    SUPG: Tezduyar-class τ_M via tau_m_metric(|U|, h, μ̂, sig²τ, dim).  The
    upwind velocity is U = −aq (CPU: U_n=+μ̂∇φ̂ for electrons, U_p=−μ̂∇φ̂):
    physically carriers move along their drift velocity U, and the conservative
    drift weight aq=sign·μ̂∇φ̂ = −U.  Test-function augmentation is (U·∇w).
    The strong residual of the conservative form includes the −/+μ̂ĉΔφ̂ term;
    since Δφ̂ at GPs is not passed to the standalone carrier kernel we stabilise
    the ADVECTIVE part U·∇ĉ = −(aq·∇ĉ) plus σ and −μ̂∇²ĉ (as the CPU effectively
    does — the ±μ̂ĉΔφ̂ divergence part of the residual is the recorded SUPG
    omission; the Galerkin drift term is exact regardless).

    ONE factory pair; sign is absorbed into aq_gp by the caller — the kernel is
    sign-agnostic.  See assemble_xdd_carrier for the calling convention.

B3  Exciton diffusion-reaction (X̂_D donor / X̂_A acceptor) — NO SUPG:
    Strong (nondim, per species i ∈ {D, A}):
      ∂_t X̂_i − ∇·(μ̂_{X,i}(x) ∇X̂_i) + σ_tot X̂_i = Ĝ_i + R̂_{feed,i}
      σ_tot = σ_BDF + 1/τ̂_{x,i} + k̂_{diss,i}(x)
    Galerkin weak form (NO SUPG — excitons are diffusion-dominated):
      σ_tot (X̂_i, w) + μ̂_{X,i}(∇X̂_i, ∇w) = (f̂_i, w)
      f̂_i = Ĝ_i + R̂_{feed,i} + σ_BDF X̂_i^n [+ BDF2 correction]

    DESIGN DECISION — one new stiffness kernel, reuse the carrier load:
    σ_tot(x) is a spatially-varying mass coefficient (k̂_diss(x) is a GP
    field), but the carrier Ae only accepts a SCALAR sigma, so reuse there
    would contort the API.  We therefore write ONE thin kernel
    make_xdd_exciton_Ae taking sigma as a GP array:
        Ae += (σ_tot_q N_b N_a + μ̂_X_q ∇N_b·∇N_a) dJxW
    The load vector reuses make_xdd_carrier_be verbatim: at aq=0, supg=0 it
    reduces to (f̂, N_a)dV exactly.  ONE new Ae factory, ZERO new be factory.

NO SUPG on excitons (recorded deviation from the carrier brick; excitons are
diffusion-dominated and the CPU code does not stabilise them).
"""
import numpy as np
import scipy.sparse as sp
import warp as wp

from ..assembly.femelm import FEMElm, fe_dN_s, fe_detJxW_s, fe_N
from ..assembly.operators import _kernel_cache
from ..api.ns_bricks import tau_m_metric

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
        half = he * wp.float64(0.5)
        # jac = (he/2)^dim — power loop (house idiom, see operators.py)
        jac = wp.float64(1.0)
        for _ in range(dim):
            jac = jac * half
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
        half = he * wp.float64(0.5)
        # jac = (he/2)^dim — power loop (house idiom, see operators.py)
        jac = wp.float64(1.0)
        for _ in range(dim):
            jac = jac * half
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


# ══════════════════════════════════════════════════════════════════════════════
# B2 — carrier drift-diffusion stiffness kernel
# ══════════════════════════════════════════════════════════════════════════════
#
# Kernel signature mirrors scalar_transport.make_scalar_ad_Ae exactly:
#   conn   [ne, nbf]     int32  — local→global DOF map
#   h      [ne]          f64    — element size
#   Ntab   [nqp, nbf]    f64    — basis values (for mass term)
#   dNtab  [nqp, nbf, d] f64    — reference basis gradients
#   lapNtab[nqp, nbf]    f64    — reference basis Laplacians (for VMS p2)
#   wtab   [nqp]         f64    — quadrature weights
#   aq     [ne*nqp, dim] f64    — advection field at GPs (sign*mu*grad_phi)
#   kq     [ne*nqp]      f64    — μ̂(x) diffusivity at GPs
#   sigma                f64    — BDF coefficient (0 for steady)
#   sig2tau              f64    — (2σ)² for tau_m_metric
#   supg                 f64    — SUPG scale (1.0 or 0.0)
#   Ae     [ne,nbf,nbf]  f64    — output (pre-zeroed)
# ══════════════════════════════════════════════════════════════════════════════

def make_xdd_carrier_Ae(nbf: int, nqp: int, dim: int):
    """Element stiffness for the XDD carrier brick (SUPG, CONSERVATIVE form).

    Implements (aq = sign·μ̂·∇φ̂ from the caller; U = −aq the drift velocity):
        σ(N_b, N_a) + (aq·∇N_a) N_b + μ̂(∇N_b, ∇N_a) + τ_M(U·∇N_a, res_b)
    where the CONSERVATIVE drift term (aq·∇N_a) N_b carries the density N_b and
    weights ∇φ̂·∇N_a against the TEST function (CPU DDEquation.h parity: the
    electron block is −μ̂_n n̂(∇φ̂·∇w), hole +μ̂_p p̂(∇φ̂·∇w)).
    res_b = σ N_b + (U·∇N_b) − μ̂ lapN_b is the VMS strong residual on the
    advective part (U·∇ĉ), the SUPG-stabilised piece; the ±μ̂ĉΔφ̂ divergence
    part is the recorded SUPG omission (Δφ̂ not available in this kernel).

    Sign is baked into aq_gp by the caller (electrons aq=−μ̂∇φ̂, holes +μ̂∇φ̂);
    U = −aq is formed inside the kernel for the SUPG upwind direction.
    ONE factory — sign is not a kernel arg; the caller varies aq_gp.
    Cache key: ("xdd_carrier_Ae", nbf, nqp, dim).
    """
    key = ("xdd_carrier_Ae", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    dim_f   = float(dim)

    @wp.kernel(module="unique", enable_backward=False,
               module_options=({"max_unroll": 0} if dim >= 3 else {}))
    def xdd_carrier_Ae(
        conn:     wp.array2d(dtype=wp.int32),
        h:        wp.array(dtype=wp.float64),
        Ntab:     wp.array2d(dtype=wp.float64),
        dNtab:    wp.array3d(dtype=wp.float64),
        lapNtab:  wp.array2d(dtype=wp.float64),
        wtab:     wp.array(dtype=wp.float64),
        aq:       wp.array2d(dtype=wp.float64),   # [ne*nqp, dim]
        kq:       wp.array(dtype=wp.float64),     # [ne*nqp]  μ̂ at GPs
        sigma:    wp.float64,
        sig2tau:  wp.float64,
        supg:     wp.float64,
        Ae:       wp.array3d(dtype=wp.float64),
    ):
        e = wp.tid()
        he = h[e]
        half = he * wp.float64(0.5)
        # jac = (he/2)^dim — power loop (house idiom, see operators.py)
        jac = wp.float64(1.0)
        for _ in range(dim):
            jac = jac * half
        dscale = wp.float64(2.0) / he

        for q in range(nqp):
            dJxW = wtab[q] * jac
            gp   = e * nqp + q
            kap  = kq[gp]       # μ̂ at this Gauss point

            # |U| for tau_m_metric (U = −aq is the drift velocity; |U|=|aq|)
            amag = wp.float64(0.0)
            for d in range(dim):
                amag += aq[gp, d] * aq[gp, d]
            amag = wp.sqrt(amag)
            tauM = supg * tau_m_metric(amag, he, kap, sig2tau,
                                       wp.float64(dim_f))

            for a in range(nbf):
                Na = Ntab[q, a]
                # aq·∇N_a  (conservative drift: ∇φ̂-weight on the TEST function)
                # U·∇N_a = −(aq·∇N_a)  (SUPG test augmentation, U = −aq)
                agNa = wp.float64(0.0)
                for d in range(dim):
                    agNa += aq[gp, d] * dNtab[q, a, d] * dscale
                Ugw = -agNa

                for b in range(nbf):
                    Nb = Ntab[q, b]
                    # U·∇N_b = −(aq·∇N_b)  (advection on trial for the residual)
                    agNb = wp.float64(0.0)
                    # ∇N_a·∇N_b (diffusion)
                    lap = wp.float64(0.0)
                    for d in range(dim):
                        agNb += aq[gp, d] * dNtab[q, b, d] * dscale
                        lap += (dNtab[q, a, d] * dNtab[q, b, d]
                                * dscale * dscale)
                    Ugu = -agNb

                    # VMS strong residual on trial N_b (advective part):
                    #   res_b = σ N_b + U·∇N_b − μ̂ lapN_b
                    # lapN is zero at p1 (Q1 Laplacians = 0) but required at p2
                    # for third-order L2 convergence (scalar_transport finding).
                    resu = (sigma * Nb + Ugu
                            - kap * lapNtab[q, b] * dscale * dscale)

                    # Galerkin: σ(N_b N_a) + (aq·∇N_a) N_b  [CONSERVATIVE drift]
                    #           + μ̂(∇N_b·∇N_a)
                    # SUPG:     τ_M (U·∇N_a) res_b
                    wp.atomic_add(
                        Ae, e, a, b,
                        (sigma * Na * Nb + agNa * Nb + kap * lap
                         + tauM * Ugw * resu) * dJxW)

    _kernel_cache[key] = xdd_carrier_Ae
    return xdd_carrier_Ae


# ══════════════════════════════════════════════════════════════════════════════
# B2 — carrier load kernel
# ══════════════════════════════════════════════════════════════════════════════

def make_xdd_carrier_be(nbf: int, nqp: int, dim: int):
    """Element load for the XDD carrier brick.

    Implements: (f, N_a) + τ_M (U·∇N_a, f),  U = −aq the drift velocity
    where f is the full source (D̂ − R̂ + BDF-history for B4;
    the manufactured source for MMS tests).  The SUPG augmentation matches the
    Ae kernel's U·∇N_a test augmentation (conservative-form consistency).

    Cache key: ("xdd_carrier_be", nbf, nqp, dim).
    """
    key = ("xdd_carrier_be", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    dim_f   = float(dim)

    @wp.kernel(module="unique", enable_backward=False,
               module_options=({"max_unroll": 0} if dim >= 3 else {}))
    def xdd_carrier_be(
        conn:    wp.array2d(dtype=wp.int32),
        h:       wp.array(dtype=wp.float64),
        Ntab:    wp.array2d(dtype=wp.float64),
        dNtab:   wp.array3d(dtype=wp.float64),
        wtab:    wp.array(dtype=wp.float64),
        aq:      wp.array2d(dtype=wp.float64),   # [ne*nqp, dim]
        kq:      wp.array(dtype=wp.float64),     # [ne*nqp]  μ̂ at GPs
        fq:      wp.array(dtype=wp.float64),     # [ne*nqp]  source
        sig2tau: wp.float64,
        supg:    wp.float64,
        be:      wp.array2d(dtype=wp.float64),
    ):
        e = wp.tid()
        he = h[e]
        half = he * wp.float64(0.5)
        jac = wp.float64(1.0)
        for _ in range(dim):
            jac = jac * half
        dscale = wp.float64(2.0) / he

        for q in range(nqp):
            dJxW = wtab[q] * jac
            gp   = e * nqp + q
            kap  = kq[gp]
            fv   = fq[gp]

            amag = wp.float64(0.0)
            for d in range(dim):
                amag += aq[gp, d] * aq[gp, d]
            amag = wp.sqrt(amag)
            tauM = supg * tau_m_metric(amag, he, kap, sig2tau,
                                       wp.float64(dim_f))

            for a in range(nbf):
                # U·∇N_a = −(aq·∇N_a)  (SUPG test augmentation, U = −aq drift vel)
                Ugw = wp.float64(0.0)
                for d in range(dim):
                    Ugw += aq[gp, d] * dNtab[q, a, d] * dscale
                Ugw = -Ugw
                # (f, N_a) + τ_M (U·∇N_a, f)  — SUPG-consistent with the Ae kernel
                wp.atomic_add(be, e, a,
                              (Ntab[q, a] + tauM * Ugw) * fv * dJxW)

    _kernel_cache[key] = xdd_carrier_be
    return xdd_carrier_be


# ══════════════════════════════════════════════════════════════════════════════
# B2 — assembly helper
# ══════════════════════════════════════════════════════════════════════════════

def assemble_xdd_carrier(dm, aq_gp, mu_gp, fq_gp, *, sigma=0.0,
                          sig2tau=None, supg=None):
    """Assemble the constrained XDD carrier system T^T K T, T^T F.

    Parameters
    ----------
    dm : DeviceMesh
    aq_gp : dict {p: np.ndarray[ngp, dim]}
        Signed drift-weight field at Gauss points (CONSERVATIVE form).  Caller
        builds this as ``sign * mu_gp * grad_phi_gp``:
          electrons → sign=-1  →  aq = −μ̂ ∇φ̂  (drift residual −μ̂n̂(∇φ̂·∇w))
          holes     → sign=+1  →  aq = +μ̂ ∇φ̂  (drift residual +μ̂p̂(∇φ̂·∇w))
        The kernel forms the conservative drift term (aq·∇w) ĉ and uses U=−aq
        as the SUPG upwind velocity.  Sign is fully absorbed by the caller;
        this helper and the kernel are sign-agnostic.
    mu_gp : dict {p: np.ndarray[ngp]}
        Mobility μ̂(x) at GPs (diffusivity in the drift-diffusion PDE).
    fq_gp : dict {p: np.ndarray[ngp]}
        Source f at GPs.  For standalone MMS: manufactured source.
        For B4 coupled solve: D̂ − R̂ + BDF-history terms.
    sigma : float
        BDF time coefficient: 0 (steady), 1/Δt (BDF1), 3/(2Δt) (BDF2).
    sig2tau : float or None
        (2σ)² passed to tau_m_metric.  Derived as (2σ)² if None.
    supg : float or None
        SUPG scale.  1.0 (on, default) or 0.0 (off = Galerkin).

    Returns
    -------
    K : scipy.sparse.csr_matrix  (constrained, n_free × n_free)
    F : np.ndarray               (constrained, length n_free)
    """
    if sig2tau is None:
        sig2tau = (2.0 * sigma) ** 2
    supg_val = 1.0 if supg is None else float(supg)

    d = dm.device
    rows, cols, vals = [], [], []
    F_full = np.zeros(dm.n_nodes)

    for pv, b in dm.bins.items():
        conn_np = dm.mesh.conn_of[pv].astype(np.int64)
        ne, nbf = conn_np.shape
        nqp     = b["nqp"]

        aq_np  = np.ascontiguousarray(aq_gp[pv],  dtype=np.float64)
        mu_np  = np.ascontiguousarray(mu_gp[pv],  dtype=np.float64)
        fq_np  = np.ascontiguousarray(fq_gp[pv],  dtype=np.float64)

        aq_d  = wp.array(aq_np,  dtype=wp.float64, device=d)
        kq_d  = wp.array(mu_np,  dtype=wp.float64, device=d)
        fq_d  = wp.array(fq_np,  dtype=wp.float64, device=d)

        Ae = wp.zeros((ne, nbf, nbf), dtype=wp.float64, device=d)
        be = wp.zeros((ne, nbf),      dtype=wp.float64, device=d)

        kA = make_xdd_carrier_Ae(nbf, nqp, dm.dim)
        kb = make_xdd_carrier_be(nbf, nqp, dm.dim)
        sg = wp.float64(supg_val)

        wp.launch(kA, dim=ne,
                  inputs=[b["conn"], b["h"], b["N"], b["dN"], b["lapN"],
                          b["w"], aq_d, kq_d,
                          wp.float64(sigma), wp.float64(sig2tau), sg, Ae],
                  device=d)
        wp.launch(kb, dim=ne,
                  inputs=[b["conn"], b["h"], b["N"], b["dN"],
                          b["w"], aq_d, kq_d, fq_d,
                          wp.float64(sig2tau), sg, be],
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


# ══════════════════════════════════════════════════════════════════════════════
# B3 — exciton diffusion-reaction stiffness kernel
# ══════════════════════════════════════════════════════════════════════════════
#
# Weak form (per species, NO SUPG):
#   σ_tot(x)(X̂, w) + μ̂_X(x)(∇X̂, ∇w) = (f̂, w)
#   f̂ = Ĝ + R̂_feed + σ_BDF·X̂ⁿ  [+ BDF2 correction when applicable]
#
# DRY decision: the load kernel (be) is identical to the carrier be with
# aq=0 and supg=0 (reduces to (f̂, N_a)dV).  Therefore we REUSE
# make_xdd_carrier_be for the load.  The stiffness (Ae) requires a NEW kernel
# because σ_tot is spatially varying (a GP field, not a scalar), and the
# carrier Ae only accepts a scalar sigma.  Writing a single make_xdd_exciton_Ae
# (< 40 lines) that promotes sigma from float64 to array(float64) is the
# correct DRY tradeoff: ONE new kernel (Ae), ZERO new load kernel (reuse
# carrier_be verbatim).  The brief explicitly allows new factories when reuse
# "contorts the API" — the spatially-varying sigma case is exactly that.
# ══════════════════════════════════════════════════════════════════════════════

def make_xdd_exciton_Ae(nbf: int, nqp: int, dim: int):
    """Element stiffness for the XDD exciton brick (reaction-diffusion, no SUPG).

    Implements per element:
        Ae[e, a, b] += (σ_tot_q N_b N_a + μ̂_X_q ∇N_b·∇N_a) dJxW

    σ_tot_q is a GP field (σ_BDF + 1/τ̂_x + k̂_diss(x) at each Gauss point).
    μ̂_X_q is the exciton diffusivity GP field.
    ONE factory for both X̂_D and X̂_A — species distinction is entirely in the
    GP coefficient fields (μ̂_X and σ_tot may differ between donor/acceptor).
    NO SUPG — excitons are diffusion-dominated (deviation from CPU recorded in
    the module docstring and the SP-1 plan's deviation ledger).

    Kernel signature:
        conn      [ne, nbf]       int32  — local→global DOF map
        h         [ne]            f64    — element size
        Ntab      [nqp, nbf]      f64    — basis values
        dNtab     [nqp, nbf, dim] f64    — reference basis gradients
        wtab      [nqp]           f64    — quadrature weights
        sigma_gp  [ne*nqp]        f64    — σ_tot(x) at GPs
        mu_gp     [ne*nqp]        f64    — μ̂_X(x) at GPs
        Ae        [ne, nbf, nbf]  f64    — output (pre-zeroed)

    Cache key: ("xdd_exciton_Ae", nbf, nqp, dim).
    """
    key = ("xdd_exciton_Ae", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique", enable_backward=False)
    def xdd_exciton_Ae(
        conn:     wp.array2d(dtype=wp.int32),
        h:        wp.array(dtype=wp.float64),
        Ntab:     wp.array2d(dtype=wp.float64),
        dNtab:    wp.array3d(dtype=wp.float64),
        wtab:     wp.array(dtype=wp.float64),
        sigma_gp: wp.array(dtype=wp.float64),   # [ne*nqp]  σ_tot(x) at GPs
        mu_gp:    wp.array(dtype=wp.float64),   # [ne*nqp]  μ̂_X(x) at GPs
        Ae:       wp.array3d(dtype=wp.float64),  # [ne, nbf, nbf]
    ):
        e = wp.tid()
        he = h[e]
        half = he * wp.float64(0.5)
        # jac = (he/2)^dim — power loop (house idiom, see operators.py)
        jac = wp.float64(1.0)
        for _ in range(dim):
            jac = jac * half
        dscale = wp.float64(2.0) / he

        for q in range(nqp):
            dJxW = wtab[q] * jac
            gp   = e * nqp + q
            sig  = sigma_gp[gp]    # σ_tot at this Gauss point
            kap  = mu_gp[gp]       # μ̂_X at this Gauss point

            for a in range(nbf):
                Na = Ntab[q, a]
                for b in range(nbf):
                    Nb = Ntab[q, b]
                    # diffusion: μ̂_X (∇N_b · ∇N_a)
                    lap = wp.float64(0.0)
                    for d in range(dim):
                        lap += (dNtab[q, a, d] * dNtab[q, b, d]
                                * dscale * dscale)
                    # mass-reaction: σ_tot N_b N_a
                    wp.atomic_add(Ae, e, a, b,
                                  (sig * Na * Nb + kap * lap) * dJxW)

    _kernel_cache[key] = xdd_exciton_Ae
    return xdd_exciton_Ae


# ══════════════════════════════════════════════════════════════════════════════
# B3 — exciton assembly helper
# ══════════════════════════════════════════════════════════════════════════════

def assemble_xdd_exciton(dm, mu_gp, sigma_gp, fq_gp):
    """Assemble the constrained XDD exciton system T^T K T, T^T F.

    Implements the weak form:
        σ_tot(x)(X̂, w) + μ̂_X(x)(∇X̂, ∇w) = (f̂, w)

    DRY NOTE — kernel reuse:
    - Stiffness (Ae): uses make_xdd_exciton_Ae (new kernel, required because
      σ_tot is a GP field — the carrier Ae only accepts a scalar sigma).
    - Load vector (be): REUSES make_xdd_carrier_be with aq=0 and supg=0.
      At these values the carrier be reduces to (f̂, N_a)dV exactly:
          tauM = supg * tau_m_metric(...) = 0   (supg=0)
          agw  = sum(aq[gp,d]*dNtab[q,a,d]*dscale) = 0   (aq=0)
          be[e,a] += (Ntab[q,a] + 0) * fv * dJxW
      This is identical to the Poisson be kernel — reuse is clean.
    Cache keys: "xdd_exciton_Ae" (new) / "xdd_carrier_be" (reused).

    Parameters
    ----------
    dm : DeviceMesh
    mu_gp : dict {p: np.ndarray[ngp]}
        Exciton diffusivity μ̂_X(x) at Gauss points.  Species distinction
        (donor vs acceptor) is in the values; use the same function for both.
    sigma_gp : dict {p: np.ndarray[ngp]}
        Effective mass coefficient σ_tot(x) = σ_BDF + 1/τ̂_x + k̂_diss(x)
        at Gauss points.  May be spatially varying (k̂_diss is a GP field).
    fq_gp : dict {p: np.ndarray[ngp]}
        Full RHS source at GPs: f̂ = Ĝ + R̂_feed + history term(s).

    Returns
    -------
    K : scipy.sparse.csr_matrix  (constrained, n_free × n_free)
    F : np.ndarray               (constrained, length n_free)
    """
    d = dm.device
    rows, cols, vals = [], [], []
    F_full = np.zeros(dm.n_nodes)

    for pv, b in dm.bins.items():
        conn_np = dm.mesh.conn_of[pv].astype(np.int64)
        ne, nbf = conn_np.shape
        nqp     = b["nqp"]
        ngp     = ne * nqp

        mu_np  = np.ascontiguousarray(mu_gp[pv],    dtype=np.float64)
        sig_np = np.ascontiguousarray(sigma_gp[pv], dtype=np.float64)
        fq_np  = np.ascontiguousarray(fq_gp[pv],    dtype=np.float64)

        mu_d   = wp.array(mu_np,  dtype=wp.float64, device=d)
        sig_d  = wp.array(sig_np, dtype=wp.float64, device=d)
        fq_d   = wp.array(fq_np,  dtype=wp.float64, device=d)

        # --- Stiffness: mass-reaction + diffusion (new exciton kernel) ---
        Ae = wp.zeros((ne, nbf, nbf), dtype=wp.float64, device=d)
        kA = make_xdd_exciton_Ae(nbf, nqp, dm.dim)
        wp.launch(kA, dim=ne,
                  inputs=[b["conn"], b["h"], b["N"], b["dN"],
                          b["w"], sig_d, mu_d, Ae],
                  device=d)

        # --- Load: reuse carrier_be with aq=0, supg=0 → (f̂, N_a) dV ---
        # aq=zeros means advection = 0; sig2tau=0 and supg=0 disable SUPG.
        # The carrier_be kernel at these values reduces to Ntab[q,a]*fv*dJxW,
        # which is exactly the exciton load (f̂, N_a)dV.
        aq_zero = wp.zeros((ngp, dm.dim), dtype=wp.float64, device=d)
        mu_be   = wp.array(mu_np, dtype=wp.float64, device=d)  # kq (unused at supg=0)

        be = wp.zeros((ne, nbf), dtype=wp.float64, device=d)
        kb = make_xdd_carrier_be(nbf, nqp, dm.dim)
        wp.launch(kb, dim=ne,
                  inputs=[b["conn"], b["h"], b["N"], b["dN"],
                          b["w"], aq_zero, mu_be, fq_d,
                          wp.float64(0.0),   # sig2tau = 0 (SUPG disabled)
                          wp.float64(0.0),   # supg    = 0 (SUPG disabled)
                          be],
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
