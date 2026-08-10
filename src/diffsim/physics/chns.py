"""CHNS physics helpers — Task 1 (SP-0): phase-mixture closures, capillary
force density, and VMS stabilization parameter for the coupled
Cahn-Hilliard–Navier-Stokes brick.

All functions are pure NumPy; Warp kernels (Tasks 3–7) call these host forms
for CPU reference paths and gate their correctness against these results.

Spec reference
--------------
  Spec §1 (task-1-brief.md), functions mix_props / capillary_gp / tau_m_gp.
  CHNSIntegrandsGenForm.hpp:125-217 pullback pattern is mirrored here for the
  phase-mixture averaging (linear-in-phi interpolation).

  Element-matrix / RHS constant-viscosity kernels being generalized:
  ``src/diffsim/api/ns_bricks.py:341-514`` (make_linear_ns_Ae / make_linear_ns_be).

tau_m_gp legacy
---------------
  Generalizes the constant-viscosity h-based tau formula in
  ``src/diffsim/physics/vms.py:48-56`` (``tau_hbased_host``) to per-GP local
  rho, eta.  Constant mapping (tau_hbased_host symbol -> chns.py symbol):

    tau_hbased_host            chns.py                    Value              Role
    ------------------         ---------------            -----              -------------------------
    (2*b0/dt)^2  (b0=1)        4.0 / dt^2                 4/dt^2             transient term
    c1=4.0                     ci0=4.0                    4.0                advective coefficient
    c2CI=CI_F*16*dim (dim-dep) ci_f*16*dim * nu^2/h^4     1152 @ dim=2       viscous coefficient
                                                           1728 @ dim=3
    nu = eta/(rho*Re)          eta_gp/(rho_gp*Re)         --                 kinematic viscosity, local

  The outer ``/ rho_gp`` converts the pressure-stabilized tauM (units of
  time/density) into the momentum stabilization form required by the CHNS
  variational residual (see spec §1).
"""

import numpy as np

__all__ = ["mix_props", "capillary_gp", "tau_m_gp", "make_chns_newton"]

# ---------------------------------------------------------------------------
# Phase-mixture linear interpolation
# ---------------------------------------------------------------------------


def mix_props(
    phi: np.ndarray,
    rho_h: float,
    rho_l: float,
    eta_h: float,
    eta_l: float,
    floor_frac: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Linear phase-mixture density and viscosity at arbitrary phi values.

    Interpolates linearly in phi: for phi in [-1, 1]
        rho = a_rho * phi + b_rho,   a = (hi-lo)/2, b = (hi+lo)/2
        eta = a_eta * phi + b_eta
    Clamps results from below at ``floor_frac * lo`` to prevent zero or
    negative material properties when phi overshoots the [-1, 1] interval.

    Parameters
    ----------
    phi : ndarray, any shape
        Phase-field values.
    rho_h, rho_l : float
        Density of the heavy (phi=+1) and light (phi=-1) phase.
    eta_h, eta_l : float
        Dynamic viscosity of the heavy (phi=+1) and light (phi=-1) phase.
    floor_frac : float, optional
        Clamping floor as a fraction of the low-phase value.  Default 1e-3.

    Returns
    -------
    rho : ndarray, same shape as phi
        Mixture density, clamped from below.
    eta : ndarray, same shape as phi
        Mixture viscosity, clamped from below.
    n_clamped : int
        Number of entries where at least one property was clamped.
    """
    phi = np.asarray(phi, dtype=float)

    a_rho = 0.5 * (rho_h - rho_l)
    b_rho = 0.5 * (rho_h + rho_l)
    a_eta = 0.5 * (eta_h - eta_l)
    b_eta = 0.5 * (eta_h + eta_l)

    rho_raw = a_rho * phi + b_rho
    eta_raw = a_eta * phi + b_eta

    rho_floor = floor_frac * rho_l
    eta_floor = floor_frac * eta_l

    rho = np.maximum(rho_raw, rho_floor)
    eta = np.maximum(eta_raw, eta_floor)

    # Count GPs where any property was clamped
    clamped_mask = (rho_raw < rho_floor) | (eta_raw < eta_floor)
    n_clamped = int(np.sum(clamped_mask))

    return rho, eta, n_clamped


# ---------------------------------------------------------------------------
# Capillary body-force density
# ---------------------------------------------------------------------------


def capillary_gp(
    mu_gp: np.ndarray,
    grad_phi_gp: np.ndarray,
    Cn: float,
    We: float,
) -> np.ndarray:
    """Capillary force density at Gauss points: f_cap = (Cn*We)^{-1} mu grad(phi).

    Parameters
    ----------
    mu_gp : ndarray, shape [ngp]
        Chemical potential at each Gauss point.
    grad_phi_gp : ndarray, shape [ngp, dim]
        Gradient of the phase field at each Gauss point.
    Cn : float
        Cahn number (interface thickness / length scale).
    We : float
        Weber number.

    Returns
    -------
    f_gp : ndarray, shape [ngp, dim]
        Capillary force density at each Gauss point.
    """
    mu_gp = np.asarray(mu_gp, dtype=float)
    assert mu_gp.ndim == 1, f"mu_gp must be 1-D, got shape {mu_gp.shape}"
    grad_phi_gp = np.asarray(grad_phi_gp, dtype=float)

    prefactor = 1.0 / (Cn * We)
    # mu_gp[:, None] broadcasts over the dim axis
    return prefactor * mu_gp[:, None] * grad_phi_gp


# ---------------------------------------------------------------------------
# Per-GP VMS stabilization parameter
# ---------------------------------------------------------------------------


def tau_m_gp(
    u_gp: np.ndarray,
    rho_gp: np.ndarray,
    eta_gp: np.ndarray,
    h: float,
    dt: float,
    Re: float,
    Ci: tuple[float, float] = (4.0, 36.0),
) -> np.ndarray:
    """Per-Gauss-point VMS momentum stabilization parameter tau_m.

    Implements the h-based form from ``src/diffsim/physics/vms.py:48-56``
    (``tau_hbased_host``) generalized to per-GP local (rho, eta):

        tau_m = [ (2*b0/dt)^2 + c1*|u|^2/h^2
                              + c2CI*nu_local^2/h^4 ]^{-1/2} / rho_gp

    where b0=1, c1=ci0=4, nu_local = eta_gp/(rho_gp*Re), dim inferred from
    u_gp.shape[-1], and c2CI = CI_F * 16 * dim = 36 * 16 * dim
    (1152 for dim=2, 1728 for dim=3).

    Constant mapping to ``tau_hbased_host`` (vms.py:48-56):
      - transient: (2*b0/dt)^2 with b0=1  <=>  4/dt^2  (Ci[0]/dt^2)
      - advective: c1*|u|^2/h^2  (c1=4)   <=>  Ci[0]*u_mag**2/h**2
      - viscous:   c2CI*nu^2/h^4           <=>  Ci[1]*16*dim*nu_local**2/h**4

    See also: ``src/diffsim/api/ns_bricks.py:341-514`` for the constant-nu
    element kernels (make_linear_ns_Ae / make_linear_ns_be) being generalized.

    Parameters
    ----------
    u_gp : ndarray, shape [ngp, dim]
        Velocity at each Gauss point.
    rho_gp : ndarray, shape [ngp]
        Local mixture density at each Gauss point.
    eta_gp : ndarray, shape [ngp]
        Local mixture viscosity at each Gauss point.
    h : float
        Element size (mesh spacing).
    dt : float
        Time step size.
    Re : float
        Reynolds number.
    Ci : tuple (ci0, ci_f), optional
        Stabilization constants: ci0=advective coefficient (default 4.0),
        ci_f=CI_F in the house constant c2CI = ci_f * 16 * dim (default 36.0).
        Default (4.0, 36.0).

    Returns
    -------
    tau : ndarray, shape [ngp]
        Stabilization parameter at each Gauss point.
    """
    u_gp = np.asarray(u_gp, dtype=float)
    rho_gp = np.asarray(rho_gp, dtype=float)
    eta_gp = np.asarray(eta_gp, dtype=float)

    ci0, ci_f = float(Ci[0]), float(Ci[1])
    dim = u_gp.shape[-1]

    u_mag = np.sqrt(np.sum(u_gp ** 2, axis=-1))  # [ngp]

    nu_loc = eta_gp / (rho_gp * Re)               # local kinematic viscosity

    c2CI = ci_f * 16.0 * dim                      # house constant: 36*16*dim
    term_trans = ci0 / dt ** 2                    # (2*b0/dt)^2 with b0=1, ci0=4
    term_adv = ci0 * u_mag ** 2 / h ** 2          # c1*|u|^2/h^2, c1=ci0=4
    term_visc = c2CI * nu_loc ** 2 / h ** 4       # c2CI*nu^2/h^4 (dim-aware)

    denom = np.sqrt(term_trans + term_adv + term_visc)
    return 1.0 / (denom * rho_gp)


# ---------------------------------------------------------------------------
# Task 7 (SP-0): monolithic (u, p, phi, mu) Warp kernel factory
# ---------------------------------------------------------------------------
def make_chns_newton(nbf: int, nqp: int, dim: int, interface: str = "ch"):
    r"""Element residual (``be = -R``) + Jacobian (``Ae``) Warp kernel for the
    coupled BDF1 CHNS system, node-major ``blk = dim + 3`` per node ordered
    ``(u_0..u_{dim-1}, p, phi, mu)``.  This is the compiled twin of
    ``diffsim.adjoint.chns.CHNSDiscrete._assemble`` — every term/block mirrors
    that numpy reference (the executable spec); parity is gated to 1e-10.

    The host interpolates the state to Gauss points and passes per-GP arrays;
    the kernel builds the local rho/eta (mix_props), the DIFFERENTIATED tau
    (Newton-consistent dtau_du, dtau_dphi), the momentum strong residual
    ``r_mom`` and the Galerkin/SUPG/PSPG rows + their exact Jacobian blocks.

    Per-GP inputs (all [ngp, ...], node-major bin GP ordering gp = e*nqp+q):
      u_gp[dim], gu_gp[dim,dim] (comp,sp), p_gp, gp_gp[dim], phi_gp,
      gphi_gp[dim], mu_gp, gmu_gp[dim], un_gp[dim], phin_gp, src_phi_gp.
    Scalars: dt, Re, We, Cn, Pe, cw_inv, agg, grav_scale, ghat[dim],
      rho_h/rho_l/eta_h/eta_l (mix_props endpoints).
    Emits Ae[ne, blk*nbf, blk*nbf], be[ne, blk*nbf] (be = -residual).

    SP-1 deposition / MMS hooks (Task 9):
      ``src_phi_gp`` — per-GP scalar source S(x, t) added to the CH phi row:
        ``R^phi = INT psi[(phi-phi_n)/dt + u.grad phi + phi div u - S]
                 + (1/Pe) INT grad psi . grad mu``,
        so S grows the phase mass by INT S dV per unit time (the deposition
        channel).
      ``fbody_gp`` — per-GP momentum source f(x, t)[dim] entering BOTH the
        Galerkin momentum body AND the strong residual r_mom (so the SUPG/PSPG
        stabilization stays CONSISTENT — required for the MMS design-order
        gate; a Galerkin-only body force leaves an O(1) stabilization
        consistency error).  Same subtraction convention as f_grav.
    Both are pure explicit RHS terms (no Jacobian; the sources do not depend on
    the unknowns), passed as zeros when unused so the signature stays uniform.

    ``interface`` (static, part of the cache key):
      "ch" (default) -> the Cahn-Hilliard 4-field brick above; the CH kernel is
        byte-identical to the pre-CAC factory.
      "cac" -> Conservative Allen-Cahn (spec §3): the phi-row becomes
        R^phi = INT psi[(phi-phi_n)/dt + u.grad phi + phi div u]
              + gamma Cn INT gpsi.gphi
              + gamma INT psi[F'(phi)/Cn + beta sqrt(F(phi))]  (- src),
        gamma = 1/Pe, F = 1/4(phi^2-1)^2; the mu-row is a trivial identity
        R^mu = INT chi mu (mu -> 0, a wasted DOF); the momentum potential
        capillary (mu grad phi) and AGG flux are dropped, and surface tension
        is the Galerkin-only Korteweg form +(Cn/We)(gphi(x)gphi):grad w.  The
        Lagrange multiplier ``beta`` is FROZEN per Newton iterate (computed
        host-side, passed as a scalar) — no dbeta/dphi block, so the sparse
        structure matches CH.  Mirrors adjoint.chns.CHNSDiscrete(interface=
        "cac") exactly (parity gated 1e-10).  ``gamma``/``beta`` scalar args
        are ignored on the CH path.
    """
    import warp as wp
    from ..assembly.operators import _kernel_cache

    if interface not in ("ch", "cac"):
        raise ValueError(f"interface must be 'ch' or 'cac', got {interface!r}")
    cac = (interface == "cac")
    key = ("chns_newton", nbf, nqp, dim, interface)
    if key in _kernel_cache:
        return _kernel_cache[key]

    blk = dim + 3
    dim_pow = float(dim)
    c2CI = 36.0 * 16.0 * dim

    @wp.kernel(module="unique", enable_backward=False,
               module_options={"max_unroll": 0})
    def chns_k(conn: wp.array2d(dtype=wp.int32),
               h: wp.array(dtype=wp.float64),
               Ntab: wp.array2d(dtype=wp.float64),
               dNtab: wp.array3d(dtype=wp.float64),
               wtab: wp.array(dtype=wp.float64),
               u_gp: wp.array2d(dtype=wp.float64),     # [ngp, dim]
               gu_gp: wp.array3d(dtype=wp.float64),    # [ngp, dim, dim] comp,sp
               p_gp: wp.array(dtype=wp.float64),       # [ngp]
               gp_gp: wp.array2d(dtype=wp.float64),    # [ngp, dim]
               phi_gp: wp.array(dtype=wp.float64),     # [ngp]
               gphi_gp: wp.array2d(dtype=wp.float64),  # [ngp, dim]
               mu_gp: wp.array(dtype=wp.float64),      # [ngp]
               gmu_gp: wp.array2d(dtype=wp.float64),   # [ngp, dim]
               un_gp: wp.array2d(dtype=wp.float64),    # [ngp, dim]
               phin_gp: wp.array(dtype=wp.float64),    # [ngp]
               src_phi_gp: wp.array(dtype=wp.float64), # [ngp] CH source S(x,t)
               fbody_gp: wp.array2d(dtype=wp.float64), # [ngp, dim] momentum src
               ghat: wp.array(dtype=wp.float64),       # [dim]
               dt: wp.float64, Re: wp.float64, We: wp.float64,
               Cn: wp.float64, Pe: wp.float64,
               cw_inv: wp.float64, agg: wp.float64,
               grav_scale: wp.float64,
               gamma: wp.float64, beta: wp.float64,
               rho_h: wp.float64, rho_l: wp.float64,
               eta_h: wp.float64, eta_l: wp.float64,
               Ae: wp.array3d(dtype=wp.float64),
               be: wp.array2d(dtype=wp.float64)):
        e = wp.tid()
        he = h[e]
        jac = wp.pow(he * wp.float64(0.5), wp.float64(dim_pow))
        dscale = wp.float64(2.0) / he
        h2 = he * he
        h4 = h2 * h2

        a_rho = wp.float64(0.5) * (rho_h - rho_l)
        b_rho = wp.float64(0.5) * (rho_h + rho_l)
        a_eta = wp.float64(0.5) * (eta_h - eta_l)
        b_eta = wp.float64(0.5) * (eta_h + eta_l)
        rho_floor = wp.float64(1e-3) * rho_l
        eta_floor = wp.float64(1e-3) * eta_l

        for q in range(nqp):
            dJxW = wtab[q] * jac
            gp = e * nqp + q

            # ---- local rho, eta (mix_props, clamp) --------------------
            phiq = phi_gp[gp]
            rho_raw = a_rho * phiq + b_rho
            eta_raw = a_eta * phiq + b_eta
            rho = wp.max(rho_raw, rho_floor)
            eta = wp.max(eta_raw, eta_floor)
            # drho/dphi, deta/dphi: slope where unclamped, exactly 0 clamped
            drho = a_rho
            if rho_raw < rho_floor:
                drho = wp.float64(0.0)
            deta = a_eta
            if eta_raw < eta_floor:
                deta = wp.float64(0.0)

            # ---- tau_m (per GP, local rho/eta) + derivatives ----------
            umag2 = wp.float64(0.0)
            for d in range(dim):
                umag2 += u_gp[gp, d] * u_gp[gp, d]
            nu_loc = eta / (rho * Re)
            A_t = wp.float64(4.0) / (dt * dt)
            Bu_t = wp.float64(4.0) * umag2 / h2
            C_t = wp.float64(c2CI) * nu_loc * nu_loc / h4
            denom = wp.sqrt(A_t + Bu_t + C_t)
            tau = wp.float64(1.0) / (denom * rho)
            # dC/deta, dC/drho -> ddenom/dphi (via C only)
            dC_deta = wp.float64(2.0) * wp.float64(c2CI) * eta \
                / (h4 * rho * rho * Re * Re)
            dC_drho = wp.float64(-2.0) * wp.float64(c2CI) * eta * eta \
                / (h4 * rho * rho * rho * Re * Re)
            ddenom_dphi = (dC_deta * deta + dC_drho * drho) \
                / (wp.float64(2.0) * denom)
            dtau_dphi = -tau * (ddenom_dphi / denom + drho / rho)
            # dtau/du_k = -tau * (4 u_k / h^2) / denom^2

            # ---- momentum strong residual r_mom[dim] ------------------
            # r_mom = rho(u-un)/dt + rho ugradu + Jgradu + gradp
            #         - fcap - fgrav
            r_mom = wp.vector(length=dim, dtype=wp.float64)
            ugradu = wp.vector(length=dim, dtype=wp.float64)
            accel = wp.vector(length=dim, dtype=wp.float64)   # (u-un)/dt+ugradu
            for d in range(dim):
                ug = wp.float64(0.0)     # (u.grad)u_d = sum_s u_s du_d/dx_s
                Jg = wp.float64(0.0)     # (J.grad)u_d = sum_s J_s du_d/dx_s
                for s in range(dim):
                    ug += u_gp[gp, s] * gu_gp[gp, d, s]
                    if not cac:
                        Jg += (agg * gmu_gp[gp, s]) * gu_gp[gp, d, s]
                ugradu[d] = ug
                # CAC drops the potential-form capillary (mu grad phi) and the
                # AGG flux J; surface tension is the Galerkin-only Korteweg term
                # added in the residual-row loop below (r_mom carries neither).
                if cac:
                    fcap_d = wp.float64(0.0)
                else:
                    fcap_d = cw_inv * mu_gp[gp] * gphi_gp[gp, d]
                fgrav_d = rho * grav_scale * ghat[d]
                fmms_d = fbody_gp[gp, d]     # MMS/SP-1 momentum source
                accel[d] = (u_gp[gp, d] - un_gp[gp, d]) / dt + ug
                r_mom[d] = rho * (u_gp[gp, d] - un_gp[gp, d]) / dt \
                    + rho * ug + Jg + gp_gp[gp, d] - fcap_d - fgrav_d \
                    - fmms_d

            # div u
            divu = wp.float64(0.0)
            for d in range(dim):
                divu += gu_gp[gp, d, d]
            # u.grad phi
            ugradphi = wp.float64(0.0)
            for s in range(dim):
                ugradphi += u_gp[gp, s] * gphi_gp[gp, s]

            # ============ RESIDUAL ROWS (be = -R) ======================
            for a in range(nbf):
                Na = Ntab[q, a]
                # physical grad of test a
                # u.grad w_a  (SUPG test)
                ugw = wp.float64(0.0)
                for s in range(dim):
                    ugw += u_gp[gp, s] * (dNtab[q, a, s] * dscale)
                # --- momentum rows d ---
                # gphi . grad N_a (reused by the CAC Korteweg term)
                gphi_gNa = wp.float64(0.0)
                for s in range(dim):
                    gphi_gNa += gphi_gp[gp, s] * (dNtab[q, a, s] * dscale)
                for d in range(dim):
                    # galerkin body: rho(u-un)/dt + rho ugradu + Jgradu
                    #                - fcap - fgrav
                    if cac:
                        fcap_d = wp.float64(0.0)
                    else:
                        fcap_d = cw_inv * mu_gp[gp] * gphi_gp[gp, d]
                    fgrav_d = rho * grav_scale * ghat[d]
                    fmms_d = fbody_gp[gp, d]
                    Jg = wp.float64(0.0)
                    if not cac:
                        for s in range(dim):
                            Jg += (agg * gmu_gp[gp, s]) * gu_gp[gp, d, s]
                    mom_body = rho * (u_gp[gp, d] - un_gp[gp, d]) / dt \
                        + rho * ugradu[d] + Jg - fcap_d - fgrav_d - fmms_d
                    Rd = Na * mom_body
                    # CAC Korteweg surface tension (Galerkin only):
                    #   +(Cn/We) (grad phi)_d (grad phi . grad N_a)
                    if cac:
                        Rd += (Cn / We) * gphi_gp[gp, d] * gphi_gNa
                    # viscous (eta/Re) symgu[d,s] dN_a,s
                    visc = wp.float64(0.0)
                    for s in range(dim):
                        symds = gu_gp[gp, d, s] + gu_gp[gp, s, d]
                        visc += symds * (dNtab[q, a, s] * dscale)
                    Rd += (eta / Re) * visc
                    # pressure -div(w) p -> -(dN_a,d) p
                    Rd += -(dNtab[q, a, d] * dscale) * p_gp[gp]
                    # SUPG tau (u.grad w_a) r_mom_d
                    Rd += tau * ugw * r_mom[d]
                    wp.atomic_add(be, e, blk * a + d, -Rd * dJxW)
                # --- continuity row (field dim) ---
                Rp = Na * divu
                gNa_rmom = wp.float64(0.0)
                for s in range(dim):
                    gNa_rmom += (dNtab[q, a, s] * dscale) * r_mom[s]
                Rp += tau * gNa_rmom
                wp.atomic_add(be, e, blk * a + dim, -Rp * dJxW)
                # --- phi row (field dim+1), conservative advection ---
                # SP-1 deposition hook: subtract source S so mass grows by
                # INT S dV (pure RHS term; no Jacobian dependence on unknowns)
                ch_body = (phiq - phin_gp[gp]) / dt + ugradphi \
                    + phiq * divu - src_phi_gp[gp]
                Rphi = Na * ch_body
                if cac:
                    # CAC flux: +gamma Cn (gNa.gphi)
                    #           +gamma N_a [F'(phi)/Cn + beta sqrt(F(phi))]
                    gNa_gphi_p = wp.float64(0.0)
                    for s in range(dim):
                        gNa_gphi_p += (dNtab[q, a, s] * dscale) * gphi_gp[gp, s]
                    Fp = phiq * phiq * phiq - phiq
                    Fq = wp.float64(0.25) * (phiq * phiq - wp.float64(1.0)) \
                        * (phiq * phiq - wp.float64(1.0))
                    sqrtF = wp.sqrt(wp.max(Fq, wp.float64(0.0)))
                    Rphi += gamma * Cn * gNa_gphi_p \
                        + gamma * Na * (Fp / Cn + beta * sqrtF)
                else:
                    gNa_gmu = wp.float64(0.0)
                    for s in range(dim):
                        gNa_gmu += (dNtab[q, a, s] * dscale) * gmu_gp[gp, s]
                    Rphi += (wp.float64(1.0) / Pe) * gNa_gmu
                wp.atomic_add(be, e, blk * a + dim + 1, -Rphi * dJxW)
                # --- mu row (field dim+2) ---
                if cac:
                    # trivial mu = 0 row: R^mu = INT N_a mu
                    Rmu = Na * mu_gp[gp]
                else:
                    fp = phiq * phiq * phiq - phiq
                    Rmu = Na * (mu_gp[gp] - fp)
                    gNa_gphi = wp.float64(0.0)
                    for s in range(dim):
                        gNa_gphi += (dNtab[q, a, s] * dscale) * gphi_gp[gp, s]
                    Rmu += -(Cn * Cn) * gNa_gphi
                wp.atomic_add(be, e, blk * a + dim + 2, -Rmu * dJxW)

            # ============ JACOBIAN BLOCKS (Ae) =========================
            fpp = wp.float64(3.0) * phiq * phiq - wp.float64(1.0)
            for a in range(nbf):
                Na = Ntab[q, a]
                ugw = wp.float64(0.0)
                for s in range(dim):
                    ugw += u_gp[gp, s] * (dNtab[q, a, s] * dscale)
                for b in range(nbf):
                    Nb = Ntab[q, b]
                    # u.grad N_b, J.grad N_b, gNa.gNb
                    uGNb = wp.float64(0.0)
                    JGNb = wp.float64(0.0)
                    gNagNb = wp.float64(0.0)
                    for s in range(dim):
                        dNb_s = dNtab[q, b, s] * dscale
                        dNa_s = dNtab[q, a, s] * dscale
                        uGNb += u_gp[gp, s] * dNb_s
                        if not cac:
                            JGNb += (agg * gmu_gp[gp, s]) * dNb_s
                        gNagNb += dNa_s * dNb_s
                    # c1 = (rho/dt) N_b + rho u.gradN_b + J.gradN_b (JGNb=0 CAC)
                    c1 = rho * (Nb / dt) + rho * uGNb + JGNb

                    # ---- momentum rows d ----
                    for d in range(dim):
                        rowd = blk * a + d
                        dNa_d = dNtab[q, a, d] * dscale
                        dNb_d = dNtab[q, b, d] * dscale
                        # (u col jc)
                        for jc in range(dim):
                            dNa_jc = dNtab[q, a, jc] * dscale
                            dNb_jc = dNtab[q, b, jc] * dscale
                            val = wp.float64(0.0)
                            # galerkin diagonal (transient+conv-second)
                            if jc == d:
                                val += Na * c1
                            # galerkin conv-first: rho N_b gu[d,jc]
                            val += Na * (rho * Nb * gu_gp[gp, d, jc])
                            # viscous: (1/Re) eta (delta gNa.gNb + gNa_jc gNb_d)
                            if jc == d:
                                val += (wp.float64(1.0) / Re) * eta * gNagNb
                            val += (wp.float64(1.0) / Re) * eta * dNa_jc * dNb_d
                            # SUPG (ii): tau ugw * [diag c1 + conv-first]
                            supg = wp.float64(0.0)
                            if jc == d:
                                supg += c1
                            supg += rho * Nb * gu_gp[gp, d, jc]
                            val += tau * ugw * supg
                            # SUPG (i): test depends on u_jc: N_b dN_a,jc
                            val += tau * (Nb * dNa_jc) * r_mom[d]
                            # SUPG (iii): dtau/du_jc N_b, weight ugw r_mom_d
                            dtau_du_jc = -tau * (wp.float64(4.0)
                                                 * u_gp[gp, jc] / h2) \
                                / (denom * denom)
                            val += ugw * r_mom[d] * dtau_du_jc * Nb
                            wp.atomic_add(Ae, e, rowd, blk * b + jc, val * dJxW)
                        # (p col dim)
                        # galerkin pressure -(dN_a,d) N_b ; SUPG tau ugw dNb_d
                        valp = -dNa_d * Nb + tau * ugw * dNb_d
                        wp.atomic_add(Ae, e, rowd, blk * b + dim, valp * dJxW)
                        # (phi col dim+1)
                        # galerkin: drho N_b accel_d - cap_cw mu dNb_d
                        #           - grav_scale ghat_d drho N_b
                        # viscous eta(phi): (1/Re) deta N_b symgu[d,s] dNa_s
                        if cac:
                            cap_mu = wp.float64(0.0)
                        else:
                            cap_mu = cw_inv * mu_gp[gp]
                        vphi = Na * (drho * Nb * accel[d]
                                     - cap_mu * dNb_d
                                     - grav_scale * ghat[d] * drho * Nb)
                        # CAC Korteweg d/dphi: (Cn/We)[dNb_d (gphi.gNa)
                        #                              + gphi_d (gNa.gNb)]
                        if cac:
                            gphi_gNa_j = wp.float64(0.0)
                            for s in range(dim):
                                gphi_gNa_j += gphi_gp[gp, s] \
                                    * (dNtab[q, a, s] * dscale)
                            vphi += (Cn / We) * (dNb_d * gphi_gNa_j
                                                 + gphi_gp[gp, d] * gNagNb)
                        visc_phi = wp.float64(0.0)
                        for s in range(dim):
                            symds = gu_gp[gp, d, s] + gu_gp[gp, s, d]
                            visc_phi += symds * (dNtab[q, a, s] * dscale)
                        vphi += (wp.float64(1.0) / Re) * deta * Nb * visc_phi
                        # SUPG: tau ugw * dr_mom/dphi
                        dr_phi = drho * Nb * accel[d] \
                            - cap_mu * dNb_d \
                            - grav_scale * ghat[d] * drho * Nb
                        vphi += tau * ugw * dr_phi
                        # SUPG tau-derivative: dtau/dphi N_b ugw r_mom_d
                        vphi += ugw * r_mom[d] * dtau_dphi * Nb
                        wp.atomic_add(Ae, e, rowd, blk * b + dim + 1,
                                      vphi * dJxW)
                        # (mu col dim+2) — zero in CAC (no mu-coupling)
                        if cac:
                            vmu = wp.float64(0.0)
                        else:
                            dNb_gu_d = wp.float64(0.0)
                            for s in range(dim):
                                dNb_gu_d += (dNtab[q, b, s] * dscale) \
                                    * gu_gp[gp, d, s]
                            vmu = Na * (-cw_inv * Nb * gphi_gp[gp, d]
                                        + agg * dNb_gu_d)
                            # SUPG: tau ugw * dr_mom/dmu
                            dr_mu = -cw_inv * Nb * gphi_gp[gp, d] \
                                + agg * dNb_gu_d
                            vmu += tau * ugw * dr_mu
                        wp.atomic_add(Ae, e, rowd, blk * b + dim + 2,
                                      vmu * dJxW)

                    # ---- continuity row (dim) ----
                    rowc = blk * a + dim
                    # d/du_jc
                    for jc in range(dim):
                        dNa_jc = dNtab[q, a, jc] * dscale
                        dNb_jc = dNtab[q, b, jc] * dscale
                        # galerkin: Na dN_b,jc
                        val = Na * dNb_jc
                        # PSPG diag: tau gNa_jc * c1
                        val += tau * dNa_jc * c1
                        # PSPG conv-first: sum_s tau gNa_s rho N_b gu[s,jc]
                        conv = wp.float64(0.0)
                        for s in range(dim):
                            conv += (dNtab[q, a, s] * dscale) \
                                * (rho * Nb * gu_gp[gp, s, jc])
                        val += tau * conv
                        # PSPG tau-deriv: dtau/du_jc N_b (gNa . r_mom)
                        gNa_rmom = wp.float64(0.0)
                        for s in range(dim):
                            gNa_rmom += (dNtab[q, a, s] * dscale) * r_mom[s]
                        dtau_du_jc = -tau * (wp.float64(4.0)
                                             * u_gp[gp, jc] / h2) \
                            / (denom * denom)
                        val += gNa_rmom * dtau_du_jc * Nb
                        wp.atomic_add(Ae, e, rowc, blk * b + jc, val * dJxW)
                    # d/dp: tau gNa . gNb
                    wp.atomic_add(Ae, e, rowc, blk * b + dim,
                                  tau * gNagNb * dJxW)
                    # d/dphi: tau gNa . dr_mom/dphi + tau-deriv
                    # (CAC: r_mom has no capillary term -> cap contribution 0)
                    cphi = wp.float64(0.0)
                    for s in range(dim):
                        if cac:
                            cap_s = wp.float64(0.0)
                        else:
                            cap_s = cw_inv * mu_gp[gp] \
                                * (dNtab[q, b, s] * dscale)
                        dr_s = drho * Nb * accel[s] - cap_s \
                            - grav_scale * ghat[s] * drho * Nb
                        cphi += (dNtab[q, a, s] * dscale) * dr_s
                    gNa_rmom = wp.float64(0.0)
                    for s in range(dim):
                        gNa_rmom += (dNtab[q, a, s] * dscale) * r_mom[s]
                    cphi = tau * cphi + gNa_rmom * dtau_dphi * Nb
                    wp.atomic_add(Ae, e, rowc, blk * b + dim + 1, cphi * dJxW)
                    # d/dmu: tau gNa . dr_mom/dmu (0 in CAC — no mu-coupling)
                    if cac:
                        cmu = wp.float64(0.0)
                    else:
                        cmu = wp.float64(0.0)
                        for s in range(dim):
                            dNb_gu_s = wp.float64(0.0)
                            for ss in range(dim):
                                dNb_gu_s += (dNtab[q, b, ss] * dscale) \
                                    * gu_gp[gp, s, ss]
                            dr_s = -cw_inv * gphi_gp[gp, s] * Nb \
                                + agg * dNb_gu_s
                            cmu += (dNtab[q, a, s] * dscale) * dr_s
                    wp.atomic_add(Ae, e, rowc, blk * b + dim + 2,
                                  tau * cmu * dJxW)

                    # ---- phi row (dim+1) ----
                    rowp = blk * a + dim + 1
                    for jc in range(dim):
                        dNb_jc = dNtab[q, b, jc] * dscale
                        # u.grad phi -> N_b dphi/dx_jc ; phi div u -> phi dNb_jc
                        val = Na * (Nb * gphi_gp[gp, jc] + phiq * dNb_jc)
                        wp.atomic_add(Ae, e, rowp, blk * b + jc, val * dJxW)
                    if cac:
                        # d/dphi: (1/dt)N_b + u.gradN_b + divu N_b
                        #   + gamma Cn (gNa.gNb)
                        #   + gamma N_a [F''/Cn + beta F'/(2 sqrt F)] N_b
                        Fp2 = phiq * phiq * phiq - phiq
                        Fq2 = wp.float64(0.25) \
                            * (phiq * phiq - wp.float64(1.0)) \
                            * (phiq * phiq - wp.float64(1.0))
                        sqrtF2 = wp.sqrt(wp.max(Fq2, wp.float64(0.0)))
                        dsqrtF = wp.float64(0.0)
                        if sqrtF2 > wp.float64(1e-12):
                            dsqrtF = Fp2 / (wp.float64(2.0) * sqrtF2)
                        cpp = Na * (Nb / dt + uGNb + divu * Nb) \
                            + gamma * Cn * gNagNb \
                            + gamma * Na * (fpp / Cn + beta * dsqrtF) * Nb
                        wp.atomic_add(Ae, e, rowp, blk * b + dim + 1,
                                      cpp * dJxW)
                        # no d/dmu coupling in CAC
                    else:
                        # d/dphi: (1/dt)N_b + u.gradN_b + divu N_b
                        cpp = Na * (Nb / dt + uGNb + divu * Nb)
                        wp.atomic_add(Ae, e, rowp, blk * b + dim + 1,
                                      cpp * dJxW)
                        # d/dmu: (1/Pe) gNa.gNb
                        wp.atomic_add(Ae, e, rowp, blk * b + dim + 2,
                                      (wp.float64(1.0) / Pe) * gNagNb * dJxW)

                    # ---- mu row (dim+2) ----
                    rowm = blk * a + dim + 2
                    if cac:
                        # trivial mu = 0 row: d/dmu = Na Nb
                        wp.atomic_add(Ae, e, rowm, blk * b + dim + 2,
                                      Na * Nb * dJxW)
                    else:
                        # d/dphi: -Na f'' Nb - Cn^2 gNa.gNb
                        cmp = -Na * fpp * Nb - (Cn * Cn) * gNagNb
                        wp.atomic_add(Ae, e, rowm, blk * b + dim + 1,
                                      cmp * dJxW)
                        # d/dmu: Na Nb
                        wp.atomic_add(Ae, e, rowm, blk * b + dim + 2,
                                      Na * Nb * dJxW)

    _kernel_cache[key] = chns_k
    return chns_k
