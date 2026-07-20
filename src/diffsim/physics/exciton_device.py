"""Task #35: XDD device assembly + device closures (the E5→100× path).

Measured motivation (R0 findings §5): after cuDSS the 2-D production step is
14.12 s with ~69% on the HOST — `_reduce_and_eliminate` (scipy `tolil` over the
5×5 bmat, 12.4 s/37%) and `jacobian_full` (numpy `einsum` closure blocks,
10.5 s/32%).  This module moves BOTH named stages onto the device:

  * ONE monolithic warp element kernel (`make_xdd_jacres_kernel`) evaluates the
    5-field GP state (values, gradients, Laplacians interpolated in-kernel from
    the nodal state), the A3 closures (Onsager–Braun k_diss with the 5th-order
    b-series and the EXACT dk/d|∇φ̂|; Langevin R = γ̂ n̂ p̂) and every Jacobian
    block + the residual, producing the element pair (Ae [ne,nl,nl],
    re [ne,nl]), nl = 5·nbf, dof-major (node·5 + field).
  * The scatter rides `DeviceNSAssembler` (M1d machinery) in node-pattern mode
    with the 5×5 XDD `blockmask` = kron(G, mask) — the four structurally-zero
    couplings (φ̂↔X̂ rows' Z blocks and X̂_D↔X̂_A) never exist in the CSR.
    `index_width` per #33 (mixed-width CSR honored).
  * Dirichlet rows use the assembler's device strong-row plan (replaces the
    host LIL surgery); the B5 log-carrier chain rule is a per-nnz column-scale
    kernel (replaces the host `A @ diags(scale)`).

CLOSURE PARITY CONTRACT (G1): the dist-dependent, field-INDEPENDENT closure
factors are precomputed host-side with numpy expressions copied VERBATIM from
`exciton_closures.py` (same grouping, same constants), so the in-kernel
field-dependent completion (Φ(b) series, dΦ/db, γ̂n̂p̂) reproduces the host
values to the reassociation-ULP level.  The A3 derivative contract
`closure(fields, aux) -> (value, d_value_d_fields)` is unchanged — the host
closures stay the parity reference.

PHYSICS CONTRACT: every term mirrors `exciton_system.py`'s host blocks
term-by-term (conservative signed drift, SUPG with frozen τ, exact
dissociation-field coupling, BDF history in the SUPG strong residual).  The
host path remains available (`XDDSystem(assembly="host")`) as the parity
reference; parity is gated in tests/test_xdd_device.py.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
import warp as wp

from ..assembly.device_assembly import DeviceNSAssembler
from ..assembly.operators import _kernel_cache
from ..physics.vms import tau_m_metric
from diffsim.xdd.params import _KB, _Q, _EPS0
from diffsim.xdd.morphology import tanh_mask, interface_mask

wp.set_module_options({"enable_backward": False})

# node-major field order (mirrors exciton_system; kept literal to avoid a
# circular import — exciton_system imports this module lazily)
IPHI, IN, IP, IXD, IXA = 0, 1, 2, 3, 4
NDOF = 5

# The 5×5 block-coupling mask (True = live block).  Mirrors the host
# `jacobian_full` bmat layout: the φ̂ row has no X̂ coupling, the X̂ rows do not
# couple to EACH OTHER (their φ̂/n̂/p̂ couplings are live).  The diagonal is
# forced live by the assembler regardless.
XDD_BLOCKMASK = np.array([
    # φ  n  p  Xd Xa
    [1, 1, 1, 0, 0],   # φ̂ :  Jφφ, Jφn, Jφp
    [1, 1, 1, 1, 1],   # n̂ :  full row
    [1, 1, 1, 1, 1],   # p̂ :  full row
    [1, 1, 1, 1, 0],   # X̂_D: Jxdφ, Jxdn, Jxdp, Jxdxd
    [1, 1, 1, 0, 1],   # X̂_A: Jxaφ, Jxan, Jxap, Jxaxa
], dtype=bool)


# ══════════════════════════════════════════════════════════════════════════════
# Host-once closure factor precomputes (dist-dependent, field-INDEPENDENT).
# numpy expressions copied VERBATIM from exciton_closures.py so the factors are
# bit-equal to what the host closures compute internally.
# ══════════════════════════════════════════════════════════════════════════════

def onsager_device_factors(onsager, dist_gp):
    """Field-independent Onsager–Braun factors at GPs, per bin.

    Host reference (OnsagerBraunDissociation.__call__):
        k_hat_d = ((prefactor*exp_EB)*Phi(b))*t0*mask*sd      (left-assoc)
        b       = (Q³*E)/(8π·eps·kT²),  E = (|∇φ̂|·phi0)/x0
        dk_d    = ((((c1*dPhi)*db_dE)*dE_dgrad)*t0*mask)*sd
    Returns (arrays, scalars):
        arrays  {pv: (c1, den, mask)}  c1 = prefactor*exp_EB,
                                       den = 8π·eps·kT², mask = interface mask
        scalars dict(t0, sd, sa, q3, phi0, x0, dEdg)
    onsager=None → zero factors (kd = ka = dkd = dka = 0), den = 1 (no div-0).
    """
    if onsager is None:
        arrays = {pv: (np.zeros_like(d, dtype=np.float64),
                       np.ones_like(d, dtype=np.float64),
                       np.zeros_like(d, dtype=np.float64))
                  for pv, d in dist_gp.items()}
        scalars = dict(t0=0.0, sd=0.0, sa=0.0, q3=0.0,
                       phi0=1.0, x0=1.0, dEdg=0.0)
        return arrays, scalars
    p = onsager.params
    s = p.scales()
    phi0 = s.phi0
    x0 = s.x0
    t0 = s.t0
    kT = _KB * p.T
    prefactor = (3.0 * s.gamma0 / (4.0 * math.pi * p.a**3))
    arrays = {}
    for pv, dist in dist_gp.items():
        # verbatim host expressions (grouping preserved):
        eps_r = p.eps_D + (p.eps_A - p.eps_D) * tanh_mask(dist, onsager.width)
        eps = _EPS0 * eps_r
        E_B = _Q**2 / (4.0 * math.pi * eps * p.a)
        exp_EB = np.exp(-E_B / kT)
        c1 = prefactor * exp_EB                      # (prefactor*exp_EB)
        den = 8.0 * math.pi * eps * kT**2            # b/db_dE denominator
        mask = interface_mask(dist, p.interface_thk / 2.0)
        arrays[pv] = (np.ascontiguousarray(c1, np.float64),
                      np.ascontiguousarray(den, np.float64),
                      np.ascontiguousarray(mask, np.float64))
    scalars = dict(t0=float(t0),
                   sd=float(p.ex_diss_d_scaling),
                   sa=float(p.ex_diss_a_scaling),
                   q3=float(_Q**3), phi0=float(phi0), x0=float(x0),
                   dEdg=float(phi0 / x0))
    return arrays, scalars


def langevin_device_factor(langevin, dist_gp):
    """γ̂(x) at GPs per bin (the field-independent Langevin factor).

    Host reference (LangevinRecombination.__call__): R̂ = (gh·n̂)·p̂ with gh the
    (possibly interface-modulated) γ̂.  langevin=None / spatial="disabled" → 0.
    """
    if langevin is None:
        return {pv: np.zeros_like(d, dtype=np.float64)
                for pv, d in dist_gp.items()}
    p = langevin.params
    s = p.scales()
    eps_bar = _EPS0 * (p.eps_A + p.eps_D) / 2.0
    if langevin.strategy == "sum":
        gamma = s.gamma0
    elif langevin.strategy == "min":
        gamma = _Q * min(p.mu_n, p.mu_p) / eps_bar
    elif langevin.strategy == "image_force":
        gamma = abs((p.eps_D - p.eps_A) / (p.eps_D + p.eps_A)) \
            * _Q * p.mu_p / (_EPS0 * p.eps_D)
    else:
        raise ValueError(f"Unknown Langevin strategy: {langevin.strategy!r}")
    gamma_hat = langevin.zeta * gamma * s.C0**2 / s.U0
    out = {}
    for pv, dist in dist_gp.items():
        if langevin.spatial == "uniform":
            gh = np.full(len(dist), gamma_hat, np.float64)
        elif langevin.spatial == "interface":
            gh = np.ascontiguousarray(
                gamma_hat * interface_mask(dist, langevin.width), np.float64)
        elif langevin.spatial == "disabled":
            gh = np.zeros(len(dist), np.float64)
        else:
            raise ValueError(f"Unknown Langevin spatial: {langevin.spatial!r}")
        out[pv] = gh
    return out


# ══════════════════════════════════════════════════════════════════════════════
# The monolithic 5-field element Jacobian + residual kernel
# ══════════════════════════════════════════════════════════════════════════════

def make_xdd_jacres_kernel(nbf: int, nqp: int, dim: int):
    """Element Jacobian Ae [ne, 5·nbf, 5·nbf] + residual re [ne, 5·nbf] for the
    monolithic XDD Newton system, dof-major local layout (node·5 + field).

    One thread per element; per GP the kernel interpolates the 5 nodal fields
    (values, gradients, n̂/p̂ Laplacians), completes the A3 closures from the
    precomputed dist factors, and accumulates EVERY host Jacobian block +
    the residual with the SAME term expressions as exciton_system.py's host
    path (see tests/test_xdd_device.py G1 parity).  supg=0 → the τ-terms are
    exactly zero (τ = supg·τ_M), matching the host's skipped blocks.

    Cache key: ("xdd_jacres", nbf, nqp, dim).
    """
    key = ("xdd_jacres", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    F = wp.float64
    dim_f = float(dim)

    @wp.kernel(module="unique", enable_backward=False,
               module_options={"max_unroll": 0})
    def xdd_jacres(
        conn:    wp.array2d(dtype=wp.int32),
        h:       wp.array(dtype=wp.float64),
        Ntab:    wp.array2d(dtype=wp.float64),
        dNtab:   wp.array3d(dtype=wp.float64),
        lapNtab: wp.array2d(dtype=wp.float64),
        wtab:    wp.array(dtype=wp.float64),
        # full nodal state (5 fields)
        phi:  wp.array(dtype=wp.float64),
        nn:   wp.array(dtype=wp.float64),
        pp:   wp.array(dtype=wp.float64),
        xdn:  wp.array(dtype=wp.float64),
        xan:  wp.array(dtype=wp.float64),
        # coefficient GP fields [ne*nqp]
        eps:   wp.array(dtype=wp.float64),
        mu_n:  wp.array(dtype=wp.float64),
        mu_p:  wp.array(dtype=wp.float64),
        mu_xd: wp.array(dtype=wp.float64),
        mu_xa: wp.array(dtype=wp.float64),
        # closure factor GP fields [ne*nqp]
        gh:   wp.array(dtype=wp.float64),   # Langevin γ̂(x)
        c1g:  wp.array(dtype=wp.float64),   # Onsager prefactor·exp(−E_B/kT)
        deng: wp.array(dtype=wp.float64),   # Onsager 8π·eps·kT²
        mskg: wp.array(dtype=wp.float64),   # Onsager interface mask
        # per-step GP sources [ne*nqp]
        hn:  wp.array(dtype=wp.float64),    # BDF history σ_BDF·n̂ⁿ at GPs
        hp:  wp.array(dtype=wp.float64),
        hxd: wp.array(dtype=wp.float64),
        hxa: wp.array(dtype=wp.float64),
        gd:  wp.array(dtype=wp.float64),    # generation Ĝ_D
        ga:  wp.array(dtype=wp.float64),    # generation Ĝ_A
        # scalars
        lam2:  wp.float64,
        sigma: wp.float64,
        s2t:   wp.float64,
        supg:  wp.float64,
        tinvd: wp.float64,
        tinva: wp.float64,
        t0k:   wp.float64,   # Onsager t0
        sdd:   wp.float64,   # ex_diss_d_scaling
        saa:   wp.float64,   # ex_diss_a_scaling
        q3:    wp.float64,   # Q³
        phi0:  wp.float64,
        x0:    wp.float64,
        dEdg:  wp.float64,   # phi0/x0
        # outputs
        Ae: wp.array3d(dtype=wp.float64),   # [ne, 5·nbf, 5·nbf]
        re: wp.array2d(dtype=wp.float64),   # [ne, 5·nbf]
    ):
        e = wp.tid()
        he = h[e]
        half = he * F(0.5)
        jac = F(1.0)
        for _ in range(dim):
            jac = jac * half
        dscale = F(2.0) / he

        for q in range(nqp):
            dJxW = wtab[q] * jac
            gp = e * nqp + q

            # ---- interpolate the 5 fields at this GP -------------------
            phi_q = F(0.0)
            n_q = F(0.0)
            p_q = F(0.0)
            xd_q = F(0.0)
            xa_q = F(0.0)
            lapn_q = F(0.0)
            lapp_q = F(0.0)
            gphi = wp.vec3d()
            gn = wp.vec3d()
            gpv = wp.vec3d()
            gxd = wp.vec3d()
            gxa = wp.vec3d()
            for c in range(nbf):
                nid = conn[e, c]
                phv = phi[nid]
                nv = nn[nid]
                pv_ = pp[nid]
                xdv = xdn[nid]
                xav = xan[nid]
                Nc = Ntab[q, c]
                phi_q += Nc * phv
                n_q += Nc * nv
                p_q += Nc * pv_
                xd_q += Nc * xdv
                xa_q += Nc * xav
                lapc = lapNtab[q, c]
                lapn_q += lapc * nv
                lapp_q += lapc * pv_
                for d in range(dim):
                    dNc = dNtab[q, c, d]
                    gphi[d] = gphi[d] + dNc * phv
                    gn[d] = gn[d] + dNc * nv
                    gpv[d] = gpv[d] + dNc * pv_
                    gxd[d] = gxd[d] + dNc * xdv
                    gxa[d] = gxa[d] + dNc * xav
            # physical scaling AFTER summation (host _gp_grad/_gp_lap order)
            lapn_q = lapn_q * dscale * dscale
            lapp_q = lapp_q * dscale * dscale
            for d in range(dim):
                gphi[d] = gphi[d] * dscale
                gn[d] = gn[d] * dscale
                gpv[d] = gpv[d] * dscale
                gxd[d] = gxd[d] * dscale
                gxa[d] = gxa[d] * dscale

            # ---- closures (device port of the A3 host arithmetic) ------
            gmag2 = F(0.0)
            for d in range(dim):
                gmag2 += gphi[d] * gphi[d]
            gmag = wp.sqrt(gmag2)
            # Onsager–Braun: b, Φ(b), dΦ/db (host groupings preserved)
            Efld = (gmag * phi0) / x0
            b = (q3 * Efld) / deng[gp]
            b2 = b * b
            b3 = b2 * b
            b4 = b3 * b
            b5 = b4 * b
            Phi = (F(1.0) + b + b2 / F(3.0) + b3 / F(18.0)
                   + b4 / F(180.0) + b5 / F(2700.0))
            dPhi = (F(1.0) + F(2.0) * b / F(3.0) + b2 / F(6.0)
                    + b3 / F(45.0) + b4 / F(540.0))
            khb = (c1g[gp] * Phi) * t0k          # k_hat_base
            kdm = khb * mskg[gp]
            kd = kdm * sdd
            ka = kdm * saa
            dkb = (((c1g[gp] * dPhi) * (q3 / deng[gp])) * dEdg) * t0k \
                * mskg[gp]                        # dk_dgrad_base
            dkd = dkb * sdd
            dka = dkb * saa
            # Langevin
            ghq = gh[gp]
            Rr = ghq * n_q * p_q
            dRn = ghq * p_q
            dRp = ghq * n_q

            # ---- derived GP quantities ---------------------------------
            eps_q = eps[gp]
            mu_nq = mu_n[gp]
            mu_pq = mu_p[gp]
            Dhat = kd * xd_q + ka * xa_q
            s_carr = Dhat - Rr
            fn = s_carr + hn[gp]
            fp = s_carr + hp[gp]
            fxd = gd[gp] + Rr + hxd[gp]
            fxa = ga[gp] + Rr + hxa[gp]
            sig_d = sigma + tinvd + kd
            sig_a = sigma + tinva + ka
            # dissociation-field coupling weights (host _safe_div guard)
            dDdmag = dkd * xd_q + dka * xa_q
            cdiss = F(0.0)
            cxd = F(0.0)
            cxa = F(0.0)
            if wp.abs(gmag) > F(1e-30):
                cdiss = -(dDdmag / gmag)
                cxd = dkd * xd_q / gmag
                cxa = dka * xa_q / gmag
            # signed drift weights aq = sign·μ̂·∇φ̂ and SUPG τ
            aqn = wp.vec3d()
            aqp = wp.vec3d()
            for d in range(dim):
                aqn[d] = -mu_nq * gphi[d]
                aqp[d] = mu_pq * gphi[d]
            amn = F(0.0)
            amp = F(0.0)
            for d in range(dim):
                amn += aqn[d] * aqn[d]
                amp += aqp[d] * aqp[d]
            amn = wp.sqrt(amn)
            amp = wp.sqrt(amp)
            taun = supg * tau_m_metric(amn, he, mu_nq, s2t, F(dim_f))
            taup = supg * tau_m_metric(amp, he, mu_pq, s2t, F(dim_f))
            # conservative Galerkin drift-density weights (∂carrier/∂φ̂)
            cdrift_n = (-mu_nq) * n_q
            cdrift_p = mu_pq * p_q
            nsmu_n = mu_nq       # −sign·μ̂  (electrons: +μ̂_n)
            nsmu_p = -mu_pq      # −sign·μ̂  (holes:    −μ̂_p)
            # SUPG strong residuals (frozen τ; U = −aq)
            Ugn = F(0.0)
            Ugp = F(0.0)
            for d in range(dim):
                Ugn += (-aqn[d]) * gn[d]
                Ugp += (-aqp[d]) * gpv[d]
            res_op_n = sigma * n_q + Ugn - mu_nq * lapn_q
            res_op_p = sigma * p_q + Ugp - mu_pq * lapp_q
            resn = res_op_n - fn
            resp = res_op_p - fp

            # ---- assemble local rows -----------------------------------
            for a in range(nbf):
                Na = Ntab[q, a]
                gNa = wp.vec3d()
                for d in range(dim):
                    gNa[d] = dNtab[q, a, d] * dscale
                agNa_n = F(0.0)
                agNa_p = F(0.0)
                gphigNa = F(0.0)
                gngNa = F(0.0)
                gpgNa = F(0.0)
                gxdgNa = F(0.0)
                gxagNa = F(0.0)
                for d in range(dim):
                    agNa_n += aqn[d] * gNa[d]
                    agNa_p += aqp[d] * gNa[d]
                    gphigNa += gphi[d] * gNa[d]
                    gngNa += gn[d] * gNa[d]
                    gpgNa += gpv[d] * gNa[d]
                    gxdgNa += gxd[d] * gNa[d]
                    gxagNa += gxa[d] * gNa[d]
                UgNa_n = -agNa_n
                UgNa_p = -agNa_p
                ra = a * 5

                # residual (operator-on-state − SUPG-consistent load)
                wp.atomic_add(re, e, ra + 0,
                              (lam2 * eps_q * gphigNa
                               - (p_q - n_q) * Na) * dJxW)
                wp.atomic_add(
                    re, e, ra + 1,
                    ((sigma * Na * n_q + agNa_n * n_q + mu_nq * gngNa
                      + taun * UgNa_n * res_op_n)
                     - (Na + taun * UgNa_n) * fn) * dJxW)
                wp.atomic_add(
                    re, e, ra + 2,
                    ((sigma * Na * p_q + agNa_p * p_q + mu_pq * gpgNa
                      + taup * UgNa_p * res_op_p)
                     - (Na + taup * UgNa_p) * fp) * dJxW)
                wp.atomic_add(
                    re, e, ra + 3,
                    ((sig_d * Na * xd_q + mu_xd[gp] * gxdgNa)
                     - Na * fxd) * dJxW)
                wp.atomic_add(
                    re, e, ra + 4,
                    ((sig_a * Na * xa_q + mu_xa[gp] * gxagNa)
                     - Na * fxa) * dJxW)

                for bb in range(nbf):
                    Nb = Ntab[q, bb]
                    gNagNb = F(0.0)
                    agNb_n = F(0.0)
                    agNb_p = F(0.0)
                    gphigNb = F(0.0)
                    gngNb = F(0.0)
                    gpgNb = F(0.0)
                    for d in range(dim):
                        gNbd = dNtab[q, bb, d] * dscale
                        gNagNb += gNa[d] * gNbd
                        agNb_n += aqn[d] * gNbd
                        agNb_p += aqp[d] * gNbd
                        gphigNb += gphi[d] * gNbd
                        gngNb += gn[d] * gNbd
                        gpgNb += gpv[d] * gNbd
                    lapNb = lapNtab[q, bb] * dscale * dscale
                    cb = bb * 5

                    # φ̂ row: λ²ε̂ K | +M | −M
                    wp.atomic_add(Ae, e, ra + 0, cb + 0,
                                  lam2 * eps_q * gNagNb * dJxW)
                    mnb = Na * Nb * dJxW
                    wp.atomic_add(Ae, e, ra + 0, cb + 1, mnb)
                    wp.atomic_add(Ae, e, ra + 0, cb + 2, -mnb)

                    # n̂ row
                    #   (n,φ): conservative drift + SUPG-advection coupling
                    #          + dissociation coupling (Galerkin + SUPG)
                    v = (cdrift_n * gNagNb) * dJxW
                    v += ((taun * nsmu_n)
                          * (resn * gNagNb + UgNa_n * gngNb)) * dJxW
                    v += (cdiss * Na * gphigNb) * dJxW
                    v += ((cdiss * taun) * UgNa_n * gphigNb) * dJxW
                    wp.atomic_add(Ae, e, ra + 1, cb + 0, v)
                    #   (n,n): carrier operator + ∂R̂/∂n̂ (SUPG-augmented)
                    resu_n = sigma * Nb + (-agNb_n) - mu_nq * lapNb
                    v = (sigma * Na * Nb + agNa_n * Nb + mu_nq * gNagNb
                         + taun * UgNa_n * resu_n) * dJxW
                    v += (dRn * Na * Nb
                          + (taun * dRn) * UgNa_n * Nb) * dJxW
                    wp.atomic_add(Ae, e, ra + 1, cb + 1, v)
                    #   (n,p) / (n,Xd) / (n,Xa): SUPG-augmented source mass
                    wp.atomic_add(Ae, e, ra + 1, cb + 2,
                                  (dRp * Na * Nb
                                   + (taun * dRp) * UgNa_n * Nb) * dJxW)
                    wp.atomic_add(Ae, e, ra + 1, cb + 3,
                                  ((-kd) * Na * Nb
                                   + (taun * (-kd)) * UgNa_n * Nb) * dJxW)
                    wp.atomic_add(Ae, e, ra + 1, cb + 4,
                                  ((-ka) * Na * Nb
                                   + (taun * (-ka)) * UgNa_n * Nb) * dJxW)

                    # p̂ row (mirror)
                    v = (cdrift_p * gNagNb) * dJxW
                    v += ((taup * nsmu_p)
                          * (resp * gNagNb + UgNa_p * gpgNb)) * dJxW
                    v += (cdiss * Na * gphigNb) * dJxW
                    v += ((cdiss * taup) * UgNa_p * gphigNb) * dJxW
                    wp.atomic_add(Ae, e, ra + 2, cb + 0, v)
                    wp.atomic_add(Ae, e, ra + 2, cb + 1,
                                  (dRn * Na * Nb
                                   + (taup * dRn) * UgNa_p * Nb) * dJxW)
                    resu_p = sigma * Nb + (-agNb_p) - mu_pq * lapNb
                    v = (sigma * Na * Nb + agNa_p * Nb + mu_pq * gNagNb
                         + taup * UgNa_p * resu_p) * dJxW
                    v += (dRp * Na * Nb
                          + (taup * dRp) * UgNa_p * Nb) * dJxW
                    wp.atomic_add(Ae, e, ra + 2, cb + 2, v)
                    wp.atomic_add(Ae, e, ra + 2, cb + 3,
                                  ((-kd) * Na * Nb
                                   + (taup * (-kd)) * UgNa_p * Nb) * dJxW)
                    wp.atomic_add(Ae, e, ra + 2, cb + 4,
                                  ((-ka) * Na * Nb
                                   + (taup * (-ka)) * UgNa_p * Nb) * dJxW)

                    # X̂_D row
                    wp.atomic_add(Ae, e, ra + 3, cb + 0,
                                  (cxd * Na * gphigNb) * dJxW)
                    wp.atomic_add(Ae, e, ra + 3, cb + 1,
                                  ((-dRn) * Na * Nb) * dJxW)
                    wp.atomic_add(Ae, e, ra + 3, cb + 2,
                                  ((-dRp) * Na * Nb) * dJxW)
                    wp.atomic_add(Ae, e, ra + 3, cb + 3,
                                  (sig_d * Na * Nb
                                   + mu_xd[gp] * gNagNb) * dJxW)

                    # X̂_A row
                    wp.atomic_add(Ae, e, ra + 4, cb + 0,
                                  (cxa * Na * gphigNb) * dJxW)
                    wp.atomic_add(Ae, e, ra + 4, cb + 1,
                                  ((-dRn) * Na * Nb) * dJxW)
                    wp.atomic_add(Ae, e, ra + 4, cb + 2,
                                  ((-dRp) * Na * Nb) * dJxW)
                    wp.atomic_add(Ae, e, ra + 4, cb + 4,
                                  (sig_a * Na * Nb
                                   + mu_xa[gp] * gNagNb) * dJxW)

    _kernel_cache[key] = xdd_jacres
    return xdd_jacres


# ── small device vector/CSR helpers ─────────────────────────────────────────

def _axpy_neg_kernel():
    """y[i] -= m[i]  (the MMS nodal-source subtraction on the residual)."""
    key = ("xdd_axpy_neg",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique", enable_backward=False)
    def axn(m: wp.array(dtype=wp.float64),
            y: wp.array(dtype=wp.float64)):
        i = wp.tid()
        y[i] = y[i] - m[i]

    _kernel_cache[key] = axn
    return axn


def _scale_cols_kernel():
    """vals[k] *= s[cols[k]] — the B5 log-carrier column chain rule
    (host reference: A = A @ diags(scale))."""
    key = ("xdd_scale_cols",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique", enable_backward=False)
    def sck(cols: wp.array(dtype=wp.int32),
            s: wp.array(dtype=wp.float64),
            vals: wp.array(dtype=wp.float64)):
        k = wp.tid()
        vals[k] = vals[k] * s[cols[k]]

    _kernel_cache[key] = sck
    return sck


# ══════════════════════════════════════════════════════════════════════════════
# XDDDeviceAssembler — per-system object: pattern + static uploads once,
# numeric Jacobian/residual fill per Newton iterate on device.
# ══════════════════════════════════════════════════════════════════════════════

class XDDDeviceAssembler:
    """Device Jacobian/residual assembly for XDDSystem (dof order NODE-major:
    row = node·5 + field, vs the host bmat's field-major).  Built once per
    system; per-iterate cost is one jacres kernel launch + scatter per bin.

    The returned (A, r) carry the SAME Newton-increment convention as the host
    `_reduce_and_eliminate`: Dirichlet rows are identity rows with r = g − u
    (log mode: ln g − ln u), and in log mode the matrix columns of the carrier
    fields are scaled by diag(n̂)/diag(p̂).
    """

    def __init__(self, sysm, index_width="auto", val_dtype="fp64"):
        dm = sysm.dm
        self.sysm = sysm
        self.dm = dm
        self.asm = DeviceNSAssembler(dm, ndof=NDOF, node_pattern=True,
                                     blockmask=XDD_BLOCKMASK,
                                     index_width=index_width,
                                     val_dtype=val_dtype)
        d = dm.device
        self._dev = d

        # static per-bin GP coefficient uploads
        def up(x):
            return wp.array(np.ascontiguousarray(x, np.float64),
                            dtype=wp.float64, device=d)

        ons_arrays, ons_scalars = onsager_device_factors(
            sysm.onsager, sysm.dist_gp)
        gh = langevin_device_factor(sysm.langevin, sysm.dist_gp)
        self._ons_scalars = ons_scalars
        self._coef = {}
        self._zero_gp = {}
        self._bufs = {}
        self._bin_keys = []
        for k_bin, (pv, b, ne, nbf, _) in enumerate(self.asm._bins):
            ngp = ne * b["nqp"]
            c1, den, msk = ons_arrays[pv]
            self._coef[pv] = dict(
                eps=up(sysm.eps_gp[pv]), mu_n=up(sysm.mu_n_gp[pv]),
                mu_p=up(sysm.mu_p_gp[pv]), mu_xd=up(sysm.mu_xd_gp[pv]),
                mu_xa=up(sysm.mu_xa_gp[pv]),
                gh=up(gh[pv]), c1=up(c1), den=up(den), msk=up(msk))
            self._zero_gp[pv] = wp.zeros(ngp, dtype=wp.float64, device=d)
            nl = NDOF * nbf
            self._bufs[pv] = (wp.zeros((ne, nl, nl), dtype=wp.float64,
                                       device=d),
                              wp.zeros((ne, nl), dtype=wp.float64, device=d))
            self._bin_keys.append(pv)

        # per-step caches (keyed by object identity of the host-side source)
        self._hist_key = None
        self._hist_dev = None
        self._gen_key = None
        self._gen_dev = None
        self._mms_key = None
        self._mms_dev = None
        self._strong_key = None
        self._strong_rows = None
        self._cols_d = None          # int32 column indices (log-scale kernel)
        self._state_dev = None       # persistent nodal-state device buffers

    # ── budget report (scaling-pathway declaration numbers) ─────────────────
    def budget(self):
        asm = self.asm
        nnz = int(asm.nnz)
        ndofs = int(asm.Nfull)
        vals_b = nnz * 8
        idx_b = nnz * 4
        ptr_b = (ndofs + 1) * 8
        return dict(nnz=nnz, dofs=ndofs, nnz_per_dof=nnz / ndofs,
                    bytes_per_dof=(vals_b + idx_b + ptr_b) / ndofs,
                    csr_gb=(vals_b + idx_b + ptr_b) / 1e9,
                    index_wide=bool(asm.index_wide))

    # ── per-step upload helpers (identity-keyed caches) ─────────────────────
    def _upload_gp_dicts(self, fields):
        """fields: list of {pv: np.ndarray} dicts (or None → zeros).
        Returns list of {pv: wp.array}."""
        out = []
        for fd in fields:
            if fd is None:
                out.append({pv: self._zero_gp[pv] for pv in self._bin_keys})
            else:
                out.append({pv: wp.array(
                    np.ascontiguousarray(fd[pv], np.float64),
                    dtype=wp.float64, device=self._dev)
                    for pv in self._bin_keys})
        return out

    def _hist_arrays(self):
        sysm = self.sysm
        hist = sysm.hist
        key = id(hist) if hist is not None else None
        if key != self._hist_key:
            if hist is None:
                self._hist_dev = None
            else:
                self._hist_dev = self._upload_gp_dicts(
                    [hist[IN], hist[IP], hist[IXD], hist[IXA]])
            self._hist_key = key
        if self._hist_dev is None:
            z = {pv: self._zero_gp[pv] for pv in self._bin_keys}
            return z, z, z, z
        return tuple(self._hist_dev)

    def _gen_arrays(self):
        sysm = self.sysm
        gdh = getattr(sysm, "_gd", None)
        gah = getattr(sysm, "_ga", None)
        key = (id(gdh), id(gah))
        if key != self._gen_key:
            self._gen_dev = self._upload_gp_dicts([gdh, gah])
            self._gen_key = key
        return tuple(self._gen_dev)

    def _mms_vec(self):
        """MMS nodal source as a node-major device vector (or None)."""
        mms = self.sysm.mms_source
        if mms is None:
            return None
        key = id(mms)
        if key != self._mms_key:
            n = self.dm.n_nodes
            v = np.zeros(n * NDOF)
            for f in range(NDOF):
                if f in mms:
                    v[f::NDOF] = mms[f]
            self._mms_dev = wp.array(v, dtype=wp.float64, device=self._dev)
            self._mms_key = key
        return self._mms_dev

    def _strong_plan(self):
        """(rows, fields, nodes) for the Dirichlet strong rows, node-major;
        re-plans the assembler spans only when the Dirichlet SET changes."""
        sysm = self.sysm
        key = tuple((f, id(nv[0]), id(nv[1]))
                    for f, nv in sorted(sysm.dirichlet.items()))
        if key != self._strong_key:
            rows, metas = [], []
            for f in sorted(sysm.dirichlet):
                nodes, vals = sysm.dirichlet[f]
                rows.append(nodes.astype(np.int64) * NDOF + f)
                metas.append((f, nodes, vals))
            if rows:
                rows = np.concatenate(rows)
                self.asm.set_strong_rows(rows)
            else:
                rows = np.zeros(0, np.int64)
                self.asm._strong = None
            self._strong_rows = (rows, metas)
            self._strong_key = key
        return self._strong_rows

    # ── the per-iterate entry point ─────────────────────────────────────────
    def assemble(self, state, need_matrix=True):
        """Assemble the Newton system at `state` on device.

        Returns (A, r): r is the host residual vector (node-major, with the
        Dirichlet g−u convention); A is a host scipy CSR when
        need_matrix="host", None when need_matrix is False (line-search
        residual-only) or "device" (the cuDSS zero-copy path — the values
        live in self.asm.vals_d, consumed via self.asm.device_csr())."""
        sysm = self.sysm
        dm = self.dm
        d = self._dev
        asm = self.asm

        # nodal state upload (5 full vectors)
        st_dev = []
        for f in range(NDOF):
            st_dev.append(wp.array(
                np.ascontiguousarray(state[f], np.float64),
                dtype=wp.float64, device=d))

        hn, hp, hxd, hxa = self._hist_arrays()
        gd, ga = self._gen_arrays()
        osc = self._ons_scalars
        s2t = sysm._sig2tau()

        asm.zero_fill()
        for k_bin, (pv, b, ne, nbf, _) in enumerate(asm._bins):
            nqp = b["nqp"]
            co = self._coef[pv]
            Ae_d, re_d = self._bufs[pv]
            Ae_d.zero_()
            re_d.zero_()
            kern = make_xdd_jacres_kernel(nbf, nqp, dm.dim)
            wp.launch(kern, dim=ne, inputs=[
                b["conn"], b["h"], b["N"], b["dN"], b["lapN"], b["w"],
                st_dev[IPHI], st_dev[IN], st_dev[IP],
                st_dev[IXD], st_dev[IXA],
                co["eps"], co["mu_n"], co["mu_p"], co["mu_xd"], co["mu_xa"],
                co["gh"], co["c1"], co["den"], co["msk"],
                hn[pv], hp[pv], hxd[pv], hxa[pv], gd[pv], ga[pv],
                wp.float64(sysm.lam2), wp.float64(sysm.sigma),
                wp.float64(s2t), wp.float64(sysm.supg),
                wp.float64(sysm.tau_inv_d), wp.float64(sysm.tau_inv_a),
                wp.float64(osc["t0"]), wp.float64(osc["sd"]),
                wp.float64(osc["sa"]), wp.float64(osc["q3"]),
                wp.float64(osc["phi0"]), wp.float64(osc["x0"]),
                wp.float64(osc["dEdg"]),
                Ae_d, re_d], device=d)
            asm.scatter_bin(k_bin, Ae_d, re_d)

        # MMS nodal source: R -= mms (host applies it before reduction)
        mms = self._mms_vec()
        if mms is not None:
            wp.launch(_axpy_neg_kernel(), dim=asm.Nfull,
                      inputs=[mms, asm.F_d], device=d)

        # B5 log-carrier column chain rule (host: A @ diags(scale))
        if need_matrix and sysm._log_carriers:
            n = dm.n_nodes
            scale = np.ones(n * NDOF)
            cur = getattr(sysm, "_current_state", state)
            for f in (IN, IP):
                scale[f::NDOF] = cur[f]
            if self._cols_d is None:
                self._cols_d = wp.array(asm.indices.astype(np.int32),
                                        dtype=wp.int32, device=d)
            s_d = wp.array(scale, dtype=wp.float64, device=d)
            wp.launch(_scale_cols_kernel(), dim=asm.nnz,
                      inputs=[self._cols_d, s_d, asm.vals_d], device=d)

        # Dirichlet strong rows (device): identity rows, r = g − u
        rows, metas = self._strong_plan()
        if len(rows):
            cur = getattr(sysm, "_current_state", state)
            bvals = []
            for f, nodes, vals in metas:
                cu = cur[f][nodes]
                if sysm._log_carriers and f in (IN, IP):
                    bvals.append(np.log(vals) - np.log(cu))
                else:
                    bvals.append(vals - cu)
            self.asm.apply_strong_rows(np.concatenate(bvals))

        # NOTE: on warp-CPU .numpy() is a zero-copy VIEW of the device buffer;
        # the next assemble would mutate previously-returned arrays.  Copy on
        # CPU (CUDA .numpy() already materializes a fresh host array).
        cpu_view = str(self._dev).startswith("cpu")
        r = asm.F_d.numpy()
        if cpu_view:
            r = r.copy()
        if need_matrix == "host" or need_matrix is True:
            vals = asm.vals_d.numpy()
            if cpu_view:
                vals = vals.copy()
            A = sp.csr_matrix((vals, asm.indices, asm.indptr),
                              shape=(asm.Nfull, asm.Nfull))
            return A, r
        return None, r
