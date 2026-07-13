"""M1c: adjoint machinery for the linearized s=1/2 NS stepper (the user's
production formulation), volume terms first.

Within a step, the linearized system is R(x; aq, nu) = A(aq, nu) x - b = 0
with x = (u, p) node-major and the advecting GP field aq + viscosity nu as
the differentiable inputs. This module provides the TAPED residual kernel
computing the volume part of A(aq, nu) x with x FROZEN — the same epoch
pattern as m1a's sbm/adjoint.py: for a QoI J,

    A^T lam = dJ/dx                (transposed solve; cuDSS/splu — routed
                                    around, never taped)
    dJ/d{aq, nu} = -lam^T dR/d{aq, nu}   (this module's tape sweep)

TAU IS FROZEN in the backward pass (its aq-dependence is NOT differentiated)
— the production adjoint pattern: tau is a stabilization parameter, and
differentiating it buys higher-order corrections to a stabilization term at
significant tape cost. Documented delta; the m1b findings 1 timestab
lesson applies to gradients too.

FINDINGS 4c CONTRACT: enable_backward=True kernels here use wp.pow
jacobians, no loop-reassigned locals, and ship with a tape-vs-kernel-FD
unit test BEFORE any physics gate.
"""
import numpy as np
import warp as wp

from ..assembly.operators import _kernel_cache
from ..physics.vms import tau_m_metric, tau_c_metric


def make_lin_ns_residual(nbf: int, nqp: int, dim: int,
                         tau_frozen: bool = True):
    """Taped volume residual in RESIDUAL FORM: field quantities from the
    frozen state are accumulated ONCE per Gauss point at loop depth 1, then
    each test function only READS them — the m1a-proven taped shape.

    WARP BACKWARD BUG #2 (measured, findings 4c amended): scalar
    accumulators initialized inside second-level unrolled loops (the
    Ae-style a/b nesting) poison the ENTIRE tape with NaN — used or dead.
    Accumulators in taped kernels must live at nesting depth 1."""
    key = ("lin_ns_res", nbf, nqp, dim, tau_frozen)
    if key in _kernel_cache:
        return _kernel_cache[key]
    if dim == 3:
        k3 = _make_lin_ns_residual_3d(nbf, nqp, tau_frozen)
        _kernel_cache[key] = k3
        return k3
    ndof = dim + 1
    dim_pow = float(dim)
    dim_f = float(dim)
    vecT = wp.vec2d if dim == 2 else wp.vec3d
    matT = wp.mat22d if dim == 2 else wp.mat33d

    if dim != 2:
        raise NotImplementedError(
            "taped NS residual: dim=2 only until the warp adjoint "
            "interaction bug is resolved upstream (findings 4c amendment); "
            "the dim-3 kernel follows the same scalar generator pattern")

    # max_unroll HIGH: warp does NOT replay dynamic (non-unrolled) loops
    # in the backward pass — intermediate values are missing and the tape
    # NaNs (documented warp limitation; THE root cause of the overnight
    # NaN hunt: the q x b x a nesting exceeds the default unroll budget of
    # 16, so the loops silently went dynamic. Every "clean" micro-repro
    # had a single small loop that stayed static.)
    @wp.kernel(module="unique", enable_backward=True,
               module_options={"max_unroll": 1024})
    def lin_ns_res(conn: wp.array2d(dtype=wp.int32),
                   h: wp.array(dtype=wp.float64),
                   Ntab: wp.array2d(dtype=wp.float64),
                   dNtab: wp.array3d(dtype=wp.float64),
                   lapNtab: wp.array2d(dtype=wp.float64),
                   wtab: wp.array(dtype=wp.float64),
                   aq: wp.array2d(dtype=wp.float64),        # DIFF
                   div_aq: wp.array(dtype=wp.float64),      # DIFF
                   aq_frozen: wp.array2d(dtype=wp.float64),  # tau source
                   nu_arr: wp.array(dtype=wp.float64),      # DIFF, len 1
                   sigma: wp.float64, sig2tau: wp.float64,
                   s_skew: wp.float64,
                   x: wp.array(dtype=wp.float64),           # frozen state
                   r: wp.array(dtype=wp.float64)):
        # FULL-SCALAR SHAPE (m1a sbm_dir_res idiom): every accumulator is a
        # depth-1 scalar; no vec/mat locals; tau inlined. This is the one
        # taped shape with a green track record (findings 4c amendment: the
        # NS kernel NaN'd in every vec/mat formulation despite each
        # construct passing in isolation — an interaction bug, upstream
        # report with repros in docs/superpowers/warp_adjoint_repros/).
        e = wp.tid()
        he = h[e]
        jac = wp.pow(he * wp.float64(0.5), wp.float64(2.0))
        dscale = wp.float64(2.0) / he
        for q in range(nqp):
            nu = nu_arr[0]
            dJxW = wtab[q] * jac
            gp = e * nqp + q
            # frozen-state fields: scalar accumulators at depth 1
            u0 = wp.float64(0.0)
            u1 = wp.float64(0.0)
            g00 = wp.float64(0.0)     # du0/dx
            g01 = wp.float64(0.0)     # du0/dy
            g10 = wp.float64(0.0)
            g11 = wp.float64(0.0)
            pv_ = wp.float64(0.0)
            gp0 = wp.float64(0.0)
            gp1 = wp.float64(0.0)
            lap0 = wp.float64(0.0)    # lap(u0): G4 residual completion
            lap1 = wp.float64(0.0)
            for b in range(nbf):
                Nb = Ntab[q, b]
                dnb0 = dNtab[q, b, 0] * dscale
                dnb1 = dNtab[q, b, 1] * dscale
                lnb = lapNtab[q, b] * dscale * dscale
                xb0 = x[conn[e, b] * 3 + 0]
                xb1 = x[conn[e, b] * 3 + 1]
                pb = x[conn[e, b] * 3 + 2]
                u0 += Nb * xb0
                u1 += Nb * xb1
                g00 += dnb0 * xb0
                g01 += dnb1 * xb0
                g10 += dnb0 * xb1
                g11 += dnb1 * xb1
                lap0 += lnb * xb0
                lap1 += lnb * xb1
                pv_ += Nb * pb
                gp0 += dnb0 * pb
                gp1 += dnb1 * pb
            divu = g00 + g11
            a0 = aq[gp, 0]
            a1 = aq[gp, 1]
            # tau source: frozen copy (production tau-frozen pattern) or
            # the DIFFERENTIABLE aq (exactness for transient chains, where
            # aq varies with earlier states and FD sees dtau/daq — measured
            # as the 4e-4 chain leak in the N=2 isolation ladder)
            if wp.static(tau_frozen):
                af0 = aq_frozen[gp, 0]
                af1 = aq_frozen[gp, 1]
            else:
                af0 = aq[gp, 0]
                af1 = aq[gp, 1]
            uGu = wp.float64(4.0) * (af0 * af0 + af1 * af1) / (he * he)
            GG = wp.float64(2.0) * wp.pow(wp.float64(2.0) / he,
                                          wp.float64(4.0))
            tauM = wp.float64(1.0) / wp.sqrt(
                sig2tau + uGu + wp.float64(36.0) * nu * nu * GG)
            tauC = wp.float64(1.0) / (tauM * wp.float64(2.0)
                                      * wp.float64(4.0) / (he * he))
            diva = div_aq[gp]
            # strong linearized momentum residual — COMPLETE (retrofit
            # G4, matches ns_bricks make_linear_ns_Ae): the -nu lap(u)
            # term is exactly 0 at p1 (lapN tables vanish) and required
            # at p2; the taped twin MUST mirror the forward operator or
            # the dnu cotangent diverges from FD at p2 (measured).
            sfac = sigma + s_skew * diva
            rm0 = sfac * u0 + a0 * g00 + a1 * g01 + gp0 - nu * lap0
            rm1 = sfac * u1 + a0 * g10 + a1 * g11 + gp1 - nu * lap1
            conv0 = a0 * g00 + a1 * g01
            conv1 = a0 * g10 + a1 * g11
            for a in range(nbf):
                Na = Ntab[q, a]
                dna0 = dNtab[q, a, 0] * dscale
                dna1 = dNtab[q, a, 1] * dscale
                agw = a0 * dna0 + a1 * dna1
                v0 = (sigma * Na * u0
                      + Na * (conv0 + s_skew * diva * u0)
                      + nu * (dna0 * g00 + dna1 * g01)
                      + tauM * agw * rm0
                      + (tauC * divu - pv_) * dna0) * dJxW
                v1 = (sigma * Na * u1
                      + Na * (conv1 + s_skew * diva * u1)
                      + nu * (dna0 * g10 + dna1 * g11)
                      + tauM * agw * rm1
                      + (tauC * divu - pv_) * dna1) * dJxW
                qv = (Na * divu
                      + tauM * (dna0 * rm0 + dna1 * rm1)) * dJxW
                wp.atomic_add(r, conn[e, a] * 3 + 0, v0)
                wp.atomic_add(r, conn[e, a] * 3 + 1, v1)
                wp.atomic_add(r, conn[e, a] * 3 + 2, qv)

    _kernel_cache[key] = lin_ns_res
    return lin_ns_res


def ns_volume_cotangents(dm, aq_by_bin, div_aq_by_bin, nu, sigma, s_skew,
                         x_full, lam_full, timestab=True,
                         tau_frozen=True):
    """(-lam^T dR/daq per bin, -lam^T dR/dnu): the tape sweep. x_full and
    lam_full are FULL node-major vectors (ndof = dim+1)."""
    d = dm.device
    dim = dm.dim
    sig2tau = (2.0 * sigma) ** 2 if timestab else 0.0
    lam_d = wp.array(np.ascontiguousarray(lam_full), dtype=wp.float64,
                     device=d)
    x_d = wp.array(np.ascontiguousarray(x_full), dtype=wp.float64, device=d)
    aq_bar = {}
    dnu = 0.0
    for pv, b in dm.bins.items():
        k = make_lin_ns_residual(b["nbf"], b["nqp"], dim,
                                 tau_frozen=tau_frozen)
        tape = wp.Tape()
        aq = wp.array(np.ascontiguousarray(aq_by_bin[pv]), dtype=wp.float64,
                      device=d, requires_grad=True)
        dq = wp.array(np.ascontiguousarray(div_aq_by_bin[pv]),
                      dtype=wp.float64, device=d, requires_grad=True)
        aq_f = wp.array(np.ascontiguousarray(aq_by_bin[pv]),
                        dtype=wp.float64, device=d)      # frozen tau source
        nu_a = wp.array(np.array([nu]), dtype=wp.float64, device=d,
                        requires_grad=True)
        r = wp.zeros(dm.n_nodes * (dim + 1), dtype=wp.float64, device=d,
                     requires_grad=True)
        with tape:
            wp.launch(k, dim=len(b["eids"]),
                      inputs=[b["conn"], b["h"], b["N"], b["dN"],
                              b["lapN"],       # G4: complete residual
                              b["w"],
                              aq, dq, aq_f, nu_a, wp.float64(sigma),
                              wp.float64(sig2tau), wp.float64(s_skew),
                              x_d, r], device=d)
        tape.backward(grads={r: lam_d})
        aq_bar[pv] = (-tape.gradients[aq].numpy(),
                      -tape.gradients[dq].numpy())
        dnu += -float(tape.gradients[nu_a].numpy()[0])
    return aq_bar, dnu


def make_lin_ns_load(nbf: int, nqp: int, dim: int):
    """Taped LOAD twin of ns_bricks.make_linear_ns_be: b contributions
    r[node*ndof+c] += (N_a + tauM agw) fq_i + PSPG. Differentiable inputs
    {aq (via tau + agw), fq, nu}; same 4c rules as the residual kernel
    (plain locals, no struct, depth-1 accumulators). dim=2 first."""
    key = ("lin_ns_load", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]
    if dim == 3:
        k3 = _make_lin_ns_load_3d(nbf, nqp)
        _kernel_cache[key] = k3
        return k3

    @wp.kernel(module="unique", enable_backward=True,
               module_options={"max_unroll": 1024})
    def lin_ns_load(conn: wp.array2d(dtype=wp.int32),
                    h: wp.array(dtype=wp.float64),
                    Ntab: wp.array2d(dtype=wp.float64),
                    dNtab: wp.array3d(dtype=wp.float64),
                    wtab: wp.array(dtype=wp.float64),
                    aq: wp.array2d(dtype=wp.float64),       # DIFF
                    fq: wp.array2d(dtype=wp.float64),       # DIFF
                    nu_arr: wp.array(dtype=wp.float64),     # DIFF len 1
                    sig2tau: wp.float64,
                    r: wp.array(dtype=wp.float64)):
        e = wp.tid()
        he = h[e]
        jac = wp.pow(he * wp.float64(0.5), wp.float64(2.0))
        dscale = wp.float64(2.0) / he
        for q in range(nqp):
            nu = nu_arr[0]
            dJxW = wtab[q] * jac
            gp = e * nqp + q
            a0 = aq[gp, 0]
            a1 = aq[gp, 1]
            amag = wp.sqrt(a0 * a0 + a1 * a1)
            uGu = wp.float64(4.0) * (a0 * a0 + a1 * a1) / (he * he)
            GG = wp.float64(2.0) * wp.pow(wp.float64(2.0) / he,
                                          wp.float64(4.0))
            tauM = wp.float64(1.0) / wp.sqrt(
                sig2tau + uGu + wp.float64(36.0) * nu * nu * GG)
            f0 = fq[gp, 0]
            f1 = fq[gp, 1]
            for a in range(nbf):
                Na = Ntab[q, a]
                dna0 = dNtab[q, a, 0] * dscale
                dna1 = dNtab[q, a, 1] * dscale
                agw = a0 * dna0 + a1 * dna1
                wp.atomic_add(r, conn[e, a] * 3 + 0,
                              (Na + tauM * agw) * f0 * dJxW)
                wp.atomic_add(r, conn[e, a] * 3 + 1,
                              (Na + tauM * agw) * f1 * dJxW)
                wp.atomic_add(r, conn[e, a] * 3 + 2,
                              tauM * (dna0 * f0 + dna1 * f1) * dJxW)

    _kernel_cache[key] = lin_ns_load
    return lin_ns_load


def ns_load_cotangents(dm, aq_by_bin, fq_by_bin, nu, sigma, lam_full,
                       timestab=True):
    """(lam^T db/daq per bin, lam^T db/dfq per bin, lam^T db/dnu): tape
    sweep over the load. NOTE the sign convention differs from the
    residual sweep: these are +lam^T d(b)/d(.) — the caller composes
    R = A x - b itself."""
    d = dm.device
    dim = dm.dim
    sig2tau = (2.0 * sigma) ** 2 if timestab else 0.0
    lam_d = wp.array(np.ascontiguousarray(lam_full), dtype=wp.float64,
                     device=d)
    aq_bar, fq_bar = {}, {}
    dnu = 0.0
    for pv, b in dm.bins.items():
        k = make_lin_ns_load(b["nbf"], b["nqp"], dim)
        tape = wp.Tape()
        aq = wp.array(np.ascontiguousarray(aq_by_bin[pv]), dtype=wp.float64,
                      device=d, requires_grad=True)
        fq = wp.array(np.ascontiguousarray(fq_by_bin[pv]), dtype=wp.float64,
                      device=d, requires_grad=True)
        nu_a = wp.array(np.array([nu]), dtype=wp.float64, device=d,
                        requires_grad=True)
        r = wp.zeros(dm.n_nodes * (dim + 1), dtype=wp.float64, device=d,
                     requires_grad=True)
        with tape:
            wp.launch(k, dim=len(b["eids"]),
                      inputs=[b["conn"], b["h"], b["N"], b["dN"], b["w"],
                              aq, fq, nu_a, wp.float64(sig2tau), r],
                      device=d)
        tape.backward(grads={r: lam_d})
        aq_bar[pv] = tape.gradients[aq].numpy()
        fq_bar[pv] = tape.gradients[fq].numpy()
        dnu += float(tape.gradients[nu_a].numpy()[0])
    return aq_bar, fq_bar, dnu


def _make_lin_ns_residual_3d(nbf: int, nqp: int, tau_frozen: bool):
    """dim-3 taped volume residual — same 4c-rule scalar shape as dim-2.
    RESIDUAL FORM keeps the unroll budget ~q*(b+a), NOT the a*b*dof
    element-matrix explosion that forced max_unroll=0 on the 3-D forward
    Ae kernels (79-min compile, findings 6): full unrolling here is
    expected to compile in O(minute) — measured by its gate."""

    @wp.kernel(module="unique", enable_backward=True,
               module_options={"max_unroll": 4096})
    def lin_ns_res3(conn: wp.array2d(dtype=wp.int32),
                    h: wp.array(dtype=wp.float64),
                    Ntab: wp.array2d(dtype=wp.float64),
                    dNtab: wp.array3d(dtype=wp.float64),
                    lapNtab: wp.array2d(dtype=wp.float64),
                    wtab: wp.array(dtype=wp.float64),
                    aq: wp.array2d(dtype=wp.float64),       # DIFF
                    div_aq: wp.array(dtype=wp.float64),     # DIFF
                    aq_frozen: wp.array2d(dtype=wp.float64),
                    nu_arr: wp.array(dtype=wp.float64),     # DIFF
                    sigma: wp.float64, sig2tau: wp.float64,
                    s_skew: wp.float64,
                    x: wp.array(dtype=wp.float64),
                    r: wp.array(dtype=wp.float64)):
        e = wp.tid()
        he = h[e]
        jac = wp.pow(he * wp.float64(0.5), wp.float64(3.0))
        dscale = wp.float64(2.0) / he
        for q in range(nqp):
            nu = nu_arr[0]
            dJxW = wtab[q] * jac
            gp = e * nqp + q
            u0 = wp.float64(0.0)
            u1 = wp.float64(0.0)
            u2 = wp.float64(0.0)
            g00 = wp.float64(0.0)
            g01 = wp.float64(0.0)
            g02 = wp.float64(0.0)
            g10 = wp.float64(0.0)
            g11 = wp.float64(0.0)
            g12 = wp.float64(0.0)
            g20 = wp.float64(0.0)
            g21 = wp.float64(0.0)
            g22 = wp.float64(0.0)
            pv_ = wp.float64(0.0)
            gp0 = wp.float64(0.0)
            gp1 = wp.float64(0.0)
            gp2 = wp.float64(0.0)
            lap0 = wp.float64(0.0)    # lap(u_c): G4 residual completion
            lap1 = wp.float64(0.0)
            lap2 = wp.float64(0.0)
            for b in range(nbf):
                Nb = Ntab[q, b]
                d0 = dNtab[q, b, 0] * dscale
                d1 = dNtab[q, b, 1] * dscale
                d2 = dNtab[q, b, 2] * dscale
                lnb = lapNtab[q, b] * dscale * dscale
                xb0 = x[conn[e, b] * 4 + 0]
                xb1 = x[conn[e, b] * 4 + 1]
                xb2 = x[conn[e, b] * 4 + 2]
                pb = x[conn[e, b] * 4 + 3]
                u0 += Nb * xb0
                u1 += Nb * xb1
                u2 += Nb * xb2
                g00 += d0 * xb0
                g01 += d1 * xb0
                g02 += d2 * xb0
                g10 += d0 * xb1
                g11 += d1 * xb1
                g12 += d2 * xb1
                g20 += d0 * xb2
                g21 += d1 * xb2
                g22 += d2 * xb2
                lap0 += lnb * xb0
                lap1 += lnb * xb1
                lap2 += lnb * xb2
                pv_ += Nb * pb
                gp0 += d0 * pb
                gp1 += d1 * pb
                gp2 += d2 * pb
            divu = g00 + g11 + g22
            a0 = aq[gp, 0]
            a1 = aq[gp, 1]
            a2 = aq[gp, 2]
            if wp.static(tau_frozen):
                af0 = aq_frozen[gp, 0]
                af1 = aq_frozen[gp, 1]
                af2 = aq_frozen[gp, 2]
            else:
                af0 = aq[gp, 0]
                af1 = aq[gp, 1]
                af2 = aq[gp, 2]
            uGu = wp.float64(4.0) * (af0 * af0 + af1 * af1 + af2 * af2) \
                / (he * he)
            GG = wp.float64(3.0) * wp.pow(wp.float64(2.0) / he,
                                          wp.float64(4.0))
            tauM = wp.float64(1.0) / wp.sqrt(
                sig2tau + uGu + wp.float64(36.0) * nu * nu * GG)
            tauC = wp.float64(1.0) / (tauM * wp.float64(3.0)
                                      * wp.float64(4.0) / (he * he))
            diva = div_aq[gp]
            # COMPLETE residual (G4; see the 2-D twin's comment)
            sfac = sigma + s_skew * diva
            rm0 = sfac * u0 + a0 * g00 + a1 * g01 + a2 * g02 + gp0 \
                - nu * lap0
            rm1 = sfac * u1 + a0 * g10 + a1 * g11 + a2 * g12 + gp1 \
                - nu * lap1
            rm2 = sfac * u2 + a0 * g20 + a1 * g21 + a2 * g22 + gp2 \
                - nu * lap2
            conv0 = a0 * g00 + a1 * g01 + a2 * g02
            conv1 = a0 * g10 + a1 * g11 + a2 * g12
            conv2 = a0 * g20 + a1 * g21 + a2 * g22
            for a in range(nbf):
                Na = Ntab[q, a]
                dna0 = dNtab[q, a, 0] * dscale
                dna1 = dNtab[q, a, 1] * dscale
                dna2 = dNtab[q, a, 2] * dscale
                agw = a0 * dna0 + a1 * dna1 + a2 * dna2
                v0 = (sigma * Na * u0
                      + Na * (conv0 + s_skew * diva * u0)
                      + nu * (dna0 * g00 + dna1 * g01 + dna2 * g02)
                      + tauM * agw * rm0
                      + (tauC * divu - pv_) * dna0) * dJxW
                v1 = (sigma * Na * u1
                      + Na * (conv1 + s_skew * diva * u1)
                      + nu * (dna0 * g10 + dna1 * g11 + dna2 * g12)
                      + tauM * agw * rm1
                      + (tauC * divu - pv_) * dna1) * dJxW
                v2 = (sigma * Na * u2
                      + Na * (conv2 + s_skew * diva * u2)
                      + nu * (dna0 * g20 + dna1 * g21 + dna2 * g22)
                      + tauM * agw * rm2
                      + (tauC * divu - pv_) * dna2) * dJxW
                qv = (Na * divu
                      + tauM * (dna0 * rm0 + dna1 * rm1 + dna2 * rm2)) \
                    * dJxW
                wp.atomic_add(r, conn[e, a] * 4 + 0, v0)
                wp.atomic_add(r, conn[e, a] * 4 + 1, v1)
                wp.atomic_add(r, conn[e, a] * 4 + 2, v2)
                wp.atomic_add(r, conn[e, a] * 4 + 3, qv)

    return lin_ns_res3


def _make_lin_ns_load_3d(nbf: int, nqp: int):
    """dim-3 taped load twin (same shape rules)."""

    @wp.kernel(module="unique", enable_backward=True,
               module_options={"max_unroll": 4096})
    def lin_ns_load3(conn: wp.array2d(dtype=wp.int32),
                     h: wp.array(dtype=wp.float64),
                     Ntab: wp.array2d(dtype=wp.float64),
                     dNtab: wp.array3d(dtype=wp.float64),
                     wtab: wp.array(dtype=wp.float64),
                     aq: wp.array2d(dtype=wp.float64),      # DIFF
                     fq: wp.array2d(dtype=wp.float64),      # DIFF
                     nu_arr: wp.array(dtype=wp.float64),    # DIFF
                     sig2tau: wp.float64,
                     r: wp.array(dtype=wp.float64)):
        e = wp.tid()
        he = h[e]
        jac = wp.pow(he * wp.float64(0.5), wp.float64(3.0))
        dscale = wp.float64(2.0) / he
        for q in range(nqp):
            nu = nu_arr[0]
            dJxW = wtab[q] * jac
            gp = e * nqp + q
            a0 = aq[gp, 0]
            a1 = aq[gp, 1]
            a2 = aq[gp, 2]
            amag2 = a0 * a0 + a1 * a1 + a2 * a2
            uGu = wp.float64(4.0) * amag2 / (he * he)
            GG = wp.float64(3.0) * wp.pow(wp.float64(2.0) / he,
                                          wp.float64(4.0))
            tauM = wp.float64(1.0) / wp.sqrt(
                sig2tau + uGu + wp.float64(36.0) * nu * nu * GG)
            f0 = fq[gp, 0]
            f1 = fq[gp, 1]
            f2 = fq[gp, 2]
            for a in range(nbf):
                Na = Ntab[q, a]
                dna0 = dNtab[q, a, 0] * dscale
                dna1 = dNtab[q, a, 1] * dscale
                dna2 = dNtab[q, a, 2] * dscale
                agw = a0 * dna0 + a1 * dna1 + a2 * dna2
                wp.atomic_add(r, conn[e, a] * 4 + 0,
                              (Na + tauM * agw) * f0 * dJxW)
                wp.atomic_add(r, conn[e, a] * 4 + 1,
                              (Na + tauM * agw) * f1 * dJxW)
                wp.atomic_add(r, conn[e, a] * 4 + 2,
                              (Na + tauM * agw) * f2 * dJxW)
                wp.atomic_add(r, conn[e, a] * 4 + 3,
                              tauM * (dna0 * f0 + dna1 * f1
                                      + dna2 * f2) * dJxW)

    return lin_ns_load3
