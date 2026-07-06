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

from ..assembly.femelm import FEMElm, fe_N, fe_dN_s, fe_detJxW_s
from ..assembly.operators import _kernel_cache
from ..physics.vms import tau_m_metric, tau_c_metric


def make_lin_ns_residual(nbf: int, nqp: int, dim: int):
    """Taped volume residual in RESIDUAL FORM: field quantities from the
    frozen state are accumulated ONCE per Gauss point at loop depth 1, then
    each test function only READS them — the m1a-proven taped shape.

    WARP BACKWARD BUG #2 (measured, findings 4c amended): scalar
    accumulators initialized inside second-level unrolled loops (the
    Ae-style a/b nesting) poison the ENTIRE tape with NaN — used or dead.
    Accumulators in taped kernels must live at nesting depth 1."""
    key = ("lin_ns_res", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]
    ndof = dim + 1
    dim_pow = float(dim)
    dim_f = float(dim)
    vecT = wp.vec2d if dim == 2 else wp.vec3d
    matT = wp.mat22d if dim == 2 else wp.mat33d

    # Inner accumulations live in wp.funcs — the m1a-proven pattern
    # (shift_fn): kernel-BODY accumulators above depth 1 poison the tape
    # (warp backward bug #2, measured both dead AND used), but the same
    # accumulation inside a wp.func differentiates correctly.
    @wp.func
    def _grad_dot_vec(dNtab: wp.array3d(dtype=wp.float64), fe: FEMElm,
                      a: int, dscale: wp.float64, v: vecT) -> wp.float64:
        s = wp.float64(0.0)
        for d in range(dim):
            s += v[d] * fe_dN_s(dNtab, fe, a, d, dscale)
        return s

    @wp.func
    def _grad_dot_gurow(dNtab: wp.array3d(dtype=wp.float64), fe: FEMElm,
                        a: int, dscale: wp.float64, gu: matT,
                        i: int) -> wp.float64:
        s = wp.float64(0.0)
        for d in range(dim):
            s += fe_dN_s(dNtab, fe, a, d, dscale) * gu[i, d]
        return s

    @wp.func
    def _vec_dot_gurow(av: vecT, gu: matT, i: int) -> wp.float64:
        s = wp.float64(0.0)
        for d in range(dim):
            s += av[d] * gu[i, d]
        return s

    if dim != 2:
        raise NotImplementedError(
            "taped NS residual: dim=2 only until the warp adjoint "
            "interaction bug is resolved upstream (findings 4c amendment); "
            "the dim-3 kernel follows the same scalar generator pattern")

    @wp.kernel(module="unique", enable_backward=True)
    def lin_ns_res(conn: wp.array2d(dtype=wp.int32),
                   h: wp.array(dtype=wp.float64),
                   Ntab: wp.array2d(dtype=wp.float64),
                   dNtab: wp.array3d(dtype=wp.float64),
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
        fe = FEMElm()
        fe.e = e
        fe.he = h[e]
        jac = wp.pow(fe.he * wp.float64(0.5), wp.float64(2.0))
        dscale = wp.float64(2.0) / fe.he
        for q in range(nqp):
            fe.q = q
            nu = nu_arr[0]
            dJxW = fe_detJxW_s(wtab, fe, jac)
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
            for b in range(nbf):
                Nb = fe_N(Ntab, fe, b)
                dnb0 = fe_dN_s(dNtab, fe, b, 0, dscale)
                dnb1 = fe_dN_s(dNtab, fe, b, 1, dscale)
                xb0 = x[conn[e, b] * 3 + 0]
                xb1 = x[conn[e, b] * 3 + 1]
                pb = x[conn[e, b] * 3 + 2]
                u0 += Nb * xb0
                u1 += Nb * xb1
                g00 += dnb0 * xb0
                g01 += dnb1 * xb0
                g10 += dnb0 * xb1
                g11 += dnb1 * xb1
                pv_ += Nb * pb
                gp0 += dnb0 * pb
                gp1 += dnb1 * pb
            divu = g00 + g11
            a0 = aq[gp, 0]
            a1 = aq[gp, 1]
            # tau INLINED from the frozen advecting copy (tau-frozen)
            af0 = aq_frozen[gp, 0]
            af1 = aq_frozen[gp, 1]
            uGu = wp.float64(4.0) * (af0 * af0 + af1 * af1) \
                / (fe.he * fe.he)
            GG = wp.float64(2.0) * wp.pow(wp.float64(2.0) / fe.he,
                                          wp.float64(4.0))
            tauM = wp.float64(1.0) / wp.sqrt(
                sig2tau + uGu + wp.float64(36.0) * nu * nu * GG)
            tauC = wp.float64(1.0) / (tauM * wp.float64(2.0)
                                      * wp.float64(4.0)
                                      / (fe.he * fe.he))
            diva = div_aq[gp]
            # strong linearized momentum residual (nu-lap dropped at p1)
            sfac = sigma + s_skew * diva
            rm0 = sfac * u0 + a0 * g00 + a1 * g01 + gp0
            rm1 = sfac * u1 + a0 * g10 + a1 * g11 + gp1
            conv0 = a0 * g00 + a1 * g01
            conv1 = a0 * g10 + a1 * g11
            for a in range(nbf):
                Na = fe_N(Ntab, fe, a)
                dna0 = fe_dN_s(dNtab, fe, a, 0, dscale)
                dna1 = fe_dN_s(dNtab, fe, a, 1, dscale)
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
                         x_full, lam_full, timestab=True):
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
        k = make_lin_ns_residual(b["nbf"], b["nqp"], dim)
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
                      inputs=[b["conn"], b["h"], b["N"], b["dN"], b["w"],
                              aq, dq, aq_f, nu_a, wp.float64(sigma),
                              wp.float64(sig2tau), wp.float64(s_skew),
                              x_d, r], device=d)
        tape.backward(grads={r: lam_d})
        aq_bar[pv] = (-tape.gradients[aq].numpy(),
                      -tape.gradients[dq].numpy())
        dnu += -float(tape.gradients[nu_a].numpy()[0])
    return aq_bar, dnu
