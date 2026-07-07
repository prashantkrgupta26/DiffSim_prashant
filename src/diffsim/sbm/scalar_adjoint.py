"""M2-A5: taped scalar advection-diffusion volume kernel + cotangents.

Mirrors ns_adjoint.py's proven taped shape (findings 4c contract:
full-scalar accumulators at depth 1, wp.pow jacobians, no structs,
max_unroll=1024, residual form). kq is a PER-GP array (the B1 field
interface); tau differentiated through (not frozen). dim=2 (the dim-3
variant follows the ns_adjoint scalar-generator pattern when needed).
"""
import numpy as np
import warp as wp

from ..assembly.operators import _kernel_cache
from ..physics.vms import CI_F


def make_scalar_residual(nbf: int, nqp: int):
    key = ("scalar_ad_res", nbf, nqp)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique", enable_backward=True,
               module_options={"max_unroll": 1024})
    def scalar_res(conn: wp.array2d(dtype=wp.int32),
                   h: wp.array(dtype=wp.float64),
                   Ntab: wp.array2d(dtype=wp.float64),
                   dNtab: wp.array3d(dtype=wp.float64),
                   lapNtab: wp.array2d(dtype=wp.float64),
                   wtab: wp.array(dtype=wp.float64),
                   aq: wp.array2d(dtype=wp.float64),     # DIFF [ngp,2]
                   kq: wp.array(dtype=wp.float64),       # DIFF [ngp]
                   sigma: wp.float64, sig2tau: wp.float64,
                   supg: wp.float64,
                   x: wp.array(dtype=wp.float64),        # frozen T (full)
                   r: wp.array(dtype=wp.float64)):
        e = wp.tid()
        he = h[e]
        jac = wp.pow(he * wp.float64(0.5), wp.float64(2.0))
        dscale = wp.float64(2.0) / he
        for q in range(nqp):
            dJxW = wtab[q] * jac
            gp = e * nqp + q
            kap = kq[gp]
            a0 = aq[gp, 0]
            a1 = aq[gp, 1]
            # frozen-state fields at depth 1
            T0 = wp.float64(0.0)
            gT0 = wp.float64(0.0)
            gT1 = wp.float64(0.0)
            lT = wp.float64(0.0)
            for b in range(nbf):
                xb = x[conn[e, b]]
                T0 += Ntab[q, b] * xb
                gT0 += dNtab[q, b, 0] * dscale * xb
                gT1 += dNtab[q, b, 1] * dscale * xb
                lT += lapNtab[q, b] * dscale * dscale * xb
            # tau_m_metric inlined EXACTLY (vms.py: 1/sqrt(sig2 +
            # 4|a|^2/h^2 + CI_F kap^2 dim (2/h)^4)), differentiable
            # through kq/aq
            uGu = wp.float64(4.0) * (a0 * a0 + a1 * a1) / (he * he)
            GG = wp.float64(2.0) * wp.pow(wp.float64(2.0) / he,
                                          wp.float64(4.0))
            tau = supg / wp.sqrt(sig2tau + uGu
                                 + wp.float64(CI_F) * kap * kap * GG)
            adv = a0 * gT0 + a1 * gT1
            res = sigma * T0 + adv - kap * lT
            for a in range(nbf):
                agw = (a0 * dNtab[q, a, 0] + a1 * dNtab[q, a, 1]) * dscale
                ra = (sigma * Ntab[q, a] * T0 + Ntab[q, a] * adv
                      + kap * (dNtab[q, a, 0] * gT0
                               + dNtab[q, a, 1] * gT1)
                      * dscale + tau * agw * res) * dJxW
                wp.atomic_add(r, conn[e, a], ra)

    _kernel_cache[key] = scalar_res
    return scalar_res


def scalar_volume_cotangents(dm, aq_by_bin, kq_by_bin, sigma, x_full,
                             lam_full, sig2tau=None, supg=1.0):
    """Per-GP cotangents (-lam^T dR/dkq, -lam^T dR/daq) per bin — the B1
    field-kappa interface (dkappa per GP) and the coupling chain (daq)."""
    d = dm.device
    if sig2tau is None:
        sig2tau = (2.0 * sigma) ** 2
    lam_d = wp.array(np.ascontiguousarray(lam_full), dtype=wp.float64,
                     device=d)
    x_d = wp.array(np.ascontiguousarray(x_full), dtype=wp.float64,
                   device=d)
    out = {}
    for pv, b in dm.bins.items():
        k = make_scalar_residual(b["nbf"], b["nqp"])
        tape = wp.Tape()
        aq = wp.array(np.ascontiguousarray(aq_by_bin[pv]),
                      dtype=wp.float64, device=d, requires_grad=True)
        kq = wp.array(np.ascontiguousarray(kq_by_bin[pv]),
                      dtype=wp.float64, device=d, requires_grad=True)
        r = wp.zeros(dm.n_nodes, dtype=wp.float64, device=d,
                     requires_grad=True)
        with tape:
            wp.launch(k, dim=len(b["eids"]),
                      inputs=[b["conn"], b["h"], b["N"], b["dN"],
                              b["lapN"], b["w"], aq, kq,
                              wp.float64(sigma), wp.float64(sig2tau),
                              wp.float64(supg), x_d, r], device=d)
        tape.backward(grads={r: lam_d})
        out[pv] = (-tape.gradients[kq].numpy(),
                   -tape.gradients[aq].numpy())
    return out
