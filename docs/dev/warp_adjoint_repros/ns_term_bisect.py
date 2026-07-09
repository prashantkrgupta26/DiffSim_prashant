"""Term-by-term bisection of the whole-value NS kernel's NaN tape.
Builds truncated variants; prints ONE verdict line per variant."""
import numpy as np, warp as wp, sys
sys.path.insert(0, "tests")
from test_ns_adjoint import _setup
from diffsim.assembly.femelm import FEMElm, fe_N, fe_dN_s, fe_detJxW_s
from diffsim.physics.vms import tau_m_metric, tau_c_metric
from diffsim.assembly.operators import _kernel_cache

def make_v(nbf, nqp, dim, LEVEL):
    # LEVEL: 1=uval only; 2=+gu(outer); 3=+gradp/pval; 4=+av/dot terms;
    # 5=+tau (frozen); 6=+resm full; 7=+all test terms (== full kernel)
    key = ("ns_term_bisect", LEVEL)
    if key in _kernel_cache:
        return _kernel_cache[key]
    ndof = dim + 1
    dim_pow = float(dim); dim_f = float(dim)
    @wp.kernel(module="unique", enable_backward=True)
    def kv(conn: wp.array2d(dtype=wp.int32), h: wp.array(dtype=wp.float64),
           Ntab: wp.array2d(dtype=wp.float64),
           dNtab: wp.array3d(dtype=wp.float64),
           wtab: wp.array(dtype=wp.float64),
           aq: wp.array2d(dtype=wp.float64),
           div_aq: wp.array(dtype=wp.float64),
           aq_frozen: wp.array2d(dtype=wp.float64),
           nu_arr: wp.array(dtype=wp.float64),
           sigma: wp.float64, sig2tau: wp.float64, s_skew: wp.float64,
           x: wp.array(dtype=wp.float64), r: wp.array(dtype=wp.float64)):
        e = wp.tid()
        fe = FEMElm(); fe.e = e; fe.he = h[e]
        jac = wp.pow(fe.he * wp.float64(0.5), wp.float64(dim_pow))
        dscale = wp.float64(2.0) / fe.he
        for q in range(nqp):
            fe.q = q
            nu = nu_arr[0]
            dJxW = fe_detJxW_s(wtab, fe, jac)
            gp = e * nqp + q
            uval = wp.vec2d()
            gu = wp.mat22d()
            pval = wp.float64(0.0)
            gradp = wp.vec2d()
            for b in range(nbf):
                Nb = fe_N(Ntab, fe, b)
                xv = wp.vec2d(x[conn[e, b] * ndof + 0],
                              x[conn[e, b] * ndof + 1])
                dnb = wp.vec2d(fe_dN_s(dNtab, fe, b, 0, dscale),
                               fe_dN_s(dNtab, fe, b, 1, dscale))
                uval += Nb * xv
                if wp.static(LEVEL >= 2):
                    gu += wp.outer(xv, dnb)
                if wp.static(LEVEL >= 3):
                    pb = x[conn[e, b] * ndof + dim]
                    pval += Nb * pb
                    gradp += pb * dnb
            av = wp.vec2d(aq[gp, 0], aq[gp, 1])
            out = wp.float64(0.0)
            if wp.static(LEVEL >= 4):
                out += wp.dot(av, uval)
            if wp.static(LEVEL >= 5):
                af = wp.vec2d(aq_frozen[gp, 0], aq_frozen[gp, 1])
                tauM = tau_m_metric(wp.sqrt(wp.dot(af, af)), fe.he, nu,
                                    sig2tau, wp.float64(dim_f))
                out += tauM * wp.dot(av, av)
            if wp.static(LEVEL >= 6):
                resm = (sigma + s_skew * div_aq[gp]) * uval + gu * av + gradp
                out += wp.dot(resm, resm) * wp.float64(1e-3)
            if wp.static(LEVEL == 1):
                out += wp.dot(av, av) * uval[0]
            for a in range(nbf):
                wp.atomic_add(r, conn[e, a] * ndof, out * dJxW
                              * fe_N(Ntab, fe, a))
    _kernel_cache[key] = kv
    return kv

dm, xq, aq, dq, x_full, lam_full = _setup(3, "cuda:0")
pv = list(dm.bins)[0]; b = dm.bins[pv]; d = dm.device
for lvl in (1, 2, 3, 4, 5, 6):
    k = make_v(b["nbf"], b["nqp"], dm.dim, lvl)
    tape = wp.Tape()
    aqd = wp.array(np.ascontiguousarray(aq[pv]), dtype=wp.float64,
                   device=d, requires_grad=True)
    dqd = wp.array(np.ascontiguousarray(dq[pv]), dtype=wp.float64,
                   device=d, requires_grad=True)
    aqf = wp.array(np.ascontiguousarray(aq[pv]), dtype=wp.float64, device=d)
    nua = wp.array(np.array([0.05]), dtype=wp.float64, device=d,
                   requires_grad=True)
    r = wp.zeros(dm.n_nodes * 3, dtype=wp.float64, device=d,
                 requires_grad=True)
    with tape:
        wp.launch(k, dim=len(b["eids"]),
                  inputs=[b["conn"], b["h"], b["N"], b["dN"], b["w"], aqd,
                          dqd, aqf, nua, wp.float64(20.0), wp.float64(1600.0),
                          wp.float64(0.5),
                          wp.array(x_full, dtype=wp.float64, device=d), r],
                  device=d)
    tape.backward(grads={r: wp.array(lam_full, dtype=wp.float64, device=d)})
    ga = tape.gradients[aqd].numpy()
    print(f"LEVEL {lvl}: aq nan = {np.isnan(ga).sum()}/{ga.size}", flush=True)
print("BISECT DONE", flush=True)
