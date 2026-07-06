"""Bisect the NaN: taped mini-kernels adding one term family at a time."""
import numpy as np, warp as wp, sys
sys.path.insert(0, "tests")
from test_ns_adjoint import _setup
from diffsim.assembly.femelm import FEMElm, fe_N, fe_dN_s, fe_detJxW_s
from diffsim.physics.vms import tau_m_metric, tau_c_metric
from diffsim.assembly.operators import _kernel_cache

def make_variant(nbf, nqp, dim, MASS, CONV, VISC, SUPG, TAUC):
    key = ("nan_bisect", MASS, CONV, VISC, SUPG, TAUC)
    if key in _kernel_cache:
        return _kernel_cache[key]
    ndof = dim + 1
    dim_pow = float(dim); dim_f = float(dim)
    @wp.kernel(module="unique", enable_backward=True)
    def kvar(conn: wp.array2d(dtype=wp.int32), h: wp.array(dtype=wp.float64),
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
        nu = nu_arr[0]
        for q in range(nqp):
            fe.q = q
            dJxW = fe_detJxW_s(wtab, fe, jac)
            gp = e * nqp + q
            amag_f = wp.float64(0.0)
            for d in range(dim):
                amag_f += aq_frozen[gp, d] * aq_frozen[gp, d]
            amag_f = wp.sqrt(amag_f)
            tauM = wp.float64(0.0)
            tauC = wp.float64(0.0)
            if wp.static(SUPG or TAUC):
                tauM = tau_m_metric(amag_f, fe.he, nu, sig2tau,
                                    wp.float64(dim_f))
                tauC = tau_c_metric(tauM, fe.he, wp.float64(dim_f))
            diva = div_aq[gp]
            for a in range(nbf):
                Na = fe_N(Ntab, fe, a)
                agw = wp.float64(0.0)
                for d in range(dim):
                    agw += aq[gp, d] * fe_dN_s(dNtab, fe, a, d, dscale)
                for b in range(nbf):
                    Nb = fe_N(Ntab, fe, b)
                    agu = wp.float64(0.0)
                    lap = wp.float64(0.0)
                    for d in range(dim):
                        agu += aq[gp, d] * fe_dN_s(dNtab, fe, b, d, dscale)
                        lap += fe_dN_s(dNtab, fe, a, d, dscale) \
                            * fe_dN_s(dNtab, fe, b, d, dscale)
                    conv = agu + s_skew * diva * Nb
                    resu = sigma * Nb + conv
                    diag = wp.float64(0.0)
                    if wp.static(MASS):
                        diag += sigma * Na * Nb
                    if wp.static(CONV):
                        diag += Na * conv
                    if wp.static(VISC):
                        diag += nu * lap
                    if wp.static(SUPG):
                        diag += tauM * agw * resu
                    diag = diag * dJxW
                    for i in range(dim):
                        ub = x[conn[e, b] * ndof + i]
                        wp.atomic_add(r, conn[e, a] * ndof + i, diag * ub)
                        if wp.static(TAUC):
                            for j in range(dim):
                                wp.atomic_add(
                                    r, conn[e, a] * ndof + i,
                                    tauC * fe_dN_s(dNtab, fe, a, i, dscale)
                                    * fe_dN_s(dNtab, fe, b, j, dscale)
                                    * dJxW * x[conn[e, b] * ndof + j])
    _kernel_cache[key] = kvar
    return kvar

dm, xq, aq, dq, x_full, lam_full = _setup(3, "cuda:0")
pv = list(dm.bins)[0]; b = dm.bins[pv]; d = dm.device
for name, flags in (("mass", (1,0,0,0,0)), ("conv", (1,1,0,0,0)),
                    ("visc", (1,1,1,0,0)), ("supg", (1,1,1,1,0)),
                    ("tauc", (1,1,1,1,1))):
    k = make_variant(b["nbf"], b["nqp"], dm.dim, *[bool(f) for f in flags])
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
    gn = float(tape.gradients[nua].numpy()[0])
    print(f"{name}: aq-grad nan = {np.isnan(ga).sum()}/{ga.size}, "
          f"nu-grad = {gn}", flush=True)
