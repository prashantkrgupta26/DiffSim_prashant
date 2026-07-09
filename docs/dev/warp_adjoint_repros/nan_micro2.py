"""Is the trigger deadness, or nested rebinding regardless of use?"""
import numpy as np, warp as wp, sys
sys.path.insert(0, "tests")
from test_ns_adjoint import _setup
from diffsim.assembly.femelm import FEMElm, fe_N, fe_dN_s, fe_detJxW_s
from diffsim.assembly.operators import _kernel_cache

def make_m2(nbf, nqp, dim, USED):
    key = ("nan_micro2", USED)
    if key in _kernel_cache:
        return _kernel_cache[key]
    ndof = dim + 1
    dim_pow = float(dim)
    @wp.kernel(module="unique", enable_backward=True)
    def km2(conn: wp.array2d(dtype=wp.int32), h: wp.array(dtype=wp.float64),
            Ntab: wp.array2d(dtype=wp.float64),
            dNtab: wp.array3d(dtype=wp.float64),
            wtab: wp.array(dtype=wp.float64),
            aq: wp.array2d(dtype=wp.float64),
            sigma: wp.float64,
            x: wp.array(dtype=wp.float64), r: wp.array(dtype=wp.float64)):
        e = wp.tid()
        fe = FEMElm(); fe.e = e; fe.he = h[e]
        jac = wp.pow(fe.he * wp.float64(0.5), wp.float64(dim_pow))
        dscale = wp.float64(2.0) / fe.he
        for q in range(nqp):
            fe.q = q
            dJxW = fe_detJxW_s(wtab, fe, jac)
            gp = e * nqp + q
            for a in range(nbf):
                Na = fe_N(Ntab, fe, a)
                agw = wp.float64(0.0)
                for d in range(dim):
                    agw += aq[gp, d] * fe_dN_s(dNtab, fe, a, d, dscale)
                for b in range(nbf):
                    Nb = fe_N(Ntab, fe, b)
                    diag = sigma * Na * Nb * dJxW
                    if wp.static(USED):
                        diag += agw * Nb * dJxW      # agw genuinely used
                    for i in range(dim):
                        wp.atomic_add(r, conn[e, a] * ndof + i,
                                      diag * x[conn[e, b] * ndof + i])
    _kernel_cache[key] = km2
    return km2

dm, xq, aq, dq, x_full, lam_full = _setup(3, "cuda:0")
pv = list(dm.bins)[0]; b = dm.bins[pv]; d = dm.device
for name, used in (("agw-used", True), ("agw-dead", False)):
    k = make_m2(b["nbf"], b["nqp"], dm.dim, used)
    tape = wp.Tape()
    aqd = wp.array(np.ascontiguousarray(aq[pv]), dtype=wp.float64,
                   device=d, requires_grad=True)
    r = wp.zeros(dm.n_nodes * 3, dtype=wp.float64, device=d,
                 requires_grad=True)
    with tape:
        wp.launch(k, dim=len(b["eids"]),
                  inputs=[b["conn"], b["h"], b["N"], b["dN"], b["w"], aqd,
                          wp.float64(20.0),
                          wp.array(x_full, dtype=wp.float64, device=d), r],
                  device=d)
    tape.backward(grads={r: wp.array(lam_full, dtype=wp.float64, device=d)})
    ga = tape.gradients[aqd].numpy()
    # FD check when used
    msg = f"{name}: nan = {np.isnan(ga).sum()}/{ga.size}"
    if used and np.isfinite(ga).all():
        eps = 1e-6
        def rdot(af):
            rr = wp.zeros(dm.n_nodes * 3, dtype=wp.float64, device=d)
            wp.launch(k, dim=len(b["eids"]),
                      inputs=[b["conn"], b["h"], b["N"], b["dN"], b["w"],
                              wp.array(np.ascontiguousarray(af),
                                       dtype=wp.float64, device=d),
                              wp.float64(20.0),
                              wp.array(x_full, dtype=wp.float64, device=d),
                              rr], device=d)
            return float(lam_full @ rr.numpy())
        ap = aq[pv].copy(); ap[0, 0] += eps
        am = aq[pv].copy(); am[0, 0] -= eps
        fd = (rdot(ap) - rdot(am)) / (2 * eps)
        msg += f"  fd={fd:.6e} tape={ga[0,0]:.6e}"
    print(msg, flush=True)
