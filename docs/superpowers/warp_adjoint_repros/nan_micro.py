"""Micro-bisect: mass-only kernels with preamble pieces toggled."""
import numpy as np, warp as wp, sys
sys.path.insert(0, "tests")
from test_ns_adjoint import _setup
from diffsim.assembly.femelm import FEMElm, fe_N, fe_dN_s, fe_detJxW_s
from diffsim.assembly.operators import _kernel_cache

def make_micro(nbf, nqp, dim, DEAD_AGW, DEAD_SQRT):
    key = ("nan_micro", DEAD_AGW, DEAD_SQRT)
    if key in _kernel_cache:
        return _kernel_cache[key]
    ndof = dim + 1
    dim_pow = float(dim)
    @wp.kernel(module="unique", enable_backward=True)
    def kmicro(conn: wp.array2d(dtype=wp.int32),
               h: wp.array(dtype=wp.float64),
               Ntab: wp.array2d(dtype=wp.float64),
               dNtab: wp.array3d(dtype=wp.float64),
               wtab: wp.array(dtype=wp.float64),
               aq: wp.array2d(dtype=wp.float64),
               nu_arr: wp.array(dtype=wp.float64),
               sigma: wp.float64,
               x: wp.array(dtype=wp.float64),
               r: wp.array(dtype=wp.float64)):
        e = wp.tid()
        fe = FEMElm(); fe.e = e; fe.he = h[e]
        jac = wp.pow(fe.he * wp.float64(0.5), wp.float64(dim_pow))
        dscale = wp.float64(2.0) / fe.he
        for q in range(nqp):
            fe.q = q
            dJxW = fe_detJxW_s(wtab, fe, jac)
            gp = e * nqp + q
            if wp.static(DEAD_SQRT):
                amag = wp.float64(0.0)
                for d in range(dim):
                    amag += aq[gp, d] * aq[gp, d]
                amag = wp.sqrt(amag)      # dead: never used below
            for a in range(nbf):
                Na = fe_N(Ntab, fe, a)
                if wp.static(DEAD_AGW):
                    agw = wp.float64(0.0)
                    for d in range(dim):
                        agw += aq[gp, d] * fe_dN_s(dNtab, fe, a, d, dscale)
                for b in range(nbf):
                    Nb = fe_N(Ntab, fe, b)
                    diag = sigma * Na * Nb * dJxW
                    for i in range(dim):
                        wp.atomic_add(r, conn[e, a] * ndof + i,
                                      diag * x[conn[e, b] * ndof + i])
    _kernel_cache[key] = kmicro
    return kmicro

dm, xq, aq, dq, x_full, lam_full = _setup(3, "cuda:0")
pv = list(dm.bins)[0]; b = dm.bins[pv]; d = dm.device
for name, flags in (("bare", (0, 0)), ("dead-agw", (1, 0)),
                    ("dead-sqrt", (0, 1)), ("both", (1, 1))):
    k = make_micro(b["nbf"], b["nqp"], dm.dim, *[bool(f) for f in flags])
    tape = wp.Tape()
    aqd = wp.array(np.ascontiguousarray(aq[pv]), dtype=wp.float64,
                   device=d, requires_grad=True)
    nua = wp.array(np.array([0.05]), dtype=wp.float64, device=d,
                   requires_grad=True)
    r = wp.zeros(dm.n_nodes * 3, dtype=wp.float64, device=d,
                 requires_grad=True)
    with tape:
        wp.launch(k, dim=len(b["eids"]),
                  inputs=[b["conn"], b["h"], b["N"], b["dN"], b["w"], aqd,
                          nua, wp.float64(20.0),
                          wp.array(x_full, dtype=wp.float64, device=d), r],
                  device=d)
    tape.backward(grads={r: wp.array(lam_full, dtype=wp.float64, device=d)})
    ga = tape.gradients[aqd].numpy()
    print(f"{name}: aq-grad nan = {np.isnan(ga).sum()}/{ga.size} "
          f"max|g| = {np.nanmax(np.abs(ga)):.2e}", flush=True)
