"""Device-resident single-sync Krylov (M1b Task 1; m1a findings 3 + spec
S16b GPU-only inner loop).

The M0 host Krylov does ~6 synchronizing dot() readbacks per iteration
(~10 ms each on WSL2) — measured fatal (71-min battery vs 89 s direct). Here
EVERY per-iteration scalar lives in a small device array; iteration
coefficients are computed by 1-thread kernels reading/writing it; the ONLY
host synchronization is the periodic convergence check (every `check_every`
iterations, one readback of the residual norm).

Reductions are two-stage: a grid-stride partial-sum kernel into
NUM_BLOCKS slots, then a single-block final sum. Deterministic for a fixed
block count (fixed reduction order per slot; final sum sequential).

Scalars array layout (FP64): [rz, pAp, alpha, beta, rz_new, rnorm2, bnorm2].

TAPE RULE (finding 4c): these kernels are never taped (solves are Tier-2
VJP boundaries) => enable_backward=False; no loop-reassigned locals are used
in any accumulation visible to an adjoint anyway.
"""
import numpy as np
import warp as wp

from ..assembly.operators import _kernel_cache

_NB = 256          # partial-reduction slots
_BLOCK = 256


def _get(name):
    return _kernel_cache.get(("kdev", name))


def _put(name, k):
    _kernel_cache[("kdev", name)] = k
    return k


def _make_kernels():
    if _get("dot_partial") is not None:
        return

    @wp.kernel(module="unique", enable_backward=False)
    def dot_partial(a: wp.array(dtype=wp.float64),
                    b: wp.array(dtype=wp.float64),
                    n: wp.int32,
                    partial: wp.array(dtype=wp.float64)):
        # grid-stride partial dot: slot t sums a[i]*b[i] for i = t, t+NB*BLK...
        t = wp.tid()
        s = wp.float64(0.0)
        i = t
        while i < n:
            s += a[i] * b[i]
            i += _NB * _BLOCK
        wp.atomic_add(partial, t % _NB, s)

    @wp.kernel(module="unique", enable_backward=False)
    def reduce_to(partial: wp.array(dtype=wp.float64),
                  scal: wp.array(dtype=wp.float64), slot: wp.int32):
        # single-thread final sum (deterministic)
        s = wp.float64(0.0)
        for i in range(_NB):
            s += partial[i]
        scal[slot] = s

    @wp.kernel(module="unique", enable_backward=False)
    def zero_arr(a: wp.array(dtype=wp.float64)):
        a[wp.tid()] = wp.float64(0.0)

    @wp.kernel(module="unique", enable_backward=False)
    def cg_alpha(scal: wp.array(dtype=wp.float64)):
        # alpha = rz / pAp
        scal[2] = scal[0] / scal[1]

    @wp.kernel(module="unique", enable_backward=False)
    def cg_update_xr(x: wp.array(dtype=wp.float64),
                     r: wp.array(dtype=wp.float64),
                     p: wp.array(dtype=wp.float64),
                     Ap: wp.array(dtype=wp.float64),
                     scal: wp.array(dtype=wp.float64)):
        i = wp.tid()
        a = scal[2]
        x[i] = x[i] + a * p[i]
        r[i] = r[i] - a * Ap[i]

    @wp.kernel(module="unique", enable_backward=False)
    def cg_beta_shift(scal: wp.array(dtype=wp.float64)):
        # beta = rz_new / rz ; rz <- rz_new ; rnorm2 mirror
        scal[3] = scal[4] / scal[0]
        scal[0] = scal[4]

    @wp.kernel(module="unique", enable_backward=False)
    def cg_update_p(p: wp.array(dtype=wp.float64),
                    z: wp.array(dtype=wp.float64),
                    scal: wp.array(dtype=wp.float64)):
        i = wp.tid()
        p[i] = z[i] + scal[3] * p[i]

    @wp.kernel(module="unique", enable_backward=False)
    def hadamard(minv: wp.array(dtype=wp.float64),
                 r: wp.array(dtype=wp.float64),
                 z: wp.array(dtype=wp.float64)):
        i = wp.tid()
        z[i] = minv[i] * r[i]

    for name, k in (("dot_partial", dot_partial), ("reduce_to", reduce_to),
                    ("zero_arr", zero_arr), ("cg_alpha", cg_alpha),
                    ("cg_update_xr", cg_update_xr),
                    ("cg_beta_shift", cg_beta_shift),
                    ("cg_update_p", cg_update_p), ("hadamard", hadamard)):
        _put(name, k)


class SyncCounter:
    """Test hook: counts host synchronizations (readbacks)."""
    def __init__(self):
        self.count = 0


def _dot_dev(a, b, n, partial, scal, slot, device):
    wp.launch(_get("zero_arr"), dim=_NB, inputs=[partial], device=device)
    wp.launch(_get("dot_partial"), dim=_NB * _BLOCK, inputs=[a, b, n, partial],
              device=device)
    wp.launch(_get("reduce_to"), dim=1, inputs=[partial, scal, slot],
              device=device)


def cg_dev(op, b, tol=1e-10, atol=1e-12, maxiter=2000, diag=None,
           check_every=10, sync_counter=None):
    """Single-sync device CG. op: .matvec(x_wp, y_wp), .n_free, .device.
    Host syncs ONLY at the periodic convergence check (and once at entry for
    bnorm). Returns (x numpy, info)."""
    _make_kernels()
    d = op.device
    n = op.n_free
    bd = wp.array(np.ascontiguousarray(b, np.float64), dtype=wp.float64,
                  device=d)
    x = wp.zeros(n, dtype=wp.float64, device=d)
    r = wp.clone(bd)
    z = wp.zeros(n, dtype=wp.float64, device=d)
    partial = wp.zeros(_NB, dtype=wp.float64, device=d)
    scal = wp.zeros(8, dtype=wp.float64, device=d)
    minv = None
    if diag is not None:
        minv = wp.array(1.0 / np.ascontiguousarray(diag, np.float64),
                        dtype=wp.float64, device=d)
        wp.launch(_get("hadamard"), dim=n, inputs=[minv, r, z], device=d)
    else:
        wp.copy(z, r)
    p = wp.clone(z)
    Ap = wp.zeros(n, dtype=wp.float64, device=d)

    _dot_dev(r, z, n, partial, scal, 0, d)          # rz
    _dot_dev(bd, bd, n, partial, scal, 6, d)        # bnorm2
    bnorm = max(np.sqrt(scal.numpy()[6]), 1e-300)   # one entry sync
    if sync_counter is not None:
        sync_counter.count += 1
    thresh2 = max(tol * bnorm, atol) ** 2

    it = 0
    rnorm2 = None
    while it < maxiter:
        # one whole check_every batch without any host sync
        for _ in range(min(check_every, maxiter - it)):
            it += 1
            op.matvec(p, Ap)
            _dot_dev(p, Ap, n, partial, scal, 1, d)
            wp.launch(_get("cg_alpha"), dim=1, inputs=[scal], device=d)
            wp.launch(_get("cg_update_xr"), dim=n, inputs=[x, r, p, Ap, scal],
                      device=d)
            if minv is not None:
                wp.launch(_get("hadamard"), dim=n, inputs=[minv, r, z],
                          device=d)
            else:
                wp.copy(z, r)
            _dot_dev(r, z, n, partial, scal, 4, d)      # rz_new
            _dot_dev(r, r, n, partial, scal, 5, d)      # rnorm2
            wp.launch(_get("cg_beta_shift"), dim=1, inputs=[scal], device=d)
            wp.launch(_get("cg_update_p"), dim=n, inputs=[p, z, scal],
                      device=d)
        rnorm2 = float(scal.numpy()[5])                 # THE periodic sync
        if sync_counter is not None:
            sync_counter.count += 1
        if rnorm2 < thresh2:
            return x.numpy(), {"iters": it,
                               "relres": np.sqrt(rnorm2) / bnorm,
                               "converged": True}
    return x.numpy(), {"iters": it,
                       "relres": np.sqrt(rnorm2 if rnorm2 is not None else np.inf) / bnorm,
                       "converged": False}


def _make_bicgstab_kernels():
    if _get("bs_beta_p") is not None:
        return
    _make_kernels()

    # scalars layout (bicgstab): [0]=rho [1]=rhat_v [2]=alpha [3]=omega
    # [4]=rho_new [5]=rnorm2 [6]=bnorm2 [7]=tt [8]=ts [9]=beta [10]=breakdown

    @wp.kernel(module="unique", enable_backward=False)
    def bs_beta(scal: wp.array(dtype=wp.float64), first: wp.int32):
        if scal[10] != wp.float64(0.0):
            return                                 # frozen after breakdown
        # beta = (rho_new/rho)(alpha/omega); breakdown flag on tiny rho_new
        # (relative to |rhat||r| ~ rnorm-scale, via bnorm2 proxy)
        eps = wp.float64(1.0e-30) * scal[6]
        if wp.abs(scal[4]) < eps:
            scal[10] = wp.float64(1.0)         # rho breakdown
        if first == 1:
            scal[9] = wp.float64(0.0)
        else:
            scal[9] = (scal[4] / scal[0]) * (scal[2] / scal[3])
        scal[0] = scal[4]

    @wp.kernel(module="unique", enable_backward=False)
    def bs_p_update(p: wp.array(dtype=wp.float64),
                    r: wp.array(dtype=wp.float64),
                    v: wp.array(dtype=wp.float64),
                    scal: wp.array(dtype=wp.float64)):
        i = wp.tid()
        if scal[10] != wp.float64(0.0):
            return
        p[i] = r[i] + scal[9] * (p[i] - scal[3] * v[i])

    @wp.kernel(module="unique", enable_backward=False)
    def bs_alpha(scal: wp.array(dtype=wp.float64)):
        if scal[10] != wp.float64(0.0):
            return
        eps = wp.float64(1.0e-30) * scal[6]
        if wp.abs(scal[1]) < eps:
            scal[10] = wp.float64(2.0)         # rhat_v breakdown
            scal[2] = wp.float64(0.0)
        else:
            scal[2] = scal[0] / scal[1]

    @wp.kernel(module="unique", enable_backward=False)
    def bs_s_update(s: wp.array(dtype=wp.float64),
                    r: wp.array(dtype=wp.float64),
                    v: wp.array(dtype=wp.float64),
                    scal: wp.array(dtype=wp.float64)):
        i = wp.tid()
        if scal[10] != wp.float64(0.0):
            return
        s[i] = r[i] - scal[2] * v[i]

    @wp.kernel(module="unique", enable_backward=False)
    def bs_omega(scal: wp.array(dtype=wp.float64)):
        if scal[10] != wp.float64(0.0):
            return
        if scal[7] > wp.float64(0.0):
            scal[3] = scal[8] / scal[7]
        else:
            scal[3] = wp.float64(0.0)

    @wp.kernel(module="unique", enable_backward=False)
    def bs_xr_update(x: wp.array(dtype=wp.float64),
                     r: wp.array(dtype=wp.float64),
                     ph: wp.array(dtype=wp.float64),
                     sh: wp.array(dtype=wp.float64),
                     s: wp.array(dtype=wp.float64),
                     t: wp.array(dtype=wp.float64),
                     scal: wp.array(dtype=wp.float64)):
        i = wp.tid()
        if scal[10] != wp.float64(0.0):
            return
        x[i] = x[i] + scal[2] * ph[i] + scal[3] * sh[i]
        r[i] = s[i] - scal[3] * t[i]

    for name, k in (("bs_beta", bs_beta), ("bs_p_update", bs_p_update),
                    ("bs_alpha", bs_alpha), ("bs_s_update", bs_s_update),
                    ("bs_omega", bs_omega), ("bs_xr_update", bs_xr_update)):
        _put(name, k)


def bicgstab_dev(op, b, tol=1e-10, atol=1e-12, maxiter=2000, diag=None,
                 check_every=10, sync_counter=None, max_restarts=50):
    """Single-sync device BiCGStab (Jacobi-preconditioned). Breakdown guards
    live ON DEVICE (scalars[10]) and FREEZE all update kernels, so the state
    at the periodic host check is the last pre-breakdown iterate; the host
    then RESTARTS (rhat <- r, scalars reset) up to max_restarts times — the
    standard cure for rho-breakdown on hard nonsymmetric systems (measured
    on the L6 cavity monolithic block). Same contract as krylov.bicgstab."""
    _make_bicgstab_kernels()
    d = op.device
    n = op.n_free
    bd = wp.array(np.ascontiguousarray(b, np.float64), dtype=wp.float64,
                  device=d)
    x = wp.zeros(n, dtype=wp.float64, device=d)
    r = wp.clone(bd)
    rhat = wp.clone(bd)
    p = wp.zeros(n, dtype=wp.float64, device=d)
    v = wp.zeros(n, dtype=wp.float64, device=d)
    s = wp.zeros(n, dtype=wp.float64, device=d)
    t = wp.zeros(n, dtype=wp.float64, device=d)
    ph = wp.zeros(n, dtype=wp.float64, device=d)
    sh = wp.zeros(n, dtype=wp.float64, device=d)
    partial = wp.zeros(_NB, dtype=wp.float64, device=d)
    scal = wp.zeros(12, dtype=wp.float64, device=d)
    minv = None
    if diag is not None:
        minv = wp.array(1.0 / np.ascontiguousarray(diag, np.float64),
                        dtype=wp.float64, device=d)

    def precond(dst, src):
        if minv is not None:
            wp.launch(_get("hadamard"), dim=n, inputs=[minv, src, dst],
                      device=d)
        else:
            wp.copy(dst, src)

    _dot_dev(bd, bd, n, partial, scal, 6, d)
    bnorm = max(np.sqrt(scal.numpy()[6]), 1e-300)
    if sync_counter is not None:
        sync_counter.count += 1
    thresh2 = max(tol * bnorm, atol) ** 2
    # rho/alpha/omega start at 1 by convention
    ones = np.zeros(12); ones[0] = ones[2] = ones[3] = 1.0
    ones[6] = bnorm ** 2
    wp.copy(scal, wp.array(ones, dtype=wp.float64, device=d))

    it = 0
    first = 1
    restarts = 0
    while it < maxiter:
        for _ in range(min(check_every, maxiter - it)):
            it += 1
            _dot_dev(rhat, r, n, partial, scal, 4, d)       # rho_new
            wp.launch(_get("bs_beta"), dim=1, inputs=[scal, first], device=d)
            first = 0
            wp.launch(_get("bs_p_update"), dim=n, inputs=[p, r, v, scal],
                      device=d)
            precond(ph, p)
            op.matvec(ph, v)
            _dot_dev(rhat, v, n, partial, scal, 1, d)       # rhat_v
            wp.launch(_get("bs_alpha"), dim=1, inputs=[scal], device=d)
            wp.launch(_get("bs_s_update"), dim=n, inputs=[s, r, v, scal],
                      device=d)
            precond(sh, s)
            op.matvec(sh, t)
            _dot_dev(t, t, n, partial, scal, 7, d)          # tt
            _dot_dev(t, s, n, partial, scal, 8, d)          # ts
            wp.launch(_get("bs_omega"), dim=1, inputs=[scal], device=d)
            wp.launch(_get("bs_xr_update"), dim=n,
                      inputs=[x, r, ph, sh, s, t, scal], device=d)
            _dot_dev(r, r, n, partial, scal, 5, d)          # rnorm2
        vals = scal.numpy()                                 # THE periodic sync
        if sync_counter is not None:
            sync_counter.count += 1
        rnorm2, flag = float(vals[5]), float(vals[10])
        if flag != 0.0:
            if restarts < max_restarts:
                restarts += 1
                wp.copy(rhat, r)                    # restart from last good r
                st = np.zeros(12)
                st[0] = st[2] = st[3] = 1.0
                st[5] = rnorm2
                st[6] = bnorm ** 2
                wp.copy(scal, wp.array(st, dtype=wp.float64, device=d))
                first = 1
                continue
            return x.numpy(), {"iters": it,
                               "relres": np.sqrt(rnorm2) / bnorm,
                               "converged": False, "restarts": restarts,
                               "breakdown": "rho" if flag == 1.0 else "rhat_v"}
        if rnorm2 < thresh2:
            return x.numpy(), {"iters": it,
                               "relres": np.sqrt(rnorm2) / bnorm,
                               "converged": True}
    return x.numpy(), {"iters": it, "relres": np.sqrt(rnorm2) / bnorm,
                       "converged": False}
