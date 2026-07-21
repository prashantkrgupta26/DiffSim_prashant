"""Task #49 — device-resident outer flexible GMRES (FGMRES).

#40 and #42 both converged on the same lever and #43 made it MANDATORY:
the rung-b film step is ~90% solve, and the residual host cost is the
host scipy `lgmres` OUTER wrapping the blockch preconditioner apply.
#40 removed launch orchestration, #42 removed per-inner-solve transfers —
each individually overlapped the async GPU queue, so neither moved the
wall.  The remaining lever is to eliminate the host<->device boundary
ENTIRELY: keep the WHOLE outer Krylov solve device-resident so ONE
region spans the outer vecops, the preconditioner apply, AND the matvec,
and the r/z host<->device copies at the apply boundaries become
intra-device / free.

This module is that outer.  A right-preconditioned (flexible) GMRES(m)
with restart whose ENTIRE state — the Krylov basis V, the preconditioned
basis Z, the Hessenberg H, the Givens rotations, the residual vector g —
lives in device arrays.  The only host contact per RESTART CYCLE is:
  * one residual-norm readback per inner iteration for the convergence
    test (the g[j+1] Givens residual estimate — the exact analogue of
    scipy lgmres's per-iteration residual, and the ONLY place a host
    scalar is needed to decide early exit);
  * the small (<= m) Hessenberg/Givens scalars, which are computed by
    1-thread device kernels and read back once per iteration to drive
    the next MGS column and the back-substitution.
The large vectors (V, Z, x, w, r — each N floats) NEVER leave the device.

Contract vs the host lgmres it replaces: FGMRES is the same flexible
right-preconditioned Arnoldi lgmres runs (scipy lgmres is GCROT/LGMRES,
a restarted flexible GMRES with an augmentation subspace; with the
augmentation off it IS restarted FGMRES).  The iterate sequence matches
the host path to few-ULP when the same preconditioner action and the
same restart length are used; where the outer-check cadence or the
LGMRES augmentation shifts the count, convergence-history equivalence is
asserted and documented (G1).

The preconditioner `apply_dev(v_in, z_out)` and the matvec
`A_matvec(z_in, w_out)` are device-in/device-out closures (wp.array ->
wp.array) supplied by the caller (linsolve.blockch_pairs_device); this
module owns only the Arnoldi/Givens machinery.
"""
import numpy as np
import warp as wp

from ..assembly.operators import _kernel_cache

_NB = 256
_BLOCK = 256


def _get(name):
    return _kernel_cache.get(("fgdev", name))


def _put(name, k):
    _kernel_cache[("fgdev", name)] = k
    return k


def _make_kernels():
    if _get("dot_partial") is not None:
        return

    @wp.kernel(module="unique", enable_backward=False)
    def dot_partial(a: wp.array(dtype=wp.float64),
                    b: wp.array(dtype=wp.float64),
                    n: wp.int32,
                    partial: wp.array(dtype=wp.float64)):
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
        # single-thread final sum (deterministic) + RE-ZERO the partial
        # buffer (invariant: partial == 0 between dots)
        s = wp.float64(0.0)
        for i in range(_NB):
            s += partial[i]
            partial[i] = wp.float64(0.0)
        scal[slot] = s

    @wp.kernel(module="unique", enable_backward=False)
    def dot_row(V: wp.array2d(dtype=wp.float64), row: wp.int32,
                w: wp.array(dtype=wp.float64), n: wp.int32,
                partial: wp.array(dtype=wp.float64)):
        # partial dot of V[row, :] . w  (rows of the 2-D basis buffer)
        t = wp.tid()
        s = wp.float64(0.0)
        i = t
        while i < n:
            s += V[row, i] * w[i]
            i += _NB * _BLOCK
        wp.atomic_add(partial, t % _NB, s)

    @wp.kernel(module="unique", enable_backward=False)
    def axpy_row(w: wp.array(dtype=wp.float64),
                 V: wp.array2d(dtype=wp.float64), row: wp.int32,
                 H: wp.array(dtype=wp.float64), hslot: wp.int32):
        # w <- w - H[hslot] * V[row, :]   (MGS projection step)
        i = wp.tid()
        w[i] = w[i] - H[hslot] * V[row, i]

    @wp.kernel(module="unique", enable_backward=False)
    def store_normed(w: wp.array(dtype=wp.float64),
                     V: wp.array2d(dtype=wp.float64), row: wp.int32,
                     scal: wp.array(dtype=wp.float64), sslot: wp.int32):
        # V[row, :] <- w / sqrt(scal[sslot])   (normalize the new basis vec)
        i = wp.tid()
        inv = wp.float64(1.0) / wp.sqrt(scal[sslot])
        V[row, i] = w[i] * inv

    @wp.kernel(module="unique", enable_backward=False)
    def copy_to_row(src: wp.array(dtype=wp.float64),
                    V: wp.array2d(dtype=wp.float64), row: wp.int32):
        i = wp.tid()
        V[row, i] = src[i]

    @wp.kernel(module="unique", enable_backward=False)
    def copy_from_row(V: wp.array2d(dtype=wp.float64), row: wp.int32,
                      dst: wp.array(dtype=wp.float64)):
        i = wp.tid()
        dst[i] = V[row, i]

    @wp.kernel(module="unique", enable_backward=False)
    def hnorm_from_scal(scal: wp.array(dtype=wp.float64), sslot: wp.int32,
                        H: wp.array(dtype=wp.float64), hslot: wp.int32):
        # H[hslot] = sqrt(scal[sslot])  (subdiagonal Hessenberg entry)
        H[hslot] = wp.sqrt(scal[sslot])

    @wp.kernel(module="unique", enable_backward=False)
    def dot_to_H(partial: wp.array(dtype=wp.float64),
                 H: wp.array(dtype=wp.float64), hslot: wp.int32):
        # final sum of a MGS dot into H[hslot], re-zero partial
        s = wp.float64(0.0)
        for i in range(_NB):
            s += partial[i]
            partial[i] = wp.float64(0.0)
        H[hslot] = s

    @wp.kernel(module="unique", enable_backward=False)
    def givens_apply_and_update(H: wp.array(dtype=wp.float64),
                                cs: wp.array(dtype=wp.float64),
                                sn: wp.array(dtype=wp.float64),
                                g: wp.array(dtype=wp.float64),
                                j: wp.int32, mp1: wp.int32):
        # single-thread: apply the j existing Givens rotations to column j
        # of H (stored contiguous, column j at [j*mp1 .. j*mp1+j+1]),
        # compute the new rotation to zero H[j+1,j], apply it to g.
        base = j * mp1
        for i in range(j):
            t0 = cs[i] * H[base + i] + sn[i] * H[base + i + 1]
            t1 = -sn[i] * H[base + i] + cs[i] * H[base + i + 1]
            H[base + i] = t0
            H[base + i + 1] = t1
        hj = H[base + j]
        hj1 = H[base + j + 1]
        denom = wp.sqrt(hj * hj + hj1 * hj1)
        if denom == wp.float64(0.0):
            cs[j] = wp.float64(1.0)
            sn[j] = wp.float64(0.0)
        else:
            cs[j] = hj / denom
            sn[j] = hj1 / denom
        H[base + j] = cs[j] * hj + sn[j] * hj1
        H[base + j + 1] = wp.float64(0.0)
        gj = g[j]
        g[j] = cs[j] * gj
        g[j + 1] = -sn[j] * gj

    @wp.kernel(module="unique", enable_backward=False)
    def back_solve(H: wp.array(dtype=wp.float64),
                   g: wp.array(dtype=wp.float64),
                   y: wp.array(dtype=wp.float64),
                   nj: wp.int32, mp1: wp.int32):
        # single-thread upper-triangular back substitution: solve
        # H[0:nj,0:nj] y = g[0:nj]  (H column-major contiguous, stride mp1)
        for ii in range(nj):
            i = nj - 1 - ii
            s = g[i]
            for k in range(i + 1, nj):
                s -= H[k * mp1 + i] * y[k]
            y[i] = s / H[i * mp1 + i]

    @wp.kernel(module="unique", enable_backward=False)
    def zero_g_set0(g: wp.array(dtype=wp.float64),
                    scal: wp.array(dtype=wp.float64), sslot: wp.int32,
                    mp1: wp.int32):
        # g <- (beta, 0, 0, ...), beta = sqrt(scal[sslot])
        for i in range(mp1):
            g[i] = wp.float64(0.0)
        g[0] = wp.sqrt(scal[sslot])

    for name, k in (
            ("dot_partial", dot_partial), ("reduce_to", reduce_to),
            ("dot_row", dot_row), ("axpy_row", axpy_row),
            ("store_normed", store_normed), ("copy_to_row", copy_to_row),
            ("copy_from_row", copy_from_row),
            ("hnorm_from_scal", hnorm_from_scal), ("dot_to_H", dot_to_H),
            ("givens_apply_and_update", givens_apply_and_update),
            ("back_solve", back_solve), ("zero_g_set0", zero_g_set0)):
        _put(name, k)


def _make_combine_kernel(m):
    """x <- x + sum_{i<nj} y[i] * Z[i, :] — one kernel, m unrolled at
    compile time so the loop bound is a constant (capturable, no host y
    readback).  nj (the actual Arnoldi length this cycle) gates the sum."""
    name = f"combine_{m}"
    k = _get(name)
    if k is not None:
        return k
    M = int(m)

    @wp.kernel(module="unique", enable_backward=False)
    def combine(x: wp.array(dtype=wp.float64),
                Z: wp.array2d(dtype=wp.float64),
                y: wp.array(dtype=wp.float64), nj: wp.int32):
        i = wp.tid()
        s = wp.float64(0.0)
        for c in range(M):
            if c < nj:
                s += y[c] * Z[c, i]
        x[i] = x[i] + s

    _put(name, combine)
    return combine


class _FGMRESWorkspace:
    """Persistent device workspace for the outer FGMRES: the Krylov basis
    V ((m+1) x N), the preconditioned basis Z (m x N), work vectors, the
    Hessenberg H (column-contiguous (m) x (m+1)), Givens cs/sn, the
    residual vector g, the small-solution y, a scalar scratch, and the
    reduction partial buffer.  Cached per (N, m, device) so the buffers
    are allocated ONCE per pattern and reused across Newton iterates /
    apply cycles (the #37/#40 residency pattern)."""

    def __init__(self, N, m, device):
        self.N = N
        self.m = m
        self.device = device
        self.V = wp.zeros((m + 1, N), dtype=wp.float64, device=device)
        self.Z = wp.zeros((m, N), dtype=wp.float64, device=device)
        self.x = wp.zeros(N, dtype=wp.float64, device=device)
        self.r = wp.zeros(N, dtype=wp.float64, device=device)
        self.w = wp.zeros(N, dtype=wp.float64, device=device)
        self.zt = wp.zeros(N, dtype=wp.float64, device=device)
        self.H = wp.zeros(m * (m + 1), dtype=wp.float64, device=device)
        self.cs = wp.zeros(m, dtype=wp.float64, device=device)
        self.sn = wp.zeros(m, dtype=wp.float64, device=device)
        self.g = wp.zeros(m + 1, dtype=wp.float64, device=device)
        self.y = wp.zeros(m, dtype=wp.float64, device=device)
        self.scal = wp.zeros(8, dtype=wp.float64, device=device)
        self.partial = wp.zeros(_NB, dtype=wp.float64, device=device)


_WS_CACHE = {}
_WS_CAP = 8


def _workspace(N, m, device):
    key = (N, m, str(device))
    ws = _WS_CACHE.get(key)
    if ws is None:
        while len(_WS_CACHE) >= _WS_CAP:
            _WS_CACHE.pop(next(iter(_WS_CACHE)))
        ws = _FGMRESWorkspace(N, m, device)
        _WS_CACHE[key] = ws
    return ws


def _dot(a, b, n, partial, scal, slot, device):
    wp.launch(_get("dot_partial"), dim=_NB * _BLOCK,
              inputs=[a, b, n, partial], device=device)
    wp.launch(_get("reduce_to"), dim=1, inputs=[partial, scal, slot],
              device=device)


def _sub(a, b, out, n, device):
    # out = a - b
    k = _get("vsub")
    if k is None:
        @wp.kernel(module="unique", enable_backward=False)
        def vsub(a: wp.array(dtype=wp.float64),
                 b: wp.array(dtype=wp.float64),
                 out: wp.array(dtype=wp.float64)):
            i = wp.tid()
            out[i] = a[i] - b[i]
        k = _put("vsub", vsub)
    wp.launch(k, dim=n, inputs=[a, b, out], device=device)


def fgmres_dev(A_matvec, b_dev, apply_dev, N, device,
               tol=1e-10, atol=1e-13, restart=30, maxiter=100,
               x0_dev=None, sync_counter=None):
    """Device-resident right-preconditioned (flexible) GMRES(restart).

    A_matvec(z_in_wp, w_out_wp): device matvec  w = A z   (in-place out).
    apply_dev(v_in_wp, z_out_wp): device precond z = M^{-1} v (in-place).
    b_dev: rhs as a device wp.array (length N).
    Returns (x_dev wp.array, info).  x_dev is the workspace's x buffer
    (device-resident); the caller downloads it ONCE (or reads on device).

    Host syncs: one residual-norm readback per inner iteration (the
    Givens residual estimate |g[j+1]|, the convergence test) + the small
    Hessenberg/Givens scalars per iteration.  No large vector ever leaves
    the device.  `maxiter` counts RESTART CYCLES (matching scipy lgmres's
    `maxiter`), each of up to `restart` inner Arnoldi steps."""
    _make_kernels()
    m = int(restart)
    ws = _workspace(N, m, device)
    combine = _make_combine_kernel(m)
    V, Z = ws.V, ws.Z
    x, r, w, zt = ws.x, ws.r, ws.w, ws.zt
    H, cs, sn, g, y = ws.H, ws.cs, ws.sn, ws.g, ws.y
    scal, partial = ws.scal, ws.partial
    mp1 = m + 1

    if x0_dev is not None:
        wp.copy(x, x0_dev)
    else:
        x.zero_()

    # ||b|| for the relative convergence threshold (one entry sync)
    _dot(b_dev, b_dev, N, partial, scal, 6, device)
    bnorm = max(float(np.sqrt(scal.numpy()[6])), 1e-300)
    if sync_counter is not None:
        sync_counter.count += 1
    thresh = max(tol * bnorm, atol)

    total_inner = [0]
    converged = False
    relres = np.inf
    outer = 0
    while outer < maxiter and not converged:
        outer += 1
        # r = b - A x   (skip the matvec on the zero initial guess)
        if x0_dev is None and outer == 1:
            wp.copy(r, b_dev)
        else:
            A_matvec(x, w)
            _sub(b_dev, w, r, N, device)   # r = b - A x
        _dot(r, r, N, partial, scal, 5, device)
        beta = float(np.sqrt(scal.numpy()[5]))
        if sync_counter is not None:
            sync_counter.count += 1
        relres = beta / bnorm
        if beta <= thresh:
            converged = True
            break
        # V[0] = r / beta ; g = (beta, 0, ...)
        wp.launch(_get("store_normed"), dim=N,
                  inputs=[r, V, 0, scal, 5], device=device)
        wp.launch(_get("zero_g_set0"), dim=1, inputs=[g, scal, 5, mp1],
                  device=device)

        nj = 0
        for j in range(m):
            # z_j = M^{-1} V[j]  (device precond); w = A z_j
            wp.launch(_get("copy_from_row"), dim=N, inputs=[V, j, zt],
                      device=device)
            apply_dev(zt, w)                 # w = M^{-1} V[j]
            wp.launch(_get("copy_to_row"), dim=N, inputs=[w, Z, j],
                      device=device)         # Z[j] = M^{-1} V[j]
            A_matvec(w, zt)                  # zt = A Z[j]
            wp.copy(w, zt)                   # w = A Z[j]  (MGS target)
            total_inner[0] += 1
            # modified Gram-Schmidt: H[i,j] = V[i].w ; w -= H[i,j] V[i]
            for i in range(j + 1):
                wp.launch(_get("dot_row"), dim=_NB * _BLOCK,
                          inputs=[V, i, w, N, partial], device=device)
                hslot = j * mp1 + i
                wp.launch(_get("dot_to_H"), dim=1,
                          inputs=[partial, H, hslot], device=device)
                wp.launch(_get("axpy_row"), dim=N,
                          inputs=[w, V, i, H, hslot], device=device)
            # H[j+1,j] = ||w|| ; V[j+1] = w / ||w||
            _dot(w, w, N, partial, scal, 0, device)
            wp.launch(_get("hnorm_from_scal"), dim=1,
                      inputs=[scal, 0, H, j * mp1 + j + 1], device=device)
            wp.launch(_get("store_normed"), dim=N,
                      inputs=[w, V, j + 1, scal, 0], device=device)
            # apply prior Givens to column j, form + apply new rotation
            wp.launch(_get("givens_apply_and_update"), dim=1,
                      inputs=[H, cs, sn, g, j, mp1], device=device)
            nj = j + 1
            # residual estimate |g[j+1]| — THE per-iteration convergence
            # readback (the only host scalar needed to decide early exit)
            resid = abs(float(g.numpy()[j + 1]))
            if sync_counter is not None:
                sync_counter.count += 1
            relres = resid / bnorm
            if resid <= thresh:
                converged = True
                break
        # back-substitute H[0:nj,0:nj] y = g[0:nj]; x += Z[:nj] y
        wp.launch(_get("back_solve"), dim=1, inputs=[H, g, y, nj, mp1],
                  device=device)
        wp.launch(combine, dim=N, inputs=[x, Z, y, nj], device=device)

    info = {"converged": converged, "relres": relres,
            "outer": outer, "inner": total_inner[0]}
    return x, info
