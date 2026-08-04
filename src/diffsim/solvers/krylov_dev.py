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

Task #40 (fused/graph-captured inner loop): the legacy loop above costs
15 (CG) / 22 (BiCGStab) launches per iteration — measured ~half the
rung-b solve bucket as per-launch Python overhead (#37 profile: 1.18M
launches/53 s, 236k _dot_dev/32 s).  The FUSED path folds the scalar
bookkeeping kernels into the reduction tails (the single-thread final
reduce also computes alpha/beta/omega and RE-ZEROS the partial buffer,
restoring the invariant "partial == 0 between dots" so the per-dot
zero_arr launch disappears) and merges back-to-back dots over the same
vector into one dual-dot pass: 6 (CG) / 13 (BiCGStab) launches per
iteration, bit-identical arithmetic per partial-sum slot.  On top of
that the whole readback-free check_every batch is CUDA-graph captured
once per (operator-buffers, n, check_every) key and replayed — one
Python call per batch.  Workspace + graph live in a module cache keyed
by the operator's device-buffer pointers (the blockch pair buffers are
persistent and refilled in place, #37 pattern, so the graph stays valid
across Newton iterates); the workspace holds strong references to those
buffers so a pointer can never be recycled under a live graph.

Knob: graph="auto"|"off"|"fused"|"graph" per solve, default from
DIFFSIM_KRYLOV_GRAPH or "auto".  auto = fused+captured on CUDA, LEGACY
path (bit-for-bit today) on CPU; off = legacy everywhere; fused =
fused kernels without capture (any device); graph = fused+captured on
any device (CPU uses warp 1.15 APIC record/replay — test hook; verified
capture records WITHOUT executing on both backends).
"""
import os

import numpy as np
import warp as wp

from ..assembly.operators import _kernel_cache

_NB = 256          # partial-reduction slots
_BLOCK = 256

_GRAPH_MODES = ("auto", "off", "fused", "graph")
_graph_mode = None          # None -> DIFFSIM_KRYLOV_GRAPH or "auto"


def set_krylov_graph(mode):
    """Set the module-wide default for the Task-#40 inner-loop path."""
    global _graph_mode
    if mode is not None and mode not in _GRAPH_MODES:
        raise ValueError(f"krylov_graph must be one of {_GRAPH_MODES}")
    _graph_mode = mode


def krylov_graph_mode():
    if _graph_mode is not None:
        return _graph_mode
    env = os.environ.get("DIFFSIM_KRYLOV_GRAPH", "auto")
    return env if env in _GRAPH_MODES else "auto"


def _resolve_path(graph, device):
    """-> (use_fused, use_capture) for this solve."""
    mode = graph if graph is not None else krylov_graph_mode()
    if mode not in _GRAPH_MODES:
        raise ValueError(f"krylov_graph must be one of {_GRAPH_MODES}")
    is_cuda = str(device).startswith("cuda")
    if mode == "off":
        return False, False
    if mode == "fused":
        return True, False
    if mode == "graph":
        return True, True
    return (True, True) if is_cuda else (False, False)   # auto


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


# ---------------------------------------------------------------------------
# Task #40: fused kernels — reduction tails absorb the scalar bookkeeping
# kernels and re-zero the partial buffer (invariant: partial == 0 between
# dots), dual-dot merges back-to-back dots.  Per-slot accumulation pattern
# is IDENTICAL to the legacy dot_partial/reduce_to pair (same thread ->
# slot mapping, same per-thread strided sub-sums, single-thread final sum
# in slot order), so fused results match the legacy loop bit-for-bit on a
# serial backend and to atomic-scheduling ULP on CUDA.
# ---------------------------------------------------------------------------

def _make_fused_kernels():
    if _get("fdot2") is not None:
        return
    _make_kernels()
    _make_bicgstab_kernels()

    @wp.kernel(module="unique", enable_backward=False)
    def fdot2(a: wp.array(dtype=wp.float64),
              b: wp.array(dtype=wp.float64),
              c: wp.array(dtype=wp.float64),
              e: wp.array(dtype=wp.float64),
              n: wp.int32,
              partial: wp.array(dtype=wp.float64)):
        # dual grid-stride dot: slots [0,NB) accumulate a.b, [NB,2NB) c.e
        t = wp.tid()
        s1 = wp.float64(0.0)
        s2 = wp.float64(0.0)
        i = t
        while i < n:
            s1 += a[i] * b[i]
            s2 += c[i] * e[i]
            i += _NB * _BLOCK
        wp.atomic_add(partial, t % _NB, s1)
        wp.atomic_add(partial, _NB + t % _NB, s2)

    @wp.kernel(module="unique", enable_backward=False)
    def freduce_to(partial: wp.array(dtype=wp.float64),
                   scal: wp.array(dtype=wp.float64), slot: wp.int32):
        # final sum + RE-ZERO (keeps the partial-buffer invariant)
        s = wp.float64(0.0)
        for i in range(_NB):
            s += partial[i]
            partial[i] = wp.float64(0.0)
        scal[slot] = s

    @wp.kernel(module="unique", enable_backward=False)
    def fcg_red_pap_alpha(partial: wp.array(dtype=wp.float64),
                          scal: wp.array(dtype=wp.float64)):
        # pAp = sum(partial); alpha = rz / pAp   (absorbs cg_alpha)
        s = wp.float64(0.0)
        for i in range(_NB):
            s += partial[i]
            partial[i] = wp.float64(0.0)
        scal[1] = s
        scal[2] = scal[0] / s

    @wp.kernel(module="unique", enable_backward=False)
    def fcg_update_xr_z(x: wp.array(dtype=wp.float64),
                        r: wp.array(dtype=wp.float64),
                        p: wp.array(dtype=wp.float64),
                        Ap: wp.array(dtype=wp.float64),
                        minv: wp.array(dtype=wp.float64),
                        z: wp.array(dtype=wp.float64),
                        scal: wp.array(dtype=wp.float64)):
        # x/r update + Jacobi apply in one pass (absorbs hadamard)
        i = wp.tid()
        a = scal[2]
        x[i] = x[i] + a * p[i]
        ri = r[i] - a * Ap[i]
        r[i] = ri
        z[i] = minv[i] * ri

    @wp.kernel(module="unique", enable_backward=False)
    def fcg_red_rz_beta(partial: wp.array(dtype=wp.float64),
                        scal: wp.array(dtype=wp.float64)):
        # rz_new = sum[0,NB); rnorm2 = sum[NB,2NB); beta = rz_new/rz;
        # rz <- rz_new   (absorbs cg_beta_shift)
        s1 = wp.float64(0.0)
        s2 = wp.float64(0.0)
        for i in range(_NB):
            s1 += partial[i]
            partial[i] = wp.float64(0.0)
        for i in range(_NB, 2 * _NB):
            s2 += partial[i]
            partial[i] = wp.float64(0.0)
        scal[4] = s1
        scal[5] = s2
        scal[3] = s1 / scal[0]
        scal[0] = s1

    # -- bicgstab fused tails; scal[11] = device-side "first" flag so the
    #    batch is uniform (capturable across restarts) --------------------

    @wp.kernel(module="unique", enable_backward=False)
    def fbs_red_rho_beta(partial: wp.array(dtype=wp.float64),
                         scal: wp.array(dtype=wp.float64)):
        s = wp.float64(0.0)
        for i in range(_NB):
            s += partial[i]
            partial[i] = wp.float64(0.0)
        scal[4] = s                      # rho_new (written even when frozen,
        if scal[10] != wp.float64(0.0):  # like the legacy reduce_to)
            return
        eps = wp.float64(1.0e-12) * scal[5]
        if wp.abs(scal[4]) < eps:
            scal[10] = wp.float64(1.0)   # rho breakdown
        if scal[11] != wp.float64(0.0):
            scal[9] = wp.float64(0.0)
            scal[11] = wp.float64(0.0)
        else:
            scal[9] = (scal[4] / scal[0]) * (scal[2] / scal[3])
        scal[0] = scal[4]

    @wp.kernel(module="unique", enable_backward=False)
    def fbs_p_update_prec(p: wp.array(dtype=wp.float64),
                          r: wp.array(dtype=wp.float64),
                          v: wp.array(dtype=wp.float64),
                          minv: wp.array(dtype=wp.float64),
                          ph: wp.array(dtype=wp.float64),
                          scal: wp.array(dtype=wp.float64)):
        i = wp.tid()
        if scal[10] != wp.float64(0.0):
            return                       # frozen: p unchanged => ph would
        pi = r[i] + scal[9] * (p[i] - scal[3] * v[i])   # recompute equal
        p[i] = pi
        ph[i] = minv[i] * pi

    @wp.kernel(module="unique", enable_backward=False)
    def fbs_red_rhatv_alpha(partial: wp.array(dtype=wp.float64),
                            scal: wp.array(dtype=wp.float64)):
        s = wp.float64(0.0)
        for i in range(_NB):
            s += partial[i]
            partial[i] = wp.float64(0.0)
        scal[1] = s
        if scal[10] != wp.float64(0.0):
            return
        eps = wp.float64(1.0e-12) * scal[5]
        if wp.abs(scal[1]) < eps:
            scal[10] = wp.float64(2.0)   # rhat_v breakdown
            scal[2] = wp.float64(0.0)
        else:
            scal[2] = scal[0] / scal[1]

    @wp.kernel(module="unique", enable_backward=False)
    def fbs_s_update_prec(s: wp.array(dtype=wp.float64),
                          r: wp.array(dtype=wp.float64),
                          v: wp.array(dtype=wp.float64),
                          minv: wp.array(dtype=wp.float64),
                          sh: wp.array(dtype=wp.float64),
                          scal: wp.array(dtype=wp.float64)):
        i = wp.tid()
        if scal[10] != wp.float64(0.0):
            return
        si = r[i] - scal[2] * v[i]
        s[i] = si
        sh[i] = minv[i] * si

    @wp.kernel(module="unique", enable_backward=False)
    def fbs_red_tt_ts_omega(partial: wp.array(dtype=wp.float64),
                            scal: wp.array(dtype=wp.float64)):
        s1 = wp.float64(0.0)
        s2 = wp.float64(0.0)
        for i in range(_NB):
            s1 += partial[i]
            partial[i] = wp.float64(0.0)
        for i in range(_NB, 2 * _NB):
            s2 += partial[i]
            partial[i] = wp.float64(0.0)
        scal[7] = s1
        scal[8] = s2
        if scal[10] != wp.float64(0.0):
            return
        if scal[7] > wp.float64(0.0):
            scal[3] = scal[8] / scal[7]
        else:
            scal[3] = wp.float64(0.0)

    for name, k in (("fdot2", fdot2), ("freduce_to", freduce_to),
                    ("fcg_red_pap_alpha", fcg_red_pap_alpha),
                    ("fcg_update_xr_z", fcg_update_xr_z),
                    ("fcg_red_rz_beta", fcg_red_rz_beta),
                    ("fbs_red_rho_beta", fbs_red_rho_beta),
                    ("fbs_p_update_prec", fbs_p_update_prec),
                    ("fbs_red_rhatv_alpha", fbs_red_rhatv_alpha),
                    ("fbs_s_update_prec", fbs_s_update_prec),
                    ("fbs_red_tt_ts_omega", fbs_red_tt_ts_omega)):
        _put(name, k)


def _fdot(a, b, n, partial, device):
    # partial is zero on entry (invariant); caller launches a fused tail
    wp.launch(_get("dot_partial"), dim=_NB * _BLOCK,
              inputs=[a, b, n, partial], device=device)


class _KrylovWorkspace:
    """Persistent per-(kind, operator-buffers, n, check_every) device
    workspace + captured graph.  Holds STRONG references to the
    operator's device buffers (op_dev) so their pointers cannot be freed
    and recycled while a captured graph referencing them is alive."""

    def __init__(self, kind, op, n, device, check_every):
        nv = 5 if kind == "cg" else 9
        self.vecs = [wp.zeros(n, dtype=wp.float64, device=device)
                     for _ in range(nv)]
        self.partial = wp.zeros(2 * _NB, dtype=wp.float64, device=device)
        self.scal = wp.zeros(12, dtype=wp.float64, device=device)
        self.minv = wp.zeros(n, dtype=wp.float64, device=device)
        self.op_dev = op._dev            # strong refs (pointer stability)
        self.graph = None
        self.graph_failed = False


_WS_CACHE = {}
_WS_CAP = 32          # FIFO cap: bounds retained device memory


def _fusable(op):
    """The fused/captured path requires a CSROperator-shaped op: stable
    device buffers (`_dev`) to key the workspace/graph on and a known
    launch-only `matvec` (`_spmv`).  Arbitrary operator-protocol objects
    (matrix-free / constrained / test wrappers) take the legacy loop —
    their matvec closures are not guaranteed capture-safe."""
    return hasattr(op, "_dev") and hasattr(op, "_spmv")


def _op_key(op):
    parts = []
    for a in op._dev:
        parts.append(a.ptr if isinstance(a, wp.array) else ("s", str(a)))
    return (id(op._spmv), tuple(parts))


def _workspace(kind, op, n, device, check_every):
    key = (kind, str(device), n, check_every, _op_key(op))
    ws = _WS_CACHE.get(key)
    if ws is None:
        while len(_WS_CACHE) >= _WS_CAP:
            _WS_CACHE.pop(next(iter(_WS_CACHE)))
        ws = _KrylovWorkspace(kind, op, n, device, check_every)
        _WS_CACHE[key] = ws
    return ws


def _try_capture(ws, device, batch, check_every):
    """Capture one readback-free check_every batch into ws.graph.
    Capture records WITHOUT executing (verified on both backends), so
    the caller replays the graph for every full batch including the
    first.  Any failure disables capture for this workspace and the
    solve proceeds on the fused launch path."""
    if ws.graph is not None or ws.graph_failed:
        return
    try:
        wp.capture_begin(device)
        ok = False
        try:
            batch(check_every)
            ok = True
        finally:
            try:
                g = wp.capture_end(device)
            except Exception:
                if ok:
                    raise
                g = None
        ws.graph = g
    except Exception:
        ws.graph = None
        ws.graph_failed = True


def _upload_into(dst, host_vec, n):
    wp.copy(dst, wp.array(np.ascontiguousarray(host_vec, np.float64),
                          dtype=wp.float64, device="cpu"), count=n)


def _cg_fused(op, b, tol, atol, maxiter, diag, check_every, sync_counter,
              capture, b_dev=None, x_out=None, diag_dev=None,
              fixed_iters=None):
    """Task-#40 fused (and optionally graph-captured) CG inner loop.
    Iterate-identical to the legacy cg_dev loop: same kernels-per-slot
    arithmetic, same batch boundaries, same convergence checks — 6
    launches/iteration instead of 15, one capture_launch per batch when
    captured.

    Task #42 device-resident options (all default off = today's path):
      b_dev      — rhs already on device (wp.array); skips the host upload.
      x_out      — device buffer to write the solution into; skips the
                   `x.numpy()` download (returns None for x, and the
                   caller reads x_out on device).
      diag_dev   — Jacobi diag already inverted on device (1/diag as a
                   wp.array in `minv`); skips the diag upload.
      fixed_iters — run EXACTLY this many iterations with NO convergence
                   readback (collapse the per-check_every sync to nothing:
                   the outer FGMRES residual is the only convergence gate).
                   Must be a multiple of check_every so a single captured
                   graph covers the whole budget."""
    _make_fused_kernels()
    d = op.device
    n = op.n_free
    ws = _workspace("cg", op, n, d, check_every)
    x, r, z, p, Ap = ws.vecs
    partial, scal, minv = ws.partial, ws.scal, ws.minv
    if b_dev is not None:
        bd = b_dev                                      # no host upload
    else:
        bd = wp.array(np.ascontiguousarray(b, np.float64), dtype=wp.float64,
                      device=d)
    if diag_dev is not None:
        wp.copy(minv, diag_dev)                         # already 1/diag
    else:
        _upload_into(minv, 1.0 / np.ascontiguousarray(diag, np.float64), n)
    x.zero_()
    wp.copy(r, bd)
    wp.launch(_get("hadamard"), dim=n, inputs=[minv, r, z], device=d)
    wp.copy(p, z)

    _fdot(r, z, n, partial, d)
    wp.launch(_get("freduce_to"), dim=1, inputs=[partial, scal, 0],
              device=d)                                 # rz

    def batch(k):
        for _ in range(k):
            op.matvec(p, Ap)
            _fdot(p, Ap, n, partial, d)
            wp.launch(_get("fcg_red_pap_alpha"), dim=1,
                      inputs=[partial, scal], device=d)
            wp.launch(_get("fcg_update_xr_z"), dim=n,
                      inputs=[x, r, p, Ap, minv, z, scal], device=d)
            wp.launch(_get("fdot2"), dim=_NB * _BLOCK,
                      inputs=[r, z, r, r, n, partial], device=d)
            wp.launch(_get("fcg_red_rz_beta"), dim=1,
                      inputs=[partial, scal], device=d)
            wp.launch(_get("cg_update_p"), dim=n, inputs=[p, z, scal],
                      device=d)

    if fixed_iters is not None:
        # Task #42: readback-free fixed-budget inner solve.  Replay the
        # captured check_every batch fixed_iters/check_every times (the
        # budget is a check_every multiple); NO scal readback at all.
        if capture:
            _try_capture(ws, d, batch, check_every)
        it = 0
        while it < fixed_iters:
            k = min(check_every, fixed_iters - it)
            if ws.graph is not None and k == check_every:
                wp.capture_launch(ws.graph)
            else:
                batch(k)
            it += k
        info = {"iters": it, "converged": True, "fixed": True,
                "graph": ws.graph is not None}
        if x_out is not None:
            wp.copy(x_out, x)                # persist out of the shared ws
            return None, info
        return x.numpy(), info

    _fdot(bd, bd, n, partial, d)
    wp.launch(_get("freduce_to"), dim=1, inputs=[partial, scal, 6],
              device=d)                                 # bnorm2
    bnorm = max(np.sqrt(scal.numpy()[6]), 1e-300)       # one entry sync
    if sync_counter is not None:
        sync_counter.count += 1
    thresh2 = max(tol * bnorm, atol) ** 2

    if capture:
        _try_capture(ws, d, batch, check_every)

    it = 0
    rnorm2 = None
    while it < maxiter:
        k = min(check_every, maxiter - it)
        if ws.graph is not None and k == check_every:
            wp.capture_launch(ws.graph)
        else:
            batch(k)
        it += k
        rnorm2 = float(scal.numpy()[5])                 # THE periodic sync
        if sync_counter is not None:
            sync_counter.count += 1
        if rnorm2 < thresh2:
            info = {"iters": it, "relres": np.sqrt(rnorm2) / bnorm,
                    "converged": True, "graph": ws.graph is not None}
            if x_out is not None:
                wp.copy(x_out, x)
                return None, info
            return x.numpy(), info
    info = {"iters": it,
            "relres": np.sqrt(rnorm2 if rnorm2 is not None
                              else np.inf) / bnorm,
            "converged": False, "graph": ws.graph is not None}
    if x_out is not None:
        wp.copy(x_out, x)
        return None, info
    return x.numpy(), info


def cg_dev(op, b, tol=1e-10, atol=1e-12, maxiter=2000, diag=None,
           check_every=10, sync_counter=None, graph=None,
           b_dev=None, x_out=None, diag_dev=None, fixed_iters=None):
    """Single-sync device CG. op: .matvec(x_wp, y_wp), .n_free, .device.
    Host syncs ONLY at the periodic convergence check (and once at entry for
    bnorm). Returns (x numpy, info).

    graph: Task-#40 knob — "auto" (default; fused+captured on CUDA,
    legacy on CPU), "off" (legacy loop bit-for-bit), "fused",
    "graph" (capture on any device).  diag=None or a non-CSROperator op
    (matrix-free protocol objects) always takes the legacy loop
    (production blockch/fused-backend inners are Jacobi-preconditioned
    CSROperators).

    Task #42 device-resident options (b_dev / x_out / diag_dev /
    fixed_iters): keep the rhs and solution on device across the whole
    blockch apply so the per-inner-solve upload/download syncs collapse
    to the outer FGMRES check cadence.  Only honored on the fused/CSR
    path (the production inner); a request for these on the legacy path
    is a programming error."""
    fused, cap = _resolve_path(graph, op.device)
    dev_resident = (b_dev is not None or x_out is not None
                    or diag_dev is not None or fixed_iters is not None)
    if dev_resident and not (fused and _fusable(op)):
        raise ValueError("cg_dev device-resident args require the fused "
                         "CSROperator path (graph!='off', diag on a "
                         "CSROperator)")
    if fused and (diag is not None or diag_dev is not None) \
            and _fusable(op):
        return _cg_fused(op, b, tol, atol, maxiter, diag, check_every,
                         sync_counter, cap, b_dev=b_dev, x_out=x_out,
                         diag_dev=diag_dev, fixed_iters=fixed_iters)
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
        # beta = (rho_new/rho)(alpha/omega); breakdown = rho_new tiny
        # RELATIVE TO THE CURRENT RESIDUAL (scal[5] = rnorm2): near
        # convergence rho shrinks legitimately with r — scaling against
        # bnorm^2 misfires there (measured: 'breakdown' at relres 1e-27)
        eps = wp.float64(1.0e-12) * scal[5]
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
        eps = wp.float64(1.0e-12) * scal[5]
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


def _bicgstab_fused(op, b, tol, atol, maxiter, diag, check_every,
                    sync_counter, max_restarts, capture,
                    b_dev=None, x_out=None, diag_dev=None,
                    fixed_iters=None):
    """Task-#40 fused (and optionally graph-captured) BiCGStab.
    Iterate-identical to the legacy loop (same per-slot arithmetic and
    freeze-on-breakdown semantics); the "first" flag moves to the device
    (scal[11]) so every batch is uniform and ONE captured graph covers
    entry and post-restart batches alike.  13 launches/iteration instead
    of 22.

    Task #42 device-resident options (b_dev/x_out/diag_dev/fixed_iters):
    see _cg_fused.  In fixed_iters mode there is NO readback, so the
    on-device breakdown FREEZE (state held at the last good iterate) is
    the only breakdown handling — no host restart; a frozen inner is a
    valid degraded preconditioner and the outer FGMRES still converges."""
    _make_fused_kernels()
    d = op.device
    n = op.n_free
    ws = _workspace("bs", op, n, d, check_every)
    x, r, rhat, p, v, s, t, ph, sh = ws.vecs
    partial, scal, minv = ws.partial, ws.scal, ws.minv
    if b_dev is not None:
        bd = b_dev                                      # no host upload
    else:
        bd = wp.array(np.ascontiguousarray(b, np.float64), dtype=wp.float64,
                      device=d)
    if diag_dev is not None:
        wp.copy(minv, diag_dev)                         # already 1/diag
    else:
        _upload_into(minv, 1.0 / np.ascontiguousarray(diag, np.float64), n)
    x.zero_()
    wp.copy(r, bd)
    wp.copy(rhat, bd)
    for a_ in (p, v, s, t, ph, sh):
        a_.zero_()

    _fdot(bd, bd, n, partial, d)
    wp.launch(_get("freduce_to"), dim=1, inputs=[partial, scal, 6],
              device=d)
    bnorm = max(np.sqrt(scal.numpy()[6]), 1e-300)
    if sync_counter is not None:
        sync_counter.count += 1
    thresh2 = max(tol * bnorm, atol) ** 2

    def _reset_scal(rn2):
        st = np.zeros(12)
        st[0] = st[2] = st[3] = 1.0     # rho/alpha/omega convention
        st[5] = rn2                     # breakdown thresholds are relative
        st[6] = bnorm ** 2              # to scal[5]; zero would disarm them
        st[11] = 1.0                    # device-side "first" flag
        _upload_into(scal, st, 12)

    _reset_scal(bnorm ** 2)

    def batch(k):
        for _ in range(k):
            _fdot(rhat, r, n, partial, d)
            wp.launch(_get("fbs_red_rho_beta"), dim=1,
                      inputs=[partial, scal], device=d)
            wp.launch(_get("fbs_p_update_prec"), dim=n,
                      inputs=[p, r, v, minv, ph, scal], device=d)
            op.matvec(ph, v)
            _fdot(rhat, v, n, partial, d)
            wp.launch(_get("fbs_red_rhatv_alpha"), dim=1,
                      inputs=[partial, scal], device=d)
            wp.launch(_get("fbs_s_update_prec"), dim=n,
                      inputs=[s, r, v, minv, sh, scal], device=d)
            op.matvec(sh, t)
            wp.launch(_get("fdot2"), dim=_NB * _BLOCK,
                      inputs=[t, t, t, s, n, partial], device=d)
            wp.launch(_get("fbs_red_tt_ts_omega"), dim=1,
                      inputs=[partial, scal], device=d)
            wp.launch(_get("bs_xr_update"), dim=n,
                      inputs=[x, r, ph, sh, s, t, scal], device=d)
            _fdot(r, r, n, partial, d)
            wp.launch(_get("freduce_to"), dim=1, inputs=[partial, scal, 5],
                      device=d)

    if capture:
        _try_capture(ws, d, batch, check_every)

    if fixed_iters is not None:
        # Task #42: readback-free fixed-budget inner solve (no restarts —
        # on-device breakdown freeze holds the last good iterate).
        it = 0
        while it < fixed_iters:
            k = min(check_every, fixed_iters - it)
            if ws.graph is not None and k == check_every:
                wp.capture_launch(ws.graph)
            else:
                batch(k)
            it += k
        info = {"iters": it, "converged": True, "fixed": True,
                "restarts": 0, "graph": ws.graph is not None}
        if x_out is not None:
            wp.copy(x_out, x)
            return None, info
        return x.numpy(), info

    it = 0
    restarts = 0
    rnorm2 = np.inf
    while it < maxiter:
        k = min(check_every, maxiter - it)
        if ws.graph is not None and k == check_every:
            wp.capture_launch(ws.graph)
        else:
            batch(k)
        it += k
        vals = scal.numpy()                             # THE periodic sync
        if sync_counter is not None:
            sync_counter.count += 1
        rnorm2, flag = float(vals[5]), float(vals[10])
        if rnorm2 < thresh2:                   # converged wins over any
            info = {"iters": it,               # concurrent breakdown flag
                    "relres": np.sqrt(rnorm2) / bnorm,
                    "converged": True, "restarts": restarts,
                    "graph": ws.graph is not None}
            if x_out is not None:
                wp.copy(x_out, x)
                return None, info
            return x.numpy(), info
        if flag != 0.0:
            if restarts < max_restarts:
                restarts += 1
                wp.copy(rhat, r)                # restart from last good r
                _reset_scal(rnorm2)
                continue
            info = {"iters": it, "relres": np.sqrt(rnorm2) / bnorm,
                    "converged": False, "restarts": restarts,
                    "breakdown": ("rho" if flag == 1.0 else "rhat_v"),
                    "graph": ws.graph is not None}
            if x_out is not None:
                wp.copy(x_out, x)
                return None, info
            return x.numpy(), info
    info = {"iters": it, "relres": np.sqrt(rnorm2) / bnorm,
            "converged": False, "graph": ws.graph is not None}
    if x_out is not None:
        wp.copy(x_out, x)
        return None, info
    return x.numpy(), info


def bicgstab_dev(op, b, tol=1e-10, atol=1e-12, maxiter=2000, diag=None,
                 check_every=10, sync_counter=None, max_restarts=50,
                 graph=None, b_dev=None, x_out=None, diag_dev=None,
                 fixed_iters=None, apply_dev=None):
    """Single-sync device BiCGStab (Jacobi-preconditioned). Breakdown guards
    live ON DEVICE (scalars[10]) and FREEZE all update kernels, so the state
    at the periodic host check is the last pre-breakdown iterate; the host
    then RESTARTS (rhat <- r, scalars reset) up to max_restarts times — the
    standard cure for rho-breakdown on hard nonsymmetric systems (measured
    on the L6 cavity monolithic block). Same contract as krylov.bicgstab.

    graph: Task-#40 knob (see cg_dev).
    Task #42 device-resident options (see cg_dev).

    apply_dev: optional right-preconditioner hook ``apply_dev(src, dst)``
        (both wp.array float64 on op.device) that computes dst = M^{-1} src.
        Applied at the two standard right-preconditioned BiCGStab insertion
        points: ph = M^{-1} p and sh = M^{-1} s.  The existing diag= fast
        path (Jacobi hadamard) is UNCHANGED when apply_dev is None.
        Only valid on the legacy (non-fused) loop; diag=, diag_dev=, and apply_dev are
        mutually exclusive (diag=/diag_dev= take the fused path, apply_dev does not)."""
    fused, cap = _resolve_path(graph, op.device)
    dev_resident = (b_dev is not None or x_out is not None
                    or diag_dev is not None or fixed_iters is not None)
    if dev_resident and not (fused and _fusable(op)):
        raise ValueError("bicgstab_dev device-resident args require the "
                         "fused CSROperator path")
    if apply_dev is not None and (diag is not None or diag_dev is not None):
        raise ValueError("bicgstab_dev: apply_dev and (diag= or diag_dev=) are mutually "
                         "exclusive; use apply_dev for a general preconditioner "
                         "or diag=/diag_dev= for the Jacobi fast path")
    if fused and (diag is not None or diag_dev is not None) \
            and _fusable(op):
        return _bicgstab_fused(op, b, tol, atol, maxiter, diag,
                               check_every, sync_counter, max_restarts,
                               cap, b_dev=b_dev, x_out=x_out,
                               diag_dev=diag_dev, fixed_iters=fixed_iters)
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
        # note: precond(dst, src) -> apply_dev(src, dst)
        if apply_dev is not None:
            apply_dev(src, dst)          # right-preconditioner hook
        elif minv is not None:
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
    ones[5] = bnorm ** 2      # rnorm2 seed: breakdown thresholds are
    ones[6] = bnorm ** 2      # RELATIVE to scal[5]; zero would disarm them
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
        if rnorm2 < thresh2:                       # converged wins over any
            return x.numpy(), {"iters": it,       # concurrent breakdown flag
                               "relres": np.sqrt(rnorm2) / bnorm,
                               "converged": True, "restarts": restarts}
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
