"""M1d D1-item-4: device GP-field kernels — the LAST host-side per-step
work in the device-assembly stepper path (the velocity/scalar
interpolation-to-Gauss-points einsums in LinearizedMonolithicStepper /
the coupled_nu-style benchmarks).

Given conn, N/dN tables and a DEVICE node-vector, these kernels produce
aq [ngp, dim], div [ngp], grad(u) [ngp, dim*dim] and grad(p) [ngp, dim]
on device — the same per-element loop shapes as the forward assembly
kernels (operators.py factory pattern: _kernel_cache keyed instantiation,
module="unique", enable_backward=False, rolled loops for nbf>4 / dim 3).
The fine-scale correction (u - tauM res_M, conventions item 1) and the
GP-array linear combinations (BDF extrapolation, f - history) are per-GP
kernels so the WHOLE advecting-field pipeline is device-resident: the
stepper uploads node vectors once per step and hands wp arrays straight
to DeviceNSAssembler.

Host mirrors (the gates): LinearizedMonolithicStepper._gp_eval /
_corrected_gp; parity 1e-12-class, tests/test_device_assembly.py.
"""
import numpy as np
import warp as wp

from .operators import _kernel_cache, csr_spmv
from ..physics.vms import tau_m_metric


def _opts(nbf, dim):
    # findings 6 / 4e rule: rolled loops beyond ~4 bf or in 3-D keep the
    # one-time nvrtc compile in seconds
    return {"max_unroll": 0} if (dim >= 3 or nbf > 4) else {}


def make_gp_interp_div(nbf: int, nqp: int, dim: int):
    """Interp a full node vector [n_nodes, dim] to GP values aq [ne*nqp,
    dim] + consistent divergence dq [ne*nqp] (dN contraction * 2/he)."""
    key = ("gp_interp_div", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique", enable_backward=False,
               module_options=_opts(nbf, dim))
    def gp_interp_div(conn: wp.array2d(dtype=wp.int32),
                      h: wp.array(dtype=wp.float64),
                      Ntab: wp.array2d(dtype=wp.float64),
                      dNtab: wp.array3d(dtype=wp.float64),
                      vals: wp.array2d(dtype=wp.float64),
                      aq: wp.array2d(dtype=wp.float64),
                      dq: wp.array(dtype=wp.float64)):
        e = wp.tid()
        dscale = wp.float64(2.0) / h[e]
        for q in range(nqp):
            gp = e * nqp + q
            div = wp.float64(0.0)
            for d in range(dim):
                acc = wp.float64(0.0)
                for a in range(nbf):
                    v = vals[conn[e, a], d]
                    acc += Ntab[q, a] * v
                    div += dNtab[q, a, d] * v
                aq[gp, d] = acc
            dq[gp] = div * dscale

    _kernel_cache[key] = gp_interp_div
    return gp_interp_div


def make_gp_grad_vec(nbf: int, nqp: int, dim: int):
    """Velocity gradient at GPs: gu[gp, d*dim + c] = d u_c / d x_d
    (matches the host einsum 'qad,eac->eqdc' * 2/he layout)."""
    key = ("gp_grad_vec", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique", enable_backward=False,
               module_options=_opts(nbf, dim))
    def gp_grad_vec(conn: wp.array2d(dtype=wp.int32),
                    h: wp.array(dtype=wp.float64),
                    dNtab: wp.array3d(dtype=wp.float64),
                    vals: wp.array2d(dtype=wp.float64),
                    gu: wp.array2d(dtype=wp.float64)):
        e = wp.tid()
        dscale = wp.float64(2.0) / h[e]
        for q in range(nqp):
            gp = e * nqp + q
            for d in range(dim):
                for c in range(dim):
                    acc = wp.float64(0.0)
                    for a in range(nbf):
                        acc += dNtab[q, a, d] * vals[conn[e, a], c]
                    gu[gp, d * dim + c] = acc * dscale

    _kernel_cache[key] = gp_grad_vec
    return gp_grad_vec


def make_gp_grad_scalar(nbf: int, nqp: int, dim: int):
    """Scalar (pressure) gradient at GPs: gp_[gp, d]."""
    key = ("gp_grad_scalar", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique", enable_backward=False,
               module_options=_opts(nbf, dim))
    def gp_grad_scalar(conn: wp.array2d(dtype=wp.int32),
                       h: wp.array(dtype=wp.float64),
                       dNtab: wp.array3d(dtype=wp.float64),
                       vals: wp.array(dtype=wp.float64),
                       gp_: wp.array2d(dtype=wp.float64)):
        e = wp.tid()
        dscale = wp.float64(2.0) / h[e]
        for q in range(nqp):
            gp = e * nqp + q
            for d in range(dim):
                acc = wp.float64(0.0)
                for a in range(nbf):
                    acc += dNtab[q, a, d] * vals[conn[e, a]]
                gp_[gp, d] = acc * dscale

    _kernel_cache[key] = gp_grad_scalar
    return gp_grad_scalar


def make_gp_multifield(nbf: int, nqp: int, dim: int, ndof: int):
    """Node-major MULTIFIELD GP eval (the M4/M5 monolithic steppers'
    layout): X [n_nodes, ndof] -> vals [ne*nqp, ndof] and grads
    [ne*nqp, ndof, dim] (reference dN scaled by 2/he — the affine-cube
    metric).  This is the device mirror of the multiphase host
    _pack_fields einsums; ndof is compile-time (the (M, K) factory
    pattern) and nbf/nqp come from the basis tabulation (A4a: p = 1
    and p = 2 both ride the same factory)."""
    key = ("gp_multifield", nbf, nqp, dim, ndof)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique", enable_backward=False,
               module_options={"max_unroll": 0})
    def gp_multi(conn: wp.array2d(dtype=wp.int32),
                 h: wp.array(dtype=wp.float64),
                 Ntab: wp.array2d(dtype=wp.float64),
                 dNtab: wp.array3d(dtype=wp.float64),
                 X: wp.array2d(dtype=wp.float64),
                 vals: wp.array2d(dtype=wp.float64),
                 grads: wp.array3d(dtype=wp.float64)):
        e = wp.tid()
        dscale = wp.float64(2.0) / h[e]
        for q in range(nqp):
            gp = e * nqp + q
            for f in range(ndof):
                acc = wp.float64(0.0)
                for a in range(nbf):
                    acc += Ntab[q, a] * X[conn[e, a], f]
                vals[gp, f] = acc
                for d in range(dim):
                    g = wp.float64(0.0)
                    for a in range(nbf):
                        g += dNtab[q, a, d] * X[conn[e, a], f]
                    grads[gp, f, d] = g * dscale

    _kernel_cache[key] = gp_multi
    return gp_multi


def make_gp_vals(nbf: int, nqp: int, ndof: int):
    """VALUES-ONLY multifield GP eval (gp_multifield sans gradients):
    X [n_nodes, ndof] -> vals [ne*nqp, ndof].  Task #37: the film BDF
    history needs only the GP VALUES of the previous-step fields, so
    the gradient loops (3/4 of the flops) are dropped.  Accumulation
    order over the basis functions is IDENTICAL to gp_multifield."""
    key = ("gp_vals", nbf, nqp, ndof)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique", enable_backward=False,
               module_options={"max_unroll": 0})
    def gp_vals_k(conn: wp.array2d(dtype=wp.int32),
                  Ntab: wp.array2d(dtype=wp.float64),
                  X: wp.array2d(dtype=wp.float64),
                  vals: wp.array2d(dtype=wp.float64)):
        e = wp.tid()
        for q in range(nqp):
            gp = e * nqp + q
            for f in range(ndof):
                acc = wp.float64(0.0)
                for a in range(nbf):
                    acc += Ntab[q, a] * X[conn[e, a], f]
                vals[gp, f] = acc

    _kernel_cache[key] = gp_vals_k
    return gp_vals_k


def make_csr_spmv_ncomp(ncomp: int):
    """Multi-component CSR SpMV: y[row, c] = sum_j A[row, j] x[j, c] —
    the constraint application T @ node_vals per velocity component."""
    key = ("csr_spmv_ncomp", ncomp)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique", enable_backward=False)
    def spmv_n(indptr: wp.array(dtype=wp.int32),
               indices: wp.array(dtype=wp.int32),
               data: wp.array(dtype=wp.float64),
               x: wp.array2d(dtype=wp.float64),
               y: wp.array2d(dtype=wp.float64)):
        row = wp.tid()
        for c in range(ncomp):
            acc = wp.float64(0.0)
            for j in range(indptr[row], indptr[row + 1]):
                acc += data[j] * x[indices[j], c]
            y[row, c] = acc

    _kernel_cache[key] = spmv_n
    return spmv_n


def make_finescale_correct(nqp: int, dim: int):
    """Per-GP fine-scale correction (conventions item 1, BE residual
    item 5): out = aq_a - tauM * [(aq_a - aq_b)/dt + (a.grad)a + grad p
    - f]; tauM = metric form with the sig2tau transient fold (mirrors
    tau_metric_host + the stepper's 1/sqrt(sig2 + 1/tau^2) fold)."""
    key = ("gp_finescale", nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]
    dim_f = float(dim)

    @wp.kernel(module="unique", enable_backward=False,
               module_options=_opts(1, dim))
    def gp_finescale(h: wp.array(dtype=wp.float64),
                     aq_a: wp.array2d(dtype=wp.float64),
                     aq_b: wp.array2d(dtype=wp.float64),
                     gu: wp.array2d(dtype=wp.float64),
                     gpq: wp.array2d(dtype=wp.float64),
                     fq: wp.array2d(dtype=wp.float64),
                     nu: wp.float64, inv_dt: wp.float64,
                     sig2tau: wp.float64,
                     out: wp.array2d(dtype=wp.float64)):
        i = wp.tid()
        e = i / nqp
        he = h[e]
        amag = wp.float64(0.0)
        for d in range(dim):
            amag += aq_a[i, d] * aq_a[i, d]
        amag = wp.sqrt(amag)
        tau = tau_m_metric(amag, he, nu, sig2tau, wp.float64(dim_f))
        for c in range(dim):
            agu = wp.float64(0.0)
            for d in range(dim):
                agu += aq_a[i, d] * gu[i, d * dim + c]
            res = ((aq_a[i, c] - aq_b[i, c]) * inv_dt + agu
                   + gpq[i, c] - fq[i, c])
            out[i, c] = aq_a[i, c] - tau * res

    _kernel_cache[key] = gp_finescale
    return gp_finescale


def make_gp_axpby(dim: int):
    """out = alpha X + beta Y on [ngp, dim] GP arrays (extrapolation
    2 c_n - c_m; rhs f - history)."""
    key = ("gp_axpby", dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique", enable_backward=False)
    def gp_axpby(alpha: wp.float64, X: wp.array2d(dtype=wp.float64),
                 beta: wp.float64, Y: wp.array2d(dtype=wp.float64),
                 out: wp.array2d(dtype=wp.float64)):
        i = wp.tid()
        for c in range(dim):
            out[i, c] = alpha * X[i, c] + beta * Y[i, c]

    _kernel_cache[key] = gp_axpby
    return gp_axpby


class DeviceGPField:
    """Per-epoch device GP-field evaluator. Node vectors go up ONCE per
    step; constraint application (T spmv), interpolation, divergence,
    gradients, fine-scale correction and GP linear algebra all run on
    device. All outputs are persistent per-(tag, bin) device buffers —
    zero per-step allocation, zero host round-trips.

    Methods return per-bin dicts {pv: wp.array} in the same shapes the
    host `_gp_eval`-style code produces, so DeviceNSAssembler consumes
    them directly (it accepts wp arrays as of M1d closure)."""

    def __init__(self, dm):
        self.dm = dm
        self.dim = dm.dim
        d = dm.device
        self._bufs = {}
        self._ne = {pv: len(b["eids"]) for pv, b in dm.bins.items()}
        self._ngp = {pv: len(b["eids"]) * b["nqp"]
                     for pv, b in dm.bins.items()}
        self._k_id = {pv: make_gp_interp_div(b["nbf"], b["nqp"], dm.dim)
                      for pv, b in dm.bins.items()}
        self._k_gv = {pv: make_gp_grad_vec(b["nbf"], b["nqp"], dm.dim)
                      for pv, b in dm.bins.items()}
        self._k_gs = {pv: make_gp_grad_scalar(b["nbf"], b["nqp"], dm.dim)
                      for pv, b in dm.bins.items()}
        self._k_fs = {pv: make_finescale_correct(b["nqp"], dm.dim)
                      for pv, b in dm.bins.items()}
        self._k_spmv = make_csr_spmv_ncomp(dm.dim)
        self._k_ax = make_gp_axpby(dm.dim)
        self.device = d

    # -- persistent buffers ---------------------------------------------
    def _buf(self, key, shape):
        b = self._bufs.get(key)
        if b is None:
            b = wp.zeros(shape, dtype=wp.float64, device=self.device)
            self._bufs[key] = b
        return b

    def _upload(self, key, host_np):
        """Host -> persistent device buffer (the once-per-step node/f
        uploads)."""
        host_np = np.ascontiguousarray(host_np, np.float64)
        dst = self._buf(key, host_np.shape)
        wp.copy(dst, wp.array(host_np, dtype=wp.float64, device="cpu",
                              copy=False))
        return dst

    # -- constraint application -----------------------------------------
    def to_full(self, free_vals, tag):
        """Free node vector [n_free, dim] (host) -> full node vector
        [n_nodes, dim] on device: one upload + T spmv."""
        dm = self.dm
        x = self._upload(("free", tag), np.asarray(free_vals))
        y = self._buf(("full", tag), (dm.n_nodes, self.dim))
        wp.launch(self._k_spmv, dim=dm.n_nodes,
                  inputs=[*dm.T_dev, x, y], device=self.device)
        return y

    def to_full_scalar(self, free_vals, tag):
        """Free scalar [n_free] -> full [n_nodes] on device."""
        dm = self.dm
        x = self._upload(("free_s", tag), np.asarray(free_vals).ravel())
        y = self._buf(("full_s", tag), dm.n_nodes)
        wp.launch(csr_spmv, dim=dm.n_nodes,
                  inputs=[*dm.T_dev, x, y], device=self.device)
        return y

    def full(self, tag):
        """The full node vector previously produced under `tag`."""
        return self._bufs[("full", tag)]

    # -- GP fields --------------------------------------------------------
    def interp_div(self, full_d, tag):
        """(aq, dq) per bin from a device full node vector."""
        dm = self.dm
        aq, dq = {}, {}
        for pv, b in dm.bins.items():
            a_ = self._buf(("aq", tag, pv), (self._ngp[pv], self.dim))
            d_ = self._buf(("dq", tag, pv), self._ngp[pv])
            wp.launch(self._k_id[pv], dim=self._ne[pv],
                      inputs=[b["conn"], b["h"], b["N"], b["dN"],
                              full_d, a_, d_], device=self.device)
            aq[pv], dq[pv] = a_, d_
        return aq, dq

    def grad_vec(self, full_d, tag):
        """gu per bin [ngp, dim*dim] (layout d*dim + c = d u_c/d x_d)."""
        dm = self.dm
        gu = {}
        for pv, b in dm.bins.items():
            g_ = self._buf(("gu", tag, pv),
                           (self._ngp[pv], self.dim * self.dim))
            wp.launch(self._k_gv[pv], dim=self._ne[pv],
                      inputs=[b["conn"], b["h"], b["dN"], full_d, g_],
                      device=self.device)
            gu[pv] = g_
        return gu

    def grad_scalar(self, full_s_d, tag):
        """grad p per bin [ngp, dim]."""
        dm = self.dm
        gp = {}
        for pv, b in dm.bins.items():
            g_ = self._buf(("gp", tag, pv), (self._ngp[pv], self.dim))
            wp.launch(self._k_gs[pv], dim=self._ne[pv],
                      inputs=[b["conn"], b["h"], b["dN"], full_s_d, g_],
                      device=self.device)
            gp[pv] = g_
        return gp

    def upload_gp(self, vals_by_bin, tag):
        """Host per-bin GP arrays (e.g. f_fn(xq, t)) -> device buffers."""
        return {pv: self._upload(("gpv", tag, pv), vals_by_bin[pv])
                for pv in vals_by_bin}

    def finescale(self, aq_a, aq_b, gu, gpq, fq, nu, dt, sig2tau, tag):
        """Corrected GP field aq_a - tauM res_M per bin (device)."""
        dm = self.dm
        out = {}
        for pv, b in dm.bins.items():
            o_ = self._buf(("fs", tag, pv), (self._ngp[pv], self.dim))
            wp.launch(self._k_fs[pv], dim=self._ngp[pv],
                      inputs=[b["h"], aq_a[pv], aq_b[pv], gu[pv],
                              gpq[pv], fq[pv], wp.float64(nu),
                              wp.float64(1.0 / dt), wp.float64(sig2tau),
                              o_], device=self.device)
            out[pv] = o_
        return out

    def axpby(self, alpha, X, beta, Y, tag):
        """Per-bin out = alpha X + beta Y on [ngp, dim] GP arrays."""
        out = {}
        for pv in X:
            o_ = self._buf(("ax", tag, pv), (self._ngp[pv], self.dim))
            wp.launch(self._k_ax, dim=self._ngp[pv],
                      inputs=[wp.float64(alpha), X[pv],
                              wp.float64(beta), Y[pv], o_],
                      device=self.device)
            out[pv] = o_
        return out

    def axpby_nodes(self, alpha, X, beta, Y, tag):
        """out = alpha X + beta Y on FULL node vectors [n_nodes, dim] —
        the BDF extrapolation / history combination moved on-device (the
        O(n_free) numpy axpys were measured host work at 2-D L8)."""
        o_ = self._buf(("axn", tag), (self.dm.n_nodes, self.dim))
        wp.launch(self._k_ax, dim=self.dm.n_nodes,
                  inputs=[wp.float64(alpha), X, wp.float64(beta), Y, o_],
                  device=self.device)
        return o_

    def zeros_gp(self, tag):
        """Persistent per-bin ZERO GP arrays [ngp, dim] (f_fn=None: zero
        body force with no per-step host eval/upload)."""
        return {pv: self._buf(("gpv", tag, pv), (self._ngp[pv], self.dim))
                for pv in self.dm.bins}
