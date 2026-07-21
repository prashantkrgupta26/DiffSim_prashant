"""Task #43: device-resident backend for ``fp64_iterative_refinement``.

#36's IR loop marshalled through host numpy every sweep — a ``.cpu()`` +
``from_numpy(...).to(dev)`` per matvec AND per correction solve, plus a fresh
``wp.array``/``wp.zeros`` allocation per matvec.  On the GH200 that per-sweep
host↔device traffic would sink the fp32 datum for a marshalling reason
unrelated to the fp32-factor thesis (brief scope 1).

``WarpIRBackend`` keeps every IR work vector DEVICE-RESIDENT and allocated
ONCE, reused across sweeps:

  * ``b_d``   fp64 rhs (adopted zero-copy from the assembler's ``F_d`` when
              possible, else uploaded once);
  * ``x_d``   fp64 iterate;
  * ``r_d``   fp64 residual;
  * ``ax_d``  fp64 shadow-matvec output (was a fresh ``wp.zeros`` per sweep);
  * ``dx_d``  fp64 correction (promoted from the fp32 cuDSS solve);
  * ``_b32``  the STABLE fp32 cuDSS rhs operand (torch), refreshed by a
              device-to-device cast — never a host round-trip.

The NUMERICS are identical to the host path: the residual ``r = b - A x`` is
computed in fp64 against the fp64 shadow operator (``device_operator`` over
``vals_d``), and the correction ``A dx = r`` is solved by the fp32 cuDSS
factor.  Only the buffer marshalling moves onto the device.  The one
unavoidable host transfer is the residual-norm SCALAR each sweep (a single
fp64 download for the stopping test), NOT the whole vector.

This module imports torch / warp / nvmath lazily inside ``__init__`` so the
pure-numpy loop (and its CPU unit tests) never pull the GPU stack.
"""
import numpy as np

import warp as wp

from . import blas


@wp.kernel
def _sub_kernel(b: wp.array(dtype=wp.float64), ax: wp.array(dtype=wp.float64),
                out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    out[i] = b[i] - ax[i]


class WarpIRBackend:
    """Device-resident IR backend for the cuDSS fp32-factor path.

    Parameters
    ----------
    op : object
        The assembler's fp64 device operator (``device_operator()``): its
        ``.matvec(x_wp, y_wp)`` applies A in fp64 against ``vals_d`` (the
        residual's shadow matvec — NOT the fp32 factor).
    solver : nvmath DirectSolver
        The fp32 cuDSS solver, ALREADY planned + factorized over the fp32
        snapshot CSR.  ``factor_solve`` refreshes its rhs operand in place
        (device-to-device fp32 cast) and calls ``solve()``.
    b32 : torch.Tensor
        The STABLE fp32 rhs operand tensor the solver was planned with.
    n : int
        System dimension (``asm.Nfull``).
    device : warp device
        The assembler's device (``dm.device``).
    rhs_d : wp.array | None
        Optional fp64 rhs already on device (``asm.F_d``): adopted zero-copy
        so the initial rhs never round-trips.  ``None`` -> uploaded from the
        host ``b`` passed to the loop.
    """

    def __init__(self, op, solver, b32, n, device, *, rhs_d=None):
        import torch

        self._torch = torch
        self._op = op
        self._solver = solver
        self._b32 = b32                      # stable fp32 cuDSS rhs operand
        self._n = int(n)
        self._device = device
        self._rhs_d = rhs_d

        # persistent fp64 device work buffers (allocated ONCE)
        self._x_d = wp.zeros(self._n, dtype=wp.float64, device=device)
        self._r_d = wp.zeros(self._n, dtype=wp.float64, device=device)
        self._ax_d = wp.zeros(self._n, dtype=wp.float64, device=device)
        self._dx_d = wp.zeros(self._n, dtype=wp.float64, device=device)
        self._b_d = None                     # set in as_rhs

        # a persistent fp64 torch VIEW over dx_d for the fp32->fp64 promote
        # (dlpack alias — no copy; the cast writes through it in place).
        self._dx_t = torch.from_dlpack(self._dx_d.__dlpack__())

    # --- buffer lifecycle -------------------------------------------------
    def as_rhs(self, b):
        """Adopt the fp64 rhs onto the device ONCE.  Prefer the assembler's
        resident ``F_d`` (zero-copy) when the caller passed a matching rhs;
        otherwise upload the host array a single time."""
        if self._rhs_d is not None:
            self._b_d = self._rhs_d
        else:
            self._b_d = wp.array(np.ascontiguousarray(b, np.float64),
                                 dtype=wp.float64, device=self._device)
        return self._b_d

    def norm(self, v):
        # device fp64 reduction; only the resulting SCALAR crosses to host.
        return float(np.sqrt(blas.dot(v, v, self._device)))

    def zeros_like(self, v):
        # zero-rhs degenerate path: return a zeroed iterate buffer.
        wp.launch(_zero_kernel(), dim=self._n, inputs=[self._x_d],
                  device=self._device)
        return self._x_d

    # --- the two caller callables, in device-buffer space -----------------
    def initial(self, b):
        """x0 == None initial fp32 solve, landed in the ITERATE buffer x_d
        (not the shared dx_d correction buffer — those alias otherwise)."""
        self._factor_solve_into(b, self._x_d)
        return self._x_d

    def set_iterate(self, x0):
        """Adopt a provided fp64 initial iterate into x_d (host upload once).
        cuDSS callers pass x0=None, so this is the rarely-used path."""
        src = wp.array(np.ascontiguousarray(x0, np.float64),
                       dtype=wp.float64, device=self._device)
        wp.copy(self._x_d, src)
        return self._x_d

    def factor_solve(self, r):
        """fp32 correction solve of A y = r into the persistent dx_d buffer,
        device-resident.  Returns dx_d."""
        self._factor_solve_into(r, self._dx_d, dx_t=self._dx_t)
        return self._dx_d

    def _factor_solve_into(self, r, out_d, dx_t=None):
        """Solve A y = r with the fp32 factor, promoting the fp32 result into
        the fp64 device buffer ``out_d`` — all device-to-device.

        ``r`` is fp64 device: cast into the STABLE fp32 cuDSS rhs operand,
        solve, then cast the fp32 result up into ``out_d`` (via its dlpack
        torch view)."""
        torch = self._torch
        r_t = torch.from_dlpack(r.__dlpack__())          # fp64 view of r
        self._b32.copy_(r_t)                             # fp64 -> fp32 cast
        self._solver.reset_operands(b=self._b32)
        y32 = self._solver.solve()                       # fp32 torch result
        if dx_t is None:
            dx_t = torch.from_dlpack(out_d.__dlpack__())
        dx_t.copy_(y32)                                  # fp32 -> fp64 cast

    def matvec(self, x):
        """fp64 shadow matvec A @ x into the persistent ``ax_d`` buffer."""
        self._op.matvec(x, self._ax_d)
        return self._ax_d

    # --- vector arithmetic (all device-resident) --------------------------
    def residual(self, b, ax, out=None):
        """r_d <- b - ax."""
        wp.launch(_sub_kernel, dim=self._n, inputs=[b, ax, self._r_d],
                  device=self._device)
        return self._r_d

    def axpy(self, x, dx, out=None):
        """x_d <- x + dx  (in place; returns x_d)."""
        blas.axpy(1.0, dx, self._x_d, self._device)
        return self._x_d

    def to_host(self, x):
        return np.ascontiguousarray(x.numpy(), np.float64)


def _zero_kernel():
    if not hasattr(_zero_kernel, "_k"):
        @wp.kernel
        def _k(a: wp.array(dtype=wp.float64)):
            i = wp.tid()
            a[i] = wp.float64(0.0)
        _zero_kernel._k = _k
    return _zero_kernel._k
