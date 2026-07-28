"""Task A1 — block-diagonal (Jacobi-by-block) saddle preconditioner.

Produces ``make_bdiag_apply(A, ndof, device) -> apply_dev``, the
block-diagonal preconditioner closure for the monolithic (u, p) saddle system
assembled by ``assemble_linear_ns``:

  Interleaved dof convention (ndof = dim+1 per node):
    velocity dofs: node i -> i*ndof + 0, ..., i*ndof + (dim-1)
    pressure dof : node i -> i*ndof + dim

  Preconditioner M^{-1}:
    velocity block   : Jacobi (1/d_u, element-wise).
    pressure block   : Jacobi with a safe floor — the PSPG-stabilized p-p
                       diagonal is nonzero but can be small, so we use
                           d_p_safe = max(|d_p|, eps_p * max_d)
                       where max_d = max(|d|) over ALL dofs and
                       eps_p = 1e-12 (documented floor, see below).
                       This prevents near-zero pressure pivots from amplifying
                       residual noise without discarding real PSPG entries.

``apply_dev(v_in, z_out)``: Warp device closure (v_in, z_out are wp.array
  of dtype float64 on `device`) that computes z = M^{-1} v in place.

Design notes
------------
- The Jacobi diagonal is extracted ONCE at construction time; per-apply is a
  single element-wise multiply (one Warp kernel launch).
- The eps_p floor is relative to max|d| so it scales with the system size and
  conditioning — it is nonzero only when |d_p| is truly negligible (< 1e-12
  of the largest diagonal entry), which never happens in practice for
  PSPG-stabilized equal-order elements at reasonable Re and mesh size.
- The apply_dev closure captures the device diagonal array; it is safe to call
  from fgmres_dev's inner loop without extra host contact.
"""
import numpy as np
import warp as wp

from ..assembly.operators import _kernel_cache


def _bdiag_kernel():
    """Element-wise z[i] = dinv[i] * v[i] (reuse the Jacobi kernel pattern
    from test_fgmres_dev.py; cached so it is compiled only once)."""
    k = _kernel_cache.get(("saddle_bdiag_apply",))
    if k is not None:
        return k

    @wp.kernel(module="unique", enable_backward=False)
    def bdiag_apply(dinv: wp.array(dtype=wp.float64),
                    v: wp.array(dtype=wp.float64),
                    z: wp.array(dtype=wp.float64)):
        i = wp.tid()
        z[i] = dinv[i] * v[i]

    _kernel_cache[("saddle_bdiag_apply",)] = bdiag_apply
    return bdiag_apply


def make_bdiag_apply(A, ndof, device):
    """Block-diagonal (Jacobi-by-block) preconditioner for the (u, p) saddle.

    Parameters
    ----------
    A : scipy CSR matrix
        The monolithic saddle system (shape N x N, N = n_nodes * ndof).
    ndof : int
        Dofs per node (dim+1 for a dim-dimensional problem).
        Velocity dofs are 0 .. dim-1 within each node block;
        pressure dof is dim (the last one).
    device : str
        Warp device string (e.g. "cpu" or "cuda:0").

    Returns
    -------
    apply_dev : callable
        ``apply_dev(v_in_wp, z_out_wp)`` — both wp.array(dtype=float64) on
        `device`.  Computes z = M^{-1} v (block-diagonal Jacobi).
    """
    A = A.tocsr()
    N = A.shape[0]
    diag = np.asarray(A.diagonal()).copy()   # shape (N,), host numpy

    # Identify pressure dofs: node i -> i*ndof + (ndof-1)  (= i*ndof + dim)
    # All dofs not at offset (ndof-1) within a node block are velocity dofs.
    p_mask = np.zeros(N, dtype=bool)
    p_mask[np.arange(N // ndof) * ndof + (ndof - 1)] = True

    # Velocity block: plain Jacobi.  Guard zeros (can arise on Dirichlet rows
    # where the row is replaced by [0,…,1,…,0]; those diagonal entries are 1.0
    # after surgery, so the guard is defensive only).
    dinv = np.empty(N, dtype=np.float64)
    d_u = diag[~p_mask].copy()
    d_u[d_u == 0.0] = 1.0
    dinv[~p_mask] = 1.0 / d_u

    # Pressure block: abs-value floor at eps_p * max|d| (whole system).
    # The PSPG stabilization makes the p-p diagonal nonzero; the floor
    # catches pathological near-zero entries without destroying real entries.
    max_d = float(np.abs(diag).max()) if N > 0 else 1.0
    eps_p = 1e-12
    d_p = diag[p_mask].copy()
    d_p_safe = np.maximum(np.abs(d_p), eps_p * max_d)
    dinv[p_mask] = 1.0 / d_p_safe

    # Upload the inverse diagonal to the device (once).
    dinv_d = wp.array(np.ascontiguousarray(dinv), dtype=wp.float64, device=device)

    kernel = _bdiag_kernel()

    def apply_dev(v_in, z_out):
        """z_out = M^{-1} v_in  (element-wise diagonal scaling)."""
        wp.launch(kernel, dim=N, inputs=[dinv_d, v_in, z_out], device=device)

    return apply_dev
