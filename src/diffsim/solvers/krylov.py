import numpy as np
import warp as wp
from . import blas

def _to_dev(v, device):
    return wp.array(np.asarray(v, np.float64), dtype=wp.float64, device=device)

def cg(op, b, tol=1e-10, atol=1e-12, maxiter=2000, diag=None):
    d = op.device
    n = op.n_free
    bd = _to_dev(b, d)
    x = wp.zeros(n, dtype=wp.float64, device=d)
    r = wp.clone(bd)
    z = wp.zeros(n, dtype=wp.float64, device=d)
    Minv = _to_dev(1.0 / np.asarray(diag), d) if diag is not None else None
    if Minv is not None:
        blas.hadamard(Minv, r, z, d)
    else:
        wp.copy(z, r)
    p = wp.clone(z)
    Ap = wp.zeros(n, dtype=wp.float64, device=d)
    rz = blas.dot(r, z, d)
    bnorm = max(np.sqrt(blas.dot(bd, bd, d)), 1e-300)
    for it in range(1, maxiter + 1):
        op.matvec(p, Ap)
        alpha = rz / blas.dot(p, Ap, d)
        blas.axpy(alpha, p, x, d)
        blas.axpy(-alpha, Ap, r, d)
        rnorm = np.sqrt(blas.dot(r, r, d))
        if rnorm < max(tol * bnorm, atol):
            return x.numpy(), {"iters": it, "relres": rnorm / bnorm, "converged": True}
        if Minv is not None:
            blas.hadamard(Minv, r, z, d)
        else:
            wp.copy(z, r)
        rz_new = blas.dot(r, z, d)
        blas.xpay(rz_new / rz, z, p, d)
        rz = rz_new
    return x.numpy(), {"iters": maxiter, "relres": rnorm / bnorm, "converged": False}

def bicgstab(op, b, tol=1e-10, atol=1e-12, maxiter=2000, diag=None):
    d = op.device
    n = op.n_free
    bd = _to_dev(b, d)
    x = wp.zeros(n, dtype=wp.float64, device=d)
    r = wp.clone(bd)
    rhat = wp.clone(bd)
    p = wp.zeros(n, dtype=wp.float64, device=d)
    v = wp.zeros(n, dtype=wp.float64, device=d)
    s = wp.zeros(n, dtype=wp.float64, device=d)
    t = wp.zeros(n, dtype=wp.float64, device=d)
    Minv = _to_dev(1.0 / np.asarray(diag), d) if diag is not None else None
    ph = wp.zeros(n, dtype=wp.float64, device=d)
    sh = wp.zeros(n, dtype=wp.float64, device=d)
    rho = alpha = omega = 1.0
    bnorm = max(np.sqrt(blas.dot(bd, bd, d)), 1e-300)
    for it in range(1, maxiter + 1):
        rho_new = blas.dot(rhat, r, d)
        beta = (rho_new / rho) * (alpha / omega) if it > 1 else 0.0
        # p = r + beta (p - omega v)
        blas.axpy(-omega, v, p, d)
        blas.xpay(beta, r, p, d)
        if Minv is not None:
            blas.hadamard(Minv, p, ph, d)
        else:
            wp.copy(ph, p)
        op.matvec(ph, v)
        alpha = rho_new / blas.dot(rhat, v, d)
        wp.copy(s, r); blas.axpy(-alpha, v, s, d)
        if Minv is not None:
            blas.hadamard(Minv, s, sh, d)
        else:
            wp.copy(sh, s)
        op.matvec(sh, t)
        tt = blas.dot(t, t, d)
        omega = blas.dot(t, s, d) / tt if tt > 0 else 0.0
        blas.axpy(alpha, ph, x, d); blas.axpy(omega, sh, x, d)
        wp.copy(r, s); blas.axpy(-omega, t, r, d)
        rnorm = np.sqrt(blas.dot(r, r, d))
        if rnorm < max(tol * bnorm, atol):
            return x.numpy(), {"iters": it, "relres": rnorm / bnorm, "converged": True}
        rho = rho_new
    return x.numpy(), {"iters": maxiter, "relres": rnorm / bnorm, "converged": False}
