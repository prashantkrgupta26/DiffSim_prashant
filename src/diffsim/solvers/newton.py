import numpy as np
from .krylov import cg, bicgstab
from .result import NonlinearSolveResult


class _NpOp:
    """Adapts a numpy-action closure to the Krylov op protocol."""
    def __init__(self, action, n, device):
        self.action, self.n_free, self.device = action, n, device

    def matvec(self, x, y):
        import warp as wp
        r = self.action(x.numpy())
        wp.copy(y, wp.array(r.astype(np.float64), dtype=wp.float64, device=self.device))


class NonlinearSolver:
    def __init__(self, residual_fn, jac_action_fn=None, snes_rtol=1e-8, snes_atol=1e-12,
                 snes_max_it=20, linesearch="bt", linear_solver=bicgstab,
                 linear_rtol=None, device="cpu"):
        self.F = residual_fn
        self.Jv = jac_action_fn
        self.rtol, self.atol, self.max_it = snes_rtol, snes_atol, snes_max_it
        self.linesearch, self.linear_solver = linesearch, linear_solver
        self.linear_rtol, self.device = linear_rtol, device

    def _jac_action(self, u, Fu):
        if self.Jv is not None:
            return lambda v: self.Jv(u, v)
        eps0 = np.sqrt(np.finfo(np.float64).eps)
        def jv(v):
            nv = np.linalg.norm(v)
            if nv == 0.0:
                return np.zeros_like(v)
            eps = eps0 * (1.0 + np.linalg.norm(u)) / nv
            return (self.F(u + eps * v) - Fu) / eps
        return jv

    def solve(self, u0):
        u = np.array(u0, np.float64)
        Fu = self.F(u)
        f0 = fprev = np.linalg.norm(Fu)
        hist = [f0]
        for it in range(1, self.max_it + 1):
            if hist[-1] < max(self.rtol * f0, self.atol):
                return NonlinearSolveResult(
                    u=u, converged=True, iterations=it - 1,
                    residual_norm=float(hist[-1]), fnorm_history=hist)
            # Eisenstat-Walker forcing unless fixed rtol given
            eta = self.linear_rtol or min(0.1, np.sqrt(hist[-1] / fprev) if it > 1 else 0.1)
            op = _NpOp(self._jac_action(u, Fu), len(u), self.device)
            du, info = self.linear_solver(op, -Fu, tol=eta, maxiter=1000)
            lam = 1.0
            while True:
                Fnew = self.F(u + lam * du)
                fnew = np.linalg.norm(Fnew)
                if self.linesearch == "basic" or fnew <= (1.0 - 1e-4 * lam) * hist[-1]:
                    break
                lam *= 0.5
                if lam < 1e-4:
                    break
            u = u + lam * du
            Fu = self.F(u)          # always consistent with the accepted step
            fnew = np.linalg.norm(Fu)
            fprev = hist[-1]
            hist.append(fnew)
        converged = hist[-1] < max(self.rtol * f0, self.atol)
        return NonlinearSolveResult(
            u=u, converged=converged, iterations=self.max_it,
            residual_norm=float(hist[-1]), fnorm_history=hist,
            reason="converged" if converged else "maxiter")
