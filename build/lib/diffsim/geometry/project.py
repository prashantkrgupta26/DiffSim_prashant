"""Newton closest-point projection with IFT adjoint (Tier-2 VJP #3).

Spec S4.1: framework-standard distance vectors via the Newton projection
y_{k+1} = y_k - psi(y_k) grad_psi(y_k) / |grad_psi(y_k)|^2 (iteration cap,
tolerance, per-point convergence mask) — correct for non-eikonal fields.
The eikonal shortcut d = -psi grad_psi / |grad_psi|^2 is used only for
backends declaring near_eikonal=True, with a runtime |grad psi| ~ 1 check.

Adjoint (spec S5.2, VJP #3): implicit-function theorem at convergence, never
unrolled. Augmented system in z = (y, s):

    F(y, s; theta, x) = [ y - x + s grad_psi(y); psi(y) ] = 0
    J_z = [[ I + s H_psi(y), grad_psi(y) ], [ grad_psi(y)^T, 0 ]]

Backward for output y with cotangent ybar: solve J_z^T mu = [ybar; 0], then
theta_bar = d<F, -mu>/d theta at fixed z (one autograd.grad on the params)
and x_bar = mu_y (since dF/dx = [-I; 0]). Batched: one (dim+1)^2 solve per
point via torch.linalg.solve; Hessians by double-grad (dim <= 3 in practice).

Downstream quantities are ordinary torch ops on y: d = y - x and
n = grad_psi(y)/|grad_psi(y)| — autograd composes them with the custom
Function, so n's direct theta-dependence and curvature effects are exact.
n points in the direction of INCREASING psi; the caller orients it out of
the computational domain per the `domain` flag.
"""
import numpy as np
import torch

_TOL = 1e-13     # |psi(y)| convergence target
_OK_TOL = 1e-12  # per-point success threshold reported in the mask
_MAXIT = 50


def _psi_grad(oracle, y):
    """Detached (psi, grad psi) at y — forward-iteration workhorse."""
    yv = y.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        psi = oracle.psi(yv)
        (g,) = torch.autograd.grad(psi.sum(), yv)
    return psi.detach(), g.detach()


def _psi_grad_hess(oracle, y):
    """Detached (psi, grad psi, Hessian) at y — augmented-Newton workhorse."""
    dim = y.shape[1]
    yv = y.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        psi = oracle.psi(yv)
        (g,) = torch.autograd.grad(psi.sum(), yv, create_graph=True)
        H = torch.stack(
            [torch.autograd.grad(g[:, i].sum(), yv, retain_graph=True)[0]
             for i in range(dim)], dim=1)
    return psi.detach(), g.detach(), H.detach()


def _augmented_residual(x, y, s, psi, g):
    """F(y, s) = [y - x + s grad_psi(y); psi(y)], batched [N, dim+1]."""
    return torch.cat([y - x + s.unsqueeze(1) * g, psi.unsqueeze(1)], dim=1)


def _jz(s, g, H):
    """J_z = [[I + s H, g], [g^T, 0]], batched [N, dim+1, dim+1]."""
    N, dim = g.shape
    Jz = torch.zeros(N, dim + 1, dim + 1, dtype=torch.float64)
    Jz[:, :dim, :dim] = torch.eye(dim, dtype=torch.float64) + s.view(N, 1, 1) * H
    Jz[:, :dim, dim] = g
    Jz[:, dim, :dim] = g
    return Jz


def _newton_iterate(oracle, x):
    """Project x onto {psi = 0} at the CLOSEST point. Returns (y, s, ok).

    The simple gradient-projection update y <- y - psi grad_psi/|grad_psi|^2
    reaches the zero set but, for non-eikonal fields, NOT the closest point —
    and its fixed point is path-dependent, so it is not a well-defined
    differentiable map. It serves only as the warm start; convergence is
    then driven by full Newton on the augmented system F(y, s) = 0 — the
    same system the IFT backward differentiates, which is what makes the
    forward/backward pair consistent.
    """
    x = x.detach()
    N, dim = x.shape
    # Phase 1: gradient projection onto the zero set (warm start).
    y = x.clone()
    for _ in range(20):
        psi, g = _psi_grad(oracle, y)
        if (psi.abs() <= _TOL).all():
            break
        denom = (g * g).sum(dim=1).clamp_min(1e-300)
        y = y - (psi / denom).unsqueeze(1) * g
    psi, g = _psi_grad(oracle, y)
    s = ((x - y) * g).sum(dim=1) / (g * g).sum(dim=1).clamp_min(1e-300)
    # Phase 2: full Newton on F(y, s) = 0 (quadratic near the warm start).
    for _ in range(_MAXIT):
        psi, g, H = _psi_grad_hess(oracle, y)
        F = _augmented_residual(x, y, s, psi, g)
        if (F.norm(dim=1) <= _TOL).all():
            break
        dz = torch.linalg.solve(_jz(s, g, H), F.unsqueeze(2)).squeeze(2)
        y = y - dz[:, :dim]
        s = s - dz[:, dim]
    psi, g, H = _psi_grad_hess(oracle, y)
    ok = _augmented_residual(x, y, s, psi, g).norm(dim=1) <= _OK_TOL
    return y, s, ok


class _ClosestPoint(torch.autograd.Function):
    """y(x; theta) with IFT backward. Outputs: (y, ok); ok is non-differentiable.
    Non-converged points get zero cotangent flow (their rows are masked)."""

    @staticmethod
    def forward(ctx, x, oracle, *params):
        with torch.no_grad():
            y, s, ok = _newton_iterate(oracle, x)
        ctx.oracle = oracle
        ctx.save_for_backward(x.detach(), y, s, ok)
        ctx.mark_non_differentiable(ok)
        return y, ok

    @staticmethod
    def backward(ctx, ybar, _okbar):
        oracle = ctx.oracle
        x, y, s, ok = ctx.saved_tensors
        N, dim = y.shape

        # Rebuild grad/Hessian graph at the converged y (fresh leaf).
        yv = y.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            psi = oracle.psi(yv)
            (g,) = torch.autograd.grad(psi.sum(), yv, create_graph=True)
            H = torch.stack(
                [torch.autograd.grad(g[:, i].sum(), yv, create_graph=True)[0]
                 for i in range(dim)], dim=1)               # [N, dim, dim]

            # J_z, batched; identity rows for non-converged points (masked rhs).
            Jz = _jz(s, g.detach(), H.detach())
            okf = ok.to(torch.float64).unsqueeze(1)
            Jz[~ok] = torch.eye(dim + 1, dtype=torch.float64)
            rhs = torch.cat([ybar * okf, torch.zeros(N, 1, dtype=torch.float64)], dim=1)
            mu = torch.linalg.solve(Jz.transpose(1, 2), rhs.unsqueeze(2)).squeeze(2)
            mu = mu.detach()

            # theta_bar = d<F, -mu>/d theta at fixed z (yv is a leaf; its grads
            # are simply not requested).
            F = torch.cat([yv - x + s.unsqueeze(1) * g, psi.unsqueeze(1)], dim=1)
            loss = (F * (-mu)).sum()
            params = list(oracle.params)
            need = [p.requires_grad for p in params]
            grads = [None] * len(params)
            if any(need):
                got = torch.autograd.grad(
                    loss, [p for p, n in zip(params, need) if n],
                    allow_unused=True)
                it = iter(got)
                grads = [next(it) if n else None for n in need]

        xbar = mu[:, :dim]                                   # dF/dx = [-I; 0]
        return (xbar, None, *grads)


def closest_point(oracle, x_t: torch.Tensor):
    """Differentiable closest-point projection: (y, ok)."""
    return _ClosestPoint.apply(x_t, oracle, *oracle.params)


def _normal_at(oracle, y):
    """n = grad_psi(y)/|grad_psi(y)|, autograd-composed with y's history."""
    needs = torch.is_grad_enabled() and (
        y.requires_grad or any(p.requires_grad for p in oracle.params))
    yv = y if y.requires_grad else y.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        psi = oracle.psi(yv)
        (g,) = torch.autograd.grad(psi.sum(), yv, create_graph=needs)
    n = g / g.norm(dim=1, keepdim=True).clamp_min(1e-300)
    return n if needs else n.detach()


def distance_torch(oracle, x_np):
    """(d, n, ok) as torch tensors, graph-connected to oracle.params.

    d = y - x (surrogate GP -> closest point on Gamma); n = normalized
    grad psi at y (direction of increasing psi).
    """
    x = torch.as_tensor(np.ascontiguousarray(x_np, np.float64))
    if oracle.near_eikonal:
        xv = x.clone().requires_grad_(True)
        with torch.enable_grad():
            psi = oracle.psi(xv)
            (g,) = torch.autograd.grad(psi.sum(), xv, create_graph=True)
        gn2 = (g * g).sum(dim=1)
        with torch.no_grad():
            eik_err = (gn2.sqrt() - 1.0).abs().max()
        if eik_err < 1e-9:
            # Exact-SDF shortcut: grad psi is constant along the normal ray,
            # so n at y equals n at x. Fully autograd, no custom Function.
            d = -(psi / gn2).unsqueeze(1) * g
            n = g / gn2.sqrt().unsqueeze(1)
            ok = torch.ones(len(x), dtype=torch.bool)
            return d, n, ok
        import warnings
        warnings.warn(
            f"near_eikonal backend has |grad psi| deviation {float(eik_err):.2e}"
            " in the query band; falling back to Newton projection")
    y, ok = closest_point(oracle, x)
    d = y - x
    n = _normal_at(oracle, y)
    return d, n, ok


def distance_numpy(oracle, x_np):
    """Detached FP64 numpy (d, n, ok) — the forward-pipeline path."""
    with torch.no_grad():
        if oracle.near_eikonal:
            d, n, ok = distance_torch(oracle, x_np)
            return d.numpy(), n.numpy(), ok.numpy()
    x = torch.as_tensor(np.ascontiguousarray(x_np, np.float64))
    y, s, ok = _newton_iterate(oracle, x)
    d = (y - x).numpy()
    _, g = _psi_grad(oracle, y)
    n = (g / g.norm(dim=1, keepdim=True).clamp_min(1e-300)).numpy()
    return d, n, ok.numpy()
