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
_OK_TOL = 1e-10  # per-point success threshold reported in the mask.
# 1e-10, not 1e-12: INR-family fields converge the augmented residual to
# ~1e-12 +- a batch-composition-dependent ulp wobble at isolated marginal
# points (measured: the same point flipping ok across two calls in one
# run, newton_ok_frac 1.0 vs 975/976). At |F| ~ 1e-10 the IFT gradient
# error is O(1e-10) relative — far below every other tolerance in the
# chain — while the flakiness disappears.
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


def _newton_iterate(oracle, x, y0=None):
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
    # y0 (FOOT CONTINUATION): warm-start from a previous nearby solve's
    # feet — keeps the BRANCH SELECTION continuous across small geometry
    # perturbations. Measured need: wrinkly provided-INR surfaces put
    # medial points chronically near the boundary; cold-started feet flip
    # branches at alpha-perturbations ~1e-5, making any objective through
    # the face terms micro-nonsmooth (J/|g| ~ 7e-6). With continuation,
    # r(alpha) is smooth on the selected branch — which is exactly what
    # the IFT backward differentiates.
    y = x.clone() if y0 is None else torch.as_tensor(
        np.ascontiguousarray(y0, np.float64)).clone()
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
    if not bool(ok.all()):
        # RESCUE (measured need: provided-INR fields — slight
        # non-monotonicity defeats undamped Newton at a handful of points,
        # e.g. 3/112 on the sphere checkpoint at L4 with c0=0.96). Damped
        # Newton with backtracking on |F|, lstsq for near-singular Jz
        # (also the evaluation's brittleness item). Subset-only: cost is
        # per-failure, not per-batch.
        idx = torch.where(~ok)[0]
        yr = y[idx].clone()
        sr = s[idx].clone()
        xr = x[idx]
        for _ in range(60):
            psi_r, g_r, H_r = _psi_grad_hess(oracle, yr)
            Fr = _augmented_residual(xr, yr, sr, psi_r, g_r)
            fn = Fr.norm(dim=1)
            if (fn <= _TOL).all():
                break
            Jz = _jz(sr, g_r, H_r)
            dz = torch.linalg.lstsq(Jz, Fr.unsqueeze(2)).solution.squeeze(2)
            # backtracking: accept the largest t in {1, 1/2, ..., 1/64}
            # that reduces |F| per point
            t = torch.ones(len(idx), dtype=torch.float64)
            best_y, best_s, best_f = yr.clone(), sr.clone(), fn.clone()
            for _bt in range(7):
                y_t = yr - t.unsqueeze(1) * dz[:, :dim]
                s_t = sr - t * dz[:, dim]
                psi_t, g_t = _psi_grad(oracle, y_t)
                f_t = _augmented_residual(xr, y_t, s_t, psi_t, g_t
                                          ).norm(dim=1)
                better = f_t < best_f
                best_y[better] = y_t[better]
                best_s[better] = s_t[better]
                best_f[better] = f_t[better]
                t = t * 0.5
            yr, sr = best_y, best_s
        y = y.clone()
        s = s.clone()
        y[idx] = yr
        s[idx] = sr
        psi, g, H = _psi_grad_hess(oracle, y)
        ok = _augmented_residual(x, y, s, psi, g).norm(dim=1) <= _OK_TOL
        if not bool(ok.all()):
            # RESCUE STAGE 2: bracket + bisect the zero crossing along the
            # gradient ray from x, then re-Newton from ON the zero set
            # (a warm start Newton demonstrably converges from; the ray
            # march is robust to the wiggle that defeats damped Newton at
            # isolated points — measured 1/976 on the sphere checkpoint).
            idx2 = torch.where(~ok)[0]
            x2 = x[idx2]
            psi0, g0 = _psi_grad(oracle, x2)
            dhat = -torch.sign(psi0).unsqueeze(1) * g0 / g0.norm(
                dim=1, keepdim=True).clamp_min(1e-300)
            # bracket: march until sign change (up to 1 domain unit)
            t_lo = torch.zeros(len(idx2), dtype=torch.float64)
            t_hi = torch.full((len(idx2),), 1e-3, dtype=torch.float64)
            # BOUNDED march: surrogate points sit within O(h) of the true
            # surface by construction; an unbounded march on an INR field
            # escapes the trained window and finds SPURIOUS periodic-sine
            # zero sheets (measured: a foot point |d| = 44.65 domains away
            # -> garbage FD). Cap the bracket at a domain-scale distance.
            T_MAX = 0.25
            for _ in range(40):
                p_hi, _ = _psi_grad(oracle, x2 + t_hi.unsqueeze(1) * dhat)
                need = ((p_hi * psi0) > 0) & (t_hi < T_MAX)
                if not bool(need.any()):
                    break
                t_lo = torch.where(need, t_hi, t_lo)
                t_hi = torch.where(need, torch.clamp(t_hi * 1.6, max=T_MAX),
                                   t_hi)
            for _ in range(60):                    # bisect
                t_mid = 0.5 * (t_lo + t_hi)
                p_mid, _ = _psi_grad(oracle, x2 + t_mid.unsqueeze(1) * dhat)
                same = (p_mid * psi0) > 0
                t_lo = torch.where(same, t_mid, t_lo)
                t_hi = torch.where(same, t_hi, t_mid)
            y2 = x2 + (0.5 * (t_lo + t_hi)).unsqueeze(1) * dhat
            _, g2 = _psi_grad(oracle, y2)
            s2_ = ((x2 - y2) * g2).sum(dim=1) / (g2 * g2).sum(
                dim=1).clamp_min(1e-300)
            for _ in range(_MAXIT):                # re-Newton from the set
                psi_r, g_r, H_r = _psi_grad_hess(oracle, y2)
                Fr = _augmented_residual(x2, y2, s2_, psi_r, g_r)
                if (Fr.norm(dim=1) <= _TOL).all():
                    break
                dz = torch.linalg.lstsq(
                    _jz(s2_, g_r, H_r), Fr.unsqueeze(2)).solution.squeeze(2)
                y2 = y2 - dz[:, :dim]
                s2_ = s2_ - dz[:, dim]
            # accept stage-2 results only when they IMPROVED on stage 1
            # and stayed domain-local (spurious far sheets are worse than
            # an honestly-failed mask)
            psi_n, g_n = _psi_grad(oracle, y2)
            F_new = _augmented_residual(x2, y2, s2_, psi_n, g_n).norm(dim=1)
            y_cur = y[idx2]
            s_cur = s[idx2]
            psi_o, g_o = _psi_grad(oracle, y_cur)
            F_old = _augmented_residual(x2, y_cur, s_cur, psi_o, g_o
                                        ).norm(dim=1)
            take = (F_new < F_old) & ((y2 - x2).norm(dim=1) < 0.5)
            y2 = torch.where(take.unsqueeze(1), y2, y_cur)
            s2_ = torch.where(take, s2_, s_cur)
            y[idx2] = y2
            s[idx2] = s2_
            psi, g, H = _psi_grad_hess(oracle, y)
            ok = _augmented_residual(x, y, s, psi, g).norm(dim=1) <= _OK_TOL
            # global domain-locality bound: any accepted foot point farther
            # than half a domain is a spurious sheet, never a projection
            ok = ok & ((y - x).norm(dim=1) < 0.5)
        if not bool(ok.all()):
            # RESCUE STAGE 3: multi-start damped Newton — robust to
            # near-tangent gradient rays on surface wrinkles (measured:
            # one GP whose ray misses the local sheet entirely). Fixed
            # deterministic jitter stencil; best-|F| local result wins.
            idx3 = torch.where(~ok)[0]
            x3 = x[idx3]
            n3 = len(idx3)
            offs = torch.tensor(
                [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                 [0, 0, 1], [0, 0, -1], [1, 1, 1], [-1, -1, -1]][:2 ** dim],
                dtype=torch.float64)[:, :dim] * 0.04
            best_y = y[idx3].clone()
            best_s = s[idx3].clone()
            psi_b, g_b = _psi_grad(oracle, best_y)
            best_F = _augmented_residual(x3, best_y, best_s, psi_b, g_b
                                         ).norm(dim=1)
            for o_ in offs:
                yj = x3 + o_.unsqueeze(0)
                for _ in range(20):                     # gradient descent
                    pj, gj = _psi_grad(oracle, yj)
                    den = (gj * gj).sum(dim=1).clamp_min(1e-300)
                    yj = yj - (pj / den).unsqueeze(1) * gj
                pj, gj = _psi_grad(oracle, yj)
                sj = ((x3 - yj) * gj).sum(dim=1) / (gj * gj).sum(
                    dim=1).clamp_min(1e-300)
                for _ in range(30):                     # damped Newton
                    pj, gj, Hj = _psi_grad_hess(oracle, yj)
                    Fj = _augmented_residual(x3, yj, sj, pj, gj)
                    if (Fj.norm(dim=1) <= _TOL).all():
                        break
                    dz = torch.linalg.lstsq(
                        _jz(sj, gj, Hj), Fj.unsqueeze(2)).solution.squeeze(2)
                    t = torch.ones(n3, dtype=torch.float64)
                    cy, cs, cf = yj.clone(), sj.clone(), Fj.norm(dim=1)
                    for _bt in range(5):
                        y_t = yj - t.unsqueeze(1) * dz[:, :dim]
                        s_t = sj - t * dz[:, dim]
                        p_t, g_t = _psi_grad(oracle, y_t)
                        f_t = _augmented_residual(x3, y_t, s_t, p_t, g_t
                                                  ).norm(dim=1)
                        better = f_t < cf
                        cy[better] = y_t[better]
                        cs[better] = s_t[better]
                        cf[better] = f_t[better]
                        t = t * 0.5
                    yj, sj = cy, cs
                pj, gj = _psi_grad(oracle, yj)
                Fj = _augmented_residual(x3, yj, sj, pj, gj).norm(dim=1)
                local = (yj - x3).norm(dim=1) < 0.5
                win = (Fj < best_F) & local
                best_y[win] = yj[win]
                best_s[win] = sj[win]
                best_F[win] = Fj[win]
            y[idx3] = best_y
            s[idx3] = best_s
            psi, g, H = _psi_grad_hess(oracle, y)
            ok = _augmented_residual(x, y, s, psi, g).norm(dim=1) <= _OK_TOL
            ok = ok & ((y - x).norm(dim=1) < 0.5)
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


def distance_numpy(oracle, x_np, y0=None):
    """Detached FP64 numpy (d, n, ok) — the forward-pipeline path."""
    with torch.no_grad():
        if oracle.near_eikonal:
            d, n, ok = distance_torch(oracle, x_np)
            return d.numpy(), n.numpy(), ok.numpy()
    x = torch.as_tensor(np.ascontiguousarray(x_np, np.float64))
    y, s, ok = _newton_iterate(oracle, x, y0=y0)
    d = (y - x).numpy()
    _, g = _psi_grad(oracle, y)
    n = (g / g.norm(dim=1, keepdim=True).clamp_min(1e-300)).numpy()
    return d, n, ok.numpy()
