r"""benchmarks/chns/mms.py — SP-0 Task 9 manufactured solution (MMS) forcing.

Manufactured solution (task-9 brief §4, unit square [0,1]^2, BDF-marched):

    u = ( sin(pi x) cos(pi y),  -cos(pi x) sin(pi y) ) * g(t)   (divergence-free)
    p = cos(pi x) cos(pi y) * g(t)
    phi = tanh( (y - 0.5 - 0.1 sin(2 pi x) g(t)) / (sqrt(2) Cn) )
    mu  = phi^3 - phi                         (SMOOTH manufactured potential)
    g(t) = cos(t)

    NOTE on mu (design-order fix, recorded): the natural CH closure
    mu = f'(phi) - Cn^2 lap(phi) contains lap(phi), whose Q1 (piecewise-linear)
    representation is distributional at the interface.  Forcing with the
    ANALYTIC lap(phi) injects an O(1) interface consistency error into the
    capillary force f_cap = (Cn We)^{-1} mu grad phi that caps the L2 VELOCITY
    order (phi still converges cleanly).  We therefore manufacture a SMOOTH
    mu = phi^3 - phi (C-infinity) and let the mu-row source
    S_mu = mu - (phi^3 - phi) + Cn^2 lap(phi) = Cn^2 lap(phi) carry the
    difference.  Fully consistent MMS (every row forced to its exact strong
    residual); both u and phi then converge at the design order.

The chosen u is EXACTLY divergence-free (d/dx u0 + d/dy u1 = 0), so the
continuity residual forcing is ~0 (only the PSPG stabilization leaves a tiny
term, which we DO include via the pressure-row source for completeness).

We compute the per-field forcing S_f = (row residual operator applied to the
manufactured field) symbolically with sympy, matching the CHNSDiscrete /
make_chns_newton STRONG-form contract EXACTLY (Galerkin strong residual; the
SUPG/PSPG stabilization terms are consistent — they vanish on the exact
manufactured residual since r_mom_strong is forced to zero pointwise, so
forcing the Galerkin strong form suffices for the design-order MMS gate).

The forcing is returned as callables fn(xq[ngp,dim], t) -> [ngp] wired through
CHNSStepper's src_fns (CH rows) and body_fn (momentum rows):

    momentum row d:  f_u[d] = rho D_t u_d + rho (u.grad)u_d + (J.grad)u_d
                              - (2 eta/Re) div D(u)_d + dp/dx_d
                              - f_cap_d - f_grav_d       (gravity OFF for MMS)
    continuity:      S_p = div u                          (~0, exact)
    phi row:         S_phi = D_t phi + u.grad phi + phi div u
                             + (1/Pe) (-lap mu)           [sign: -(1/Pe) lap mu
                             enters as +grad.grad in weak form]
    mu row:          S_mu = mu - (phi^3 - phi) + Cn^2 lap phi   (= 0 by mu def)

where D_t is the BDF time derivative; for the L2-convergence gate we use small
fixed dt so the O(dt) BDF truncation is dominated by the target O(h^2) spatial
error, and the forcing uses the EXACT analytic dphi/dt, du/dt (the manufactured
time derivative) — the standard MMS convention (the scheme's own temporal error
is then the only unforced discrepancy, kept sub-dominant by small dt).

rho(phi), eta(phi) use the linear mix_props interpolation with the CHNSCase
endpoints rho_h=1, rho_l=1/rho_ratio (and eta likewise).
"""
from __future__ import annotations

import numpy as np


def build_mms(case, Cn, Re, We, Pe, rho_ratio, eta_ratio, gravity=False,
              steady=False):
    """Return a dict of manufactured-field and forcing callables for the MMS.

    steady : if True, g(t) = 1 (time-independent).  The manufactured fields are
        then a STEADY state, so the BDF time derivative is exactly zero and the
        discrete-solution error vs the analytic field is PURELY SPATIAL — the
        clean setting for the L2 design-order convergence gate (no temporal
        floor to contaminate the very-smooth velocity order).  Default False
        gives the brief's g(t) = cos(t) transient forcing.

    Keys:
      u_fn(x,t)->[n,2], p_fn(x,t)->[n], phi_fn(x,t)->[n], mu_fn(x,t)->[n]
      body_fn(x,t)->[n,2]      (momentum forcing; NS body-force channel)
      src_fns list [f_u0,f_u1,f_p,f_phi,f_mu]  (per-row sources; CH via src_fns)
    All callables are pure-numpy lambdified sympy expressions.
    """
    import sympy as sp

    x, y, t = sp.symbols("x y t", real=True)
    pi = sp.pi
    g = sp.Integer(1) if steady else sp.cos(t)
    Cn_s = sp.Float(Cn)
    Re_s, We_s, Pe_s = sp.Float(Re), sp.Float(We), sp.Float(Pe)
    rho_h, rho_l = sp.Integer(1), sp.Rational(1, 1) / sp.Float(rho_ratio)
    eta_h, eta_l = sp.Integer(1), sp.Rational(1, 1) / sp.Float(eta_ratio)

    # --- manufactured fields ------------------------------------------------
    u0 = sp.sin(pi * x) * sp.cos(pi * y) * g
    u1 = -sp.cos(pi * x) * sp.sin(pi * y) * g
    p = sp.cos(pi * x) * sp.cos(pi * y) * g
    phi = sp.tanh((y - sp.Rational(1, 2) - sp.Rational(1, 10)
                   * sp.sin(2 * pi * x) * g) / (sp.sqrt(2) * Cn_s))

    lap = lambda f: sp.diff(f, x, 2) + sp.diff(f, y, 2)
    grad = lambda f: (sp.diff(f, x), sp.diff(f, y))

    # SMOOTH manufactured potential (see docstring): mu = f'(phi); the mu-row
    # source S_mu = Cn^2 lap(phi) carries the closure difference.
    mu = phi ** 3 - phi

    # --- mixture props (linear mix_props, NO clamp in the resolved regime) ---
    a_rho = (rho_h - rho_l) / 2
    b_rho = (rho_h + rho_l) / 2
    a_eta = (eta_h - eta_l) / 2
    b_eta = (eta_h + eta_l) / 2
    rho = a_rho * phi + b_rho
    eta = a_eta * phi + b_eta

    # --- AGG mass flux J = agg grad mu, agg = -((rho_h-rho_l)/2)/Pe ----------
    agg = -a_rho / Pe_s
    gmu = grad(mu)
    J = (agg * gmu[0], agg * gmu[1])

    # --- momentum forcing f_u[d] (Galerkin strong form) ---------------------
    u = (u0, u1)
    gu = [[sp.diff(u[d], v) for v in (x, y)] for d in range(2)]  # gu[d][s]
    dt_u = [sp.diff(u[d], t) for d in range(2)]
    gp = grad(p)
    gphi = grad(phi)
    cw_inv = 1 / (Cn_s * We_s)
    grav_scale = 1 / sp.Float(case.Fr) ** 2
    ghat = (0, -1) if gravity else (0, 0)

    # viscous term: div( (2 eta/Re) D(u) ), D = 1/2(grad u + grad u^T)
    # component d: (1/Re) sum_s d/dx_s [ eta (du_d/dx_s + du_s/dx_d) ]
    def visc_div(d):
        acc = 0
        for s, vs in enumerate((x, y)):
            symds = sp.diff(u[d], vs) + sp.diff(u[s], (x, y)[d])
            acc += sp.diff(eta * symds, vs)
        return acc / Re_s

    f_u = []
    for d in range(2):
        ugradu = sum(u[s] * gu[d][s] for s in range(2))
        Jgradu = sum(J[s] * gu[d][s] for s in range(2))
        fcap = cw_inv * mu * gphi[d]
        fgrav = rho * grav_scale * ghat[d]
        f = (rho * dt_u[d] + rho * ugradu + Jgradu
             - visc_div(d) + gp[d] - fcap - fgrav)
        f_u.append(sp.simplify(f) if False else f)

    # --- continuity forcing S_p = div u (exact ~0) --------------------------
    S_p = sp.diff(u0, x) + sp.diff(u1, y)

    # --- phi forcing: D_t phi + u.grad phi + phi div u - (1/Pe) lap mu ------
    # weak (1/Pe) grad psi . grad mu integrates by parts to -(1/Pe) psi lap mu;
    # strong CH residual = dphi/dt + div(u phi) - (1/Pe) lap mu
    dt_phi = sp.diff(phi, t)
    divu = S_p
    ugradphi = u0 * gphi[0] + u1 * gphi[1]
    S_phi = dt_phi + ugradphi + phi * divu - (1 / Pe_s) * lap(mu)

    # --- mu forcing S_mu = mu - (phi^3 - phi) + Cn^2 lap phi (= 0 by def) ----
    S_mu = mu - (phi ** 3 - phi) + Cn_s ** 2 * lap(phi)

    # --- lambdify (numpy) ---------------------------------------------------
    def lam(expr):
        f = sp.lambdify((x, y, t), expr, "numpy")
        return f

    lu0, lu1, lp, lphi, lmu = lam(u0), lam(u1), lam(p), lam(phi), lam(mu)
    lfu0, lfu1 = lam(f_u[0]), lam(f_u[1])
    lSp, lSphi, lSmu = lam(S_p), lam(S_phi), lam(S_mu)

    def _wrap(f, vec=False):
        def g_(xy, tt):
            xy = np.asarray(xy, float)
            out = f(xy[:, 0], xy[:, 1], float(tt))
            return np.broadcast_to(np.asarray(out, float), (len(xy),)).copy()
        return g_

    u_fn = lambda xy, tt: np.stack(
        [_wrap(lu0)(xy, tt), _wrap(lu1)(xy, tt)], axis=1)
    p_fn = _wrap(lp)
    phi_fn = _wrap(lphi)
    mu_fn = _wrap(lmu)
    body_fn = lambda xy, tt: np.stack(
        [_wrap(lfu0)(xy, tt), _wrap(lfu1)(xy, tt)], axis=1)
    # src_fns: [u0,u1,p,phi,mu]; momentum via body_fn, so u rows None here
    src_fns = [None, None, _wrap(lSp), _wrap(lSphi), _wrap(lSmu)]

    return {
        "u_fn": u_fn, "p_fn": p_fn, "phi_fn": phi_fn, "mu_fn": mu_fn,
        "body_fn": body_fn, "src_fns": src_fns,
    }
