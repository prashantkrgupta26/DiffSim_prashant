"""P11 — VITRIFICATION / composition-dependent mobility arrest (1-D Cahn-Hilliard).

A self-contained, tutorial-local numerical brick (NOT a core ``src/`` change).
It pairs with P10's drying film: as an organic film dries the polymer-rich phase
concentrates and its molecular mobility collapses at a glass composition -- the
morphology *freezes*.  We add a composition-dependent mobility with a smooth
vitrification *arrest* factor to a standard 1-D Cahn-Hilliard (CH) solver, derive
its ANALYTIC Jacobian by hand, and verify everything (analytic-derivative checks,
a hand-derived Jacobian FD-check, a limiting-case CH dispersion, MMS spatial
convergence, temporal order, and the coarsening-arrest scientific result).

Model (1-D, periodic domain ``[0, L)``, ``N`` nodes, spacing ``h = L / N``)
--------------------------------------------------------------------------
* Order parameter ``phi in [0, 1]``.
* Double-well free-energy density  ``f(phi) = phi^2 (1-phi)^2`` (minima 0, 1).
    ``f'(phi)  = 2 phi (1-phi)(1-2 phi)``
    ``f''(phi) = 2 (1 - 6 phi + 6 phi^2)``
    ``f'''(phi)= 12 (2 phi - 1)``
* Chemical potential   ``mu = f'(phi) - kappa * Lap(phi)`` with the periodic
  second difference ``Lap(phi)_i = (phi_{i+1} - 2 phi_i + phi_{i-1}) / h^2``.
* Mobility ``M(phi) = M0 * a(phi)`` with the vitrification *arrest* factor
    ``a(phi) = 1 / (1 + exp((phi - phi_g)/w)) = sigmoid((phi_g - phi)/w)``.
  ``a ~ 1`` for ``phi < phi_g`` (mobile / solvent-rich), ``a -> 0`` for
  ``phi > phi_g`` (vitrified / arrested).  Its derivative is
    ``a'(phi) = -(1/w) a (1-a)``,   so  ``M'(phi) = -(M0/w) a (1-a)``.
  LIMITING CASE: as ``phi_g -> +inf`` we have ``a -> 1``, ``M -> M0`` constant
  and the brick reduces to STANDARD constant-mobility CH -- used for the
  dispersion verification.
* Face-averaged flux divergence (arithmetic face mean -> clean analytic Jacobian)
    ``div(M grad mu)_i = [ Mf_{i+1/2}(mu_{i+1}-mu_i)
                           - Mf_{i-1/2}(mu_i-mu_{i-1}) ] / h^2``,
  with ``Mf_{i+1/2} = 0.5 (M_i + M_{i+1})``.
* Time: backward Euler.  Residual
    ``R(phi) = (phi - phi_old)/dt - div(M(phi) grad mu(phi))``.

Everything runs in torch float64, preferring CUDA (falls back to CPU).
"""
from __future__ import annotations

import math

import torch

# --------------------------------------------------------------------------
# device / dtype helpers
# --------------------------------------------------------------------------
DTYPE = torch.float64


def pick_device(prefer="cuda"):
    """Return a torch device, preferring CUDA, falling back to CPU."""
    if prefer.startswith("cuda") and torch.cuda.is_available():
        return torch.device(prefer if ":" in prefer else "cuda:0")
    return torch.device("cpu")


def as_tensor(x, device=None):
    """Coerce ``x`` to a 1-D float64 tensor.

    An existing tensor keeps its OWN device unless ``device`` is given
    explicitly (so a CPU input stays on the CPU even when CUDA is available);
    only a freshly-built tensor from a list/array defaults to ``pick_device()``.
    """
    if torch.is_tensor(x):
        return x.to(dtype=DTYPE) if device is None else x.to(device=device,
                                                             dtype=DTYPE)
    return torch.as_tensor(x, dtype=DTYPE, device=device or pick_device())


# --------------------------------------------------------------------------
# free energy and its analytic derivatives
# --------------------------------------------------------------------------
def f(phi):
    """Double-well free-energy density f(phi) = phi^2 (1-phi)^2."""
    return phi ** 2 * (1.0 - phi) ** 2


def fprime(phi):
    """f'(phi) = 2 phi (1-phi)(1-2 phi)  (hand-derived, unit-tested vs CD of f)."""
    return 2.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi)


def fpp(phi):
    """f''(phi) = 2 (1 - 6 phi + 6 phi^2)."""
    return 2.0 * (1.0 - 6.0 * phi + 6.0 * phi ** 2)


def fppp(phi):
    """f'''(phi) = 12 (2 phi - 1)  (used only by the MMS analytic source)."""
    return 12.0 * (2.0 * phi - 1.0)


# --------------------------------------------------------------------------
# vitrification arrest factor and mobility
# --------------------------------------------------------------------------
def arrest(phi, phi_g, w):
    """Smooth vitrification arrest a(phi) = sigmoid((phi_g - phi)/w) in (0, 1].

    a ~ 1 (mobile) for phi < phi_g;  a -> 0 (arrested) for phi > phi_g.
    Uses ``torch.sigmoid`` for overflow-free evaluation at large |phi-phi_g|/w.
    """
    return torch.sigmoid((phi_g - phi) / w)


def Mmob(phi, M0, phi_g, w):
    """Composition-dependent mobility M(phi) = M0 * a(phi)."""
    return M0 * arrest(phi, phi_g, w)


def Mprime(phi, M0, phi_g, w):
    """Analytic dM/dphi = -(M0/w) a (1-a)  (a = arrest factor)."""
    a = arrest(phi, phi_g, w)
    return -(M0 / w) * a * (1.0 - a)


# --------------------------------------------------------------------------
# discrete spatial operators (periodic)
# --------------------------------------------------------------------------
def laplacian(phi, h):
    """Periodic second difference (phi_{i+1} - 2 phi_i + phi_{i-1}) / h^2."""
    return (torch.roll(phi, -1) + torch.roll(phi, 1) - 2.0 * phi) / (h * h)


def chem_potential(phi, kappa, h):
    """mu = f'(phi) - kappa * Lap(phi)."""
    return fprime(phi) - kappa * laplacian(phi, h)


def flux_divergence(phi, kappa, M0, phi_g, w, h):
    """div(M(phi) grad mu(phi)) with arithmetic face-averaged mobility.

    ``Mf_{i+1/2} = 0.5 (M_i + M_{i+1})``; the divergence is the difference of
    the +face and -face fluxes over ``h^2``.
    """
    mu = chem_potential(phi, kappa, h)
    M = Mmob(phi, M0, phi_g, w)
    mu_ip = torch.roll(mu, -1)                 # mu_{i+1}
    M_ip = torch.roll(M, -1)                    # M_{i+1}
    mface_plus = 0.5 * (M + M_ip)               # Mf_{i+1/2}
    flux_plus = mface_plus * (mu_ip - mu)       # face flux at i+1/2
    # div_i = (flux_{i+1/2} - flux_{i-1/2}) / h^2 ;  flux_{i-1/2} = roll(flux,1)
    return (flux_plus - torch.roll(flux_plus, 1)) / (h * h)


def residual(phi, phi_old, dt, kappa, M0, phi_g, w, h):
    """Backward-Euler residual R(phi) = (phi - phi_old)/dt - div(M grad mu)."""
    return (phi - phi_old) / dt - flux_divergence(phi, kappa, M0, phi_g, w, h)


# --------------------------------------------------------------------------
# ANALYTIC Jacobian  dR/dphi  (hand-derived, periodic, dense N x N)
# --------------------------------------------------------------------------
def jacobian(phi, dt, kappa, M0, phi_g, w, h):
    """Dense analytic Jacobian J = dR/dphi (NOT autograd).

    Derivation (chain rule).  With ``D_i = div(M grad mu)_i`` and
    ``R_i = (phi_i - phi_old_i)/dt - D_i``,

        J = I/dt - dD/dphi ,   dD/dphi = A_mob + Aop @ J_mu ,

    where
      * ``J_mu = diag(f''(phi)) - kappa * Lap`` is d(mu)/d(phi);
      * ``Aop`` is the flux-divergence operator with the CURRENT face mobilities
        frozen (D = Aop @ mu), so its chain-rule contribution is ``Aop @ J_mu``;
      * ``A_mob`` is the extra term from differentiating the face mobilities
        ``Mf = 0.5(M_i + M_{i+-1})`` through ``M'(phi)``:
            A_mob[i,i]  += 0.5 M'_i (g+_i - g-_i)/h^2
            A_mob[i,i+1]+= 0.5 M'_{i+1} g+_i /h^2
            A_mob[i,i-1]+= -0.5 M'_{i-1} g-_i /h^2
        with g+_i = mu_{i+1}-mu_i and g-_i = mu_i-mu_{i-1}.
    """
    phi = as_tensor(phi)
    device = phi.device
    N = phi.shape[0]
    h2 = h * h
    idx = torch.arange(N, device=device)
    ip = (idx + 1) % N
    im = (idx - 1) % N

    # --- periodic Laplacian matrix and J_mu = d mu / d phi -----------------
    Lmat = torch.zeros((N, N), dtype=DTYPE, device=device)
    Lmat[idx, idx] = -2.0 / h2
    Lmat[idx, ip] += 1.0 / h2
    Lmat[idx, im] += 1.0 / h2
    J_mu = torch.diag(fpp(phi)) - kappa * Lmat

    # --- face mobilities and Aop (flux-divergence, mobilities frozen) ------
    M = Mmob(phi, M0, phi_g, w)
    mface_plus = 0.5 * (M + torch.roll(M, -1))          # Mf_{i+1/2}
    mface_minus = torch.roll(mface_plus, 1)             # Mf_{i-1/2}
    Aop = torch.zeros((N, N), dtype=DTYPE, device=device)
    Aop[idx, idx] = -(mface_plus + mface_minus) / h2
    Aop[idx, ip] += mface_plus / h2
    Aop[idx, im] += mface_minus / h2

    # --- mobility-derivative term A_mob ------------------------------------
    mu = chem_potential(phi, kappa, h)
    gplus = torch.roll(mu, -1) - mu                     # mu_{i+1} - mu_i
    gminus = mu - torch.roll(mu, 1)                     # mu_i - mu_{i-1}
    Mp = Mprime(phi, M0, phi_g, w)
    Mp_ip = torch.roll(Mp, -1)                          # M'_{i+1}
    Mp_im = torch.roll(Mp, 1)                           # M'_{i-1}
    A_mob = torch.zeros((N, N), dtype=DTYPE, device=device)
    A_mob[idx, idx] += 0.5 * Mp * (gplus - gminus) / h2
    A_mob[idx, ip] += 0.5 * Mp_ip * gplus / h2
    A_mob[idx, im] += -0.5 * Mp_im * gminus / h2

    dDdphi = A_mob + Aop @ J_mu
    eye = torch.eye(N, dtype=DTYPE, device=device)
    return eye / dt - dDdphi


# --------------------------------------------------------------------------
# Newton step / time marching
# --------------------------------------------------------------------------
def step(phi_old, dt, kappa, M0, phi_g, w, h, source=None,
         tol=1e-10, maxit=40):
    """One backward-Euler step: solve R(phi) - source = 0 by damped Newton.

    ``source`` (optional, MMS) is a fixed vector subtracted from the residual;
    it is phi-independent so it does not enter the Jacobian.  Convergence stops
    at an absolute residual ``tol`` OR when Newton stagnates at its roundoff
    floor (the Jacobian's large ~kappa/h^4 entries limit the achievable
    residual).  Returns ``(phi, info)`` with convergence diagnostics.
    """
    def _resid(p):
        R = residual(p, phi_old, dt, kappa, M0, phi_g, w, h)
        return R if source is None else R - source

    phi = phi_old.clone()
    last = float("inf")
    for it in range(maxit):
        R = _resid(phi)
        rn = float(torch.linalg.vector_norm(R))
        if rn < tol:
            return phi, {"iters": it, "resid": rn, "converged": True}
        # stagnation at the roundoff floor -> accept (residual can't drop more)
        if rn > 0.5 * last:
            return phi, {"iters": it, "resid": rn, "converged": True,
                         "stalled": True}
        J = jacobian(phi, dt, kappa, M0, phi_g, w, h)
        delta = torch.linalg.solve(J, R)
        alpha = 1.0                          # backtracking line search
        for _ in range(20):
            if float(torch.linalg.vector_norm(_resid(phi - alpha * delta))) \
                    < rn or alpha < 1e-4:
                break
            alpha *= 0.5
        phi = phi - alpha * delta
        last = rn
    return phi, {"iters": maxit, "resid": float(torch.linalg.vector_norm(
        _resid(phi))), "converged": False}


def march(phi0, dt, nsteps, kappa, M0, phi_g, w, h, L,
          record_every=1, source_fn=None, t0=0.0):
    """March ``nsteps`` backward-Euler steps.

    Returns a dict with ``t`` (times), ``traj`` (N_rec x N field snapshots),
    ``Lt`` (domain length scale at each recorded time), ``mass`` (mean phi,
    a conservation diagnostic) and ``max_iters`` (worst Newton iteration count).
    ``source_fn(t)`` -> vector adds an MMS source at the step's end time.
    """
    phi = phi0.clone()
    ts, traj, Lt, mass = [], [], [], []
    max_iters = 0

    def _record(tcur):
        ts.append(float(tcur))
        traj.append(phi.detach().clone())
        Lt.append(domain_scale(phi, L))
        mass.append(float(phi.mean()))

    _record(t0)
    t = t0
    for n in range(nsteps):
        t_next = t0 + (n + 1) * dt
        src = None if source_fn is None else source_fn(t_next)
        phi, info = step(phi, dt, kappa, M0, phi_g, w, h, source=src)
        max_iters = max(max_iters, info["iters"])
        t = t_next
        if (n + 1) % record_every == 0 or (n + 1) == nsteps:
            _record(t)
    return {
        "t": torch.tensor(ts, dtype=DTYPE),
        "traj": torch.stack(traj) if traj else phi[None],
        "Lt": torch.tensor(Lt, dtype=DTYPE),
        "mass": torch.tensor(mass, dtype=DTYPE),
        "max_iters": max_iters,
        "phi_final": phi,
    }


# --------------------------------------------------------------------------
# coarsening length-scale diagnostic L(t)
# --------------------------------------------------------------------------
def domain_scale(phi, L):
    """Characteristic domain size = 2*pi / <k>, the inverse first spectral moment.

    ``<k> = sum_k k S(k) / sum_k S(k)`` over the non-DC modes, with the periodic
    structure factor ``S(k) = |FFT(phi - mean)|^2`` and ``k = 2*pi n / L``.  As
    domains coarsen, spectral weight moves to low k and this length grows.
    """
    phi = as_tensor(phi)
    N = phi.shape[0]
    pf = phi - phi.mean()
    Fk = torch.fft.rfft(pf)
    S = (Fk.conj() * Fk).real
    n = torch.arange(S.shape[0], dtype=DTYPE, device=phi.device)
    k = 2.0 * math.pi * n / L
    num = (k[1:] * S[1:]).sum()
    den = S[1:].sum()
    if float(den) <= 0.0:
        return 0.0
    k1 = float(num / den)
    if k1 <= 0.0:
        return 0.0
    return 2.0 * math.pi / k1


# --------------------------------------------------------------------------
# Method-of-Manufactured-Solutions (MMS) analytic fields + source
# --------------------------------------------------------------------------
def mms_phi(x, t, phi_bar, A, q, lam):
    """Manufactured solution phi*(x,t) = phi_bar + A sin(q x) exp(-lam t)."""
    return phi_bar + A * torch.sin(q * x) * math.exp(-lam * t)


def mms_source(x, t, phi_bar, A, q, lam, kappa, M0, phi_g, w):
    """Exact CONTINUOUS source s = d phi*/dt - div(M(phi*) grad mu(phi*)).

    With phi* = phi_bar + A sin(qx) e^{-lam t}:
      phi_x  =  A q cos(qx) e^{-lam t},
      phi_xx = -q^2 (phi* - phi_bar),
      mu = f'(phi*) + kappa q^2 (phi* - phi_bar)  (since -kappa phi_xx = +kappa q^2 (.)),
      grad mu = (f''(phi*) + kappa q^2) phi_x,
      div(M grad mu) = G'(phi*) phi_x^2 + G(phi*) phi_xx,
      G(phi)  = M(phi)(f''(phi) + kappa q^2),
      G'(phi) = M'(phi)(f''(phi)+kappa q^2) + M(phi) f'''(phi).
    Evaluated analytically at the node points -> a clean 2nd-order MMS target.
    """
    E = math.exp(-lam * t)
    phi = phi_bar + A * torch.sin(q * x) * E
    phi_x = A * q * torch.cos(q * x) * E
    phi_xx = -(q * q) * (phi - phi_bar)
    phi_t = -lam * (phi - phi_bar)
    kq2 = kappa * q * q
    M = Mmob(phi, M0, phi_g, w)
    Mp = Mprime(phi, M0, phi_g, w)
    G = M * (fpp(phi) + kq2)
    Gp = Mp * (fpp(phi) + kq2) + M * fppp(phi)
    div_exact = Gp * phi_x ** 2 + G * phi_xx
    return phi_t - div_exact


__all__ = [
    "DTYPE", "pick_device", "as_tensor",
    "f", "fprime", "fpp", "fppp",
    "arrest", "Mmob", "Mprime",
    "laplacian", "chem_potential", "flux_divergence", "residual",
    "jacobian", "step", "march", "domain_scale",
    "mms_phi", "mms_source",
]
