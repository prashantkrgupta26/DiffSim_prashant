"""P11 verification tests for the vitrification / mobility-arrest 1-D CH brick.

All tests must PASS.  They are real numerical checks (tolerance-gated, honest):
  (a) analytic f', f'' match central differences of f;
  (b) the HAND-DERIVED analytic Jacobian matches a column-wise central-difference
      Jacobian on a random smooth state (THE contributor-workflow gate) and an
      autograd cross-check;
  (c) LIMITING CASE (arrest off -> constant mobility): the measured single-mode
      CH growth rate matches the analytic dispersion sigma(k);
  (d) MMS spatial convergence is ~2nd order (L2 error drops ~4x per doubling);
  (e) temporal order of backward Euler is ~1;
  (f) SCIENTIFIC RESULT: vitrification arrest HALTS coarsening (L plateaus and
      is markedly smaller than the un-arrested run).

Run:  PYTHONPATH=<repo>/src python -m pytest test_vitrification.py -x -q
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import vitrification as V

DEV = V.pick_device()


def _grid(N, L=1.0):
    h = L / N
    x = torch.arange(N, dtype=V.DTYPE, device=DEV) * h
    return x, h


# --------------------------------------------------------------------------
# (a) analytic derivatives vs central differences of f
# --------------------------------------------------------------------------
def test_analytic_derivatives():
    phi = torch.linspace(0.02, 0.98, 50, dtype=V.DTYPE, device=DEV)
    eps = 1e-6
    fp_cd = (V.f(phi + eps) - V.f(phi - eps)) / (2 * eps)
    fpp_cd = (V.fprime(phi + eps) - V.fprime(phi - eps)) / (2 * eps)
    e1 = float((V.fprime(phi) - fp_cd).abs().max())
    e2 = float((V.fpp(phi) - fpp_cd).abs().max())
    print(f"\n[a] f'_err={e1:.3e}  f''_err={e2:.3e}")
    assert e1 < 1e-6
    assert e2 < 1e-6
    # f''' too (used by the MMS source): central diff of f''
    fppp_cd = (V.fpp(phi + eps) - V.fpp(phi - eps)) / (2 * eps)
    assert float((V.fppp(phi) - fppp_cd).abs().max()) < 1e-5


def test_arrest_limits_and_derivative():
    """a(phi): ~1 below phi_g, ~0 above; analytic M'(phi) matches central diff;
    the phi_g -> inf limit gives constant mobility M0."""
    phi_g, w, M0 = 0.7, 0.05, 1.3
    lo = V.arrest(torch.tensor(0.2, dtype=V.DTYPE), phi_g, w)
    hi = V.arrest(torch.tensor(0.95, dtype=V.DTYPE), phi_g, w)
    assert float(lo) > 0.99 and float(hi) < 0.01
    phi = torch.linspace(0.05, 0.95, 40, dtype=V.DTYPE, device=DEV)
    eps = 1e-7
    Mp_cd = (V.Mmob(phi + eps, M0, phi_g, w)
             - V.Mmob(phi - eps, M0, phi_g, w)) / (2 * eps)
    assert float((V.Mprime(phi, M0, phi_g, w) - Mp_cd).abs().max()) < 1e-5
    # limiting case: phi_g huge -> M -> M0 everywhere
    Mlim = V.Mmob(phi, M0, 1e6, w)
    assert float((Mlim - M0).abs().max()) < 1e-9


# --------------------------------------------------------------------------
# (b) Jacobian FD-check — THE key contributor-workflow gate
# --------------------------------------------------------------------------
def test_jacobian_fd_check():
    N, L = 16, 1.0
    x, h = _grid(N, L)
    kappa, M0, phi_g, w, dt = 1e-4, 1.0, 0.7, 0.05, 1e-3
    torch.manual_seed(0)
    phi = (0.5 + 0.2 * torch.sin(2 * math.pi * x / L)
           + 0.05 * torch.cos(6 * math.pi * x / L)
           + 0.02 * torch.randn(N, dtype=V.DTYPE, device=DEV))
    phi_old = phi.clone()
    Jana = V.jacobian(phi, dt, kappa, M0, phi_g, w, h)
    Jfd = torch.zeros_like(Jana)
    eps = 1e-6
    for j in range(N):
        dp = torch.zeros(N, dtype=V.DTYPE, device=DEV)
        dp[j] = eps
        Rp = V.residual(phi + dp, phi_old, dt, kappa, M0, phi_g, w, h)
        Rm = V.residual(phi - dp, phi_old, dt, kappa, M0, phi_g, w, h)
        Jfd[:, j] = (Rp - Rm) / (2 * eps)
    max_err = float((Jana - Jfd).abs().max())
    print(f"\n[b] jacobian FD-check max abs err = {max_err:.3e} "
          f"(|J|max={float(Jana.abs().max()):.1f})")
    assert max_err < 1e-6


def test_jacobian_vs_autograd():
    """The hand-assembled analytic Jacobian equals autograd's (independent
    cross-check that the by-hand chain rule is right)."""
    N, L = 20, 1.0
    x, h = _grid(N, L)
    kappa, M0, phi_g, w, dt = 3e-4, 1.1, 0.65, 0.04, 2e-3
    torch.manual_seed(1)
    phi = 0.5 + 0.25 * torch.sin(2 * math.pi * x / L) + \
        0.03 * torch.randn(N, dtype=V.DTYPE, device=DEV)
    phi_old = phi.clone()
    Jana = V.jacobian(phi, dt, kappa, M0, phi_g, w, h)
    Jag = torch.autograd.functional.jacobian(
        lambda p: V.residual(p, phi_old, dt, kappa, M0, phi_g, w, h),
        phi.clone())
    err = float((Jana - Jag).abs().max())
    print(f"\n[b'] jacobian vs autograd max err = {err:.3e}")
    assert err < 1e-9


# --------------------------------------------------------------------------
# (c) limiting-case CH dispersion (arrest off -> constant mobility)
# --------------------------------------------------------------------------
def test_dispersion_limiting_case():
    N, L = 64, 1.0
    x, h = _grid(N, L)
    kappa, M0, w = 1e-4, 1.0, 0.05
    phi_g = 1e6                     # arrest OFF -> constant mobility
    phi_bar, A, dt = 0.5, 1e-3, 1e-4
    q = 2 * math.pi / L
    phi0 = phi_bar + A * torch.sin(q * x)
    phi1, _ = V.step(phi0, dt, kappa, M0, phi_g, w, h)
    proj0 = float((phi0 * torch.sin(q * x)).sum() * 2 / N)
    proj1 = float((phi1 * torch.sin(q * x)).sum() * 2 / N)
    sigma_meas = math.log(proj1 / proj0) / dt
    fpp_bar = float(V.fpp(torch.tensor(phi_bar, dtype=V.DTYPE)))
    sigma_ana = -M0 * q * q * (fpp_bar + kappa * q * q)
    rel = abs(sigma_meas - sigma_ana) / abs(sigma_ana)
    print(f"\n[c] sigma_ana={sigma_ana:.5f} sigma_meas={sigma_meas:.5f} "
          f"rel_err={rel:.3e}")
    assert sigma_ana > 0                       # phi_bar=0.5 is spinodal (grows)
    assert rel < 0.01


# --------------------------------------------------------------------------
# (d) MMS spatial convergence (~2nd order)
# --------------------------------------------------------------------------
def test_mms_spatial_convergence():
    L = 1.0
    kappa, M0, phi_g, w = 3e-4, 1.0, 0.7, 0.05
    phi_bar, A, lam = 0.5, 0.1, 1.0
    q = 2 * math.pi / L
    dt, nsteps = 1e-5, 4
    meshes = [32, 64, 128]
    errs = []
    for N in meshes:
        x, h = _grid(N, L)
        phi = V.mms_phi(x, 0.0, phi_bar, A, q, lam)
        for n in range(nsteps):
            tn = (n + 1) * dt
            src = V.mms_source(x, tn, phi_bar, A, q, lam, kappa, M0, phi_g, w)
            phi, _ = V.step(phi, dt, kappa, M0, phi_g, w, h, source=src)
        exact = V.mms_phi(x, nsteps * dt, phi_bar, A, q, lam)
        errs.append(float(torch.sqrt(((phi - exact) ** 2).mean())))
    rates = [math.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    print(f"\n[d] MMS errs={['%.3e' % e for e in errs]} rates={np.round(rates,3)}")
    assert min(rates) > 1.9            # clean 2nd-order spatial convergence


# --------------------------------------------------------------------------
# (e) temporal order (backward Euler ~1st order)
# --------------------------------------------------------------------------
def test_temporal_order():
    N, L = 48, 1.0
    x, h = _grid(N, L)
    kappa, M0, w = 1e-4, 1.0, 0.05
    phi_g = 1e6
    torch.manual_seed(2)
    phi0 = (0.15 + 0.05 * torch.sin(2 * math.pi * x / L)
            + 0.02 * torch.cos(4 * math.pi * x / L)
            + 0.01 * torch.randn(N, dtype=V.DTYPE, device=DEV))
    T = 0.05

    def run_dt(nst):
        dt = T / nst
        phi = phi0.clone()
        for _ in range(nst):
            phi, _ = V.step(phi, dt, kappa, M0, phi_g, w, h)
        return phi

    ref = run_dt(2048)
    nsts = [16, 32, 64, 128]
    errs = [float(torch.sqrt(((run_dt(n) - ref) ** 2).mean())) for n in nsts]
    rates = [math.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    print(f"\n[e] temporal errs={['%.3e' % e for e in errs]} "
          f"rates={np.round(rates,3)}")
    assert min(rates) > 0.85 and np.mean(rates) < 1.25   # ~1st order


# --------------------------------------------------------------------------
# (f) SCIENTIFIC RESULT: vitrification arrest halts coarsening
# --------------------------------------------------------------------------
def test_coarsening_arrest_halts():
    N, L = 64, 1.0
    x, h = _grid(N, L)
    kappa, M0, phi_g, w = 3e-4, 1.0, 0.7, 0.05
    dt, nsteps = 1e-3, 300
    mean = phi_g                       # polymer-rich matrix vitrifies
    torch.manual_seed(11)
    phi0 = mean + 0.05 * (2 * torch.rand(N, dtype=V.DTYPE, device=DEV) - 1)

    free = V.march(phi0, dt, nsteps, kappa, M0, 1e6, w, h, L, record_every=10)
    arr = V.march(phi0, dt, nsteps, kappa, M0, phi_g, w, h, L, record_every=10)
    Lf, La = float(free["Lt"][-1]), float(arr["Lt"][-1])
    Lt_a = arr["Lt"]
    half = len(Lt_a) // 2
    plateau = (float(Lt_a[-1]) - float(Lt_a[half])) / float(Lt_a[-1])
    # mass (mean phi) must be conserved by the conservative flux-divergence form
    mass_drift = float(np.abs(np.asarray(arr["mass"])
                              - float(arr["mass"][0])).max())
    print(f"\n[f] L_free={Lf:.4f} L_arrest={La:.4f} ratio={Lf/La:.3f} "
          f"plateau_last_half_growth={plateau:+.3f} mass_drift={mass_drift:.2e}")
    assert Lf > La * 1.3               # arrest clearly halts coarsening
    assert plateau < 0.10              # arrested domain scale plateaus
    assert mass_drift < 1e-10          # conservative form conserves mass


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-x", "-q", "-s"]))
