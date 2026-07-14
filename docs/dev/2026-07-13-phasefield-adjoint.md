# Differentiable phase-field adjoint — dev ledger (2026-07-13)

The crown differentiable capability: a genuine, three-way-verified discrete
adjoint through DiffSim's phase-field stack, giving gradients w.r.t. material
parameters, processing parameters, free-energy coefficients, and process
schedules.  Mirrors the SBM transient adjoint (`sbm/transient_adjoint.py`) but
for the mixed Cahn-Hilliard / multiphase operator.

## Design

- **Module**: `src/diffsim/adjoint/` — `phasefield.py` (discrete forward +
  IFT adjoint), `torch_twin.py` (autograd reference, tests only).
- **Self-contained discrete forward** (`CHDiscrete`/`CHForward`): a vectorised
  numpy reimplementation of the production CH residual/Jacobian
  (`physics/cahn_hilliard.make_ch_newton`), pulling the identical basis tables
  (N, dN, w, h) and dof layout off the `DeviceMesh`.  It is **bit-parity** with
  the production `CahnHilliardStepper` (measured `max|dc| ~ 1e-16`), so the
  adjoint differentiates the real brick, not a look-alike.  The production
  bricks are UNCHANGED — the adjoint records alongside.
- **IFT adjoint** (`CHAdjoint`): the clean implicit-function-theorem adjoint.
  We do not tape Newton; we differentiate only the converged relation
  `R_n(x_n; x_hist, p) = 0`.  For `J = sum_n j(x_n)`:

      J_n^T lam_n = dj/dx_n  -  sum_{k>=1} (dR_{n+k}/dx_n)^T lam_{n+k}
      dJ/dp       = - sum_n lam_n^T (dR_n/dp)

  `J_n` is the exact forward Jacobian at convergence; `dR_n/dp` is analytic.
  BDF history enters `R_{n+k}` only through the mass/time term, so the history
  cotangent is `(ch_k/dt) * Mass * lam[c-rows]` mapped to step `n-k`'s
  c-columns — the CH analogue of the transient adjoint's `sigma/b1/b2` chain.
  Variable-coefficient BDF2 is covered: `ch = [1+r, -r^2/(1+r)]` gives both
  history slots.

- **Analytic parameter derivatives** (`CHDiscrete.dR_dparam`):
  - `dR_c/dM   = Int gradN . grad mu`
  - `dR_mu/dkappa = -Int gradN . grad c`
  - `dR_mu/dp (bulk coeff) = -Int N (df'/dp)` (FH: `df'/dB = 1-2c`,
    `df'/dA = ln c - ln(1-c)`; B is the Flory interaction chi).

## Verification gate (THREE-WAY, non-negotiable)

Objective `J = 0.5 |c_N - 0.5|^2`, FH bulk energy (A=1, B=2.5, interior so the
log regularisation is inactive), level-3 uniform mesh (81 nodes), dt=0.01.

### G1 — CH BDF1, 3 steps (`test_g1_ch_bdf1_material_params`)

| param | adjoint | autograd twin | central FD | adj/twin | adj/FD |
|-------|---------|---------------|------------|----------|--------|
| M       | +3.076615e-01 | +3.076615e-01 | +3.076615e-01 | 1.8e-16 | 3.3e-10 |
| kappa   | -7.356749e+00 | -7.356749e+00 | -7.356749e+00 | 1.2e-16 | 1.5e-09 |
| B (chi) | +8.195832e-01 | +8.195832e-01 | +8.195832e-01 | 1.4e-16 | 4.1e-11 |
| A       | -1.667729e+00 | -1.667729e+00 | -1.667729e+00 | 0.0     | 3.6e-11 |

### G2a — variable-coefficient BDF2, 4 steps (`test_g2a_ch_bdf2_material_params`)

| param | adjoint | autograd twin | central FD | adj/twin | adj/FD |
|-------|---------|---------------|------------|----------|--------|
| M       | +4.745873e-01 | +4.745873e-01 | +4.745873e-01 | 3.5e-16 | 2.0e-10 |
| kappa   | -1.152959e+01 | -1.152959e+01 | -1.152959e+01 | 4.6e-16 | 6.6e-10 |
| B (chi) | +1.270706e+00 | +1.270706e+00 | +1.270706e+00 | 1.8e-16 | 2.2e-11 |
| A       | -2.586881e+00 | -2.586881e+00 | -2.586881e+00 | 3.4e-16 | 2.9e-11 |

**Locked tolerances** (>=2x headroom): adj-vs-twin `< 1e-10` (measured ~1e-16),
adj-vs-FD `< 1e-6` (measured ~1e-9..1e-11).

### Forward parity (`test_forward_parity_production`)
poly-BDF1 `max|dc| = 1.1e-16`; FH-BDF2 `max|dc| = 3.3e-16`.  Locked `< 1e-12`.

### G2b/G2c — coupled crystallisation CH x AC (M=K=1)

`adjoint/crystallization.py`: node block (phi, mu, psi), representative
Turnbull/r14 free energy `f = f_FH(phi) + phi[q(psi)dsig + p(psi)drive]`,
`drive = dh(T/Tm-1)`.  Both phi and psi are BDF-stepped so the history
cotangent couples two slots.  Params: M, kappa, eps2, L, dsig, dh, Tm, A, B.

BDF1, 3 steps (`test_g2b_crystallization_bdf1`) — worst adj/twin `6.0e-16`,
worst adj/FD `8.9e-10` (all 9 params).
BDF2, 4 steps (`test_g2c_crystallization_bdf2`) — worst adj/twin `1.7e-15`,
worst adj/FD `4.1e-09`.  Sample (BDF1): dh `+2.661164e-01`, Tm `+4.435273e-02`,
dsig `-3.913890e-02`, eps2 `-2.743690e-01`, L `-5.534772e-02`.

## Capability status

- (a) MATERIAL PARAMETERS — **DELIVERED & VERIFIED.** Binary CH (M, kappa,
  Flory chi B, A) and coupled CH x AC crystallisation (dh, Tm, dsigma, eps2, L),
  BDF1 and variable-coefficient BDF2, all three-way to machine precision.
- (b) processing / (c) learn-free-energy / (d) design-route: pending.

## Scope notes
- Gate meshes are uniform (constraints.T == identity), natural no-flux BCs.
- The crystallisation gate is the (M,K)=(1,1) retained-species core; orientation
  theta (the 4th dof of the full 2M+2K block, KG-regularised |grad theta|) is a
  documented frontier — the crystallisation *parameters* live in the phi/psi
  coupling, which is fully covered.  Multi-species M>1 / matrix mobilities
  reuse the same IFT sweep with a larger block.

## Frontiers / next
- G3 processing params (evap rate k_e, quench schedule T(t) — a time series;
  note `dR/dT = dh/Tm` per GP is already implemented as the "T" param deriv).
- G4 learn f(c) from a trajectory; G5 design a process schedule.
