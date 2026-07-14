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

### G3 — processing param: quench schedule T(t) (`test_g3_temperature_schedule`)

`CACHForward.set_schedule(T_list, bT, Tref)` drives the march from a per-step
temperature control; T enters R_n through the crystallisation drive dh(T/Tm-1)
AND the Flory chi B(T)=B0+bT(T-Tref).  `CACHAdjoint.temperature_gradient` returns
the TIME SERIES dJ/dT_n = -lam_n^T(dR_n/dT|drive + bT dR_n/dB).  4-step ramp
[0.5,0.6,0.7,0.55], target on phi & psi:

BDF1 worst adj/twin `6.2e-16`, adj/FD `8.0e-10`; BDF2 worst adj/twin `1.0e-15`,
adj/FD `6.8e-10`.  Sample (BDF2): dJ/dT = [+3.83e-1, +3.00e-1, +3.33e-1,
+2.82e-1].

## Capability status

- (a) MATERIAL PARAMETERS — **DELIVERED & VERIFIED.** Binary CH (M, kappa,
  Flory chi B, A) and coupled CH x AC crystallisation (dh, Tm, dsigma, eps2, L),
  BDF1 and variable-coefficient BDF2, all three-way to machine precision.
- (b) PROCESSING PARAMETERS — **quench schedule T(t) delivered & verified**
  (time-series dJ/dT_n, three-way).  Evaporation-rate k_e through the film march
  (top-flux + frame velocity) is a documented frontier (needs the wodo_film face
  infrastructure; the same IFT sweep applies with a boundary dR/dk_e term).
- (c) LEARN FREE ENERGY — **delivered & verified** (G4 below).
- (d) DESIGN PROCESS ROUTE — **delivered & verified** (G5 below).

### G4 — learn a free-energy functional from a trajectory (`test_g4_learn_free_energy`)

`PolyBasisEnergy`: bulk f'(c) = sum_i a_i c^{p_i} on a monomial basis;
df'/da_i = c^{p_i} feeds the adjoint.  Truth f'(c)=c^3-c (a=[1,-1]).  Objective
= L2 misfit over a TRAJECTORY of snapshots; gradient via CHAdjoint; fit by
L-BFGS.
- trajectory-loss gradient vs FD: rel `5.4e-10`, `3.2e-11`.
- clean recovery (10 snaps): a=[0.99999986, -0.99999999], |a-truth| `1.4e-7`,
  loss `1.5e-14`.
- noisy data (sigma=0.01): recovery error K=1 `0.183` -> K=4 `0.0048`
  (~40x better with more snapshots).

### G5 — design a processing route (`test_g5_design_process_route`)

PDE-constrained optimisation: optimise the temperature schedule T(t) to hit a
TARGET crystalline fraction (mean psi) using the G3 schedule gradient + L-BFGS
(bounded T in [0.2, 2.0]).  Target set by a ground-truth cold ramp (reachable).
- achieved converges to target: target `0.23877`, achieved `0.23877`,
  |diff| `4.2e-15`; objective `1.94e-04 -> 8.67e-30`.  Optimised schedule is
  interior (not pinned at the bounds).

## Scope notes
- Gate meshes are uniform (constraints.T == identity), natural no-flux BCs.
- The crystallisation gate is the (M,K)=(1,1) retained-species core; orientation
  theta (the 4th dof of the full 2M+2K block, KG-regularised |grad theta|) is a
  documented frontier — the crystallisation *parameters* live in the phi/psi
  coupling, which is fully covered.  Multi-species M>1 / matrix mobilities
  reuse the same IFT sweep with a larger block.

## Frontiers / next
- Evaporation-rate k_e through the wodo_film march (top-flux + frame velocity):
  the only named processing param not yet wired — needs the face-integral
  infrastructure; the same IFT sweep applies with a boundary dR/dk_e term.
- Orientation theta (4th block dof) — KG-regularised |grad theta|, its own gate.
- Multi-species M>1 / matrix Onsager mobilities: same sweep, larger block.
- NN free-energy (MLP f(c)) — the G4 basis generalises; autograd twin already
  differentiates arbitrary f' so an MLP parametrisation drops in.

## Module layout
- `src/diffsim/adjoint/phasefield.py` — CHDiscrete/CHForward/CHAdjoint,
  PolyEnergy/FHEnergy/PolyBasisEnergy (binary CH + material params + G4 basis).
- `src/diffsim/adjoint/crystallization.py` — CACHDiscrete/CACHForward/
  CACHAdjoint (coupled CH x AC, crystallization params, G3 schedule + G5).
- `src/diffsim/adjoint/torch_twin.py` — CHTwin, CACHTwin (autograd reference).
- `tests/test_phasefield_adjoint.py` — 8 gates (G1, G2a/b/c, G3, G4, G5,
  forward parity).

## Merge checklist (supervisor)
1. `PYTHONPATH=<worktree>/src pytest tests/test_phasefield_adjoint.py` — 8 green.
2. Forward bricks unchanged: `git diff` touches only `adjoint/` + the test +
   this note (no edit to physics/cahn_hilliard.py or multiphase.py).
3. Forward parity gate confirms the numpy operator == production stepper.
4. All three-way tolerances locked with >=2x headroom (adj/twin < 1e-10 vs
   measured ~1e-16; adj/FD < 1e-6 vs measured ~1e-9).
