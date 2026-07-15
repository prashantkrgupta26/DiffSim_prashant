# Beyond-Flory–Huggins learned thermodynamics (Learning rung 2, groundwork)

**Date:** 2026-07-14
**Status:** enabling machinery built + verified; gauge-identifiability
demonstrated. Composition-diverse / instrument-space recovery on the *production
ternary film* is the recorded next step.

The roadmap "Forward" item is: *"Learning rung 2: instrument-space observables
(S(q,t), film height h(t), PSF) + composition-diverse protocols → genuinely
beyond-Flory–Huggins recovery."* M4 learned the **parametric** FH interaction
(χ_pf/χ_ps/χ_fs, k_e) to 1e-7 and mapped a hard gauge/identifiability structure.
This note records the machinery that makes a **functional** (beyond-parametric)
free-energy correction well-posed, and a deterministic demonstration of why it
is needed.

## The gauge structure (recap, from m4_learn_fmix.py / m4-milestone-report.md §2)

A correction to f'(c) learned on top of Flory–Huggins has two structurally
unidentifiable modes:

- **T0 (constant in f'):** shifts μ by a constant → invisible to the conserved
  c-dynamics. No data fixes it.
- **T1 (linear in f'):** integrates to a *quadratic* free energy already spanned
  by the FH χ term. Concretely ∂f'/∂B = (1 − 2c) is **proportional to the
  linear basis mode P1**, so χ and any linear correction coefficient are exactly
  aliased (M4 measured the trajectory difference along this mode at 5e-15).

So a learnable **beyond-FH** correction must live in the subspace orthogonal to
{1, c} — cubic-and-up in f (quadratic-and-up in f').

## What was built (`src/diffsim/adjoint/neural_energy.py`)

Two energy heads, both exposing the `.fp(c)/.fpp(c)` contract that
`adjoint/torch_twin.CHTwin` now consumes when its `energy` is an object (a
one-line generalisation of `_assemble` alongside the existing `"fh"`/`"poly"`
tags):

- **`NeuralCHEnergy`** — FH base + a small MLP correction to f'(c), **L2-projected
  orthogonal to {1, c} over the composition domain every evaluation** (the
  projection coefficients are differentiable in the weights, so the optimiser
  never sees the aliased directions). The MLP carries an **analytic
  input-derivative** (chain rule through tanh) so f''(c) is closed form — no
  nested autograd inside the Newton solve. The general functional head.
- **`BasisCorrEnergy`** — FH base + a low-dimensional shifted-Legendre correction
  on degrees k≥2. Because Legendre P_k (k≥2) are L2-orthogonal to {1, c} on the
  domain **by construction**, the gauge anchoring is exact with no projection.
  The interpretable sibling: a handful of coefficients whose Jacobian
  conditioning is a clean identifiability number.

### Verification (`tests/test_neural_energy.py`, BDF1 + BDF2)

- **Gauge anchoring** ⟨corr,1⟩ = ⟨corr,c⟩ ≈ 1e-19 (MLP, machine zero) / < 1e-12
  (basis, exact Gauss–Legendre) — the correction carries no T0/T1 content.
- **Curvature consistency** f''(c) (closed form) == d f'/dc (autograd): MLP rel
  1.4e-16, basis exact.
- **Adjoint vs FD** the autograd-through-convergence dJ/d{A,B,θ} matches central
  finite differences along a random direction: rel **1.4e-8 (BDF1)** /
  **5.7e-9 (BDF2)**. This is the many-weight head's correctness bar (the
  per-scalar-name numpy hand adjoint does not scale to a network; the twin's
  autograd-through-the-converged-solve is the engine, FD the independent check).

## The gauge-identifiability demonstration (`benchmarks/phase-field/beyond_fh_identifiability.py`)

Deterministic, fit-free, on the verified Stack-B binary-CH adjoint
(`tests/test_beyond_fh_identifiability.py` gates it):

| parameterisation | observation-Jacobian cond |
|---|---|
| **un-anchored** {χ(=B), P1, P2, P3} | **1.4e16** (χ and P1 aliased → singular) |
| **anchored** {χ(=B), P2, P3}        | **1.8e1** (well posed) |

A **7.6e14×** conditioning gap — the linear mode makes the inverse problem
exactly rank-deficient (reproducing the M4 5e-15 alias), and the gauge-anchored
head removes it. On the anchored (identifiable) parameterisation the beyond-FH
coefficients are then recovered from a **single interior trajectory** to
rel-err **5.6e-8** (truth (0.30, 0.15) → (0.30, 0.15000002)).

This is the load-bearing result: **gauge anchoring is what makes learning a
free-energy *functional* on top of FH well-posed.** Without it, the beyond-FH
term is not merely hard to fit — it is unidentifiable.

### Composition coverage (the other half of rung 2)

Even a *gauge-anchored* higher-degree basis {P2, P3, P4, P5} is only weakly
identifiable from ONE shallow trajectory: over the narrow composition range it
visits, the high Legendre modes are near-collinear. `run_coverage_demo`
(fit-free, conditioning only) measures the d(snapshots)/d(coeffs) Jacobian:

| protocol | visited composition | cond |
|---|---|---|
| single (narrow) | 0.46–0.56 | ~2e2–5e2 |
| composition-diverse | 0.21–0.85 | ~14–18 |

a **~15–30× conditioning improvement** from composition-diverse protocols
(multiple means/quench depths) — the recorded rung-2 prescription, shown
deterministically. Gated at `test_composition_coverage_conditioning`.

## Recorded next steps

1. **Differentiable instrument-space observable — DONE for S(q)**
   (`src/diffsim/diagnostics/structure_factor_torch.py`,
   `tests/test_structure_factor_torch.py`). A Torch twin of the numpy
   `structure_factor` estimator (torch.fft + index_add radial binning): value
   parity with numpy ~1e-16 (2-D/3-D), S(q) gradient vs FD 1.2e-9, and — wired
   through the autograd twin — an `||S_sim(q)-S_data(q)||^2` misfit backprops to
   the beyond-FH coefficients matching FD to **1.8e-11**. A `grid_index_from_coords`
   helper gathers a uniform-mesh nodal field onto the FFT grid (exact). **h(t)**
   is next (it is already a forward state `WodoFilmStepper.h_curr`; needs the
   taped film path — step 2).
2. **End-to-end recovery of a *higher-order* functional** (not just
   conditioning). The conditioning improvement from composition coverage is
   shown (above); the remaining step is a robust recovery *fit* of {P2..P5}
   from a composition-diverse ensemble and/or the differentiable S(q,t) loss,
   demonstrating the coefficients are actually recovered where a single
   narrow trajectory fails. Note: naive fitting through the stiff CH forward
   with near-wall ICs hits FH-log NaN; keep ICs interior and regularise, or fit
   through S(q,t) which is smoother.
3. **Port the head to the production ternary film** (`physics/wodo_film.py`).
   Today the differentiable adjoint (Stack B) covers binary CH + coupled CH×AC;
   the ternary film uses FD Gauss–Newton (`m4_learn_fmix.py`). The MLP head is
   the general drop-in once the film has a taped/adjoint path (or via FD-GN on
   the anchored coefficients as an interim).
4. **k_e on the IFT sweep** (the one processing param still FD-only).
