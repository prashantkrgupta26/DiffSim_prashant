# NeuralSDF Oracle — Implementation Spec (M1c/M2)

Status: DRAFT for review. Parent: `2026-07-02-diffsim-design.md` §4
(oracle contract, epoch pipeline, shape gradients). Measured motivation:
the L6 shape-optimization demo (`benchmarks/shape_opt_cylinder.py`) —
exact within-epoch gradients are not descent-stable across
reclassification epochs; a parameter-rich, smoothly deformable geometry
representation is the prerequisite for every mitigation strategy below.

## N1. Scope and non-goals

In scope: a `NeuralSDFOracle` implementing the S4.1 contract; fitting
machinery with admissibility-gated training; the AD path to network
weights θ; gates through Poisson → NS → drag gradient. NOT in scope
(deferred, N6): relaxed/differentiable classification — NeuralSDF makes
geometry *parameters* smooth; it does not by itself make the *retained
set* smooth. The spec is explicit about this to avoid over-claiming.

## N2. The oracle class

```
class NeuralSDFOracle(SDFOracle):
    near_eikonal = False          # promoted per-instance after audit
    psi:   torch MLP, float64, host/GPU torch (geometry-epoch layer)
    params: list(psi.parameters())   # thousands of theta
```

- **Architecture (baseline, locked until measured otherwise):** MLP
  x → ψ, width 64 × depth 4, sine or softplus activations (C² needed —
  the p2 shift consumes the Hessian through d's IFT derivative; ReLU is
  inadmissible), fp64 weights. Fourier/positional encoding optional
  behind a flag (helps sharp features; hurts c₀ if overdone — audit
  decides).
- `classify` = sign(ψ) batched; `distance_vector` = the framework
  Newton+IFT projector (`geometry/project.py`) unchanged — GridSDF
  already exercises the generic non-eikonal path, NeuralSDF reuses it
  verbatim. `distance_torch` comes for free (ψ is torch end-to-end).
- **Admissibility is a runtime contract, not a hope** (S4.1 diagnostics):
  every epoch evaluates ĉ₀ = min‖∇ψ‖ over the band and the Newton
  success mask; an epoch that violates (ĉ₀ < c₀_min or mask failures)
  raises with the report attached — the neural-geometry paper's
  Assumption 3.1/Lemma 3.4 as executable checks (findings 5c).

## N3. Fitting (geometry → weights)

`fit_neural_sdf(target, band_pts, h, order) -> NeuralSDFOracle`

- Supervision: ψ-values (+ optional ∇ψ) from any existing oracle
  (CSG/TriMesh/Grid) sampled in a tubular band ∪ ambient sprinkling —
  fit-to-oracle first; point-cloud fitting later.
- Loss: SDF regression + **eikonal regularizer** λ_e(‖∇ψ‖−1)² *in the
  band only* (keeps ĉ₀ healthy where the projector works; global
  eikonal fights expressiveness).
- **Stopping rule = the spec's ε∞ ~ h^(k+1) tolerance** (S4.1): train
  until the band sup-error clears the mesh's geometric-accuracy budget,
  else surface the geometry-limited-refinement-plateau warning. The fit
  gate LOCKS measured (ε̂∞, ĉ₀) for the reference shapes.

## N4. Gradients

Within an epoch, θ enters only through step-4 geometry data (S4.3):
d(θ), n(θ), corr(θ) at surrogate GPs. The chain is the m1a/m1c pipeline
unchanged — face-residual cotangents (dbar, nbar, corrbar, gbar/qbar) →
torch backward through the Newton IFT into ψ's autograd graph → θ.grads.
Nothing new is taped; the only new code is ψ itself.

Gates (measure-then-lock, in order):
1. **Fit gate:** sphere + cylinder-2D fits meet ε∞(h) at L4-L6;
   admissibility report locked.
2. **Forward equivalence:** SBM Poisson MMS through NeuralSDF matches
   the analytic-oracle solution to the geometry budget (the plateau rule
   made a test).
3. **AD gate (4c contract, oracle edition):** d(probe-QoI)/dθ_i for ~8
   sampled weights, three-way (adjoint vs FD-on-θ_i vs torch dense twin
   where feasible), β-on and β-off per the evaluation-response rule.
4. **NS composition:** Re=20 cylinder smoke with the fitted-cylinder
   NeuralSDF; Cd within the locked baseline's tolerance.
5. **Drag gradient:** d(Cd)/dθ gate vs FD on a θ-subset — the hero-demo
   ingredient.

## N5. Shape optimization: what NeuralSDF changes, honestly

The L6 finding stands regardless of representation: the objective is
piecewise-smooth in θ with jumps at reclassification. NeuralSDF changes
three practical things: (i) thousands of θ instead of 2 — descent
directions exist that deform the shape *without* moving the interface
across cell boundaries as fast as rigid translation does; (ii) the
deformation is global and smooth — epoch-transition jumps are typically
smaller for equal ‖δθ‖; (iii) it enables N6's relaxations. The BASELINE
optimizer spec'd here: epoch trust region (step accepted by OBJECTIVE
decrease, not gradient faith), with re-carve per accepted step and
h-continuation (refine when the epoch noise floor dominates the
gradient signal — the band-study "elements across the feature" rule
applied to optimization).

## N6. Deferred: differentiable classification (research fork)

Two candidate relaxations, spec'd as QUESTIONS with gates, not
commitments: (a) λ-relaxation — retained-set weights w(ψ/h) smooth in a
narrow band (changes assembly: per-element volume weights; conditioning
audit required); (b) objective smoothing — expectation over dithered
grid offsets (embarrassingly parallel; no formulation change; N×
forward cost). Decision point AFTER the N4 gates exist, using the L6
demo as the benchmark problem: whichever relaxation first turns its
measured Cd-ascent into descent wins a milestone.

## N7. Order of work

1. `NeuralSDFOracle` + fit-to-oracle + gates 1-2 (one session).
2. AD gate 3 (the 4c ritual; expect the FD-on-θ trust region to need
   the fit's ε∞ headroom — document the epsilon choice).
3. Gates 4-5 (compose with the existing NS/drag machinery — no new
   adjoint code expected).
4. N5 baseline optimizer on the L6 demo problem; measure.
5. N6 decision with data.
