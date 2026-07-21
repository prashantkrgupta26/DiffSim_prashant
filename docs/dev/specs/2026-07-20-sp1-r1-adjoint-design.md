# SP-1 R1 — Differentiable XDD: the adjoint-machinery rung

**Design spec.** Milestone P1 of the reconciled roadmap
(`docs/dev/2026-07-20-roadmap-reconciliation-strawman.md`). Builds directly
on the merged SP-1 R0 forward substrate. Ratified by Baskar 2026-07-20;
brainstormed same day.

## 1. Goal & scope

R1 turns the R0 forward XDD stack into a **differentiable** one: it delivers
the adjoint machinery that produces verified gradients of instrument
observables with respect to physical design variables — and, deliberately,
the *foundation* on which two later rungs (R2 learned closures, R3 optimal
experimental design) are built without re-architecture.

**In scope (R1):**
- Two adjoint modes, both gradient-verified: **implicit steady-state** and
  **taped transient (frozen-dt)**.
- A **unified control interface** through which design variables register a
  residual-VJP contribution, with **three control classes** wired:
  closure parameters, material/transport scalars, illumination/generation.
- Instrument observables (from R0's `observables.py`) made into
  differentiable QoIs; steady mode pairs with J–V-derived QoIs, transient
  mode with J(t) light-modulation and TRPL/PL.
- **One end-to-end inverse per mode** (recover a planted parameter).
- **Two forward-compatibility requirements** (below) so R2/R3 plug in.

**Explicitly OUT of scope (later rungs, own specs):**
- **Morphology / interface-geometry design variables** — deferred to a next
  phase; Baskar has a separate AI-based morphology-design approach. R1 does
  NOT build an XDD shape-adjoint.
- **R2 — learned closures** (NN-weight closures, pySR symbolic route).
- **R3 — optimal experimental design** (Fisher-information / expected-
  information-gain objectives to design illumination profiles and pcAFM
  probes).
- Any hero-scale (100M-DOF) differentiable run — R1 gates run at 2-D
  production scale.

## 2. The two forward-compatibility requirements (load-bearing)

R1 does not implement R2/R3, but its interfaces MUST admit them, or the
later rungs pay for a re-architecture. Two requirements, both testable in
R1:

1. **Vector / NN-weight controls.** The control interface accepts a control
   whose parameter is a *vector* (not only a scalar), exposing
   `∂R/∂p` as a Jacobian-vector/vector-Jacobian hook rather than a scalar
   derivative. R1 exercises this with a small vector control (e.g. a
   piecewise/low-rank illumination profile) so the path is proven; R2's NN
   closure is then "a control whose vector is the network weights."
2. **Exposed sensitivity map.** The adjoint exposes the per-parameter
   **sensitivity rows** ∂(observable)/∂p — the Jacobian of the
   observable(s) w.r.t. the unknown physical parameters — NOT only a
   collapsed scalar `dJ/dp`. R3's Fisher information matrix is assembled
   directly from these rows. R1 returns them as a first-class output of
   each adjoint solve.

## 3. Architecture

### 3.1 The control interface (`Control`)
A control is the unit of "a thing we differentiate with respect to." It
exposes:
- `value` / `set(p)` — read/write the current parameter (scalar or vector).
- `residual_vjp(state, seed) -> dp` — given an adjoint seed on the residual,
  return the contribution to `∂R/∂pᵀ · seed` (the VJP). For scalar controls
  this is a number; for vector controls, a vector.
- `sensitivity_rows(state) -> array` — ∂R/∂p as the rows needed for the
  exposed sensitivity map (requirement 2).

The three R1 control classes implement this interface:
- **`ClosureControl`** — dissociation (Onsager-Braun prefactor, D/A
  scalings) and recombination (Langevin coefficient). The A3 contract
  already returns `(value, d_value_d_fields)`; R1 adds `d_value_d_param`
  (analytic where available, else the existing closure autodiff path) and
  routes it through `residual_vjp`.
- **`MaterialControl`** — μ_n, μ_p, exciton μ, E_g, binding energy,
  permittivities, lifetimes. Direct residual derivatives (the parameter
  enters `exciton_dd`/`exciton_system` residual terms).
- **`IlluminationControl`** — generation profile and intensity (enters the
  generation source term). Includes at least one *vector* instance to
  satisfy requirement 1.

Controls are registered with the adjoint driver; both modes consume the
identical interface, so a future `MorphologyControl` (next phase) or
`NNClosureControl` (R2) is a plug-in.

### 3.2 Mode A — implicit steady-state adjoint
For steady operating points (a converged J–V point / steady solve):
1. Forward solves to convergence: `R(u; p) = 0`.
2. Given a QoI `J(u; p)`, solve the **transposed** linear system once:
   `Aᵀ λ = ∂J/∂u`, where `A = ∂R/∂u` is the converged Newton Jacobian
   (already device-assembled by R0; the transposed solve reuses the merged
   cuDSS / fp32-IR / graph-captured Krylov path — dot-product test gates
   the transpose).
3. Gradient: `dJ/dp = ∂J/∂p − λᵀ ∂R/∂p`, assembled per registered control
   via `residual_vjp`.
4. Sensitivity map: for observable vector **o**, the rows ∂o/∂p come from
   `(∂o/∂u)(−A⁻¹ ∂R/∂p) + ∂o/∂p` — computed by reusing the same factored
   `Aᵀ` across observable seeds.

Reuses the proven NS implicit-steady adjoint pattern
(`src/diffsim/adjoint/` + the `sbm/ns_adjoint` lineage); no time axis.

### 3.3 Mode B — taped transient (frozen-dt) adjoint
For transient observables (J(t) light-modulation, TRPL/PL):
1. Forward BDF march is **checkpointed** (converged state per step +
   step metadata); `dt` is recorded per step.
2. Reverse sweep accumulates the adjoint with **dt frozen on the backward
   pass** (the rule the existing `transient_adjoint` tests enforce — an
   adaptive-dt forward is fine; the backward pass uses the recorded dt's
   and must not re-adapt).
3. Per-step VJPs route through the same control interface; the observable
   is a functional of the trajectory (a J(t)/TRPL misfit).
4. Sensitivity rows accumulate across the reverse sweep.

Reuses the transient-adjoint checkpointing lineage
(`sbm/transient_adjoint`, `adjoint/`); the BDF1/BDF2 cross-matrix rule and
history handling are inherited from R0's stepper.

### 3.4 Differentiable observables
R0's `observables.py` already computes forward J–V curves
(`build_jv_curve`, `extract_jsc_ff`), transient traces (`capture_trace`,
`nirmal_features`), and the TRPL/PL gating (IRF convolution, bandwidth).
R1 adds, for the QoIs the adjoint flows back from:
- **Steady QoIs:** a J–V point / Jsc / fill-factor scalar, and a
  full-curve misfit `Σ (J_model(V) − J_data(V))²`.
- **Transient QoIs:** J(t) light-modulation misfit and a TRPL decay misfit.
Each QoI exposes `∂J/∂u` and `∂J/∂p` (the seeds the adjoint consumes) and
its **sensitivity rows** (requirement 2).

## 4. Verification & gates

**House gates (mandatory, all must pass):**
- **G1 — three-way gradient check** (adjoint vs unrolled tape vs central
  FD, rel. error ≤1e-6, FP64) for **every control class** in each
  applicable mode: steady × {closure, material, illumination};
  transient × {closure, material, illumination}.
- **G2 — dot-product test** ⟨Av,w⟩ = ⟨v,Aᵀw⟩ to machine precision for the
  transposed operator used by Mode A (guards the transpose).
- **G3 — transient checkpointing correctness:** frozen-dt reverse sweep
  reproduces the tape gradient; a checkpoint-vs-full-store equivalence
  check; adaptive-forward / frozen-backward respected.
- **G4 — vector-control gradient check:** the vector `IlluminationControl`
  passes a three-way check on every component (proves requirement 1).
- **G5 — sensitivity-map correctness:** the exposed ∂o/∂p rows match FD of
  each observable w.r.t. each parameter (≤1e-6); this is the R3-enabling
  output, gated now (proves requirement 2).

**Inverse demos (end-to-end, one per mode):**
- **D1 — steady inverse:** plant a known Langevin recombination coefficient
  (and, as a second variant, a dissociation prefactor); generate a
  synthetic light J–V; recover the planted value from a random start via
  gradient-based optimization to a tight tolerance (report recovered vs
  planted, and the misfit trajectory).
- **D2 — transient inverse:** plant a known recombination/lifetime;
  generate a synthetic TRPL decay; recover it.

Honest-reporting rule (house): the demos report recovered-vs-true and the
optimizer trajectory; if a parameter is weakly identifiable from the chosen
observable, that is reported (it is exactly the signal R3's OED will later
optimize away).

## 5. Scaling-pathway declaration (mandatory, standing rule)

- **Stage residency:**
  - Adjoint assembly (∂R/∂u, ∂R/∂p) — **device** (same 5-field block
    system as R0's forward; reuse `exciton_device` assembly).
  - Transposed / linear solve — **device** (merged cuDSS / fp32-IR /
    graph-captured Krylov).
  - Checkpoint storage (Mode B) — **host**, with an explicit budget line:
    O(steps) converged states at 2-D production size; the transient gates
    run at a step count whose host memory is bounded (declare the number).
    Long horizons at hero scale = a later rung (binomial/Revolve noted, not
    built).
  - Observables + sensitivity rows — **device** (assembled with the
    forward observables).
- **100M-DOF budget note:** the adjoint is the *same* sparse system as the
  forward, so the #35 device assembly, #38 ChunkedCSR, and #36 fp32-IR
  machinery carry it unchanged; R1 introduces no new nnz-space arrays. The
  item to declare honestly is the **transient checkpoint budget**, not the
  solve — at hero scale it is the binding host resource and is deferred to
  the hero rung.
- **Deployment tiers:** workstation single-GPU (gpubox) for all R1 gates
  and inverse demos; the hero-scale differentiable run is a later rung.

## 6. File structure (planned)

- `src/diffsim/xdd/adjoint.py` — the `Control` interface, the three control
  classes, and the two adjoint drivers (steady, transient) specialized to
  the XDD 5-field system. Consumes `exciton_system` / `exciton_device`.
- `src/diffsim/xdd/observables.py` — extend with the differentiable QoIs
  and their `∂J/∂u`, `∂J/∂p`, sensitivity-row outputs (do not duplicate the
  forward observable code; add the derivative faces).
- `tests/test_xdd_adjoint.py` — G1–G5 gate battery.
- `tests/test_xdd_inverse.py` — D1/D2 inverse demos.
- Reuse (do not fork): `src/diffsim/adjoint/*`, the `sbm/*_adjoint`
  patterns, the solver stack.

## 7. Compute placement

Per the standing preference: development and all gates run on **gpubox**
(the CPU-backend Mac only for quick file-level checks). The three-way
gradient checks and inverse demos are small (2-D production scale) and
run comfortably on one Ada GPU; use `scripts/remote/gpubox-test.sh`.

## 8. What R1 explicitly enables (the payoff, later)

- **R2 learned closures:** `NNClosureControl` is a vector control (weights);
  its `residual_vjp` is autodiff of the network; the adjoint already flows
  `dLoss/dweights`. pySR uses the differentiable solver in the loop to score
  candidate symbolic forms. Nothing in R1 needs changing.
- **R3 optimal experimental design:** the exposed sensitivity rows assemble
  the Fisher information matrix; illumination profiles and pcAFM probe
  protocols become the *design* variables of an information-gain objective
  (A-/D-optimality / EIG). The vector-control path (illumination) and the
  sensitivity map are the two hooks R1 gates today.
