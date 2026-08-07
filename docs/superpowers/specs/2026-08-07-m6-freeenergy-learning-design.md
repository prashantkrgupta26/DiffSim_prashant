# M6 — Free-energy learning from MD trajectories (φ-only, K=0)

Design ratified by Baskar 2026-08-07 (brainstorming session). Builds on the
delivered OrgElMorph adjoint (rungs 1–2, three-way verified) and the binary
neural-energy precedent `src/diffsim/adjoint/neural_energy.py`. This is the
"second application" the rung-1 design (§9) was built to accommodate.

## Mission

Learn a **beyond-Flory-Huggins free energy** (and a named mobility closure) for
the M-component blend by **trajectory-matching a differentiable multi-Cahn-Hilliard
march against voxelized MD morphology snapshots**. The free energy is a learnable
`MultiEnergy` object; gradients of a trajectory-matching loss w.r.t. its
parameters flow through the *already-verified* K=0 adjoint
(`dJ/dθ = −Σₙ λₙᵀ ∂Rₙ/∂θ`), so no adjoint-engine rework is required — only a new
energy object, a mobility closure, a twin extension, and the MD data/loss layer.

## Status this design builds on

- **Adjoint engine (delivered, K=0):** `src/diffsim/adjoint/multiphase.py` —
  `MultiCHForward` / `MultiCHAdjoint`, node-major 2M block, three-way verified
  (hand adjoint == torch twin == FD) for χ/N/mobility/κ/mean-φ. The bulk energy
  sits behind the **`MultiEnergy` protocol** (`mu`, `dmu_dphi`, `dmu_dparam`) so a
  learned energy is a drop-in subclass. Rung-2 cuDSS backend gives GPU solves.
- **Binary precedent:** `neural_energy.py` `NeuralCHEnergy` — FH base + a
  **gauge-anchored MLP correction** to f′(c): `f′ = f′_FH(c;A,B) + [g(c;θ) − a₀ − a₁c]`,
  where `a₀,a₁` project the correction off the `{1, c}` gauge modes
  (constant + linear-in-c are unidentifiable from conserved dynamics — the "T0
  gauge") via a differentiable Gram solve over the composition domain. FD-gated
  in `tests/test_neural_energy.py`. M6 is the M-component generalization.
- **The data:** time-resolved MD snapshots, voxelized into volume-fraction (φ)
  and crystallization (ψ) fields. **v1 uses φ only** (see Scope).

## Scope decisions (ratified)

- **v1 physics = φ-only, K=0.** Reuse the verified `MultiCHForward`/`MultiCHAdjoint`
  unchanged. Crystallization (ψ, the Allen-Cahn block, 2M→2M+2K) is the **v2
  extension** (§Designed-for) — the MD ψ fields are ignored in v1. This de-risks
  the learn-from-MD pipeline on machinery we already trust before the large
  crystallization-adjoint build.
- **Learning signal = trajectory-matching through the march.** Multi-snapshot:
  the loss compares the simulated φ-field to the MD φ-field at each snapshot time,
  summed over snapshots — a full-trajectory objective `J = Σₖ jₖ(x_{n(tₖ)})`,
  which the adjoint already supports.
- **Loss = hybrid: coarse-grained field L2 (primary) + descriptor regularizers.**
  `jₖ = α‖CG(φ_sim,k) − CG(φ_MD,k)‖² + β·D(descr(φ_sim,k), descr(φ_MD,k))`, with
  descriptors = structure factor S(k) / domain-size / composition histogram
  (translation/rotation-invariant). The CG (coarse-grain) kernel maps both fields
  to a common resolution; descriptors keep the fit honest where the raw field
  mismatch is MD thermal noise / the mean-field gap. All terms differentiable in
  the CH φ-field.
- **Learned parameters:**
  - **Free energy θ** — always learned (`NeuralMultiEnergy`).
  - **Mobility** — learned through an **assumed NAMED functional closure**
    `M(φ; a)` with a few coefficients `a` (NOT a free matrix), per the
    mobility-closures-stay-NAMED house rule; sidesteps the energy↔mobility
    degeneracy a free matrix would create.
  - **MD→CH time-scale τ** — **configurable**: pre-calibrated (fixed) by default,
    optionally learnable. **Identifiability note:** a global τ is degenerate with
    the mobility closure's overall magnitude (rescaling time ≡ scaling M
    uniformly), so **learn at most one of {τ, mobility-magnitude}**; a
    normalization on the closure magnitude, plus the descriptor terms + Gramian
    pre-check, break the degeneracy. This is a Gramian-before-compute gate.
  - **κ (gradient energy)** — **fixed** in v1 (estimable from the MD interface
    width); learnable-κ deferred (§Designed-for).
- **Verification = synthetic-recovery first.** Gate the whole pipeline by
  generating data from a *known* `NeuralMultiEnergy` + closure and confirming the
  fit recovers them (three-way `dJ/dθ`, `dJ/da` gate + a recovery test) **before**
  any real-MD fit — so imperfect real-MD fits implicate the physics/data, not a
  gradient bug.
- **Deliverable boundary:** the learnable energy + closure objects, the verified
  `dJ/dθ`/`dJ/da` (+ optional `dJ/dτ`) gates, the MD-ingest + hybrid-loss layer,
  and the synthetic-recovery demo. The **real-MD training campaign** (optimizer
  loop, hyperparameters, the actual MD dataset) is the *application*, run
  downstream (like the BO campaign, DAISY/analysis-side), not part of the engine
  deliverable.

## Architecture

A differentiable trajectory-matching loop:

```
MD φ-snapshots {φ_MD,k @ t_k}  ──ingest──▶  IC = φ_MD,0 on the CH mesh
                                            │
  θ (energy), a (mobility), τ (opt) ──▶  MultiCHForward march (K=0)
                                            │  records committed states
  loss J = Σ_k [ CG-L2 + descriptor ]  ◀────┘  compare φ_sim at step n(t_k)
                                            │            to φ_MD,k
  dJ/dx_n  ──▶  MultiCHAdjoint (verified)  ──▶  dJ/dθ, dJ/da, (dJ/dτ)
                                            │
  optimizer step on {θ, a, (τ)}  ◀──────────┘   (downstream campaign)
```

The **only new differentiable physics** is inside the energy object and the
mobility closure; the march and adjoint are unchanged. The loss `j(φ)` is an
arbitrary differentiable scalar of the φ-field (the adjoint obtains `∂j/∂φ` by
autograd and maps it into the 2M cotangent, μ-rows = 0), so the CG-L2 +
descriptor loss plugs in with no engine change.

## The math

**Trajectory loss.** With the time map `t_CH = t_MD / τ` selecting the committed
step `n(tₖ)` nearest each snapshot (linear interpolation between committed states
when τ is learnable, so `∂j/∂τ` flows through the interpolation):

```
J = Σ_k [ α ‖CG φ_sim(n_k) − CG φ_MD,k‖² + β D(descr φ_sim(n_k), descr φ_MD,k) ]
dJ/dx_n = ∂j/∂x_n   (nonzero only at snapshot steps; μ-rows 0)
```

**Parameter gradients** (reverse sweep, unchanged engine):

```
Jₙᵀ λₙ = ∂j/∂xₙ − Σ_{k≥1}(∂R_{n+k}/∂xₙ)ᵀ λ_{n+k}
dJ/dθ  = − Σₙ λₙᵀ (∂Rₙ/∂θ)      # θ via the MultiEnergy object (dmu_dparam)
dJ/da  = − Σₙ λₙᵀ (∂Rₙ/∂a)      # a via the mobility closure's ∂R/∂a
dJ/dτ  = Σₙ (∂J/∂n)(∂n/∂τ)       # optional; through the snapshot-interpolation
```

**M-component gauge (inside the energy object).** Each exchange potential
correction is projected off the `{1, φ₀,…,φ_{M-1}}` modes (the (M+1)-dim
constant+linear space unidentifiable from conserved φ-dynamics; the quadratic
modes are already spanned by the identifiable χ-matrix, so the neural correction
carries cubic-and-higher). The projection is a differentiable Gram solve over a
composition-domain quadrature — the M-generalization of `neural_energy._gauge_coeffs`.
The exact (M+1)-mode projection is finalized in the plan against the binary
precedent.

## Components (each independently testable)

1. **`NeuralMultiEnergy(MultiEnergy)`** (`neural_multiphase.py`) — FH base
   (χ, N) + gauge-anchored MLP correction; exposes `mu` (M potentials),
   `dmu_dphi` (M×M Hessian), `dmu_dparam(name)` for every learnable θ (FH params
   + network weights). Gauge projection differentiable in θ. Complex-step / FD
   gated like `FHMultiEnergy`.
2. **Named mobility closure** `M(φ; a)` — an engine-level transport closure with
   a few learnable coefficients and analytic `∂R/∂a` (mirrors the existing
   `onsager_*` `dR_dparam` path, generalized through the closure). NAMED
   (`const | <closure-name>`); the magnitude coefficient carries the rate when τ
   is fixed.
3. **`MultiCHTwin` energy-object extension** — consume a generic `MultiEnergy`
   (as `CHTwin` already does via its `obj_energy` branch) + the mobility closure,
   so the three-way gate covers `dJ/dθ`, `dJ/da`, `dJ/dτ`.
4. **MD ingest + hybrid loss** (`md_ingest.py`, `traj_loss.py`) — voxel→CH-mesh
   mapping, IC from the first snapshot, fixed/learnable time-scale, and the
   `α·CG-L2 + β·descriptor` loss (S(k)/domain-size/composition-PDF), differentiable
   through φ.
5. **Gramian / identifiability pre-check** — before any fit, assemble the
   sensitivity Gramian over {θ, a, (τ)} and report rank/conditioning (flag the
   τ↔mobility-magnitude degeneracy and gauge-null directions). Gramian-before-compute.

## Verification gate (house rule)

`tests/test_neural_multiphase.py` and `tests/test_m6_recovery.py`:
1. **Energy-object correctness** — `NeuralMultiEnergy` `mu`/`dmu_dphi`/`dmu_dparam`
   complex-step/FD gated; gauge residual ≈ 0 (correction orthogonal to
   `{1, φᵢ}`).
2. **Three-way gradient** — hand IFT adjoint == `MultiCHTwin` == FD for `dJ/dθ`,
   `dJ/da`, and `dJ/dτ` on a ternary case, BDF1+BDF2, tight tolerances
   (adj/twin<1e-10, adj/fd<1e-6).
3. **Synthetic recovery** — data generated from a known `(θ*, a*)`; the fit
   recovers them to tolerance (gauge-invariant comparison for θ). This is the
   end-to-end pipeline gate.

## Module layout (v1)

```
src/diffsim/adjoint/
  neural_multiphase.py   # NeuralMultiEnergy (MultiEnergy) + mobility closure
  torch_twin.py          # extend MultiCHTwin: generic energy + closure + tau
src/diffsim/learn/       # (or daisy-side) MD ingest + hybrid trajectory loss
  md_ingest.py
  traj_loss.py
tests/test_neural_multiphase.py   # energy-object + three-way dJ/dθ,da,dτ gate
tests/test_m6_recovery.py         # synthetic-recovery pipeline gate
examples/m6_learn_from_md.py      # synthetic-recovery demo (then real-MD hook)
```

## §Designed-for extensions

- **Crystallization (v2, the ψ block).** Extend K=0 → (M,K): add the ψ
  Allen-Cahn block (2M→2M+2K) exactly as binary `crystallization.py`
  (CACHDiscrete/Adjoint) + `CACHTwin` extend the binary CH adjoint. Then the MD
  ψ fields enter the loss and the energy couples φ↔ψ. This is the largest single
  build in the OrgElMorph effort and gets its own spec.
- **Learnable κ.** A learnable gradient-energy (per-species or φ-dependent);
  engine-level param with analytic `∂R/∂κ` (already present as `kappa_*`).
- **Learnable τ via full trajectory interpolation** (if the degenerate
  magnitude-shortcut proves insufficient).

## House rules honored

Measure-then-lock; commit-per-green; three-way-verified gate; Gramian-before-
compute (the identifiability pre-check + the τ↔mobility degeneracy guard);
mobility closures stay NAMED; the gauge-anchor lives inside the energy object
(differentiable, invisible to the engine); findings-log honesty on the
mean-field↔atomistic gap (why the loss is CG+descriptor, not raw-field);
no dyadic feature dims in gates.

## Deliverables checklist

- [ ] `NeuralMultiEnergy` (`MultiEnergy` subclass) + M-component gauge, complex-step/FD gated.
- [ ] Named mobility closure `M(φ;a)` with analytic `∂R/∂a`.
- [ ] `MultiCHTwin` extension (generic energy + closure + τ).
- [ ] Three-way gate for `dJ/dθ`, `dJ/da`, `dJ/dτ` (ternary, BDF1+BDF2).
- [ ] MD ingest + hybrid CG-L2 + descriptor loss (differentiable).
- [ ] Gramian / identifiability pre-check (incl. τ↔mobility degeneracy).
- [ ] Synthetic-recovery demo (`examples/m6_learn_from_md.py`) + recovery gate.
- [ ] (v2) crystallization ψ block; (later) learnable κ, full-interpolation τ.
