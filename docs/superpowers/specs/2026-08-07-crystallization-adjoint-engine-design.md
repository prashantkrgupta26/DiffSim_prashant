# K>0 ψ-Crystallization Adjoint Engine (sub-project ①) — Design

Design ratified by Baskar 2026-08-07 (brainstorming). First of three sub-projects
toward **learning the free energy for K>0** (crystallization): ① the adjoint
engine (this spec) → ② a general non-parametric learnable coupled φ–ψ energy →
③ MD φ+ψ trajectory-matching. Mirrors the K=0 arc (rung-1 engine → M6 learnable
energy → MD ingest).

## Mission

Build the **M-generic ψ-crystallization adjoint**: a discrete IFT adjoint through
the coupled multi-Cahn-Hilliard × multi-Allen-Cahn BDF march (phase separation +
crystallinity, **no orientation θ**), giving exact, three-way-verified gradients
of a morphology objective w.r.t. the crystallization parameters (dσ, dh, Tm, ε²,
L) and the existing design parameters (χ, N, mobility, κ). The bulk free energy
sits behind a **`CrystalEnergy` protocol** so a general non-parametric learnable
energy (sub-project ②) drops in with no engine rework — the same discipline that
let M6 reuse the K=0 adjoint.

## Status this design builds on

- **K=0 M-component adjoint (delivered, merged):** `adjoint/multiphase.py`
  (`MultiCHForward`/`MultiCHAdjoint`, 2M block, `MultiEnergy` protocol,
  `MobilityClosure`), three-way verified, GPU cuDSS backend. **M6 learnable
  energy** (`BasisMultiEnergy`) proves the protocol-boundary pattern this spec
  reuses one level up.
- **Binary crystallization adjoint (delivered):** `adjoint/crystallization.py`
  — a three-way-verified (M,K)=(1,1) coupled CH×AC adjoint, node block
  `(φ, μ, ψ)`, crystallization params (dh, Tm, dσ, ε², L) differentiable,
  **θ orientation deferred** (documented frontier). `torch_twin.CACHTwin` is its
  autograd twin. This is the exact template to M-generalize.
- **Production K>0 (delivered):** `physics/multiphase.MultiPhaseStepper` is
  (M,K)-generic (`2M+2K` block incl. θ) with the full **4-fold ψ-dependent χ_eff**
  (`(1−ψ_i)(1−ψ_j)χ_aa + …`), crystal bulk, ψ gradients. Theory:
  `docs/theory/crystallization_formulation_p1.md`.
- **The data (drives ②/③):** MD time-snapshots voxelized to φ AND ψ (crystallinity
  fraction) — no orientation, confirming the θ deferral.

## Scope decisions (ratified)

- **v1 physics = φ + ψ, θ DEFERRED.** Block = `(φ_i, μ_i)` for i=0..M-1 (conserved
  CH pair) + `ψ_k` for k in the crystallizable subset K ⊆ {0..M-1} (non-conserved
  Allen-Cahn, single dof). No orientation θ. Matches the binary template and the
  MD data.
- **Coupling = ADDITIVE crystal-bulk, χ ψ-INDEPENDENT (first impl).**
  `f = f_FH(φ; χ) + Σ_{k∈K} φ_k [ q(ψ_k) dσ_k + p(ψ_k) drive_k ]`, with the fixed
  polynomials `q(ψ)=ψ²(1−ψ)²`, `p(ψ)=ψ²(3−2ψ)` and `drive_k = dh_k(1 − T/Tm_k)`.
  De-risks the block + Allen-Cahn + adjoint machinery on a clean, proven form.
  The **4-fold ψ-dependent χ_eff and production-parity are ①b** (§Designed-for);
  the *general* ψ-dependence of χ is learned non-parametrically in ② rather than
  hard-coded.
- **Energy behind the `CrystalEnergy` protocol FROM THE START.** The engine
  differentiates the energy opaquely; `AdditiveCrystalEnergy` is its first
  implementation, the non-parametric `NeuralCrystalEnergy` (②) the second — no
  engine rework, exactly as `MultiEnergy` → M6.
- **General crystallizable subset K.** Any subset of species may crystallize;
  non-crystallizing species carry no ψ dof.
- **Verification = three-way self-verified.** Hand adjoint == torch twin == finite
  differences, per parameter, ternary + one quaternary (with a crystallizable
  subset), BDF1+BDF2. Production-parity to the 4-fold χ_eff is ①b (the additive
  form intentionally does NOT match production K>0).
- **Deliverable boundary:** the coupled CH×AC adjoint engine + `CrystalEnergy`
  protocol + additive impl + twin, all three-way verified. Learning (②) and MD
  ingest (③) are the downstream sub-projects.

## Architecture

A discrete IFT adjoint through the coupled CH×AC BDF march, generalizing the
binary `crystallization.py` to the M-generic `2M+K` block and routing the bulk
energy through a protocol.

```
node-major block per node:  [(φ_0,μ_0)…(φ_{M-1},μ_{M-1}), ψ_{k0}, ψ_{k1}, …]
                             └── 2M conserved CH ──────┘ └── K non-cons. AC ──┘
```

**Dynamics (K=0 branch unchanged; the ψ block is new):**
```
CH_i:  φ_i conserved       R_φi = ∫N(σφ_i − hist_φi) + Σ_j M_ij ∫∇N·∇μ_j
       μ_i = ∂f/∂φ_i        R_μi = ∫N(μ_i − ∂f/∂φ_i) − κ_i ∫∇N·∇φ_i
AC_k:  ψ_k non-conserved    R_ψk = ∫N(σψ_k − hist_ψk)
                                   + L_k[ ∫N ∂f/∂ψ_k + ε_k² ∫∇N·∇ψ_k ]
```
ψ is a SINGLE dof (no μ-splitting — the Allen-Cahn relaxation is 2nd-order, not
4th). Its time term mirrors φ's (σψ − hist), so the adjoint history cotangent for
ψ is a mass term on the ψ-rows (no gradient-flux history, unlike φ's transport).

**Adjoint (reverse sweep, IFT):** `Jₙᵀλₙ = ∂j/∂xₙ − Σ_{k≥1}(∂R_{n+k}/∂xₙ)ᵀλ_{n+k}`,
`dJ/dp = −Σₙ λₙᵀ(∂Rₙ/∂p)`. `Jₙ = ∂Rₙ/∂xₙ` is the converged forward Jacobian; the
BDF history couples the conserved φ-fields (mass-coupled φ-rows, as K=0) AND the
non-conserved ψ-fields (mass-coupled ψ-rows). Structure identical to
`crystallization.CACHAdjoint`, generalized to M species + a K subset.

## The `CrystalEnergy` protocol

Every bulk free energy (parametric or learned) exposes, on Gauss-point
`phis` (length-M) and `psis` (length-K) arrays, preserving input dtype
(complex-step):
- `dfdphi(phis, psis)` → the M exchange potentials `∂f/∂φ_i` (→ μ).
- `dfdpsi(phis, psis)` → the K Allen-Cahn driving forces `∂f/∂ψ_k`.
- `d2f` Hessian blocks: `∂²f/∂φ_i∂φ_j` (M×M), `∂²f/∂φ_i∂ψ_k` (M×K cross),
  `∂²f/∂ψ_k∂ψ_l` (K×K) — for the Newton Jacobian.
- `dfdphi_dparam` / `dfdpsi_dparam` for each learnable parameter — the
  per-parameter cotangents the adjoint reduces against.
- `param_names`, `crystallizable` (the set K).

**`AdditiveCrystalEnergy`** (first impl): `f = f_FH(φ;χ) + Σ_k φ_k[q(ψ_k)dσ_k +
p(ψ_k)drive_k]`. Reuses `FHMultiEnergy` for the φ part; the crystal part gives
`∂f/∂φ_k += q(ψ_k)dσ_k + p(ψ_k)drive_k`, `∂f/∂ψ_k = φ_k[q'(ψ_k)dσ_k + p'(ψ_k)
drive_k]`, the cross Hessian `∂²f/∂φ_k∂ψ_k = q'dσ + p'drive`, and
`∂²f/∂ψ_k∂ψ_l = δ_kl φ_k[q''dσ + p''drive]`. `param_names` add `dsig_k`,
`dh_k`, `Tm_k` (T external), `eps2_k`, `L_k`.

## Components (each independently testable)

1. `CrystalCHDiscrete` (`adjoint/crystallization_multi.py`) — vectorised numpy
   residual/Jacobian, node-major `2M+K` block; M-generalization of
   `CACHDiscrete`. Mobility via the existing `MobilityClosure`; crystallization
   coupling via the `CrystalEnergy` object. `dR_dparam` covers `dsig_k`/`dh_k`/
   `Tm_k`/`eps2_k`/`L_k` (crystallization) delegating bulk params to the energy
   object, plus the existing χ/N/mobility/κ.
2. `CrystalEnergy` protocol + `AdditiveCrystalEnergy` (same module) — complex-step
   gated (the protocol contract above).
3. `CrystalCHForward` / `CrystalCHAdjoint` — Newton BDF march + IFT reverse sweep
   with the ψ Allen-Cahn history coupling.
4. `CrystalCHTwin` (`torch_twin.py`) — M-generalize `CACHTwin` to consume a
   generic `CrystalEnergy` + the mobility closure; `grads()` handles the
   crystallization param names.

## Verification gate (house rule)

`tests/test_crystallization_multi.py`:
1. **Energy-object correctness** — `AdditiveCrystalEnergy` `dfdphi`/`dfdpsi`/`d2f`/
   `dparam` complex-step gated (1e-30 imag perturbation).
2. **Three-way gradient** — hand IFT adjoint == `CrystalCHTwin` == FD for the
   crystallization params (dσ, dh, Tm, ε², L) AND χ/N/mobility/κ, on a ternary and
   one quaternary case with a crystallizable subset, BDF1+BDF2. Tolerances
   adj/twin < 1e-10, adj/fd < 1e-6.
3. **Reduction check** — with K=∅ (no crystallizing species), `CrystalCHForward`
   reproduces the K=0 `MultiCHForward` trajectory (the ψ machinery is a strict
   superset).

## Module layout (①)

```
src/diffsim/adjoint/
  crystallization_multi.py   # CrystalCHDiscrete/Forward/Adjoint,
                             # CrystalEnergy protocol + AdditiveCrystalEnergy
  torch_twin.py              # + CrystalCHTwin (extend; consumes CrystalEnergy)
tests/test_crystallization_multi.py
```

## §Designed-for extensions

- **② General non-parametric coupled energy** (`NeuralCrystalEnergy`, the
  "very general non-parametric form"). An MLP `f_θ(φ, ψ)` over the (M+K) inputs,
  gauge-anchored: project `{1, φ_i}` out of the μ-directions (conserved φ), leave
  the **ψ-directions largely free** (ψ is non-conserved → its potential ∂f/∂ψ is
  directly identifiable from the AC relaxation, much less gauge freedom than φ).
  Subsumes the 4-fold χ_eff(ψ) and the crystal bulk — a drop-in `CrystalEnergy`,
  no engine rework. Own spec (sub-project ②).
- **①b 4-fold χ_eff(ψ) + production parity.** Add the ψ-dependent
  `chi_eff(i,j) = (1−ψ_i)(1−ψ_j)χ_aa + …` (new ∂R_μ/∂ψ cross-blocks) and
  forward-parity to `MultiPhaseStepper` K>0. (Largely obviated if ② learns the
  ψ-dependence directly.)
- **Orientation θ** (the 2M+2K block): the KWC `|∇θ|` grain-boundary term — its own
  gate; not in the MD data, so unlearnable from it.
- **③ MD φ+ψ trajectory-matching** (extends M6 Plan B to ψ).

## House rules honored

Measure-then-lock; commit-per-green; three-way-verified gate; Gramian-before-
compute (deferred to ②'s learning); mobility closures stay NAMED; the energy
gauge-anchor lives inside the energy object (differentiable, invisible to the
engine); findings-log honesty on the additive-vs-production-form gap and the
φ/ψ gauge asymmetry; no dyadic feature dims in gates.

## Deliverables checklist

- [ ] `CrystalEnergy` protocol + `AdditiveCrystalEnergy` (complex-step gated).
- [ ] `CrystalCHDiscrete` (2M+K block, Newton Jacobian incl. φ–ψ cross).
- [ ] `CrystalCHForward` / `CrystalCHAdjoint` (ψ Allen-Cahn history coupling).
- [ ] `CrystalCHTwin` (M-generalized CACHTwin, generic energy + closure).
- [ ] Three-way gate (crystallization params + χ/N/mob/κ; ternary + quaternary;
      BDF1+BDF2) + the K=∅ reduction-to-K=0 check.
- [ ] (②) `NeuralCrystalEnergy` non-parametric coupled energy; (①b) 4-fold χ(ψ)
      + production parity; (③) MD φ+ψ ingest; (later) orientation θ.
