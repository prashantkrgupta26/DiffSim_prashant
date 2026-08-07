# Differentiable OrgElMorph — M-component phase-separation adjoint

Design ratified by Baskar 2026-08-06 (brainstorming session). Milestone
context: M5 OrgElMorph (`docs/dev/specs/2026-07-10-m5-orgelmorph.md`);
formulation contract `docs/theory/crystallization_formulation_p1.md`.

## Mission

Make the M-component phase-separation morphology simulation **differentiable**:
gradients of an arbitrary scalar morphology objective with respect to the blend
**design parameters** (χ-matrix, degrees of polymerization N, Onsager mobility
M, gradient-energy κ), delivered as a genuine discrete implicit-function-theorem
(IFT) adjoint through the multi-Cahn-Hilliard BDF march and three-way verified.

Two consumers, one engine:
1. **Gradient-based BO / inverse design** (first application) — gradients w.r.t.
   design parameters to steer campaigns and enable knowledge-gradient
   acquisition. This is what `daisy-morph`'s reserved `gradients=` knob points at.
2. **Free-energy learning** (M6, explored next) — gradients w.r.t. a learnable
   free-energy functional. Accommodated by construction (§9), not a rebuild.

## Status this design builds on

- **Forward (delivered):** `physics/multiphase.py` `MultiPhaseStepper` — the
  (M,K)-generic multi-CH × multi-AC brick; ternary→pentanary through one code
  path, full `(M+1)×(M+1)` χ-matrix. `daisy-morph` drives it (`core.py`) at K=0
  for pure demixing. `core.py` already records "the M-component adjoint through
  MultiPhaseStepper is pending."
- **Differentiability (delivered, binary):** `src/diffsim/adjoint/` —
  `phasefield.py` (discrete IFT adjoint through **binary** M=1 CH, three-way
  verified: hand IFT adjoint = torch autograd twin = finite differences),
  `crystallization.py` ((M,K)=(1,1)), `neural_energy.py` (gauge-anchored
  learnable free energy, M=1), `torch_twin.py` (autograd reference).
- **The gap:** the production forward `daisy-morph` runs is M-component, but
  `gradients=` raises `NotImplementedError`; gradient work falls back to finite
  differences (N+1 forward runs per gradient).
- **Downstream:** `daisy-morph` (executor) + `daisy-hull` (GPU phase-diagram
  oracle) drive DAISY BO campaigns; acquisition/campaign logic is DAISY-side.

## Scope decisions (ratified)

- **v1 physics = phase-separation only** (M-component multi-CH, K=0; φᵢ, μᵢ). No
  crystallization ψ/θ in v1, but the block/coupling machinery is **designed-for**
  crystallization extension (§7).
- **v1 parameters = residual (bulk/transport) parameters only:** χ-matrix, N,
  mobility M, κ. **Mean-φ** (initial-condition path) is the immediate
  fast-follow (§7), not v1.
- **Objective = arbitrary user-supplied differentiable scalar** `j(φ-field)`;
  first verification objective = interfacial (gradient) energy.
- **Execution = two rungs.** Rung 1 CPU reference-grade + three-way verified;
  rung 2 GPU production 256×128 (committed, not optional).
- **Deliverable boundary:** validated gradient API + one inverse-design demo.
  KG-BO acquisition/campaign loops stay DAISY-side.

## Architecture

A **discrete IFT adjoint** through the M-component multi-CH BDF march, reusing
the proven binary pattern (`phasefield.py`): a self-contained discrete forward
so cotangents flow and every gradient is three-way verifiable, generalized to a
`2M`-DOF node block and the full χ-matrix.

### Rung plan

- **Rung 1 — CPU, reference-grade, verified.** Vectorised numpy/scipy
  M-component discrete forward + adjoint. Forward-parity to `MultiPhaseStepper`
  (K=0) at Newton tolerance. Three-way gradient gate at 64² (spot-checks to
  128²). Wire `daisy-morph gradients=` → CPU engine. Ship one inverse-design
  demo. **Documented limitation:** gradients at reference resolution (≤128²)
  until rung 2.
- **Rung 2 — GPU, production 256×128 (committed).** On-device `Jᵀ`/cuDSS adjoint
  reusing the production converged Jacobian; committed-state trajectory
  checkpointing; parameter derivatives as device arrays; same `gradients=`
  signature. Gate: matches rung-1 CPU gradients at overlapping resolution + the
  6 daisy-morph GPU integration tests → bump the DiffSim pin.
- **Fast-follow (after rung 1) — mean-φ initial-condition gradient** (§7).

## The math

Per committed BDF step `n`, the forward Newton drives `Rₙ(xₙ; x_hist, p) = 0`
over the node-major `2M` block `[(φ₀,μ₀), …, (φ_{M-1},μ_{M-1})]`; solvent
`φ_s = 1 − Σφᵢ` eliminated. Weak residual (K=0):

```
R_{φ_i} = ∫ N (σ φ_i − hist_i) + Σ_j M_ij ∫ ∇N·∇μ_j
R_{μ_i} = ∫ N (μ_i − ∂f/∂φ_i) − κ_i ∫ ∇N·∇φ_i
f = Σ_i (φ_i/N_i) ln φ_i + Σ_{i<j} φ_i φ_j χ(i,j)
    + Σ_i (κ_i/2)|∇φ_i|² + b Σ_i 1/φ_i
```

All species sums include the eliminated solvent (index M), so `∂f/∂φ_i` carries
the `∂φ_s/∂φ_i = −1` chain. The reference discrete forward must reproduce
`MultiPhaseStepper` (K=0) to Newton tolerance.

Adjoint for objective `J = Σₙ j(xₙ)`, marched backward:

```
Jₙᵀ λₙ = ∂j/∂xₙ − Σ_{k≥1} (∂R_{n+k}/∂xₙ)ᵀ λ_{n+k}
dJ/dp  = − Σₙ λₙᵀ (∂Rₙ/∂p)
```

`Jₙ = ∂Rₙ/∂xₙ` is **exactly the converged forward Jacobian** (refactor via
`splu`; do not tape Newton). Generalization vs. binary: the BDF history couples
**all M** conserved φ-fields (binary's single-`c` history term → M mass-coupled
φ-rows; mirrors `sbm/transient_adjoint.py`'s history coupling), and `∂R/∂p` is
analytic per parameter.

### Parameter derivatives `∂R/∂p` (rung-1 set)

| param | enters | `∂R/∂p` |
|---|---|---|
| **χ(i,j)** | `∂f/∂φ` in `R_μ` | closed-form, bilinear in φ (per pair, incl. solvent pairs; full symmetric matrix) |
| **Nᵢ** | `(φᵢ/Nᵢ) ln φᵢ` in `R_μ` | analytic (`∂/∂Nᵢ` of the log term) |
| **mobility Mᵢⱼ** | transport term in `R_φ` | `∫ ∇N·∇μⱼ` (linear) |
| **κᵢ** | gradient term in `R_μ` | `−∫ ∇N·∇φᵢ` (linear) |

χ and N are **bulk-energy** parameters routed through the `MultiEnergy` object
(§9); mobility M and κ are **engine-level** (transport / gradient energy).

## Objective interface

Arbitrary user-supplied differentiable scalar `j(φ-field)` (written in torch).
The engine obtains `∂j/∂φ` by autograd on the field and maps it into the
`2M`-DOF cotangent (μ-rows = 0). Full-trajectory objectives `Σₙ j(xₙ)` are
supported (needed for learning; §9).

**First verification objective:** interfacial (gradient) energy
`∫ (κ/2) Σ|∇φᵢ|²` — smooth, and exactly the early-stop signal `daisy-morph`
already tracks. Total free energy `F[φ]` is the second.

## Verification gate (three-way — house rule)

`tests/test_multiphase_adjoint.py`:
1. **Forward parity** — reference discrete forward == `MultiPhaseStepper` (K=0)
   to Newton tolerance.
2. **Three-way gradient** — hand IFT adjoint == torch twin (`MultiCHTwin`,
   autograd-through-convergence) == finite differences, per parameter, for a
   ternary and one quaternary case, at 64² (spot-check 128²). Tight tolerances;
   commit-per-green.

## Module layout (rung 1)

```
src/diffsim/adjoint/
  multiphase.py   # MultiCHDiscrete, MultiCHForward, MultiCHAdjoint,
                  # FHMultiEnergy (a MultiEnergy implementation, §9)
  torch_twin.py   # + MultiCHTwin (extend existing; consumes a MultiEnergy)
tests/test_multiphase_adjoint.py
```

`daisy-morph`: implement `gradients=` in `core.py` (currently
`NotImplementedError`) → run the reference differentiable forward on CPU, return
`r.gradients` dict `{param: dJ/dp}` alongside `SimResult`; retain the FD
fallback. Document the reference-resolution limitation in `AGENTS.md`/`README`.

**Inverse-design demo:** gradient descent on the χ-matrix to hit a target
interfacial-energy / descriptor, preceded by a **Gramian / identifiability
pre-check** (house rule: Gramian-before-compute for inverse work; χ carries the
known gauge/aliasing structure documented in `neural_energy.py`).

## §7. Designed-for extensions

- **Rung 2 (GPU, committed).** The production forward already assembles `Jₙ` at
  convergence; re-form/transpose and solve `Jᵀλ` with cuDSS on-device. Store the
  committed-state trajectory `{xₙ}` and recompute the factorization on the
  backward pass — the main memory design point at 256×128×2M. Parameter `∂R/∂p`
  as device arrays. Same `gradients=` signature.
- **Mean-φ initial-condition gradient (fast-follow).** Cotangent into `x₀` then
  `∂x₀/∂φ_mean` (uniform shift of the φ-field mean under a fixed noise seed) — a
  distinct mechanism from the residual-parameter path, isolated and added once
  the engine is verified.
- **Crystallization (later milestone).** The `2M` block-factory and
  history-coupling machinery extend to `2M+2K` (add the ψ Allen-Cahn block)
  exactly as binary `crystallization.py` extends `phasefield.py` — extension,
  not rewrite.

## §9. Extensibility to free-energy learning (M6)

The binary code never hard-codes the bulk energy: `phasefield.py`/`torch_twin.py`
consume an **energy object** exposing `f`, its φ-derivatives, and its own
parameter-cotangents. The M-component engine carries the identical discipline.

**`MultiEnergy` protocol** — every bulk free energy (parametric FH *or* learned)
implements:
- `f(φ)`, the M chemical potentials `∂f/∂φᵢ`, the Hessian `∂²f/∂φᵢ∂φⱼ` (for the
  Jacobian);
- `∂(∂f/∂φᵢ)/∂θ` for each of *its own* learnable parameters θ.

The adjoint engine differentiates the energy object **opaquely** via
`dJ/dθ = −Σₙ λₙᵀ (∂Rₙ/∂θ)`, where `∂Rₙ/∂θ` comes entirely from the energy
object.

| application | energy object | learnable θ | engine change |
|---|---|---|---|
| Rung-1 BO | parametric FH (`FHMultiEnergy`) | χ-matrix, N | — |
| Free-energy learning (M6) | neural / extended-FH-basis (M-generalization of `neural_energy.py`) | network / basis weights | **none** |

Pinned now so learning lands cleanly later:
- **χ and N route through the `MultiEnergy` object even in rung 1** (they are
  bulk-energy params), while **mobility M and κ stay engine-level**. Parametric
  FH is then literally "a learned energy with χ,N exposed," so M6 reuses the
  identical cotangent path.
- **The gauge-anchor lives inside the energy object.** The M-component
  generalization of the `{1, φᵢ}`-orthogonality projection (constant + linear-in-φ
  modes unidentifiable from conserved dynamics; quadratic already spanned by χ)
  is the energy object's responsibility — differentiable, invisible to the
  engine. The Gramian/identifiability pre-check in the BO demo is the same guard
  the learning loop needs.

Net: reaching free-energy learning is a new `MultiEnergy` subclass + the
already-planned twin extension — **no engine rework**. The rung-1 cost is only
discipline: honor the protocol boundary instead of inlining χ/N into the
residual.

## House rules honored

Measure-then-lock; commit-per-green; three-way-verified gate; Gramian-before-
compute on the inverse-design demo; mobility closures stay NAMED
(`wodo | negi | …`); findings-log honesty on the rung-1 resolution limitation;
no dyadic feature dims in gates.

## Deliverables checklist

- [ ] `src/diffsim/adjoint/multiphase.py` — `MultiCHDiscrete`, `MultiCHForward`,
      `MultiCHAdjoint`, `FHMultiEnergy` (via `MultiEnergy` protocol).
- [ ] `MultiCHTwin` in `torch_twin.py` (energy-object aware).
- [ ] `tests/test_multiphase_adjoint.py` — forward parity + three-way gradient
      gate (ternary + quaternary).
- [ ] `daisy-morph` `gradients=` wired to the CPU engine (`r.gradients`), FD
      fallback retained, limitation documented.
- [ ] Inverse-design demo (χ → target objective) with Gramian pre-check.
- [ ] Rung 2: GPU `Jᵀ`/cuDSS adjoint at 256×128, pin bump through the 6 GPU tests.
- [ ] Fast-follow: mean-φ initial-condition gradient.
