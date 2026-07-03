# DiffSim: GPU-Native Differentiable Adaptive Octree FEM Framework

**Design specification — 2026-07-02**
**Status:** Approved design, pre-implementation
**Relationship to other efforts:** This document is (a) the architecture blueprint for the production CUDA C++ framework (cuFEM) and (b) the specification for the DiffSim prototype built in NVIDIA Warp. It serves as the shared-substrate design for the FASTEST research program (Tracks S and MF, plus the differentiability thread D1).

---

## 1. Goals, Scope, and Deliverables

### 1.1 What DiffSim is

A GPU-native, natively differentiable, adaptive octree FEM framework for immersed-boundary multiphysics — the next-generation successor to the group's Dendro-KT / Dendrite-KT / TalyFEM CPU stack. Two coupled deliverables:

1. **The blueprint** (this document + per-milestone chapters): the design the production CUDA C++ framework (cuFEM) is implemented against — data structures, kernel designs, module boundaries, adjoint strategy, multi-GPU model — with the CPU stack's proven algorithms (SFC partitioning, 2:1 balance, traversal-based assembly, incomplete octrees) mapped to their GPU-native forms.
2. **The prototype**: a working NVIDIA Warp implementation whose kernels are written to translate ~1:1 to CUDA C++. It de-risks the blueprint: every architectural claim is validated by running prototype code before the corresponding blueprint chapter is frozen.

### 1.2 Requirements (from stakeholder discussion)

- CUDA-native; single-node multi-GPU over NVLink only (no network transport).
- Adaptive, incomplete (carved) octrees; 2:1 balanced; hanging-node constraints.
- Linear (p1) and quadratic (p2) bases; both in the prototype from milestone 1; mixed p1/p2 in one mesh designed for, staged later.
- Shifted Boundary Method (SBM) as the immersed method (IMGA documented as a non-differentiable attachment point only; see §1.5).
- Geometry backends: neural SDF (INR), STL/triangle mesh, analytic CAD/CSG primitives, discrete SDF grids.
- Lego-like multiphysics: NS + X where X ∈ {thermal, mass transport, Cahn–Hilliard, PNP, elasticity, thin shells, AM-thermal}; FSI/conjugate problems as the trajectory.
- Natively differentiable: shape/geometry optimization, neural-closure training in the loop, inverse problems / data assimilation, control/RL rollouts (all four confirmed as primary use cases).
- Neural constitutive models and neural signed distance fields as first-class citizens.
- Solution-adaptive mesh refinement with a pluggable indicator interface (user-specified or a posteriori), requiring k-ring (1-ring, optionally 2-ring) octree neighbor queries.
- Mixed-precision arithmetic: per-element precision, iterative refinement (IR) and GMRES-IR, tensor-core paths (FASTEST Track MF).
- Adaptive additive-manufacturing (FDM printing) simulation: element activation, layer-synchronous adaptivity, digital-twin (faster-than-real-time) target.
- Backward Euler (BDF1) and BDF2 time integration (plus θ-method, matching existing codes).
- Nomenclature consistent with Dendrite-KT / TalyFEM / Proteus (Hughes FEM textbook + DiffPack heritage): `Integrands` / `Integrands4side` as the user-facing weak-form API.
- Reuse NVIDIA-native components wherever they exist (AMGX, cuDSS, cuSPARSE, cuBLAS/CUTLASS, NCCL, Warp mesh/BVH/volume primitives, OptiX where warranted).

### 1.3 Milestone-1 definition (prototype)

End-to-end differentiable Octree-SBM incompressible Navier–Stokes around a static complex 3D geometry — VMS-stabilized, p1 and p2 bases, both stepping strategies (Leray projection and monolithic semi-implicit), BDF1/BDF2, with validated adjoint gradients w.r.t. (a) geometry parameters and (b) a constitutive parameter. Single GPU. The hero demonstration uses a neural SDF; all four geometry backends are exercised through the oracle verification suite (SBM1–7 + AD tier).

### 1.4 Out of scope

Multi-node / MPI (the halo-exchange interface is the documented attachment point, nothing more); body-fitted mesh generation; production I/O beyond VTU; IMGA cut-cell quadrature in the differentiable core; CPU execution paths (verification comparisons run against the existing CPU stack, not a CPU port of this framework).

### 1.5 Why SBM over IMGA — the differentiability argument

Both methods immerse geometry in a non-conforming octree and enforce BCs weakly (Nitsche). The difference is *where geometry enters the discrete residual*:

**IMGA: geometry enters through quadrature.** Cut elements are recursively subdivided; sub-cell Gauss points are classified in/out; volume integrals run over "in" points only. The residual therefore depends on geometry through the *active set of quadrature points* — a piecewise-constant function of shape. An infinitesimal boundary motion either flips no Gauss point (volume-term shape derivative exactly zero — the derivative misses the physics) or flips one (residual jumps — Dirac spike). These discontinuities occur at sub-cell scale h/2^s throughout every intercepted element and are dense in design space. Additional non-smoothness: sliver-cut conditioning spikes, penalty floors (h-clamping is a discrete switch), and discrete subdivision-depth choices. A differentiable IMGA requires smeared-Heaviside quadrature weights, reintroducing diffuse-interface error into every volume integral.

**SBM: geometry enters through a smooth field.** All volume integrals are over whole, uncut elements with fixed, geometry-independent quadrature. Geometry appears in the residual through one continuous channel: the distance-vector field **d**(x̃) at surrogate-boundary Gauss points — in the Taylor shift (u + ∇u·d), the mapped boundary data u_D(M(x̃)), and the penalty terms. **d** is a smooth function of the geometry representation (analytic for SDF backends; piecewise-smooth for STL closest-point). ∂R/∂(shape) is obtained by differentiating a thin set of boundary-face terms only, mirroring the continuous Hadamard shape-derivative structure. The only discrete dependence is element classification (λ-criterion) — quantized at whole-element granularity and localized: for shape perturbations smaller than an element crossing, the active set is constant and gradients are exact. Freezing the surrogate per design iteration gives consistent gradients within an O(h) trust region (the same contract as frozen-mesh adjoints in body-fitted shape optimization). Reclassification events coincide with remesh events, so one epoch mechanism (§7) handles both.

Secondary: fixed quadrature ⇒ fixed sparsity ⇒ adjoint solves reuse forward operators/preconditioners; no sliver cells ⇒ adjoint solves as well-conditioned as forward; SBM's information requirement (distance vectors only) is exactly what every geometry backend produces natively, whereas IMGA sub-cell in/out tests against a neural SDF would differentiate a sign function.

### 1.6 Success criteria

- Prototype matches published group benchmarks (cylinder Re=100 C_d/St; sphere Re=300; lid-driven cavity with immersed obstacles; p2 convergence orders; per-brick benchmarks in §6).
- Gradient checks pass (custom-VJP vs unrolled-tape vs finite differences; rel. error < 1e-6 in FP64 on small problems).
- One demonstration inverse problem per capstone (electrode shape, AM process parameters, one control rollout).
- The blueprint is implementable milestone-for-milestone without reverse-engineering the prototype.
- Performance gates in §9 pass.

---

## 2. System Architecture

### 2.1 Layer map

Five layers, strict downward dependencies. Physics and geometry are data-driven plugins.

```
┌─────────────────────────────────────────────────────────────────┐
│ L5  ORCHESTRATION      Scenario / Optimizer / RL-rollout loops   │
│     TimeLoop · CheckpointManager · AdaptationController ·        │
│     PrintScheduleDriver (AM)                                     │
├─────────────────────────────────────────────────────────────────┤
│ L4  COUPLING           BlockIterativeCoupler (NS↔X fixed point)  │
│     TimeStepper: {LerayProjection | MonolithicSemiImplicit}      │
│     TimeScheme: {BDF1 | BDF2 | Theta}                            │
├─────────────────────────────────────────────────────────────────┤
│ L3  PHYSICS BRICKS     one CEquation module per equation:        │
│     NS-VMS · Heat-SUPG · Mass-SUPG · CahnHilliard · PNP ·        │
│     Elasticity · ThinShell · AMThermal · …                       │
│     each brick = {Integrands_Ae/be, Integrands4side_Ae/be,       │
│                   closure hooks (analytic | tabulated | neural)} │
├─────────────────────────────────────────────────────────────────┤
│ L2  DISCRETIZATION     ElementKernels (p1/p2 tensorized, typed   │
│     per (p, precision) bin) · SurrogateBoundary · Assembler      │
│     (matrix-free + assembled CSR) · LinearSolvers (Warp Krylov,  │
│     AMGX, cuDSS; IR/GMRES-IR wrappers) · NonlinearSolver         │
│     (SNES-like Newton–Krylov) · PrecisionManager ·               │
│     Adaptivity (indicators, refine/coarsen, transfer)            │
├─────────────────────────────────────────────────────────────────┤
│ L1  FOUNDATION         Octree (Morton, 2:1, k-ring neighbors,    │
│     incomplete/carving, activation masks) · GeometryOracle       │
│     (4 backends) · FieldStorage (SoA, ghost layers, NodeData) ·  │
│     DevicePartition (NVLink)                                     │
└─────────────────────────────────────────────────────────────────┘
         ⇅ adjoint counterpart of every layer (two-tier: tape/AD
           for kernels, custom VJPs at solve/transfer boundaries)
```

### 2.2 Cross-cutting contracts

1. **Topology epochs.** L1 structures (classification, surrogate faces, neighbor tables, constraints, sparsity, preconditioner setups) are immutable between adaptation events and versioned by an epoch counter. Everything downstream keys off the epoch. This is what makes both the CUDA port and adjoint bookkeeping tractable.
2. **Linear epoch-transition operators.** Every operator that crosses an epoch boundary (intergrid transfer, history extrapolation, Leray correction) must be linear. Adjoint = transpose. Discrete decisions (indicator flags, classification changes, activation) sit *between* linear operators, never inside a differentiated segment.
3. **Pure integrands.** `Integrands_*` functions are pure (no hidden state access); all Gauss-point data arrives through the `FEMElm` view. Purity is what makes them auto-adjointable (Warp tape) and Enzyme-differentiable (CUDA).
4. **FP64 spine.** Residuals, reductions, geometry data (distance vectors, normals, penalty scalings), and conservation diagnostics are always FP64, regardless of element precision bins.
5. **Buy, don't build.** Krylov loops are written in-framework (they are small and the tape must see them); AMG, sparse direct, batched GEMM, BVH queries, and collectives are always library calls (§5.4).

### 2.3 L1 Foundation

- **Octree:** GPU-resident sorted Morton-key leaf array (CUB radix sort in CUDA; Warp sort in prototype). Max level 21 (64-bit keys, 3D). Incomplete via carving against the GeometryOracle. 2:1 balance by iterated neighbor-key insertion + re-sort. Refinement predicate = user callback on octant coordinates (Dendro convention).
- **Neighbor table (new vs cuFEM's current design):** assembly does NOT need neighbor lists (traversal/scatter-gather stands). Explicit k-ring tables are built on demand, per epoch, only when an indicator/estimator requests them: Morton-key arithmetic (same-level candidates + parent fallback) + binary search over the sorted leaf array; face/edge/vertex 1-ring in CSR-style adjacency; 2-ring by composition, cached only if profiling justifies. Cost: one sorted-array sweep, epoch-amortized.
- **Hanging-node constraints:** stored as an explicit sparse interpolation operator (GPU-resident), making constraints data rather than control flow; adjoint is the stored transpose. No hanging nodes at carved boundaries (cancellation-node rule from the SC'21 paper).
- **Activation masks (AM support):** 7 bits per element (1 active + 6 face-boundary flags), updated incrementally on voxel activation by checking six neighbors. Inactive elements are masked/Dirichlet-frozen, not compacted: activation is not a topology event; sparsity and DOF maps are stable within a layer.
- **FieldStorage:** struct-of-arrays, per-device ghost layers in Dendro pre-ghost/local/post-ghost layout; NodeData slot semantics (§3).
- **DevicePartition:** SFC-contiguous leaf ranges per GPU (§8). A 1-device partition is the degenerate case; no code path forks.

### 2.4 L2 Discretization

- **ElementKernels:** tensorized reference-element operators parameterized by basis order p (GEMM-cast assembly; Warp tiles / tensor cores for p2). Kernels are *typed per (p, precision) bin* (§2.5). Mixed p1/p2 in one mesh: per-element order tags + order-transition constraints — designed here, prototyped post-M1.
- **SurrogateBoundary:** element classification (λ-criterion via Gauss-point oracle queries), face extraction, distance-vector evaluation (§4).
- **Assembler:** matrix-free actions (Krylov operator applies) and assembled CSR (for AMGX/cuDSS and preconditioners), both from the same integrands. Deterministic mode: no floating-point atomics (coloring or gather-based assembly), pinned reduction order.
- **LinearSolvers:** §5.
- **PrecisionManager:** §2.5.
- **Adaptivity:** §7.

### 2.5 Mixed precision (FASTEST Track MF)

Precision is a per-element property, exactly like basis order:

- Elements are binned by (p, precision) tuples; each bin dispatches to a typed kernel variant (FP64 / FP32 / TF32-BF16 tensor-core). Batched low-precision element applies are cast as matrix-matrix products for tensor cores (MF2).
- **IR / GMRES-IR wrappers** around any inner solver: inner solve/preconditioner in bin precision; residual, correction, and all reductions in FP64. GMRES-IR for the stiff regime.
- **Precision selection:** starts as a config table; graduates to the MF3 law — plain-IR / GMRES-IR / FP64-fallback selected where κ(A)·u_low crosses O(1), with κ(A) driven by p, the Nitsche penalty α/h, and physics stiffness.
- **Geometry data is always FP64** (precomputed once per epoch and frozen — the FASTEST S1 pattern), because penalty terms feed the conditioning cliff.
- **Adjoint precision policy is separate from forward** (D1): gradient-precision floor ≠ forward floor. Checkpoint storage precision and adjoint-solve precision are independent knobs; the prototype measures gradient error vs checkpoint precision (M3/M6 experiments) and feeds MF4's budget.
- **Sentinels:** mass/energy drift diagnostics (FP64) are the canaries that precision bins must respect within declared budgets.

The Track S (assembled / AMGX / FP64) and Track MF (matrix-free / mixed-precision) paths are two implementations behind the same `Assembler`/`LinearSolver` interfaces: the module boundary is the program's track boundary.

---

## 3. User-Facing API and Nomenclature

A student who has written a Dendrite/TalyFEM app must be able to write a DiffSim brick with zero relearning. Heritage: Hughes' Linear FEM textbook symbols + DiffPack simulator-class conventions, as carried by TalyFEM (`talylite`), Dendrite-KT, and Proteus.

### 3.1 The Integrands contract

The framework owns the element loop and the Gauss-point loop; user code is called per integration point:

```python
class HTEquation(CEquation):                    # CEquation base, DiffPack/TalyFEM
    @wp.func
    def Integrands_Ae(fe: FEMElm, Ae: ElementMatrix, h: float):
        detJxW = fe.detJxW()
        for a in range(fe.nbf()):
            for b in range(fe.nbf()):
                M = fe.N(a) * fe.N(b) * detJxW
                K = wp.float64(0.0)
                for k in range(fe.nsd()):
                    K += k_val * fe.dN(a, k) * fe.dN(b, k) * detJxW
                Ae[a, b] += bdf_c0 * M / dt + K

    @wp.func
    def Integrands_be(fe: FEMElm, be: ElementVector, h: float): ...
    @wp.func
    def Integrands4side_Ae(fe: FEMElm, sideInd: int, id: int,
                           Ae: ElementMatrix, h: float): ...
    @wp.func
    def Integrands4side_be(fe: FEMElm, sideInd: int, id: int,
                           be: ElementVector, h: float): ...
```

Preserved verbatim from the existing stack:

- `FEMElm` accessors: `N(a)`, `dN(a, i)`, `d2N(a, i, j)`, `detJxW()`, `nsd()`, `nbf()`, `position()`, `surface().normal()`.
- Hughes symbols: `Ae`, `be`; `a`/`b` trial/test indices; `M`/`K`/`R` term names; `tauM`, `tauC`; `nsd`/`nbf`/`ndof`.
- Node-major block indexing: `Ae(ndof*a + dof_i, ndof*b + dof_j)`, `be(ndof*a + dof_i)`.
- `NodeData` contract: `Vars` enum (`VEL_X, …, PHI, MU, …`), `value(i)`, `name(i)`, `valueno()`; history slots `*_PRE1/_PRE2/_PRE3` rotated at L5.
- `VecInfo(vec, ndof, nodeDataOffset)` maps solver vectors into NodeData slots; `PLACEHOLDER_GUESS` marks the nonlinear iterate.
- Dirichlet BCs: position-lambda returning `Boundary` with `addDirichlet(dof, value)`; `sideInd` from `BoundaryTypes::WALL` (`X_MINUS … Z_PLUS`, then `VOXEL::SPHERE/BOX/CIRCLE/GEOMETRY/FUNCTION`).
- `InputData` reads `config.txt` (libconfig-style); `solver_options_*` blocks apply to per-solver option prefixes (`"-momentum_"`, `"pp_"`, `"ch_"`); AMGX JSON configs follow the same per-block convention.
- `TimeInfo` (t0, dt vector, totalT vector); `dendrite_init/finalize` bracketing; `PrintInfo/PrintStatus/PrintWarning/PrintError` rank-aware logging; VTU output per `petscVectopvtu` naming (`results_%05d`).
- Canonical sub-solve block names: **CH-solve / VP-solve / PP-solve / VU-solve / Remesh**.
- Style: Google C++ base (trailing-underscore members, CamelCase classes, `k`-prefix constants in framework code, UPPER_SNAKE enums in apps), per talylite `docs/style.md`.

### 3.2 Deliberate deltas (documented as such)

1. **Pure integrands.** No `p_data_` member access inside integrands; Gauss-point field values arrive via `fe.value(PHI)` / `fe.valueDerivative(PHI, i)` (same semantics as `valueFEM`/`valueDerivativeFEM`, backed by NodeData slots). Purity ⇒ Warp-tape adjoint in the prototype, Enzyme in CUDA. This is the D1 division of labor: AD on element residuals; implicit adjoints on solves.
2. **SBM side variant.** `Integrands4side` gains an SBM form receiving the distance vector `d`, true normal `n`, and area-correction factor `(n̄·n)` at the surrogate Gauss point. The shift operator and Nitsche term scaffolding live in the framework; the brick supplies physics coefficients.
3. **Time-scheme coefficients as data.** `bdf_c = {1.5, −2.0, 0.5}` (BDF2) / `{1.0, −1.0, 0.0}` (BDF1), θ-method as in Proteus's `setIntegrandsFlags()`; first BDF2 step bootstrapped at order 1; variable-Δt BDF2 coefficient table from the thermal-SBM paper.

---

## 4. Geometry Pipeline & SBM Core

### 4.1 GeometryOracle contract

```
classify(points[N])        → sign[N]              (in/out; sign of ψ)
distance_vector(points[N]) → d[N], normal[N]      (surrogate GP → true boundary)
velocity(points[N], t)     → v_Γ[N]               (moving geometry; zero if static)
```

| Backend | Implementation | Differentiable w.r.t. |
|---|---|---|
| NeuralSDF (INR) | MLP (PyTorch interop or Warp-native) | network weights θ and query point |
| STL / TriangleMesh | `wp.Mesh` BVH: `mesh_query_point` (closest point + sign), `mesh_query_ray` | mesh vertex positions (built-in adjoints) |
| AnalyticCSG | primitive SDFs + smooth min/max blends | primitive parameters |
| GridSDF | `wp.Volume` (NanoVDB), trilinear/tricubic | voxel values (= level-set shape optimization) |

**Distance vectors:** framework-standard **Newton closest-point projection** `x_{k+1} = x_k − ψ∇ψ/‖∇ψ‖²` (iteration cap, tolerance, per-point convergence mask) — correct for non-eikonal fields; the eikonal shortcut `d = −ψ∇ψ/‖∇ψ‖²` is an optimization for verified near-eikonal backends. Newton derivative via implicit-function theorem at convergence (never unrolled).

**Admissibility diagnostics (runtime, per epoch):** ε̂∞ estimate; ĉ₀ = min‖∇ψ‖ in the band; Hausdorff bound d_H ≤ ε̂∞/min(1, ĉ₀); Newton success masks; the ε∞ ~ h^(k+1) training-tolerance rule surfaced as a geometry-limited-refinement-plateau warning.

### 4.2 Pipeline per topology epoch

1. **Carve:** top-down construction, per-octant oracle classification, Exterior subtrees pruned. Refinement predicate: user callback or narrowband on |ψ|.
2. **Classify elements:** Gauss-point sign counts → Interior / TrueIntercepted / FalseIntercepted / Exterior via the **λ-criterion**. λ is per-brick: 0.5 (optimal surrogate; Dirichlet-dominated flow) or 1 (flux-accurate; Neumann/thermal).
3. **Extract surrogate boundary:** faces between retained and dropped elements (all-nodes-flagged face rule); two-sided Γ̃⁺/Γ̃⁻ classification (sign of mean n·ñ) for thin shells.
4. **Evaluate geometry data:** d, n, (n̄·n) at every surrogate-face Gauss point — one batched oracle call, cached for the epoch, always FP64.
5. **Assemble boundary terms:** shift operator inside framework `Integrands4side` scaffolding — Dirichlet (consistency + adjoint-consistency + penalty with Taylor shift S u = u + ∇u·d), Neumann (primal form + area correction; the π/4 failure without it is a locked test), Robin (AM convection, PNP Stern layers), two-sided shell.

### 4.3 Shape gradients

Within an epoch, geometry appears in the residual only through step-4 outputs: ∂R/∂θ = tape through boundary integrands → chain through d(θ), n(θ) → oracle backward (PyTorch autograd / wp.Mesh adjoints / analytic CSG). Steps 1–3 are frozen per epoch (piecewise-constant in θ; exact gradients within the O(h) trust region). Shape-optimization loop: carve → converge forward → adjoint → θ-step → re-carve; epoch-transition operators (linear) connect states when the active set changes.

### 4.4 Moving geometry & AM activation

Same pipeline, different epoch triggers. Rigid motion: re-carve when displacement > tol·h; transferred BDF history receives the **pressure-robust Leray correction** (one PPE solve with homogeneous-Neumann BC on the surrogate Dirichlet boundary — the BC choice is load-bearing: it forbids the correction pushing mass through the body; φ=0 on outflow, zero-mean otherwise; one solve per event covers all BDF history by linearity). AM activation: bitset flips within an epoch (no topology change); remesh once per layer.

---

## 5. Solver Core & Two-Tier Adjoint Architecture

### 5.1 Time steppers (both ship; one interface)

- **LerayProjection (default):** VP-solve (Oseen-linearized predictor; advecting velocity â = 2uⁿ − uⁿ⁻¹; skew-symmetric form s=½ default for unconditional energy stability) → PP-solve (variable-coefficient incremental pressure Poisson with VMS fine-scale RHS term) → VU-solve (per-component mass-matrix L² projection, reusing one assembled mass matrix). All linear; PP/VU SPD.
- **MonolithicSemiImplicit:** one coupled (u,p) linear solve per step with the exact-adjoint linearized VMS. Contrast case for MF3 conditioning analysis and splitting-error checks.

Stabilization: elementwise-constant tauM/tauC (no ∇τ, no residual-derivative terms — deliberate, keeps integrands differentiable). Nonlinear bricks (CH, PNP) use the NonlinearSolver (§5.1.1) inside the block coupler.

#### 5.1.1 NonlinearSolver (SNES-like Newton–Krylov)

A first-class, brick-agnostic Newton solver mirroring PETSc SNES, wrapping any `LinearSolver` for the inner solve:

- **Structure per iteration:** residual F(u) from `Integrands_be`; Jacobian J from `Integrands_Ae` (analytic, the existing convention) or by AD of the residual integrands (Tier-1 forward-mode on the pure integrands — consistent linearization for free, including closure input-derivatives per §6.3); solve J δu = −F via any inner `LinearSolver` (Warp Krylov, AMGX, cuDSS); update with line search.
- **Variants:** full Newton with assembled J; **JFNK** (Jacobian-free Newton–Krylov: J·v by AD directional derivative of the residual — the matrix-free Track-MF citizen, preconditioned by the low-order-refined proxy of §5.4); modified Newton (frozen J across iterations/steps, the cheap default for mildly nonlinear bricks).
- **Line search:** backtracking (`bt`) and critical-point (`cp`) to start, matching SNES defaults; none (`basic`) for well-behaved problems.
- **Inexact Newton:** Eisenstat–Walker forcing terms for the inner tolerance (loose early, tight late) — composes with IR/GMRES-IR (the inner solve's precision policy is the bin policy; Newton residual norms always FP64).
- **Nomenclature:** `NonlinearEquation` base (as in TalyFEM) and `setNonLinearSolver(eq, octDA, ndof, mfree)` construction; options via the same `solver_options_*` config blocks with SNES-style keys (`snes_rtol`, `snes_atol`, `snes_max_it`, `snes_linesearch_type`) under per-solver prefixes (`"ch_"`, `"pnp_"`), exactly the Proteus convention.
- **Adjoint:** unchanged Tier-2 story — implicit-function adjoint at convergence (one transposed-Jacobian solve at the converged state); Newton iterations are never unrolled. The convergence tolerance bounds the adjoint consistency error, so `snes_rtol` participates in the gradient-precision budget (D1/MF4).

### 5.2 Two-tier adjoints

Rule: **tape what's element-local, invert what's global.**

- **Tier 1 (automatic):** all integrands, geometry evaluations, closures, transfer kernels — pure Warp functions, adjoint by `wp.Tape` (prototype) / Enzyme on the same contract (CUDA).
- **Tier 2 (custom VJPs — exactly four):**
  1. *Linear solve* x = A⁻¹b → Aᵀ-solve. SPD blocks: Aᵀ = A, same setup/preconditioner reused. Oseen: adjoint operator **assembled directly** via the exact-adjoint identity M*ₐ,ₛ = −Mₐ,₁₋ₛ (swap s → 1−s in the same integrand); own AMGX setup, amortized over the backward sweep (linearization point frozen per step).
  2. *Newton solve* (NonlinearSolver, §5.1.1): implicit-function adjoint at convergence — one transposed-Jacobian solve.
  3. *Newton closest-point projection*: IFT at the converged point.
  4. *Block-iterative coupler*: reverse-sweep vs coupled-IFT at convergence — both implemented; prototype benchmarks the crossover (expected: IFT wins when block iterations > 2).

### 5.3 Checkpointing

Baseline: checkpoint converged state per timestep + epoch metadata; recompute assembly in the backward sweep (adjoint ≈ 2× forward compute; memory O(steps)). Long horizons: binomial/Revolve at L5. Checkpoint storage precision is a policy knob (D1 gradient floor; measured in M3/M6).

### 5.4 Linear algebra & the solver reuse map

Framework-written: Krylov loops (CG, BiCGStab, GMRES — small, tape-visible, FP64 reductions), Jacobi/Chebyshev preconditioners, IR/GMRES-IR wrappers.

| Component | Library | Use | Adjoint story |
|---|---|---|---|
| AMG | **AMGX** (pyAMGX in prototype; C API in cuFEM; device CSR zero-copy via `__cuda_array_interface__`) | Track-S elliptic + nonsymmetric solves; multi-GPU native | SPD: reuse setup; nonsym: assemble adjoint operator (§5.2.1) |
| Sparse direct | **cuDSS** | stiff-regime fallback (the observed ASM→LU crossover as PNP ε shrinks); coarse solves in custom MG | native transpose-solve flag |
| SpMV, ILU0/IC0 | **cuSPARSE** | assembled applies; simple preconditioners | native transpose SpMV |
| Krylov/AMG (portable, mixed-precision) | **Ginkgo** | production CUDA C++ alternative for Track MF | operators ours |
| Batched dense | **cuBLAS/CUTLASS**, Warp tiles | tensorized p2 assembly, per-element ops | tape/Enzyme |
| Collectives | **NCCL** | dot products, gradient allreduce | symmetric |

**Matrix-free preconditioning (Track MF recommended strategy):** low-order-refined proxy — assemble the p1 (or low-precision) sparse operator on the same nodes, hand it to AMGX as preconditioner; true operator apply stays matrix-free and mixed-precision. Plus Chebyshev/Jacobi as portable base.

---

## 6. Multiphysics Bricks & Coupling

### 6.1 Brick catalog (defaults from the group's papers)

| Brick | Fields (ndof) | Stabilization | SBM λ / BC types | Solve type | Source |
|---|---|---|---|---|---|
| NS-VMS | u, p (d+1) | RB-VMS, s=½, backflow | 0.5 / Dirichlet shift | VP+PP+VU or monolithic | Octree-SBM JCP; solver papers |
| Heat / Mass-SUPG | θ or c (1) | SUPG + backflow | 1 / Dirichlet, Neumann (n̄·n), Robin | linear | SBM thermal flows |
| CahnHilliard | φ, μ (2) | — (mixed form) | wetting (stretch) | Newton (CH-solve) | CPC/JCP CHNS |
| PNP | φₑ, c₁…c_N (N+1) | SUPG on NP | 1 / Dirichlet, no-flux, Robin (Stern) | Newton, monolithic PNP block | JCP NS-PNP; weak-BC CMAME |
| Elasticity | u (d) | — | 0.5 / Dirichlet, Neumann | linear (Newton later) | Neural-SBM CAD |
| ThinShell (modifier) | — | — | two-sided Γ̃⁺/Γ̃⁻ | — | ThinShell |
| AMThermal | T (1) | SUPG if advective | activation masks / Robin (h_infill, h_out), plate Dirichlet | linear | FEAD AM |

Milestone 1 ships NS-VMS + Heat. Every later brick is a pure L3 addition (the lego test: no L1/L2 diffs in the PR), validated by its paper's benchmarks.

### 6.2 Coupling contract

A brick declares `fields` (NodeData slots + ndof), `reads` (e.g., NS reads θ for buoyancy Ri·θ·ê_g, φ for ρ(φ), η(φ); Heat/PNP read u), `writes`, `solve type`. `BlockIterativeCoupler` topologically orders blocks and runs the fixed-point loop with the ‖U_k − U_{k−1}‖ convergence check; one-way couplings degenerate to a single sweep. Coupling data flows only through NodeData/VecInfo; bricks never call bricks. CHNS = NS-VMS + CahnHilliard + mixture closures + local-Cahn coefficient field (a learnable Gauss-point coefficient).

### 6.3 Closure hooks (neural constitutive models)

Any Gauss-point coefficient may be declared a `Closure`: η(γ̇), κ(T), ψ′(φ), mobility m(φ), tauM (learned-subgrid slot), AM h(T), κ(T), C_p(T). A closure is analytic | tabulated | neural. Prototype: PyTorch modules, batched per Gauss-point array, zero-copy; gradients flow to inputs (Jacobian) and weights (training-in-the-loop). Production: ONNX Runtime / TensorRT (continuing the neural-viscosity deployment pattern). Required training contract: Lipschitz regularization (log-space training + input-gradient penalty) — bounded closure Jacobians for stable Newton and stable end-to-end gradients. When a closure sits inside an implicit solve, its input-derivative enters the Jacobian (consistent linearization by AD; lagging optional).

### 6.4 Per-brick verification

Each brick carries its paper's benchmarks as acceptance tests plus one gradient benchmark (NS: dC_d/dθ_shape; Heat: dNu/dκ; PNP: d(flux)/d(electrode potential); AM: dT_history/d(fan h)) validated against FD.

---

## 7. Adaptivity & Topology Epochs

### 7.1 Indicator interface

`RefinementIndicator: (fields, octree, neighbors, geometry) → target level per leaf`. Four built-in families:

1. **User/feature-based:** vorticity→level linear map (ω_min/ω_max/l_min guidance from the AMR-SBM paper); interface band |φ| ≤ δ with (R_b, R_i) two-level strategy; geometry narrowband on |ψ|; IPDPS morphology detector (erosion/dilation as local matvec-style ops; also drives local-Cahn).
2. **Recovery-based (ZZ):** gradient recovery over 1-ring (optionally 2-ring) patches. Needs neighbor table.
3. **Residual-based:** element residuals + face jumps. Needs 1-ring.
4. **Adjoint-weighted (DWR / MF4):** dual weight ω_K from an adjoint solve against a QoI — reuses the Tier-2 transposed-solve infrastructure. One budget co-allocating h, p, precision, shift order (allocation theory is MF4's contribution; the prototype provides the composition).

Indicators are outside the gradient path by design (they select the discretization; they don't parameterize physics).

### 7.2 Refine / coarsen / transfer

Multi-level single-pass refine and consensus coarsen (IPDPS Algorithms 5–7); 2:1 rebalance; SFC repartition. The **epoch transition operator** (always linear; adjoint = transpose) composes: P2C shape-function interpolation (refine); **field-conserving L² -projection coarsening** (per-field selectable: φ conservative, others injection or conservative — closes the AMR mass-drift caveat from the JCP projection paper); **Leray correction** of NS BDF history when crossing epochs (§4.4).

### 7.3 Epoch unification

Solution-AMR event = moving-body re-carve = AM layer change: flag → refine/coarsen → rebalance → repartition → transfer(+Leray) → rebuild epoch structures → resume. One code path, one adjoint treatment (checkpoint at every epoch boundary), one test set (O7 idempotence + D6 memory stability extended to the full cycle). AM adds coarsening guards (never coarsen boundary octants or current print height).

---

## 8. Single-Node Multi-GPU over NVLink

- **Design point:** 4–8 GPUs, one node, NVLink/NVSwitch peer access. No network transport. `DevicePartition` (L1) owns everything; layers above are device-count-agnostic.
- **Partitioning:** SFC-contiguous leaf ranges per GPU; repartition only at epoch boundaries. Load weights per element = f(p, precision bin, has-surrogate-faces), measured per (kernel, GPU) once.
- **Exchange:** per-device SoA arrays + Dendro-layout ghost layers (pre-ghost/local/post-ghost; halos are few large contiguous P2P copies). Prototype: Warp per-device arrays + peer access + streams. Production: `cudaMemcpyPeerAsync`/NVSHMEM + NCCL reductions. SC'21 independent/dependent element split for overlap (interior assembly on compute stream, halos on copy stream, boundary elements last).
- **Solvers:** device-local matvec + halo per apply + NCCL allReduce dot products (FP64). AMGX multi-GPU native for Track-S PP-solve. Determinism: deterministic mode pins NCCL algorithm and forbids atomics assembly (interacts with test D3).
- **Adjoint:** transposed solves reuse halo machinery (read-ghosts ⟷ accumulate-ghosts duality); per-device tapes independent; shared parameters broadcast forward, gradient-allReduced backward.
- **Staging:** M1 single-GPU against the DevicePartition interface; multi-GPU is M5. Out of scope: multi-node, GPUDirect RDMA, CPU fallback. The halo interface is the documented multi-node attachment point.

---

## 9. Verification & Performance Program

### 9.1 Correctness

Adopt the cuFEM verification plan wholesale: tiers 1–9, same test IDs (O1–O9, N1–N6, C1–C6, Q1–Q4, B1–B9, M1–M2, G1–G5, P1–P4, V1–V7, SBM1–SBM7, S1–S6, D1–D6, CONS1–2, X1–X2, SCAL1–4, IO1–2), same operational pass criteria (machine-precision patch tests; observed order within ±0.10 over last three of five levels; **P4 — SBM linear patch on rotated geometries at all λ — as the keystone diagnostic**). Additions:

- **Tier AD:** per Tier-2 VJP and per brick: three-way gradient checks (custom VJP vs unrolled tape vs central FD; rel. error < 1e-6 FP64); dot-product tests ⟨Av, w⟩ = ⟨v, Aᵀw⟩ for every operator pair; Nitsche adjoint-consistency check; epoch-crossing gradient continuity (QoI gradient across a remesh with conservative transfer vs no-remesh reference).
- **Tier MP:** FP64-parity of IR/GMRES-IR at declared tolerances; conditioning-cliff reproduction (accuracy/iterations vs κ(A)·u_low under p and α/h sweeps) on Poisson and Oseen operators; conservation sentinels (mass/energy drift budgets).
- **Physics acceptance:** per-brick paper benchmarks locked as numerical regression baselines (not just order checks).
- **X2 bidirectional:** DiffSim-Warp ⟷ cuFEM-CUDA exchange assembled operators, solutions, and locked baselines on shared configs.

### 9.2 Performance gates (block their milestones)

- O(N) assembly/matvec on single GPU to memory capacity (SCAL4 analogue).
- Per-kernel achieved-bandwidth vs roofline model (blueprint carries the model; ≥10× per-node over the published CPU-stack numbers as floor).
- Tensor-core utilization evidence on the p2 GEMM path.
- ≥80% weak/strong scaling to 4 GPUs on epoch-free workload.
- AM digital twin: faster than real print time on the Bunny benchmark, one workstation GPU.
- Adjoint cost: backward ≤ 2.5× forward wall-clock.
- CI: Tiers 1–6 + AD per commit on workstation GPU; scaling/hero tiers per milestone on the NVLink node.

---

## 10. Roadmap (prototype milestones ↔ FASTEST program)

| Milestone | Deliverable | Gates | Feeds |
|---|---|---|---|
| M0 | Octree foundation: Morton build, carve, 2:1, hanging constraints, p1/p2 tensorized assembly, CG/BiCGStab, NonlinearSolver (Newton–Krylov, validated on Bratu) | Tiers 1–3; patch tests | S1/MF1 substrate |
| M1 | Differentiable octree-SBM Poisson → NS (both steppers, BDF1/BDF2), static 3D, all four geometry backends; shape + viscosity gradients | Tiers 4–7 + AD; cylinder/sphere/cavity; gradients vs FD | S1, D1 spike |
| M2 | Heat + Mass bricks, block coupler, neural closures in-the-loop (viscosity retrain demo) | Nu/Sh benchmarks; dNu/dκ | S2 |
| M3 | Mixed precision: (p, precision) binning, IR/GMRES-IR, tensor-core p2 path | Tier MP; cliff reproduction; parity | MF1, MF2 kernels, MF3 evidence |
| M4 | Topology epochs: AMR (4 indicator families incl. DWR), moving rigid body + Leray-corrected transfer, AMThermal brick | epoch idempotence/memory; Dütsch cylinder; Bunny faster-than-print | MF4, moving-body paper, AM twin |
| M5 | Multi-GPU NVLink: partition, halo overlap, NCCL, multi-device adjoint | ≥80% to 4 GPUs; deterministic mode | MF2 |
| M6 | Capstone inverses: electrode-shape design (PNP), AM process-parameter optimization, one control rollout | end-to-end gradient demos at scale | D1, MF6, flagship |

Blueprint chapters ship with their milestones (design → prototype evidence → frozen chapter). PNP and CHNS bricks slot after M2 as pure L3 additions; elasticity/shells prioritized by student need. Stretch (post-M6, from FASTEST): THB/C¹ bases for primal Cahn–Hilliard; mixed p1/p2 meshes in production.

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Warp ecosystem gaps (no native AMG; API churn) | AMGX via pyAMGX; solver interfaces isolate churn; kernels are 1:1-translatable so a forced CUDA retreat loses little |
| Gradient noise at epoch boundaries | linear transfers + frozen-surrogate trust region + Leray correction; Tier-AD epoch-crossing test guards regressions |
| Mixed-precision divergence on stiff PNP regimes | MF3 selector with FP64 fallback; cuDSS direct fallback; conservation sentinels |
| Splitting error (projection stepper) | monolithic stepper behind same interface; head-to-head in every physics acceptance test |
| Tape memory on long rollouts | per-step checkpointing baseline; Revolve; checkpoint-precision policy |
| Neural-geometry inadequacy (thin features, corners) | admissibility diagnostics + plateau detector; STL/CSG backends as fallback on the same oracle contract |
| AM activation non-smoothness in gradients | activation times outside the differentiated segment (epoch contract); relaxed activation documented as a research extension |

---

## Appendix A: Source basis

Design synthesized from the group's papers (octree-SBM NS; SBM thermal; INR-SBM flow; Neural-SBM elasticity; neural-geometry theory; IMGA LES; SC'21 incomplete octrees; IPDPS Proteus; CPC/JCP CHNS; NS-PNP JCP + weak-BC CMAME; conservative AMR; neural viscosity closure; thin shells; pressure-robust transfer (in progress); linearized-VMS exact adjoint; Helmholtz–Leray projection VMS; FEAD AM thermal), the cuFEM verification plan, the FASTEST program document, and the Proteus/Dendrite-KT/talylite codebases (nomenclature mined from source).
