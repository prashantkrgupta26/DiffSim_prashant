# DiffSim

**GPU-native differentiable finite-element multiphysics on adaptive octrees.**

DiffSim solves PDEs on complex, moving, *optimizable* geometries — and computes
exact gradients of the results with respect to the geometry, material
parameters, and (eventually) boundary data and closures. It is built by the
Baskar research group as the successor generation to our production C++ stack
(TalyFEM / Dendrite / DendrIon), rethought from scratch for three things at
once:

1. **GPU-first execution** — production runs live on the GPU; host↔device
   traffic is reserved for output and checkpointing.
2. **End-to-end differentiability** — every solve is a node in a gradient
   graph, verified to machine precision, so shape optimization, inverse
   problems, and learned closures are first-class citizens rather than
   afterthoughts.
3. **Pedagogical transparency** — the code is written to be *read*. Every
   numerical decision is documented where it lives, every claimed property has
   a test that measures it, and the `tutorials/` ladder takes a student from
   "solve Poisson on a disk" to "recover a shape from sensor data" in an
   afternoon.

---

## The vision

Classical simulation answers *"what happens?"*. Differentiable simulation
answers *"what should I change?"* — it turns a solver into an instrument for
design, inference, and discovery:

- **Shape optimization**: gradients with respect to geometry parameters flow
  through the mesh-generation → assembly → solve → observable chain
  (demonstrated today: recovering a circle's center and radius from nine probe
  readings, `tutorials/E_differentiable/E1_shape_optimization.py`).
- **Inverse problems**: unknown conductivities, viscosities, boundary data.
- **Hybrid ML–physics**: neural SDF geometries and neural constitutive
  closures dropped into the weak form, trained end-to-end against simulation
  outputs.

The core design bet is that **immersed methods + octrees + adjoints** are the
right substrate for this: geometry changes never remesh a body-fitted grid
(there isn't one), so gradients never differentiate through remeshing.

## How it works — the five pillars

### 1. Adaptive octree meshes (dimension-independent)

The domain is always the unit cube `[0,1]^k` for `k = 2, 3, 4` — the 4D case
exists because space-time formulations are on the roadmap, and every piece of
machinery is tested at k = 2, 3, 4 from day one (the *dim-coverage policy*).
Elements are Morton-indexed octree leaves with 2:1 balance; hanging nodes and
p-transitions are handled by a constraint operator `T` (free DOFs → full
DOFs), so every solver sees an unconstrained square system
`T^T A T u_free = T^T b`.

### 2. Geometry as signed-distance oracles

A geometry is anything that answers ψ(x) (negative inside the shape) and can
be queried for closest points. Backends today: **analytic CSG** (spheres,
boxes, rotations, smooth blends), **triangle meshes / STL** (warp BVH for
candidates, FP64 torch refinement — differentiable in vertex positions), and
**voxel grids** (differentiable in the voxel values = level-set optimization).
Neural SDFs join in M1c. Closest points are computed by Newton on the
augmented system `F(y, s) = [y − x + s∇ψ; ψ] = 0` — the *same* system the
implicit-function-theorem backward pass differentiates, so forward and
backward are consistent by construction.

### 3. The Shifted Boundary Method (SBM)

We never cut elements. The mesh keeps only elements sufficiently inside the
domain (the **λ-criterion**, with production semantics: retain an intercepted
element iff its inactive volume fraction ≤ λ), and the boundary conditions are
imposed on the resulting *surrogate boundary* Γ̃ — a staircase of whole
element faces. Accuracy is recovered by a Taylor **shift**: for each surrogate
face quadrature point, the distance vector `d` to the true boundary maps data
and (to second order) fields:

    S u = u + ∇u·d + ½ dᵀ H(u) d        (the ½ dᵀHd term only at p = 2 —
                                          "shift order = basis order, never
                                          above it", a measured rule)

Dirichlet conditions enter through a Nitsche form on shifted test functions;
Neumann conditions through the flux form with the area correction
`a = n·ñ` that maps staircase area to true-boundary area (a 4/π effect in
2D — unambiguously visible in our convergence-vs-stall regression test).

### 4. GPU execution: NVIDIA Warp kernels + PyTorch geometry

- **Warp** generates the hot kernels (element matrices, face terms, residuals,
  matrix-free operators) — FP64 throughout, one kernel module per variant so
  nothing recompiles when something unrelated changes.
- **PyTorch** owns the differentiable geometry pipeline (oracles, closest
  points, boundary data) in FP64 on the host at prototype scale.
- The **solver layer** is single-sync by design: our fused conjugate-gradient
  and BiCGStab keep every per-iteration scalar on the device and read back one
  number per convergence check (measured 4.6× per-iteration vs the naive
  host-sync loop at 200k DOFs — and the gap grows with queue depth).

### 5. Differentiability by architecture, not by taping everything

A simulation is a sequence of **epochs**. Within an epoch the mesh topology
and element classification are *frozen* (they are piecewise-constant in the
parameters anyway — we assert this in tests), so the solution is a smooth
implicit function of the parameters. Gradients then flow through three
**Tier-2 vector–Jacobian products** stitched together:

    adjoint solve:      A^T λ = ∂J/∂u
    parameter sweep:    ∂J/∂θ = −λ^T ∂R/∂θ     (a Warp tape over pure
                                                 face-residual kernels)
    geometry chain:     ∂R/∂θ reaches θ through {d, n, ḡ, q̄} — a torch
                        graph over the closest-point projection (IFT)

Krylov iterations, preconditioners, and direct solves are *routed around*
(adjoint = one transposed solve), never differentiated through.

**Verification is three-way**: the custom adjoint, a dense PyTorch "twin" of
the whole pipeline (autograd end-to-end, the readable reference
implementation), and central finite differences must agree — adjoint↔twin to
1e-8, both↔FD to 1e-6. This gate has already caught a real compiler bug: warp
1.14's adjoint of loop-reassigned locals scales gradients by an exact,
innocent-looking constant factor (32.000× in our case). No single-method
check would have survived that.

---

## What works today (all numbers are measured, locked in tests)

| Capability | Evidence |
|---|---|
| SBM Poisson, Dirichlet + Neumann, k = 2,3,4 | patch tests to machine precision; MMS orders 2 (p1) and 3 (p2); Neumann order 2 with the node-adjacent p2 band |
| Spatially-varying conductivity | MMS order 2.02 |
| Three geometry backends | cross-backend suite: same solve through CSG / STL / voxels |
| Shape + parameter gradients | three-way gate green; voxel- and vertex-level gradients vs FD |
| Shape inversion capstone | circle recovered from 9 probes, J ↓ 1000× |
| Incompressible Navier–Stokes (VMS-stabilized, equal-order) | steady Stokes/Oseen order 2; transient order 2 (BDF2) |
| Two time steppers | monolithic linearized *and* Leray pressure-projection — both validated on the same benchmarks (a standing discipline) |
| Energy stability | the s = ½ skew advection is exactly skew-symmetric discretely; kinetic energy monotone on an under-resolved decay |
| **Lid-driven cavity, Re = 100** | centerline profiles vs Ghia et al. (1982): monolithic within 0.06, projection within **0.0035** |
| **Immersed cylinder, Re = 20** | C_d = 2.847 (confined band), C_l = −3e-5 (machine-symmetric wake), forces via shifted-traction extraction |
| Single-sync device Krylov | sync count asserted exactly; 4.6×/iteration at 200k DOFs |
| Locked numerical baselines | `tests/baselines/*.json`, rtol 1e-6 regression |

Suite: ~250 tests. Every claimed property above is an assertion somewhere.

## The user-facing equation API

Physics is authored in the **Integrands style** familiar from
TalyFEM/Dendrite/DiffPack: you write what happens at one integration point;
the framework owns the element loop, quadrature, assembly, constraints,
solvers, and gradients.

```python
class PoissonBrick(CEquation):
    ndof = 1

    @staticmethod
    @wp.func
    def Integrands_Ae(fe, Ntab, dNtab, detJxW, dscale, nbf, dim, ndof, Ae, e):
        for a in range(nbf):
            for b in range(nbf):
                K = wp.float64(0.0)
                for k in range(dim):
                    K += fe_dN_s(dNtab, fe, a, k, dscale) \
                         * fe_dN_s(dNtab, fe, b, k, dscale)
                Ae[e, ndof * a, ndof * b] += K * detJxW
```

The contract ("the lego gate"): this brick, fed through the generic factory,
reproduces the hand-written production path bit-for-bit and the locked
regression baselines. If you can write the weak form, you can extend DiffSim.

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # warp-lang, torch, numpy, scipy
pytest -q -m "not tier5"    # fast tiers (~minutes)
pytest -q                   # everything (~1 h; first-ever run pays one-time
                            # kernel compiles, cached on disk afterwards)
python tutorials/A_foundations/A1_mms_convergence.py
```

Requires an NVIDIA GPU (FP64-capable), CUDA 12+, Python 3.12.
Note: the 3D Navier–Stokes element kernel is a ~1 h one-time compile; the
test that needs it is opt-in via `DIFFSIM_RUN_3D_NS=1`.

## Repository map

    src/diffsim/
      octree/      Morton keys, balanced refinement, leaf lookup
      mesh/        nodes, hanging/p-transition constraints, basis & face tables,
                   point evaluation
      geometry/    SDF oracles: CSG, STL/trimesh, voxel grids; Newton
                   closest-point projection with IFT backward
      sbm/         surrogate extraction, λ-classification, SBM Dirichlet/Neumann
                   (scalar + vector), taped adjoint kernels, dense torch twin
      assembly/    element kernels, CSR & matrix-free operators, FEMElm accessors
      api/         the Integrands brick layer (CEquation) + NS bricks
      physics/     VMS stabilization (τ_M/τ_C, M_{a,s}), Poisson helpers
      solvers/     single-sync device Krylov, BDF time integration
      steppers/    monolithic linearized NS, Leray pressure projection
    tutorials/     the pedagogical ladder (start here)
    benchmarks/    literature-anchored validation drivers
    tests/         the ~250-test suite; baselines/ holds locked values
    docs/superpowers/
      specs/       the design document (the source of truth)
      plans/       per-milestone implementation plans with progress ledgers
      *findings*.md  measured findings logs — read these; they are the lab
                     notebook (compiler bugs, convergence traps, and the
                     test-design lessons we paid for)

## Roadmap (checkmarks = gated + committed)

- ✅ **M0 — octree foundations**: SFC build, 2:1 balance, node dedup,
  hanging constraints, k-generic kernels (k = 2,3,4).
- ✅ **M0.5 — k-generic refactor**: per-axis periodic topology, mixed
  p1/p2 constraints (one-knob rule), machine-precision patch/trace/fuzz
  batteries.
- ✅ **M1a — SBM foundations**: surrogate boundary + Taylor shift,
  Nitsche Dirichlet, Eq.-21 Neumann w/ area correction (π/4 locks),
  P4 rotated-patch keystone (k = 2/3/4), the p2 Neumann band
  (node-band(3): 2-D orders 1.9–2.1; **3-D asymptotic 2nd order to L7
  on Nova — 1.97/1.94/2.03**; dyadic-radius halo rule, canonical
  r = 0.19).
- ✅ **M1b — incompressible NS**: VMS-stabilized p1, both steppers
  (Leray + monolithic), BDF1/2; cavity Re 100/1000, cylinder Re 20
  (C_d = 1.352) + Re 100 Strouhal, sphere Re 300; GPU solves (fused
  Krylov, cuDSS, AMGX).
- ✅ **M1c — adjoints + neural-SDF geometry**: taped kernels
  (findings-4c rules), transient chains (1e-8-class), Leray adjoint,
  shape + constitutive gradients; PROVIDED INR checkpoints (GENIE edit
  modes, window contract, projection hardening); heroes: **sphere
  steady 1.8e-3 ✅, sphere transient 2.6e-4 ✅**.
- ✅ **M1d — device migration** (GH200 capacity probe pending):
  cuDSS 300× over host splu; device CSR assembly 4–40× (uniform +
  hanging; strong rows folded; ndof-generic); zero-copy
  assemble→solve; steppers `use_device_assembly=True` (2.2× end-to-end,
  solver-bound); cuDSS mtlayer (~40× on refactorization loops).
  **FORMALLY CLOSED 2026-07-08** (m1d-milestone-report.md): device
  GP-field kernels (last per-step host compute migrated); D3 trace —
  per-step host work 0.81% / 0.04% / 0.03% of step at 2-D L8 / 3-D L5 /
  3-D L6 (< 5% bar); H1 hero epoch 44.3× (27 s vs the M1c-era
  ~20 min). Verdict: pure-device is THE profile; coherent =
  capacity-only (measured: 3-D L6 direct factorization exceeds 48 GB —
  fused device Krylov carries it).
- ✅ **M2 — Heat/Mass + closures + p2-NS**: scalar brick (orders
  2.00/3.00 EXACT, VMS-complete residual), coupler (de Vahl Davis Nu
  **0.05% / 0.02%**), SBM-thermal composition (Péclet-aware Nitsche,
  consistent-flux extraction), closures-in-the-loop (S2 retrain demo,
  RMSE 0.049), pure-p2 framework (cylinder C_d = 1.334; **unified
  penalty law α ~ Pe × p²**).
- ✅ **M3 rungs 1–2 — differentiable adaptivity**: explicit transfer
  operators (adjoint EXACT across a re-carve), epoch continuation
  (exact at 2.7× trust region where plain GN stalls); **THE BUNNY
  HEADLINE: cell-scale GENIE ear-edit recovered from transient flow,
  err 1.23e-3** (six-run mechanism ladder). Rung 3 (relaxed
  classification) = research brief `docs/n6_relaxed_classification_brief.md`.
- 🔶 **M4 — differentiable phase-field & learned thermodynamics**
  (in progress): AC + CH bricks ✅ (orders exact; mass 1.9e-15),
  spatial adaptivity ✅ (zero-drift nested transfer), temporal
  adaptivity ✅ (178× Δt growth; evaporation dt-cap), ternary
  Onsager CH ✅, **Wodo CMS-2012 evaporating film: Fig-3 replicated
  first-run** (Landau frame; solute conserved 4e-16) ✅; F-series
  tutorial + LaTeX course doc ✅. Remaining: 2-D figure campaign
  (local sweep + Nova kit), device-bound film marches, the learning
  arc (basis/MLP f_mix — the milestone's title).
- **Forward**: PNP electrokinetics (production DendrIon physics),
  CH block preconditioner (full-res 3-D films), space-time (k = 4),
  multi-GPU, THB.

## Development culture

Three rules we actually follow, learned the hard way (see the findings logs):

1. **Measure, then lock.** No tolerance enters a test unless a diagnostic
   printed the number first. Regression locks are measured values, not hopes.
2. **Assert the mechanism is alive.** A test whose target quantity is
   secretly zero (zero-net-flux data, linear patch fields with zero shape
   sensitivity) proves nothing. We assert the observable is nonzero *before*
   asserting it is right.
3. **Verify gradients three ways.** Adjoint, autograd twin, finite
   differences. Two of the three have independently been wrong this month.

---

*DiffSim is research software from the Baskar group (Iowa State University /
private repository). Production heritage: TalyFEM, Dendrite, DendrIon,
proteus. Contact: baskigs@gmail.com.*
