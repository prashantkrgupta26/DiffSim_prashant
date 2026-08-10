# Differentiable Extrusion-Based AM (3D Concrete Printing) — Program Roadmap

**Date:** 2026-08-09
**Status:** Approved roadmap (program-level decomposition; each sub-project gets its own spec → plan → build cycle)
**Scope of this doc:** sub-project boundaries, order, interfaces, and cross-cutting standards for the AM simulation program. It is *not* the design of any single sub-project.

## 1. Purpose and drivers

Build a differentiable, GPU-resident simulator for extrusion-based additive manufacturing on the DiffSim stack, concrete (3DCP) first, polymer FFF as a later branch. Drivers:

- **Grant (Sriram/UA 2-pager):** Aim 1 — validate the differentiable 3DCP framework against printing experiments (adjoint calibration of rheology); Aim 2 — Pareto maps over mix rheology × process parameters (buildability × printability × speed); Aim 3 — effective R-value of printed wall webbing designs.
- **Research foundation:** "Governing Equations for High-Fidelity Extrusion-Based AM Simulation in DiffSim" (in `local_code_old/AM/`, PDF + markdown) — defines the coupled system (variable-property NS+VMS, phase-field interface, scalar transport, internal-state rheology), the prescribed-deposition-source nozzle model, and the M-print-1→6 milestone ladder with quantitative benchmarks. This roadmap adopts that ladder, restructured into sub-projects.
- **Funded/committed posture:** ordered by technical dependency and validation rigor; no demo-deadline shortcuts.

## 2. Decisions of record

| Question | Decision |
|---|---|
| Material priority | Concrete (3DCP) on the critical path; polymer FFF branches later off the shared core |
| Code home | Hybrid: physics bricks in DiffSim; application in a new separate package `diffsim-3dprint` |
| Differentiability cadence | Adjoint ships with every milestone; three-way gradient gate (adjoint/FD/twin) per rung |
| Dimensionality | Develop + gradient-gate in 2-D; promote to 3-D on GPU for each rung's quantitative benchmark |
| R-value thread (Aim 3) | After the deposition ladder, so as-printed geometry (with deformation/defects) feeds the thermal model |
| Visualization | First-class citizen: every rung's exit artifact includes rendered movie + quantitative plot, produced by a shared viz module |
| Legacy C++ codes | Reference and regression anchors only; never a dependency, never ported wholesale |

## 3. Legacy asset inventory (surveyed 2026-08-09)

Sources in `local_code_old/AM/` (Bitbucket zips, extracted and surveyed):

- **`chns_nonnewtonian` (≡ `proteus`, byte-identical physics):** Dendrite-KT projection-based CHNS (Khanwale lineage): VMS/SUPG, Korteweg surface tension, BDF1/BDF2/θ, polynomial + Flory–Huggins free energies, interface-band octree AMR, ~10.3k LOC. Validated configs: dam break 2-D/3-D, jet atomization, RT instability, bubble rise, Couette, MMS. **Contains no non-Newtonian rheology** (pure linear φ-mixture viscosity), no deposition machinery, no thermal coupling, no contact angle. The name refers to its `.gitmodules` pin of a `suresh-dendrite-kt` branch of the dendrite-kt submodule, which is **not in the zip** (submodules aren't bundled). If that branch is obtained, survey it; until then Suresh's rheology work is paper-only (Neural Constitutive Equations PDF). Value: validation configs + MMS cases + projection-scheme reference. DiffSim already replicates this physics lineage.
- **`admanufacturing` (ADM, ~2.7k LOC):** voxel-activation transient heat conduction with convection+radiation BCs on tracked external surfaces; per-voxel diffusivity switching on activation; `.ctr`/`.csv` toolpath schedule formats (layer/contour/interior voxel sequences, N timesteps per voxel); ABS/PEKK material configs; octree refine-around-active-voxel and coarsen-behind; VTU output + checkpointing. Value: **toolpath/schedule format to adopt**, deposition-scheduling reference, thermal-rung reference, dynamic-adaptivity pattern.
- **Paraview pipeline** (`config/Paraview/` in chns): PVD generation, φ=0 isocontour time series and video, slice extraction, energy `.plt` logs. Seed for the viz module.

Greenfield (in no legacy code): deposition source/moving nozzle, yield-stress + thixotropic rheology, contact angle/wall energy, thermal–flow–hydration coupling, differentiability.

## 4. Program architecture

### 4.1 Two homes, one boundary

- **DiffSim (`src/diffsim/`):** physics bricks that must compose with NS+VMS, CH, SBM, and the adjoint machinery — deposition source term, `RheologyClosure` protocol + closures (HB/Papanastasiou, thixotropy), conservative Allen–Cahn (if adopted in SP-1), thermal–hydration coupling. Tested in DiffSim's suite.
- **`diffsim-3dprint`** (new repo, Python package `diffsim_3dprint`; daisy-morph pattern — imports DiffSim as engine): toolpaths and print schedules (adopting ADM's `.ctr`/`.csv` formats), case runner and benchmark definitions, the viz module, sweep/calibration orchestration, result artifacts.
- **Boundary test:** "would a non-AM DiffSim user want it?" → DiffSim; else → app.

### 4.2 Rheology as a protocol

From SP-1, the momentum brick consumes a `RheologyClosure` protocol — η_eff from frame-invariant inputs (shear rate γ̇, structure parameter λ, temperature T, cure α), tensor-basis form per the research doc — following the `CrystalEnergy` protocol pattern from crystallization. Newtonian is the trivial instance; HB, thixotropy, and eventually neural closures slot in without touching the solver. This is what makes "concrete first, polymer later" cheap.

### 4.3 Formulation notes (binding on SP-1+)

- **Air is a real, resolved phase.** One momentum equation spans both domains: φ=+1 concrete, φ=−1 air, with ρ(φ), η(φ) interpolated across the diffuse interface. Bead spreading, layer coalescence, and trapped air cavities emerge without interface tracking.
- **"Numerical air":** true contrasts (ρ ~2000:1, η ~10⁷:1) are capped — air density/viscosity raised so ratios sit near ~10²–10³ — keeping air dynamically passive but the linear systems solvable (standard in the Comminal lineage). The cap is a numerical parameter, checked for insensitivity, not a physics claim.
- **Density treatment:** quasi-incompressible variable density (not Boussinesq, which fails at these ratios); Abels–Garcke–Grün mass-flux term if the interface model stays Cahn–Hilliard.
- **Mass bookkeeping with a source:** deposition injects φ inside the domain; phase, mass, and momentum sources must be paired consistently. The conservation diagnostic is "mass matches ∫source dt," not "mass constant." The CAC-vs-CH decision (SP-1) must account for this: CAC's Lagrange-multiplier conservation term needs modification to respect the source.
- **Substrate:** wall BC (SBM/no-slip; wall energy for contact angle when needed), not a third phase. Ternary Onsager–CH is available in DiffSim if substrate interaction ever needs to be thermodynamic.

### 4.4 Meshing strategy (octrees)

The toolpath is prescribed, so the region needing resolution is known *a priori* — an advantage neither legacy code exploited fully. Three-stage strategy:

1. **SP-1/SP-2 — static graded octree:** pre-refine a corridor around the known toolpath + substrate (bead-scale fine, interface ε ≈ 1–1.5·h_min; coarse far-field air), fixed for the run. Most of adaptivity's win at zero differentiability cost: fixed DOF set, checkpointed adjoint unchanged. SP-1 design-time check: exercise DiffSim's CH/CHNS bricks on a *non-uniform* octree (all phase-field runs to date used uniform grids; static graded refinement exists in the truck-bringup nested-banding work).
2. **SP-3+ — interval-based dynamic adaptivity** (ADM-style refine-ahead/coarsen-behind), introduced only if the static corridor is the measured bottleneck (likely for tall walls, e.g. 46-layer Wolfs/Suiker prints). Remesh at layer boundaries (natural checkpoints), conservative field transfer; adjoint handles remesh events by transposing the prolongation/restriction operators between intervals. The mesh sequence is treated as fixed — correct, since it derives from the prescribed toolpath, not the solution.
3. **Never** solution-adaptive error estimators in the differentiable path: solution-dependent refinement makes the mesh a discontinuous function of parameters and poisons gradients.

### 4.5 Cross-cutting standards (every sub-project)

- **Differentiability gate:** forward + adjoint + three-way verification (adjoint/FD/twin) before a rung is done — the crystallization discipline.
- **2-D develop / 3-D validate:** physics + gradient gates in 2-D (Mac/gpubox-cheap); each rung's quantitative benchmark runs 3-D on GPU. Large 3-D: `blockch_dev` + device assembly (proven daisy path); Nova A100-80/H200 beyond 48 GB.
- **Visualization first-class:** rung exit artifacts include VTU/VTI + PVD time series, interface isosurface/contour movie, and the rung's quantitative overlay plot (bead cross-section vs Comminal, slump vs analytic, wall-failure height vs Wolfs/Suiker). The viz module is built in SP-1 (seeded from legacy Paraview scripts + daisy's `.vti` writer) and extended per rung — movies are one function call, not per-run scripting.
- **Legacy as reference:** chns configs → regression anchors; ADM → toolpath format + thermal reference; C++ never becomes a dependency.
- **Governance:** Mac is git source of truth; gpubox/Nova are compute-only (see `~/Dropbox/work/Projects/ClaudeCode/GPU-COMPUTE-SETUP.md`).

## 5. Sub-project ladder

Dependencies are linear SP-1→2→3→4→5; SP-6 can start after SP-2 (calibration needs rheology gradients, not thermal).

### SP-1 · Deposition core (≙ M-print-1)
The keystone rung: isothermal two-phase **Newtonian** deposition with a moving volumetric source following a toolpath.
- **DiffSim brick:** smoothed deposition source — phase + mass + momentum injection tracking nozzle position, differentiable w.r.t. print speed, extrusion rate, standoff, layer height. Interface-model decision (conservative Allen–Cahn vs existing CH) made here via a 2-D spike, honoring the source-term mass bookkeeping (§4.3). Non-uniform-octree CH check (§4.4).
- **App scaffold:** `diffsim-3dprint` repo — toolpath/schedule ingest (ADM `.ctr`/`.csv`), case runner, viz module (§4.5).
- **Validation:** single strand on a plate; bead cross-section (W/D, H/D) vs Comminal/Serdeczny across layer-height/nozzle-diameter and nozzle-velocity/extrusion-velocity ratios; round/spread regimes. Target ~10 %; tripwire: >15 % miss → revisit ε/mobility/source calibration before proceeding.
- **Exit gate:** 3-way-verified gradients of bead shape w.r.t. process parameters (2-D); 3-D strand validated on GPU; strand movie + cross-section overlay plot shipped; legacy regression anchors (dam break, jet) passing on the DiffSim side.

### SP-2 · Yield-stress rheology (≙ M-print-2)
- **Scope:** Herschel–Bulkley/Bingham + Papanastasiou regularization behind `RheologyClosure`.
- **Validation:** slump / slump-flow ↔ yield-stress inversion (Roussel–Coussot, τ₀ = ρVg/(√3·πR²)); deposited-bead shape retention.
- **Exit gate:** gradients w.r.t. (τ₀, K, n) verified; calibration demo — recover τ₀ from a synthetic slump observation.

### SP-3 · Thixotropy + multi-layer buildability (≙ M-print-3)
- **Scope:** Roussel λ-kinetics + A_thix structuration; multi-layer stacking with layer cycle times; dynamic adaptivity if triggered (§4.4).
- **Validation:** Wolfs/Suiker wall failure — failure after 21/27/46 layers (199.5/256.5/437 mm; 3.5/21.6/76.5 min prints), target ±2 layers; plastic-collapse vs elastic-buckling competition.
- **Exit gate:** buildability gradients verified; layer-stack deformation movie; failure-height benchmark plot.

### SP-4 · Thermal + hydration (≙ M-print-4)
- **Scope:** energy equation + hydration kinetics (Cervera/Ulm–Coussy) with latent heat; hydration → yield-stress feedback. ADM as thermal reference; de Vahl Davis-validated thermal-flow coupling reused.
- **Validation:** temperature history vs calorimetry-style references; hydration-coupled strength gain.
- **Exit gate:** thermal-coupled adjoint verified; temperature-field visualization added to viz module.

### SP-5 · As-printed thermal R-value (grant Aim 3)
- **Scope:** 3-D thermal analysis of double-wythe wall sections using **as-printed geometry from SP-3/4 outputs**; cavity convection (correlations, direct CFD where warranted); parametric study over webbing spacing/thickness/pattern.
- **Deliverable:** ranked designs by R-value per unit material; three extreme webbing candidates for experimental validation.

### SP-6 · Calibration + Pareto tooling (grant Aims 1–2; M-print-6 core)
- **Scope:** adjoint calibration workflows against experimental bead/layer data (yield stress, plastic viscosity, structuration rate); sweep orchestration; Pareto-front construction over buildability × printability × production speed.
- **Startable:** after SP-2 (synthetic calibration), production-ready after SP-3.

### Later branches (off the concrete critical path)
- **Polymer FFF:** viscoelastic log-conformation (Fattal–Kupferman), WLF/Arrhenius + Castro–Macosko cure viscosity — new `RheologyClosure` instances; thermal rung shared with SP-4; ADM ABS/PEKK configs as references.
- **Drying/moisture (M-print-5):** Bažant–Najjar transport, evaporation as interface flux, shrinkage eigenstrains; Wodo film benchmark.
- **Neural closures in-the-loop (M-print-6 full):** neural `RheologyClosure` with thermodynamic-consistency constraints; toolpath optimization demos.

## 6. Next step

Brainstorm → spec → plan SP-1 (deposition core) as its own sub-project, starting with the two SP-1-internal design decisions flagged here: CAC-vs-CH interface model under a mass source, and CH-on-nonuniform-octree verification.
