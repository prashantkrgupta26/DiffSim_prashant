# P2-R2 — NS-SBM forward maturation: stable, scalable, literature-validated

**Design spec.** Second built rung of milestone P2
(`docs/dev/specs/2026-07-21-p2-nsht-sbm-shell-milestone.md`), following R0
(`01546f8`). Brainstormed + ratified 2026-07-21. **Reorders ahead of the
co-dim-1 shell** (the umbrella's R1): the shell must build on a *stable,
validated* volumetric NS-SBM stepper, not the de-risk one.

## 1. Goal

Mature the VMS-projection + volumetric-SBM forward stepper (delivered at the
de-risk bar in R0) into one that is **stable in 3-D**, **scalable on a single
GPU to validation-scale meshes**, and **converged to the literature** on the
cylinder/sphere/slender benchmarks. This is the forward engine the shell,
the heroes, and the steady-adjoint (R3) all build on.

**Not in R2** (scope fences): the steady-adjoint / differentiable design
(deliverable C → R3); the co-dim-1 two-sided shell (after R2); HT/Boussinesq;
multi-GPU / 100M-DOF (S4 / hero rungs). R2's success = *a 3-D-stable,
single-GPU-scalable, literature-validated volumetric NS-SBM projection
stepper.*

## 2. Why R2 exists — the R0 findings it resolves

R0 verified the discretization (MMS 2.01/2.07, BDF2 1.996, SBM consistency
1e-12) and 2-D faithfulness to the monolithic solver, but left three open
items (documented in R0's scaling declaration, not hidden):
1. **3-D projection instability** — the split's drag diverges in 3-D across
   every setting probed, *even at Stokes (Re=1)* → localized to the
   projection **pressure coupling** in 3-D, not advection. Monolithic 3-D is
   stable (Cd=0.381).
2. **Literature Cd/Strouhal unreachable** — the unit-box octree can't resolve
   paper-[A]'s 5%-blockage cylinder at feasible levels; coarse meshes carry
   leak-drag (blockage-dependent) that masks the true drag; steady Cd needs a
   true BDF2 transient march, not a BDF1 pseudo-steady fixed point.
3. **Host/splu only** — the device port (SPD-PPE → AMG) is the scalability
   lever, unbuilt in R0.

## 3. Sub-phases (dependency order)

### R2a — 3-D projection stability (the gate)
Nothing 3-D works via projection until this is fixed. Approach (Baskar,
2026-07-21): **diagnostic first, then the penalty law** — the discipline
that resolved the 2-D "divergence" as a wrong-metric non-issue.
- **Diagnostic:** isolate *why* the 3-D pressure coupling diverges at Stokes.
  Candidate mechanisms to test: the pressure-Poisson operator conditioning at
  the immersed boundary in 3-D; the surrogate-consistent (homogeneous-Neumann)
  PPE BC's 3-D consistency; the pressure null-space / outflow-boundary
  handling; a missing/mis-scaled term specific to 3-D. Produce a committed
  measurement (the three-number style of the 2-D diagnostic) that names the
  mechanism.
- **Targeted fix — the penalty law α~Pe·p²:** implement the principled
  Nitsche-penalty scaling (Péclet × polynomial-order / h) replacing the fixed-
  α stable window R0-Task-5 found narrow, so no-penetration AND the traction
  observable stay consistent as h→0. This is the leading fix; the diagnostic
  confirms it is sufficient or points elsewhere.
- **Gate:** the projection stepper marches a stable, physical 3-D sphere whose
  Cd MATCHES the 3-D monolithic reference on the same mesh (the R0
  faithfulness bar, now met in 3-D), weak-div machine-zero, BDF2 engaged.
- **Fallback (documented):** if the diagnostic shows the 3-D projection
  instability is fundamental, the monolithic-block-preconditioner path
  (`solvers/block_precond.py`, already AMGX-backed) is the stable 3-D solver;
  R2 would then validate via monolithic and note the projection limitation.
- **Prerequisite note:** if the diagnostic points to the implicit fine-scale
  PPE, the latent dead-branch bug (`ppe_finescale=True` computes τ_m with
  `dt=Δt/b0`, over-scaling by b0² for BDF2 — filed in the R0 parity audit)
  MUST be fixed first.

### R2b — device port (scalability to validation meshes)
Port the projection march onto a single GPU, enough to reach the L8–L9
validation meshes (a few million cells) — NOT the 100M multi-GPU hero (S4).
- **SPD pressure-Poisson → AMGX** — the scalable win of the projection choice
  (the reason it was chosen over the monolithic saddle). Reuse the
  `solvers/block_precond.py::_AMGXCycle` pattern (AMGX AMG as the elliptic
  solver/preconditioner).
- **Predictor (nonsymmetric) → device Krylov** — the #49 device-FGMRES
  pattern (`solvers/fgmres_dev.py`) / a device BiCGStab.
- **Device assembly** — reuse the `assembly/device_assembly.py` machinery;
  mixed-width CSR (#33) + ChunkedCSR (#38) auto-apply at the larger meshes.
- **Gate:** the device projection march reproduces the host/splu result to
  few-ULP (parity), and reaches an L8–L9 mesh at a reported s/step; the SPD-PPE
  AMGX solve scales (iteration count ~mesh-independent). Default host path
  preserved (opt-in device, per house pattern).

### R2c — literature validation
With R2a (stable 3-D) + R2b (finer meshes) in hand, satisfy the Task-9/10
requirements and converge to the literature:
- **Cylinder:** Cd → 1.352 (Re20) and Strouhal → literature (Re100), via a
  properly-sized domain matching paper-[A]'s ~5% blockage + a true BDF2
  transient developed march. Mesh-convergence table showing monotone
  approach.
- **Sphere:** Cd → the Re300 reference (3-D, converged mesh).
- **3-D slender object:** a thin/slender volumetric body (the single-object
  precursor to the shell/canopy) marched to a physical, converged state.
- **Gate:** monotone mesh-convergence to each reference within a stated
  tolerance (not a single-mesh number; the R0 lesson — coarse leak-drag can
  cancel errors).

## 4. Reference anchors

The group papers (`local_code_old/`): the **VMS-projection** paper (the
stepper), the **SBM-NS-octree** paper [A] `S0021999125006163` (the cylinder/
sphere octree-SBM validation values + the 5%-blockage benchmark — the primary
literature target), the **solution-transfer** paper (BDF2 transient, penalty),
**Dokken** `1912.06392` (cut-mesh projection + penalty). The in-repo M1b
cylinder/sphere Cd + the R0 monolithic-same-mesh references.

## 5. Scaling-pathway declaration (mandatory)

- **Stage residency (R2b target):** octree + surrogate extraction + distance —
  host, per-epoch (static geometry). Predictor assembly + device Krylov,
  SPD-PPE + AMGX, correction — **device** (single GPU). Observables — device.
- **Budget:** R2 targets L8–L9 (a few million cells) on ONE GPU — well within
  HBM, no ChunkedCSR needed yet (auto-applies if a mesh crosses 2³¹ nnz). The
  100M multi-GPU hero (ChunkedCSR + fp32-IR + multi-GPU S4 + the #49 device
  outer) is R3/hero, NOT R2.
- **Tiers:** single big-node GPU (gpubox Ada / GH200) for R2's device
  validation; multi-node deferred.

## 6. Gates (summary)

| Sub-phase | Gate |
|---|---|
| R2a | 3-D sphere: projection Cd matches monolithic same-mesh (faithfulness in 3-D), weak-div machine-zero, BDF2; diagnostic names the mechanism; penalty-law α~Pe·p² non-vacuity (mutation) |
| R2b | device march = host parity (few-ULP); L8–L9 reached, s/step reported; AMGX-PPE mesh-independent iteration count |
| R2c | cylinder Cd→1.352/Strouhal→ref, sphere Re300, slender — MONOTONE mesh-convergence to each reference within tolerance |

Standing contracts throughout: MMS ±0.10, gate-hygiene (independent
reference + mutation/planted-break, no vacuous gates — the R1/R0 lesson),
honest measured perf, no baseline corruption.

## 7. Process & compute placement

Subagent-driven, spec→plan→gated-subphases, strong tier (Opus) for the
stability/physics work, mid tier for mechanical. REPO-IDENTITY GUARD +
worktree isolation for any concurrent agents. Compute: gpubox (device port +
finer-mesh validation on the 40-core CPU / GPU) + Nova GH200 for the largest
validation meshes if needed (≤2–3 GPUs). The C++/Dendrite reference + the
group papers are the parity/validation oracles.

## 8. What R2 unblocks

A stable, scalable, literature-validated volumetric NS-SBM forward stepper —
the foundation the **shell** (next), the **heroes** (B1 truck / B2 maize),
and the **steady-adjoint** (R3, deliverable C — the differentiable maize-
ideotype design engine) all require.

---

## ADDENDUM 2026-07-22 — R2a outcome & re-scope to the monolithic 3-D engine

R2a (3-D stability) is closed with a **pivot**, recorded in
`docs/dev/2026-07-22-p2-r2a-projection-3d-findings.md`. Summary:

- The full diagnostic chain proved the 3-D failure is the **projection split**,
  not the SBM. **Control:** the monolithic reference uses the *identical*
  `sbm_vector_dirichlet` block and is correct (`Cd=40.04` at Re=1, `+0.96` at
  Re=100 on level-4). The split converges to a weak, wrong fixed point
  (lagged-pressure predictor never builds the driving pressure); worst at low
  Re, not eliminated at Re=100. This is a numerical-methods problem, not a knob.

**Re-scope of the sub-phases:**

- **R2a → DONE (pivoted).** 3-D forward engine = the **monolithic SBM-NS saddle
  solve** (proven faithful with our own SBM). The projection-split-in-3-D moves
  to a dedicated research track (`docs/dev/2026-07-22-projection-sbm-3d-research-item.md`),
  still the intended 100M-scalability path.
- **R2b (device port) — re-scoped to the monolithic path.** The scalability
  lever is now the **monolithic block-preconditioner** (`solvers/block_precond.py`,
  already AMGX-backed) rather than the SPD-PPE→AMGX projection port. Port the
  monolithic 3-D march to a single GPU to reach the L8–L9 validation meshes.
  (The projection SPD-PPE→AMGX port is deferred with the projection research.)
- **R2c (literature validation) — on the monolithic engine.** Cylinder Cd /
  Strouhal and sphere Cd mesh-convergence, marched with the monolithic solver.
  The gate (monotone convergence to each reference within tolerance) is
  unchanged; only the engine changed.

The shell, the heroes (truck/maize), and the steady-adjoint (R3) all build on
the monolithic 3-D forward engine — none required the split specifically.

## ADDENDUM 2026-07-22 (2) — R2b.1 banked: cuDSS delivered, block-precond → overnight handoff

R2b.1 progress this session:
- **cuDSS unblock DELIVERED** (the banked win): monolithic sphere Cd, Re=100 —
  L4=0.964 (==splu, 4.7× faster), **L5=1.448** (clears the CPU-splu wall). L6
  (1.09M) cuDSS `ALLOC_FAILED` (direct GPU-memory wall). Baseline
  `tests/baselines/p2r2c_monolithic_sphere_convergence.json`. `monolithic_cd`
  now takes `solver=` routing through `solve_linear` (`e72121d`).
- **AMGX rebuilt on gpubox** (CUDA 12.4/sm_89; was missing — also fixes the 2
  suite amgx-parity failures).
- **Block-preconditioner (the L6+ scalability path): NOT YET SCALING.** It
  converges on a tiny L3 saddle but cliff-fails at L5 (143k) regardless of F
  strength / Schur mode (Cahouet-Chabard AND PSPG C-block) / GMRES restart — a
  structural, scale-specific breakdown, not a tuning weakness. Deferred to a
  focused instrumented investigation: **`docs/dev/2026-07-22-block-precond-scaling-handoff.md`**
  (detailed cold-start notes; branch `r2b1`, all knobs committed).

R2b.1 status: cuDSS validation banked; the iterative block-precond scalability
path is the open R&D item (handoff). All on branch `r2b1` (unmerged, unpushed).
