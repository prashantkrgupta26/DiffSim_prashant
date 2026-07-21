# P2-R0 — VMS-projection stepper on volumetric SBM

**Implementation design spec.** First rung of milestone P2
(`docs/dev/specs/2026-07-21-p2-nsht-sbm-shell-milestone.md`, §5 staging).
The stepper de-risk rung: prove the VMS-incremental-projection stepper,
composed with **volumetric** SBM, reproduces the M1b flow-past-a-body
benchmarks — *before* the co-dim-1 shell (R1) is introduced. Brainstormed
+ scoped 2026-07-21.

## 1. Goal

Establish the P2 forward engine on a trusted footing: the group's
VMS-incremental-projection method (equal-order, BDF2) running flow past an
**immersed volumetric obstacle** via SBM, validated against the M1b
cylinder/sphere drag references and the VMS-projection paper. No shell, no
adjoint, no HT — just the stepper + volumetric SBM, gated.

## 2. Key finding — the stepper already exists

`src/diffsim/steppers/leray.py::LerayProjectionStepper` is already an
incremental VMS-Helmholtz-Leray projection stepper ("production stepper
#2" from M1b):
- Three sub-solves per step — nonlinear **predictor** (Picard/Newton over
  the block, pressure pinned to `p*`), **pressure-Poisson** (the Leray
  projection, `ppe_finescale` VMS option), **correction** (mass solves to
  the divergence-free velocity).
- **Incremental** pressure (`p* ← p_hat`, "the draft's choice"), BDF
  order=2 default (`bdf_coeffs`/`bdf_order_now`, `History` in
  `solvers/timestepping.py`), the projection-draft τ in
  `physics/vms.py::tau_hbased_host` ("the projection draft's Eq. 45").
- `divergence_l2()` solenoidality diagnostic; per-step adjoint in
  `sbm/leray_adjoint.py` (not used in R0).

**What it is NOT:** SBM-coupled. It takes body-fitted `f_fn`/`g_fn`
boundary data, not an immersed obstacle. **That coupling is R0's real
work.**

So R0 is not "build a projection stepper" — it is: (a) **audit** the
existing stepper against the published VMS-projection paper and fix any
formulation deviation; (b) **compose it with volumetric SBM**; (c)
**validate** against M1b.

## 3. What exists vs what R0 adds

**Reuse as-is:**
- `steppers/leray.py::LerayProjectionStepper` — the three-sub-solve engine.
- `physics/vms.py` — τ_M/τ_C (metric + h-based projection-draft forms).
- `sbm/vector.py` — `sbm_vector_dirichlet` (shifted-Nitsche vector BC,
  penalty on shifted test×trial, backflow, `surrogate_traction`).
- `sbm/surrogate.py` — `extract_surrogate` (volumetric carve-out),
  `classify_lambda`, `face_gauss_points`.
- `api/ns_bricks.py` — `assemble_linear_ns` (the predictor's volume block).
- M1b cylinder/sphere validation harness (in-repo tests).

**R0 adds (the integration):**
1. **Volumetric-SBM boundary in the three sub-solves.** Thread the
   shifted-Nitsche vector Dirichlet (`sbm/vector.py`) through the
   predictor's block, and — critically — impose the **surrogate-consistent
   BC on the pressure-Poisson and the correction** so the projection
   preserves the no-penetration (blockage) property. This is exactly the
   BC-consistency the solution-transfer paper proves necessary; R0 wires
   it for the *static* immersed body (the moving/adaptive transfer
   projection is R3).
2. **A composed driver** — `steppers/leray_sbm.py` (or a mode of the
   existing stepper) that takes an immersed-geometry oracle (the existing
   TriMesh/SDF surrogate) instead of body-fitted `g_fn`, extracts the
   volumetric surrogate, and runs the projection march past the body.
3. **The validation harness** for Cd/Strouhal extraction via the
   surrogate traction (`sbm/vector.py::surrogate_traction`).

## 4. Formulation-parity audit (the VMS-projection paper)

Before validating, confirm (and fix if needed) that the existing stepper
matches `ns_projection_vms_paper.pdf`:
- The predicted velocity's **fine scales** (residual-based, modeled via
  τ_M) supply both advective and pressure stability → equal-order u/p.
- The **three subproblems** and the incremental pressure update match the
  paper's algorithm (predictor uses `p^{n-1}`; PPE solves the increment;
  correction projects).
- BDF-r discretization (r∈{1,2}) with the paper's σ = β₀/Δt and the
  cross-matrix history rule.
Deviations found → fix, with a one-line note in the R0 report citing the
paper equation.

## 5. Gates (all must pass)

- **G1 — NS MMS:** manufactured-solution convergence, observed order
  within ±0.10 of theoretical for **both velocity and pressure**, p1 (and
  p2 if in the fixture), 2-D and a small 3-D case. Planted coupling-break
  collapses the order (real MMS, not tautological).
- **G2 — BDF2 temporal order:** time-refinement study shows 2nd-order
  convergence of the projection march (the solution-transfer paper's
  temporal-order check on a static mesh).
- **G3 — volumetric-SBM consistency:** the shifted-Nitsche vector BC on an
  immersed body passes a patch/blockage test (no-penetration enforced;
  surrogate traction well-defined); divergence L2 within the projection
  tolerance.
- **G4 — cylinder:** reproduce the M1b references via the
  projection+volumetric-SBM path — **Cd=1.352 at Re20** (steady) and the
  **Strouhal shed-frequency at Re100** (unsteady wake, exercises the
  transient march).
- **G5 — sphere Re300:** Cd reproduces the M1b reference (3-D volumetric
  SBM, projection stepper).

## 6. Scaling-pathway declaration (mandatory)

R0 runs at 2-D / small-3-D validation scale, but declares the pathway the
hero rungs inherit:
- **Stage residency:** octree + volumetric surrogate extraction + distance
  — host, per-epoch (static geometry ⇒ acceptable). Predictor assembly,
  PPE, correction — the scaling-critical solves; R0 validates on the
  host/reference solve, and **declares the device port + SPD-PPE→AMG/AMGX
  as the R2 prerequisite** (the SPD pressure-Poisson is the scalable win
  of the projection choice; the predictor is the nonsymmetric solve). No
  new nnz-space arrays beyond the standard NS system.
- **100M note:** at hero scale the NS system hits the same ChunkedCSR
  (#38) + fp32-IR (#36) + multi-GPU (S4) requirements as any 100M solve
  (umbrella §8); R0 introduces nothing that violates them.
- **Tiers:** workstation single-GPU for all R0 gates.

## 7. File structure (planned)

- `src/diffsim/steppers/leray_sbm.py` — the SBM-composed projection driver
  (immersed-geometry oracle → volumetric surrogate → SBM-BC'd three
  sub-solves). Thin: composes `LerayProjectionStepper` + `sbm/vector.py` +
  `sbm/surrogate.py`; does not fork the stepper.
- `src/diffsim/steppers/leray.py` — modify only if the parity audit (§4)
  requires it; note each change against the paper.
- `tests/test_p2r0_projection_sbm.py` — G1–G5 gate battery.
- Reuse (do not fork): `sbm/vector.py`, `sbm/surrogate.py`, `vms.py`,
  `api/ns_bricks.py`, the M1b validation fixtures.

## 8. Explicitly out of R0 scope

- The co-dim-1 two-sided **shell** surrogate (R1).
- Any **adjoint** / differentiability (deliverable A lands in R1).
- **HT/Boussinesq**, scalar transport (later rung).
- **Moving-body / adaptive** transfer projection (R3; R0 is static
  geometry, fixed mesh or single-epoch).
- **Device port** of the projection stepper if the host reference suffices
  for the R0 gates — declared as the R2 prerequisite, not built here
  (unless the parity audit makes the device path free to include).

## 9. Process & compute placement

Subagent-driven, TDD, commit-per-green, agents never push. REPO-IDENTITY
GUARD in all prompts. Compute on **gpubox** for anything beyond file-level
CPU checks (the standing preference); the Mac only for quick unit checks.
The VMS-projection paper (`local_code_old/ns_projection_vms_paper.pdf`) and
the C++/Dendrite reference are the parity oracles — consult, never commit.

---

## P2-R0 Scaling-pathway declaration

**Recorded at Task 11 close (2026-07-21). Binding per `production-code-conventions.md` §Scaling-pathway declaration (ratified 2026-07-19).**

### R0 honest status

R0 validated the discretization:
- G1 MMS spatial order: velocity 2.01 / 2.07, pressure 2.07 / 2.02 (2-D p1).
- G2 BDF2 temporal order: 1.996 (projection march).
- G3 SBM consistency: surrogate-consistent no-penetration 1e-12, planted-break decisive.
- 2-D faithfulness to the monolithic saddle-point stepper confirmed.

**OPEN items deferred to R2:**
- 3-D projection has an open pressure-coupling stability item (stabilized config: principled penalty α~Pe·p² or implicit fine-scale PPE). Not hidden — it is the R2 task.
- Literature Cd/Strouhal convergence (G4 cylinder, G5 sphere Re300) deferred to R2 (Task-9/10 requirements). The immersed-body benchmarks require the device-AMG path to reach the mesh resolutions the references demand.

**R0 device status:** HOST/splu-reference only. The device port is the R2 prerequisite (per resolution 3 + Task-9/10 findings). R0 does NOT claim device-readiness.

### Stage residency table

| Stage | Residency | Notes |
|-------|-----------|-------|
| Octree build + mesh | host, per-epoch | static geometry — acceptable; never on the critical-step path |
| Volumetric surrogate extraction (`classify_lambda`, `extract_surrogate`) | host, per-epoch | geometry is static in R0; R3 introduces adaptive transfer |
| Distance/normal evaluation (`GeometryData.evaluate`) | host, per-epoch | KD-tree closest-point; amortized over all steps in the epoch |
| SBM face-block assembly (`sbm_vector_dirichlet`) | host, once-per-epoch | geometry-only block cached; backflow increment per-step host |
| Predictor sub-solve (nonlinear VMS momentum, nonsymmetric) | **host (R0)** → **device (R2)** | R0: splu reference; R2: device FGMRES (#49) + block-ILU/AMG; the nonsymmetric Oseen system is the harder device solve |
| Pressure-Poisson (PPE) sub-solve (SPD, scalar Laplacian) | **host (R0)** → **device/AMG (R2)** | **The AMG/AMGX-scalable lever** — the SPD pressure-Poisson is the primary reason projection was chosen over the monolithic saddle: it admits AMG/CG, replaces the indefinite Schur complement, and is the unlock for 100M-DOF throughput. R0 validates on splu; R2 ports to AMG/AMGX. cuDSS cannot reach the hero (#43) — iterative device solve (device-FGMRES #49 / AMGX) is the mandated path. |
| Velocity-update (correction) sub-solve (mass matrix, SPD, H¹) | **host (R0)** → **device (R2)** | Diagonal-dominant mass; cheapest of the three sub-solves; lumped mass is an admissible approximation at scale |
| BDF history rotate / `divergence_l2` diagnostic | host | negligible cost; diagnostic only |
| Surrogate traction / Cd extraction | host, per-output step | face-GP loop; small; amortized |

**No new nnz-space arrays beyond the standard NS system** (verified by `tests/test_p2r0_scaling.py::test_no_new_nnz_space`): the SBM face block `Af_c` scatters only into existing node-pair slots already present in the `assemble_linear_ns` COO graph. The SBM composition is pattern-conservative.

### 100M-DOF budget line

- **NS system:** `ndof = dim + 1` (3 in 2-D, 4 in 3-D); p1 hex; nnz/dof ≈ 27 × ndof² / ndof ≈ 27 × ndof (element bandwidth); index width: mixed-width CSR per the P0-2 templating.
- **SBM addition:** zero new nnz outside the NS pattern (face-block node pairs ⊆ element-connectivity pairs). The PPE and mass sub-solves are scalar (nnz/dof ≈ 27 for p1 hex), standard ChunkedCSR-compatible.
- **At 100M DOFs:** same ChunkedCSR (#38) + fp32-IR (#36) + multi-GPU halos (S4) requirements as any 100M NS solve (umbrella §8). R0 introduces nothing that violates them — the SBM composition is additive in values, not in structure.
- **Multi-GPU comms:** halos at partition boundaries for the predictor + PPE as in the monolithic NS; the SBM face block is local to the patch (surrogate faces are a local subset; no inter-partition face communication beyond the standard NS halo).

### Deployment tiers

| Tier | Hardware | Scope |
|------|----------|-------|
| Workstation / Mac | CPU (host splu) | R0 gate battery — MMS, BDF2 order, SBM consistency, scaling assertion |
| Workstation single-GPU (gpubox) | A100 / H100 | R0 cylinder/sphere benchmarks (G4/G5 nightly, when R2 device path lands) |
| Single big node | GH200 / Horizon NVL4 | Hero 3-D runs at 10M–100M DOFs (R2+); PPE-AMG the unlock |
| Multi-node | Horizon gb-large / AWS on-demand | Multi-GPU 100M+ DOF studies (R3+); standard NS halo + SBM local patch |

### The scalable lever: SPD pressure-Poisson → AMG/AMGX (R2 prerequisite)

The projection split replaces the indefinite monolithic Schur-complement solve with three sequential sub-solves of which the **pressure-Poisson (PPE) is SPD and scalar** — the exact system class that AMG/CG (and AMGX) accelerate most effectively. This is the quantitative reason projection was chosen over the monolithic saddle for the P2 hero path:

- Monolithic saddle at 100M DOFs requires an indefinite block preconditioner (Schur-complement approximation) with no proven scalable AMG path.
- The PPE Laplacian admits algebraic multigrid directly; the AMGX stub (`solvers/amgx.py`) already identifies it as the target.
- R2's device-PPE task is therefore the single highest-leverage item in the P2 scaling plan.

The predictor (nonsymmetric Oseen, step 1) requires device FGMRES (#49) + block-ILU or AMG as preconditioner — tractable but harder than the PPE. The correction (mass matrix, step 3) is SPD diagonal-dominant and cheapest of the three.

**R2 dependencies (not hidden):**
1. Device port of the predictor → FGMRES #49 + block-AMG/ILU.
2. Device port of the PPE → AMGX AMG/CG (the scalable lever).
3. 3-D pressure-coupling stability: principled penalty α~Pe·p² or implicit fine-scale PPE (open item from R0 3-D formulation).
4. G4/G5 literature Cd/Strouhal convergence at reference mesh resolutions.
