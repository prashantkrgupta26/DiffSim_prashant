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
