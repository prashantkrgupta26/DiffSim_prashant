# Projection Validation Ladder — verdict

**Date:** 2026-07-23
**Status:** ladder concluded at rung A. The projection defect is UPSTREAM of SBM —
it is a base pressure-projection ↔ monolithic **operator inconsistency on open-outflow
external flow**. Rungs B/C (Nitsche, shift) are moot (nothing to add above a rung-A
failure). Validates the monolithic pivot at the 2-D body-fitted level.

## Purpose (recap)
Baskar's a→b→c decomposition: isolate the three scaffolded concepts — (1) pressure
projection, (2) weak Nitsche imposition, (3) SBM shifting — on an EXACT body-fitted
carved-octree mesh, to find where the 3-D flow under-development originates.

## Rung results

| Rung | Test | Result |
|---|---|---|
| **0** | Lid-driven cavity (closed), projection vs monolithic vs Ghia | **PASS** — base projection sound: reproduces Ghia AND matches monolithic same-mesh. Plumbing (predictor/PPE/correct/pressure-pin) works. |
| **A** | Body-fitted square (`d=0`), STRONG Dirichlet, open channel | **FAIL** — single-pass diverges (Re=40 Cd=−61, ‖div‖=45, ‖p‖=833; Re=100 blows up). Obstacle-free open channel fails identically ⇒ **breaking ingredient = OPEN OUTFLOW**, not the obstacle. |
| **A-fix** | Stabilized inner iteration; consistent PPE operator | **FAIL** — neither closes it. |
| **B, C** | Weak Nitsche; SBM shift | **NOT RUN** — moot; the break is below them. |

## The mechanism (decisive diagnostic)

Rung 0 (closed cavity) passes; rung A (open outflow) fails. The difference is the open
outflow + lagged pressure. The clinching test (Task 4): **seed the projection with the
EXACT monolithic (u, p) and take one step.**
- The predictor reproduces the seeded state (residual 3.6e-6) — the predictor is consistent.
- **PPE + correction throws it:** Cd +4.18 → −55, and ‖div‖ *rises* 1.22 → 6.0.

So the monolithic steady state is **not a fixed point of the lagged-p\* split**. Root:
the monolithic (equal-order PSPG) enforces incompressibility via its PSPG pressure block
(weak-div ~1.22, not pointwise-zero); the projection's PPE enforces a *different* discrete
incompressibility (via `K_p`, the FE Laplacian; measured ‖L−K_p‖/‖K_p‖ = 0.67 against the
consistent `L=GᵀM⁻¹G`). Two different incompressibility notions ⇒ two different solutions ⇒
not mutual fixed points. The split's correction, applied to the monolithic solution,
over-corrects and corrupts it.

## Fixes attempted (Task 4, all default-off knobs, bit-for-bit, tests green)
- **Stabilized inner predictor↔PPE iteration** (`inner_relax` ω, `inner_accel="anderson"`,
  divergence guard): the within-step map is **non-contractive** on the open outflow — the
  inner residual never converges at any ω or with Anderson.
- **Consistent PPE operator** (`consistent_ppe`, `L=GᵀM⁻¹G`): statically idempotent
  (‖Bᵀu‖→1e-15) but **diverges faster live** — idempotency of one projection ≠ stability
  of the coupled predictor↔PPE map on external flow.

## Conclusion & implication

- **The SBM/Nitsche/shift layer was never the problem.** The projection under-development
  is a *base* pressure-projection ↔ monolithic operator inconsistency that manifests the
  moment there is an open outflow (external flow). This is why the sphere (open domain)
  failed and the cavity (closed) did not.
- **This validates the monolithic pivot** at the 2-D body-fitted level (not just 3-D SBM):
  the monolithic is faithful; the lagged-p\* split is not, for external flow, in our
  current formulation.
- **NOT a claim that projection is impossible** — classical projection works for external
  flow in the literature. It means OUR equal-order projection needs a **consistency
  co-design**: a pressure-Poisson operator + outflow BC that share the monolithic's
  (PSPG) discrete incompressibility notion. The two obvious fixes don't achieve it; this
  is genuine formulation research, not a knob.

## Decision required
Whether to (a) pursue the consistent-operator + outflow co-design (hard, uncertain payoff,
but the real 100M-scalability path), or (b) accept the validated monolithic engine and
redirect effort (R3 steady-adjoint, block-precond L6 scaling), keeping projection as a
documented open research item.

## Artifacts
Branch `projection-ladder` (unmerged): fixtures + rung-0/rung-A drivers + the `inner_relax`/
`inner_accel`/`consistent_ppe` knobs, all default-off with green regressions. Predecessor:
`2026-07-23-projection-sbm-weak-fixed-point-verdict.md` (+ Task-4 addendum). Spec/plan:
`docs/dev/specs|plans/2026-07-23-projection-validation-ladder-*.md`.
