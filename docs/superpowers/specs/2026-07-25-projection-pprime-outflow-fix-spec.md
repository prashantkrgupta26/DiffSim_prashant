# Projection p′-scheme Outflow-BC Fix — 3-D Thin-Plate Cd

**Date:** 2026-07-25
**Status:** APPROVED (Baskar chose "fix projection split (p′-scheme)").
**Authoritative reference:** `local_code_old/ns_projection_vms_paper (1).pdf` — Khara,
Murugaiyan, Khanwale, Ganapathysubramanian, *A Helmholtz–Leray projection method
with VMS stabilization for the Navier–Stokes equations*. Companion memory:
`projection-outflow-bc-scheme`.
**Dispatch on Opus** (design/numerical-correctness tier, not mechanical).

## Problem
The 3-D thin-plate projection stepper (`LeraySBMShellStepper`, `solver="gpu_cg"`)
produces **wrong-signed, diverging** drag (Cd −67→−496 over 7 steps) where the
monolithic driver gives the correct decaying startup (Cd +237→+62). The two-sided
shell assembly is NOT the bug (monolithic uses the same `extract_two_sided_surrogate`
+ `sbm_vector_dirichlet_twosided` and is correct); `gpu_cg` is NOT the bug (`splu`
and `gpu_cg` give identical Cd).

## Root cause (high-confidence hypothesis)
The base `LerayProjectionStepper` (leray.py) **already implements the incremental
(van Kan) pressure-correction p′-scheme with the correct outflow BCs** — it was
validated in 2-D (cylinder, projection ladder). The scheme (paper §2.4, Algorithm 1,
Eq. 68):
- **Incremental**: `p* = p̂ⁿ` (order-1 extrapolation); the predictor sees `∇p*` as
  known data (Eq. 44a / 29a). NOT Chorin `p*=0`.
- **PPE solves for the increment** `φ = p̂ − p*` (Eq. 37/44b), with boundary
  conditions (Eq. 68, flow-past-cylinder — the same class as flow-past-plate):
  - **`∇φ·n = 0` (natural Neumann)** on every velocity-Dirichlet boundary — inlet,
    lateral/top/bottom free-stream, AND the immersed body (plate).
  - **`φ = 0` (Dirichlet)** on the **outflow face only** (Eq. 68d: `p − p* = 0` at
    `x = L`), where the velocity is do-nothing (`n·∇u = 0`, Eq. 68e).
- **Velocity update** uses `∇φ = ∇(p̂ − p*)`, not `∇p̂` (Eq. 38/44c).

The base stepper exposes this via `pressure_outflow_nodes` (the outlet free-node
set gets the PPE `φ=0` Dirichlet), plus velocity do-nothing + backflow on the same
outlet, and an optional `rotational_pin_outflow` (consistent projection).
leray.py:215-221 documents the failure mode when this is absent: *"leaving the
outflow pressure floating for an EXTERNAL flow with a free outflow face → the
incremental p* drifts (no stable config)."*

**The 3-D thin-plate path never supplied `pressure_outflow_nodes`.** The monolithic
3-D driver (`tests/p2r1c_thin_plate_flow_3d.py`) pins pressure at a **single
outflow-low-back corner node** (line 23, 244) — correct for the coupled saddle, but
for the projection PPE it leaves the outlet face floating → `p*` drifts every step →
the predictor never builds the driving pressure → wrong-signed, diverging Cd. This
is the documented drift, NOT the deeper `p2-r2a-monolithic-pivot` weak-fixed-point
defect (that verdict predates the outflow-BC fix; the 2-D cylinder projection
validated *with* these BCs).

## The fix
Wire the **existing, 2-D-validated** outflow-BC machinery into the 3-D thin-plate
projection driver + confirm `LeraySBMShellStepper` forwards it end-to-end.

### Task 1 — outlet-face node set + driver wiring
**File:** `tests/p2r1c_thin_plate_flow_3d_projection.py` (the honest-gate driver the
landing agent is creating).
- Compute `pressure_outflow_nodes` = the FREE nodes on the outlet face `x = x_max`
  (within a geometric tol), analogous to the 2-D cylinder projection case. This is a
  2-D sheet of nodes in 3-D (generalize the 2-D outlet-line logic).
- Pass `pressure_outflow_nodes=<that set>` into `LeraySBMShellStepper`.
- Confirm the predictor outlet is do-nothing: the plate (shell) is the only strong
  velocity Dirichlet (SBM two-sided no-slip); inlet + lateral + top/bottom carry the
  free-stream `u_inf` Dirichlet; the outlet velocity is FREE (natural
  `n·(−ν∇ũ)=0`) with backflow stabilization (`beta_backflow`) on the outlet face
  (Eq. 68e).
- The plate/shell gets **NO** pressure Dirichlet — `φ` is natural-Neumann there
  (Eq. 68: `∇(p−p*)·n̂ = 0` on the body). Verify nothing in the SBM path pins
  pressure at the shell.

### Task 2 — drag from the corrected fields
- Compute Cd via `surrogate_traction` using the **corrected** pressure `p̂` and
  corrected velocity `û` (Eq. 70: full traction `(−pI + ν(∇u+∇uᵀ))·n`), NOT `p*`
  or `φ`, NOT the predictor `ũ`. Confirm the stepper exposes `p̂`/`û` at end-of-step.

### Task 3 — validation (Mac CPU, small)
- Run the SAME tiny 3-D case through BOTH the monolithic driver and the fixed
  projection driver. **Gate: projection Cd is positive, non-diverging, and agrees
  with monolithic Cd within tolerance** (the both-solver deliverable). This REPLACES
  the honest "finite-only" gate once the fix lands.
- Keep `test_p2r0_projection_sbm.py` green (the 2-D one-sided path unregressed).
- Assert the PPE ran on `gpu_cg` and converged.

### Fallback levers (only if drift persists after Task 1)
Already in the base stepper — try in this order, documenting which was needed:
1. `rotational_pin_outflow=True` (consistent projection: pins the rotational
   correction `q` to 0 on the outflow rows so `p̂ = p* + φ − νq` keeps `φ=0` AND
   `q=0` at the outlet — leray.py:125-152).
2. The velocity directional-do-nothing / backflow-velocity-stabilization on the
   outlet (leray.py:74-119) if the wake reverses through the outlet.
3. `graddiv_dynamic` if divergence in the near-outlet band is high.
Escalate to Baskar only if all three fail — that would revive the deeper-defect
hypothesis.

## Deliverable
3-D thin-plate projection Cd that is physical (positive, non-diverging, ≈ monolithic
on the same mesh) → then the **L9-near-plate GH200 run with `gpu_cg`** yields a
faithful, scalable 3-D drag: correct physics AND the scalable SPD-PPE solver, the
ideal outcome. Update the runbook + the driver docstring to remove the
"blocked by projection-split defect" caveat once validated.

## Notes / subtleties
- Adaptive octree + hanging nodes: `pressure_outflow_nodes` must be FREE nodes
  (post-constraint); the τ_m element length `h` varies (fine near the plate) — the
  base stepper handles this, but confirm the outlet-node set is computed on the
  constrained (free) numbering the PPE actually solves.
- Boundary classification: all of inlet/lateral/top/bottom are velocity-Dirichlet
  (`u_inf`), so they correctly get natural `∇φ·n=0`; only `x=x_max` is the outlet.
- This is a DRIVER + wiring fix leveraging validated stepper physics — the numerical
  risk is in getting the outlet-face node set and the do-nothing/backflow pairing
  right in 3-D, which is why it is Opus-tier, not mechanical.
