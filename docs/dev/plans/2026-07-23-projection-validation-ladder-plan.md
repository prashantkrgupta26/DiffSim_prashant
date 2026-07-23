# Projection Validation Ladder — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the three scaffolded concepts of pressure-projection+SBM
(projection / weak-Nitsche / shift) in isolation on an exact body-fitted
carved-octree mesh, in both steady and shedding wake regimes, and identify/recover a
working scalable projection scheme.

**Architecture:** A ladder of rungs (0, A, B, C in 2-D; A′, C′ in 3-D), each a driver
that runs the projection stepper and the monolithic stepper on the SAME mesh and
compares Cd (same-mesh oracle, box-free) + literature. The ladder is a parameter
progression on the existing stepper stack: box half-width alignment (`d=0` body-fitted
vs `d≠0` shifted) × BC treatment (strong Dirichlet vs weak Nitsche). Each rung runs two
solver variants (single-pass; stabilized inner iteration) and two Re (steady; shedding).

**Tech Stack:** existing `diffsim` — `octree/build.py` (`build_uniform` dim=2/3),
`geometry/csg.py` (`Box`), `sbm/surrogate.py` (`classify_lambda`/`extract_surrogate`/
`GeometryData`), `steppers/leray_sbm.py` (`LeraySBMStepper`), monolithic via
`tests/p2r0_task10_sphere_derisk.py::monolithic_cd`. Warp; gpubox two-lane compute.

## Global Constraints

- **Same-mesh monolithic oracle is the PRIMARY, box-free bar**: each rung's projection
  Cd must equal the monolithic Cd on the identical mesh within R0 tolerance. Literature
  (Ghia cavity; Breuer/Okajima square cylinder; cube refs) is the secondary absolute anchor.
- **Both wake regimes at rungs A/B/C (and A′):** a **steady** Re (below square-cylinder
  shedding onset — confirm onset, use ~Re 30–40) AND a **shedding** Re=100 (von Kármán;
  report mean Cd over ≥1 period + Strouhal St from the lift signal).
- **Both solver variants per rung:** `inner_iterate=False` (single-pass, lagged-p*) AND
  the stabilized inner iteration (Task 4). Report which achieves the pass bar.
- **Exact body-fitted = `Box` half-width `k/2^level` (⇒ d=0, corr=1.0, SBM≡Nitsche);
  shift = non-aligned half-width (⇒ d≠0).** Verify `|d|_max` per fixture (==0 for
  body-fitted rungs; >0 for shift rungs) — this is the anti-vacuity check that the rung
  tests what it claims.
- **Branch `projection-ladder`** off master; two-lane gpubox (`DiffSim`→GPU0 monolithic
  cuDSS, `DiffSim-proj`→GPU1 projection CPU-splu); REPO-IDENTITY guard.
- Bit-for-bit: any new stepper knob defaults OFF and leaves existing paths unchanged.
- Each rung FAIL localizes the breaking concept; record it and STOP advancing the ladder
  until that concept is understood (the controller decides, per findings).

---

### Task 1: Shared fixtures + benchmark references

**Files:**
- Create: `tests/ladder_fixtures.py`
- Create: `docs/dev/2026-07-23-ladder-benchmark-refs.md`
- Test: `tests/test_ladder_fixtures.py`

**Interfaces:**
- Produces: `build_cavity_2d(level, Re, device)` → (dm, cons, oracle, lid_mask, …);
  `build_square_channel_2d(level, Re, half, offset, device)` →
  (dm, cons, oracle, obstacle_node_mask, inflow_mask, outflow_nodes, geo, dmax);
  `build_cube_channel_3d(level, Re, half, offset, device)` (3-D analogue);
  `obstacle_boundary_nodes(mesh, oracle)` → node ids on the carved obstacle faces.
- Consumes: `build_uniform`, `Box`, `classify_lambda`, `extract_surrogate`,
  `GeometryData.evaluate`, `build_mesh`, `build_constraints`, `DeviceMesh.from_mesh`.

- [ ] **Step 1: Write failing tests** — assert (a) `build_square_channel_2d` with
  `half=0.25, level=4, offset=0` yields `geo.d` max == 0 and `geo.corr`≈1.0 (exact
  body-fitted); (b) with `offset=0.03` (non-aligned) yields `|d|_max`∈(0, h); (c) the
  obstacle mask is non-empty and lies on the box faces; (d) fluid node count matches the
  carved cell count. 3-D analogue for `build_cube_channel_3d`. Cavity: lid mask = top row.
- [ ] **Step 2: Run tests, verify they fail** (fixtures not defined).
- [ ] **Step 3: Implement fixtures** — channel domain `[0,Lx]×[0,Ly]` (Lx≥ downstream
  clearance from the benchmark), `Box` obstacle at the benchmark location/size; inflow
  Dirichlet at x=0 (uniform U_IN), no-slip channel walls or slip per benchmark, outflow
  free at x=Lx. Re set via `nu = U_IN·D/Re` (D = obstacle side). `offset` shifts the box
  center by a sub-cell fraction to break alignment (rung C). Reuse the sphere-fixture
  mesh-build chain verbatim; swap `Sphere`→`Box`.
- [ ] **Step 4: Pin benchmark references** in the refs doc: square-cylinder shedding
  onset Re_crit; steady Cd at the chosen steady Re; Re=100 mean Cd + St; blockage β,
  domain extents; cavity Ghia Re=100/400 centerline tables; cube-in-channel refs. Cite
  sources (Breuer et al. 2000; Okajima 1982; Ghia-Ghia-Shin 1982).
- [ ] **Step 5: Run tests, verify pass. Commit.**

### Task 2: Rung 0 — lid-driven cavity (base projection soundness)

**Files:**
- Create: `tests/ladder_rung0_cavity.py`
- Test: `tests/test_ladder_rung0.py`

**Interfaces:** Consumes `build_cavity_2d`. Produces the rung-0 verdict (projection vs
monolithic vs Ghia).

- [ ] **Step 1: Write failing test** — a fast small-level cavity (e.g. level 5, Re=100):
  assert the projection stepper reaches a steady state whose centerline u(y) matches the
  monolithic on the SAME mesh within tol, and both track Ghia within a few % at the
  tabulated points. (Keep the asserted case cheap; the driver runs Re=100 and Re=400.)
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement the driver** — march projection (single-pass) + monolithic to
  steady on the cavity; extract centerline u(y), v(x); compare to each other and to Ghia.
  No obstacle, strong lid BC. This validates the base projection independent of any
  obstacle/Nitsche/shift.
- [ ] **Step 4: Run on gpubox; record centerline match + steady ‖div u‖.** If projection
  fails the cavity, the base projection impl is broken — report and STOP (fix before A).
- [ ] **Step 5: Commit** (driver + verdict note in the refs doc).

### Task 3: Rung A — body-fitted square, STRONG Dirichlet (the decisive control)

**Files:**
- Create: `tests/ladder_rungA_square_strong.py`
- Test: `tests/test_ladder_rungA.py`

**Interfaces:** Consumes `build_square_channel_2d(half=aligned, offset=0)`,
`obstacle_boundary_nodes`. Produces the rung-A verdict at Re-steady and Re=100.

- [ ] **Step 1: Write failing test** — at the steady Re (small level), assert projection
  (single-pass) steady Cd == monolithic same-mesh Cd within R0 tol, AND mean|u| reaches
  the monolithic magnitude (no weak plateau), AND `|d|_max==0` (body-fitted guard).
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement the driver** — strong no-slip on `obstacle_boundary_nodes`
  (add to strong_mask), inflow/outflow per fixture. Run projection (single-pass) and
  monolithic at BOTH Re-steady (steady Cd) and Re=100 (mean Cd over ≥1 shedding period +
  St from lift). Log per step: mean|u|, ‖p‖, weak-div, Cd, (Cl for St).
- [ ] **Step 4: Run on gpubox (both lanes: monolithic GPU / projection CPU).** Record the
  decisive read: does strong-Dirichlet body-fitted projection match monolithic and
  develop the flow? This confirms/refutes "projection is structurally sound."
- [ ] **Step 5: Commit** (driver + verdict). If single-pass FAILS here, the fix is the
  stabilized inner iteration (Task 4) — proceed there before B.

### Task 4: Stabilized inner predictor↔PPE iteration

**Files:**
- Modify: `src/diffsim/steppers/leray.py` (the `_projection_pass` / `step` inner loop)
- Modify: `src/diffsim/steppers/leray_sbm.py` (thread the knob)
- Test: `tests/test_leray_inner_stab.py`

**Interfaces:** Produces knobs `inner_relax` (ω∈(0,1], default 1.0 = current behavior)
and `inner_accel` ("none" default | "anderson"), used by the inner loop when
`inner_iterate=True`. Bit-for-bit unchanged at defaults.

- [ ] **Step 1: Write failing test** — on a tiny fixture, assert that with
  `inner_iterate=True, inner_relax=ω` the inner update is `p* ← p* + ω(p_hat − p*)`
  (relaxed), and that `inner_accel="anderson"` applies the Anderson mixing of the last m
  residuals; assert defaults (`ω=1, accel=none`) reproduce the current ν-loop bit-for-bit.
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** damped relaxation + optional Anderson acceleration (small m,
  e.g. 2–3) on the predictor↔PPE fixed-point map, with a divergence guard (break to the
  last bounded iterate if the inner residual rises for 2 consecutive iters).
- [ ] **Step 4: Run tests (defaults bit-for-bit; relaxed/Anderson change the trajectory).
  Commit.**
- [ ] **Step 5: Re-run Rung A** with the stabilized inner iteration at both Re; record
  whether it recovers the strong branch (mean|u|→monolithic, Cd match). Append to the
  rung-A verdict.

### Task 5: Rung B — body-fitted square, WEAK Nitsche

**Files:**
- Create: `tests/ladder_rungB_square_nitsche.py`
- Test: `tests/test_ladder_rungB.py`

**Interfaces:** Consumes the SAME `build_square_channel_2d(half=aligned, offset=0)` but
imposes no-slip WEAKLY via the SBM Nitsche block (with `d=0`, so it is standard Nitsche).

- [ ] **Step 1: Write failing test** — assert projection (best variant from A) with weak
  Nitsche matches monolithic-with-weak-Nitsche on the same mesh within tol at the steady
  Re, `|d|_max==0`, and the Nitsche penalty is load-bearing (removing it breaks it).
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement the driver** — weak no-slip (SBM Nitsche, d=0) instead of strong
  mask; both Re, best solver variant (+ single-pass for contrast). Compare to monolithic
  same-mesh and to rung A (isolates the Nitsche concept: does weak imposition alone change
  the development?).
- [ ] **Step 4: Run on gpubox; record verdict** (does Nitsche preserve faithfulness?).
- [ ] **Step 5: Commit.**

### Task 6: Rung C — offset square, SBM shift

**Files:**
- Create: `tests/ladder_rungC_square_shift.py`
- Test: `tests/test_ladder_rungC.py`

**Interfaces:** Consumes `build_square_channel_2d(half=aligned, offset=δ)` with δ a
sub-cell shift so `|d|_max`∈(0,h) — the genuine SBM case.

- [ ] **Step 1: Write failing test** — assert `|d|_max`>0 (shift active), and projection
  (best variant) Cd == monolithic same-mesh Cd within tol at the steady Re; the shift
  correction (`geo.corr`, `d`) is load-bearing (zeroing d breaks the match).
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement the driver** — offset box, weak Nitsche + shift; both Re, best
  variant. Compare to monolithic same-mesh (which also uses the shift). Isolates the
  shifting concept: does adding the surrogate shift (on top of a working projection+Nitsche)
  degrade faithfulness?
- [ ] **Step 4: Run on gpubox; record verdict.**
- [ ] **Step 5: Commit.**

### Task 7: Rungs A′/C′ — 3-D cube confirmation

**Files:**
- Create: `tests/ladder_rung3d_cube.py`
- Test: `tests/test_ladder_rung3d.py`

**Interfaces:** Consumes `build_cube_channel_3d(half=aligned|offset)`. Monolithic runs
cuDSS on GPU; projection CPU-splu.

- [ ] **Step 1: Write failing test** — at a modest 3-D level, steady Re: projection (best
  variant) Cd == monolithic same-mesh Cd within tol, for body-fitted (A′) and offset/SBM
  (C′). Guard `|d|_max` (==0 for A′, >0 for C′).
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement the driver** — A′ (strong Dirichlet, body-fitted) and C′ (offset
  SBM), steady Re primary; add Re=100 shedding if the steady rungs pass. Two-lane compute.
- [ ] **Step 4: Run on gpubox; record verdict** (does the 2-D conclusion hold in 3-D — the
  original sphere regime).
- [ ] **Step 5: Commit.**

### Task 8: Synthesis + recommendation

**Files:**
- Create: `docs/dev/2026-07-23-projection-ladder-verdict.md`

- [ ] **Step 1:** Consolidate every rung's verdict (steady + shedding, both variants) into
  one table: where development breaks, which concept is responsible, and which solver
  variant achieves faithfulness. State whether a scalable projection+SBM scheme is
  recovered (and its exact config) or whether a concept is proven infeasible (validating
  the monolithic engine). Recommend the next step (productionize the working config, or
  close the projection track). Commit.

---

## Self-Review notes
- Spec coverage: rungs 0/A/B/C/A′/C′, both Re regimes (steady+shedding), both solver
  variants, same-mesh oracle + literature, the stabilized inner iteration, the enabler
  (Box+aligned half-width, d=0 guard) — all mapped to tasks.
- Decision points are explicit (rung FAIL → localize + stop; rung A single-pass fail →
  Task 4). The controller adapts per finding (research plan, not a rigid pipeline).
- Types consistent: fixtures (Task 1) are consumed by all rung drivers; the inner-iter
  knobs (Task 4) are consumed by rungs A(re-run)/B/C/3-D.
