# Computational C3 — Octree refinement and temporal adaptivity

Resolution should follow the physics. This tutorial measures octree
refinement and temporal adaptivity — **honestly**:

- **Octree refinement + hanging-node constraints** — `build_adaptive`
  puts small elements only where the field is sharp. Measured ≈ 3.6× fewer
  nodes than a uniform mesh at the same finest $h$, and the CH brick runs
  on the hanging-node mesh. **This refines to a *static* geometric
  criterion — it is NOT solution-adaptive AMR.**
- **What true dynamic AMR needs** — estimate → mark → refine/coarsen →
  2:1 balance → rebuild → conservative transfer → BDF-history transfer →
  continue. We measure the correctness-critical piece: **conservative
  transfer** (naive injection loses ~20% of sub-cell mass; cell averaging
  is exact). Full dynamic AMR is a Phase-3 deliverable.
- **Temporal (LTE step ladder)** with **real cost accounting** —
  accepted/rejected steps, full+half solves, Newton iterations, wall time,
  and a **matched-accuracy fixed-dt sweep** (NOT the fictional
  horizon/min-dt ratio the old version reported).
- **Why variable-coefficient BDF2** — growing $\Delta t$ forces BDF2 to
  use coefficients built from the actual $(\Delta t, \Delta t_{\text{prev}})$;
  the constant-step coefficients collapse to order ≈ 1. **Both orders are
  MEASURED here** (variable ≈ 2.01; constant ≈ 0.93, via a tutorial-local
  forced-`r=1` march — no dependence on an inaccessible dev note).

**Read** the course document, Computational Chapter *"Octree refinement
and temporal adaptivity"* (start with `adaptivity.py`).

**Run:**
```bash
python run.py                 # octree + transfer + real cost + BDF2 order
```
`run.py` prints eight `PASS/FAIL` checks; compare with
[`EXPECTED.md`](EXPECTED.md).

**Regenerate the document's figures + numbers** (optional):
```bash
python gen_figures.py         # writes ../../latex/figures/c3_*.png + numbers/c3.tex
```

| file | role |
|------|------|
| `adaptivity.py` | the core: `octree_refinement`, `transfer_error`, `adaptive_cost`, `instrumented_march`, `bdf2_variable_order` — read this first |
| `run.py` | the driver you run; prints the four studies |
| `gen_figures.py` | regenerates the octree/transfer/ladder/cost figures and `numbers/c3.tex` |
| `EXPECTED.md` | reference numbers your run should reproduce |

The tutorial uses the production brick
`src/diffsim/physics/cahn_hilliard.py` and its `adaptive_march`. The
constant-coefficient BDF2 degradation is **measured locally** (a
`_const_march` that forces the coefficient ratio `r=1` on a varying
history), not cited — so the comparison is reproducible from this folder
alone.

## Learning objectives

By the end you can:

- Distinguish octree refinement to a *fixed geometric criterion* (here,
  a band around a static circle) from **solution-adaptive AMR**, and
  list the additional steps dynamic AMR needs that static refinement
  does not — estimate, mark, refine/coarsen, 2:1 balance, rebuild the
  mesh and constraint operator, *transfer the state and the BDF
  history*, continue.
- Explain how hanging-node constraints let the *same* CH stepper run
  unchanged on an adaptive mesh, and quote the measured savings: 1,173
  adaptive nodes vs 4,225 uniform nodes at the same finest `h`
  (3.6× fewer dofs).
- Show why *conservative* transfer (cell averaging) between mesh levels
  is exact while *naive* transfer (nodal injection) is not: on a field
  of sub-coarse-cell droplets, injection loses ≈20% of the mass while
  averaging conserves it exactly — and say why this is the
  correctness-critical piece of dynamic AMR's transfer step.
- Build a *real* (not fictional) cost accounting for adaptive
  time-stepping — accepted/rejected steps, full+half Newton solves (each
  accepted step costs three solves), Newton iterations, wall time — and
  compare it against a matched-accuracy fixed-`dt` sweep to get an
  honest speed-up (measured ≈1.85× Newton work and ≈1.79× wall time to
  `t=0.6`), not the fictional "horizon / smallest-step" ratio.
- Explain why a *growing* `dt` forces BDF2 into its variable-coefficient
  form, and reproduce both measured regimes on the same alternating-`dt`
  sequence: variable-coefficient order ≈2.01 (preserved) vs a forced
  constant-coefficient (`r≡1`) order ≈0.93 (collapsed toward 1).
- State why the LTE step-doubling controller and the
  variable-coefficient BDF2 are a matched pair — adaptivity is worthless
  if it silently destroys the order verified in Chapter C1.

## Prerequisites

- **Concepts:** local-truncation-error estimation by step-doubling;
  constant- vs variable-step BDF coefficients; conservative vs
  non-conservative restriction between mesh levels; counting Newton
  work honestly (iterations, solves, wall time) instead of asserting a
  speed-up.
- **Chapters:** C1 (`01_convergence` — the BDF1/BDF2 order this chapter
  must preserve under a varying step); C2 (`02_boundary_conditions` —
  the constraint operator `T` and `A = T^T K T` that hanging-node
  refinement reuses unchanged); Physics P2 (*Adaptive time stepping* —
  the error-controlled step-doubling idea this chapter applies to the
  octree/BDF2 setting). Chapter 00 (`00_setup_and_smoke_test`,
  environment green) is always assumed.

## Expected cost

- **Device:** any CUDA GPU; the meshes here are 2-D (max octree level 6,
  4,225 nodes uniform at most) and use well under 1 GB of device memory,
  so an 8 GB laptop card is ample. Solver `splu` (the CH saddle is
  indefinite, as in C1/C2/P1/P2 — cuDSS is not used here).
- **Live runs are comparable to physics P1's quick mode** (`≈ 75 s` wall
  measured on an RTX 6000 Ada, mostly Warp kernel compile + Python
  start-up — see `p1.tex`). Within `run.py`'s own studies, the two
  march-heavy pieces have measured wall times in `EXPECTED.md`: the
  adaptive ladder to `t=0.6` takes `~60 s` and the matched-accuracy
  fixed-`dt` sweep takes `~108 s`; the octree/hanging-node and
  conservative-transfer studies are comparatively quick.
- First run of a session pays a one-time Warp kernel-compile cost before
  any of the above.

## Required deliverable

Submit the eight-item report of `../../ASSESSMENT.md`, with the
chapter-specific content:

1. The `config.resolved.yaml` + `metadata.json` from a reference run
   (or the run's console log if the harness has no separate config for
   this driver-style chapter).
2. The eight `[PASS]` checks from `run.py` green (`ALL CHECKS: PASS`),
   or a documented deviation.
3. The four figures (`c3_octree`, `c3_transfer`, `c3_ladder`,
   `c3_cost`) regenerated via `gen_figures.py` from your own run.
4. **Headline (three parts, all measured here):**
   (a) the conservative-transfer error — injection ≈20% mass loss vs
   averaging exact (0) on sub-coarse-cell droplets;
   (b) the *real* adaptive cost accounting to `t=0.6` — accepted/
   rejected steps, full+half solves, Newton iterations, wall time,
   final error, against the matched fixed-`dt` run, with the honest
   ≈1.85×/≈1.79× (Newton/wall) saving stated as *modest*, not the
   fictional horizon/min-dt ratio;
   (c) the measured BDF2 order under a varying step: variable-
   coefficient ≈2.01 vs forced constant-coefficient ≈0.93.
5. **Verification:** reproduce the conservative-transfer radius sweep
   (`transfer_error_sweep`) and confirm injection error shrinks toward 0
   as the droplet is fully resolved while averaging stays exact at every
   size; *and* reproduce the variable-vs-constant BDF2 order comparison
   on the alternating-`dt` sequence.
6. **Failure:** force the constant-coefficient bug (`_const_march`, `r`
   forced to 1) and report the collapsed order, or drive the octree
   refinement predicate to under-resolve the interface and show the
   node savings and step outcome degrade.
7. **Exploration:** answer one "Explore on your own" question from
   `c3.tex` with a plot (the transfer-error radius sweep or the
   tightened-LTE-tolerance cost comparison are the recommended ones).
8. **Research bridge:** one paragraph on what the conservative-transfer
   result and the measured adaptive-cost saving predict for a real
   dynamic-AMR implementation on a production CH morphology run.
