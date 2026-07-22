# Research track — pressure-projection + SBM in 3-D

**Type:** deferred research item (not on the R2 critical path).
**Opened:** 2026-07-22, out of P2-R2a (see
`2026-07-22-p2-r2a-projection-3d-findings.md`).
**Why it matters:** the projection split (SPD pressure-Poisson → AMGX) is the
intended **100M-DOF scalability path**. The monolithic saddle solve is the R2
3-D engine, but it does not scale to the hero meshes. Getting projection+SBM
faithful in 3-D is what unblocks the largest heroes on the projection path.

## The problem, precisely

On a coarse-mesh immersed-boundary sphere, our VMS-incremental-projection +
volumetric-SBM stepper converges to a **weak, wrong steady state**: the
momentum predictor never builds enough pressure to drive the flow around the
body, and the split settles into a self-consistent low-energy fixed point. The
SBM block is proven correct (the monolithic uses the identical block and gives
the right drag). The pressure gauge is correct (outflow Dirichlet). The defect
is in the **split's pressure–velocity coupling**, worst at low Re, present at
Re=100.

## Hypotheses to test (roughly in order)

1. **Steady-state consistency.** Measure `‖u_proj^∞ − u_mono^∞‖` vs dt and vs
   mesh. A consistent incremental scheme should share the monolithic fixed
   point; if the error does not vanish as dt→0, the split is inconsistent on
   the immersed problem (not just slow).
2. **Coupled / Uzawa predictor.** Add an inner velocity↔pressure sub-iteration
   (or a few Uzawa sweeps) per step so the predictor sees a
   nearly-simultaneous pressure. Cheapest structural fix to try; directly
   attacks the weak fixed point.
3. **Full-pressure re-solve (bkhara pattern).** Re-solve the *full* pressure
   each step with 1st-order pressure extrapolation (`p*=p^n`) instead of
   accumulating `p*+=φ`, matching the working body-fitted projection reference
   (`bkhara ns_vms`, `Proj_NL_PPE`). Pair with the residual-based VMS
   fine-scale threaded consistently through momentum/PPE/velocity-update.
4. **Predictor no-penetration on the surrogate.** Verify the predictor actually
   enforces `ñ·ũ≈0` on the 3-D staircase surrogate; if not, the homogeneous-
   Neumann PPE BC has nothing to preserve. Consider a Nitsche normal-penalty or
   the monolithic continuity-row coupling in the PPE.
5. **Re-sweep.** Characterise the deficit at Re ∈ {1, 20, 100, 300} to separate
   the (expected) splitting-error component from any residual structural defect,
   and to find the Re band where the split is already usable.

## Assets ready to reuse

- Diagnostics on branch `p2-r2a`: `p2r2a_field_compare.py` (field + drag
  decomposition), `p2r2a_velocity_deficit_probe.py` (predictor vs projection,
  interior samples, `RE`/`SBM_PCOUPLE`/`STEPS` env toggles),
  `p2r2a_bakeoff_3d.py` (lever sweep).
- Flagged, off-by-default knobs on the stepper: `pressure_update`
  (standard/rotational/chorin), `ppe_fine_scale`, `pressure_outflow_nodes`,
  `sbm_pressure_coupling`. All verified bit-for-bit no-ops when off.
- References in `local_code_old/`: `ns_projection_vms_paper.pdf`,
  `Suresh_Pressure_Projection_Octree_SBM_Moving_Rigid_Body.pdf`, and the two
  C++ codes (bkhara body-fitted projection, chenghauy monolithic SBM).

## Definition of done

The projection stepper reproduces the monolithic same-mesh 3-D sphere Cd within
the R0 faithfulness tolerance across Re ∈ {20, 100}, weak-div decaying, on the
level-4→5 meshes — then the R2b device port (SPD-PPE → AMGX) can proceed on the
projection path for the hero scales.
