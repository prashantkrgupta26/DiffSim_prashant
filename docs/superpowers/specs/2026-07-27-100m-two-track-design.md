# 100M Engine — Two-Track Design (Track A: R2b saddle / Track B: projection rescue)

**Date:** 2026-07-27. **Approved:** Baskar ("two-track as laid out — proceed").
**Context:** `docs/dev/2026-07-27-100m-gh200-readiness.md` (post-WP0: mesh build
exonerated; the solver is the 100M critical path). The 100M engine = whichever
track clears its gate first; the monolithic remains the sub-hero oracle
regardless of outcome.

---

## Track A — R2b saddle-preconditioner campaign (monolithic path)

**Question:** does a scalable iterative solve exist for the VMS-stabilized
(u,p) saddle with the two-sided Nitsche shell block, at 1→10M DOF, with
near-mesh-independent iteration counts?

**Known ground:** BlockAMG cliff-fails at 143k DOF (R2b handoff §0/§4, no
tuning response); AMGX diverges on the raw saddle (valid only as an inner
W-factor); unpreconditioned `fused` BiCGStab converged on one L6 3-step probe
but diverged elsewhere — fragile; `blockch_dev` two-factor Schur proven
202k→10.6M on CH pairs (NOT the NS saddle); matrix-free J·v proven to 84.5M
(film operators).

**Candidates to evaluate (in order):**
1. **blockch-style two-factor Schur adapted to (u,p):** velocity block M+K+C
   (σ-dominated at transient dt — AMG/Jacobi-friendly) + pressure Schur
   approximated by the pressure mass/stiffness (the PCD/LSC family — pick ONE:
   PCD first, it matches the σ-dominated regime). Reuses the blockch outer
   machinery.
2. **Field-split FGMRES:** `fgmres_dev` outer, velocity-block inner AMG(X)
   (valid — the velocity block is NOT a saddle), Schur inner CG on the
   approximate complement.
3. **fused BiCGStab + simple block-diagonal preconditioner** — the cheap
   baseline that quantifies what preconditioning buys.

**Campaign:** iteration-count ladder at fixed physics (the corrected Re=250
2-D config, then 3-D thin-plate) across 143k → 1M → 3M → 10M DOF, per
candidate; measure iterations/step, s/step, memory. All runs monolithic march,
existing drivers + solver routing knobs.

**KILL-GATE (time-boxed, binding):** if no candidate demonstrates
iteration growth ≤ ~2× across the 1M→10M decade by the end of the campaign
(budget: ONE plan of ≤ 6 tasks incl. implementation of candidates 1–2),
Track A reports negative-with-measurements and the 100M engine defaults to
Track B. A negative result is a deliverable, not a failure.

---

## Track B — projection rescue (PP path)

**Question:** can the projection split be made stable AND fixed-point-faithful
for thin-shell at production Re/mesh, so its decisively-scalable sub-solves
(σ-dominated momentum + SPD PPE with NCCL-CG) carry the 100M hero?

**Known ground:** 2-D Re=250/L9 structural divergence (both configs/assemblies;
O6 P0); 3-D thin-plate ~40% Cd gap (lagged-p* weak fixed point; ladder-proven
formulation-faithful, so efficiency not correctness); the consistent-projection
fix-set (F1/F2/F3b/FN pins) exists but the 2-D thin-shell path may lack parts
of the 3-D p′-outflow scheme (O6's suspect); the reaction arbiter + term
decomposition now exist as instruments.

**Workpackages (in order):**
1. **p′-scheme completeness audit + port:** diff the 2-D projection path
   against the 3-D `LeraySBMShellStepper` p′-outflow fixes (whole-outflow-line
   pin, consistent_projection internals, rotational pins); port what is
   missing; gate = the O6 P0 config (corrected units, Re=250/L9) runs BOUNDED
   for 1000 steps.
2. **Stability envelope:** if WP1 alone doesn't bound it, the O6 knob triage
   (inner_relax, inner_max, dt, consistent_projection ablation) with
   early-exit-on-divergence probes (the O6 lesson).
3. **Fixed-point gap attack:** coupled/Uzawa predictor sub-iteration (the
   documented candidate) measured by Cd_reaction vs the same-mesh monolithic
   (the arbiter transfers verbatim); gate = 3-D thin-plate gap ≤ 10% at the
   smoke scale, THEN the 2-D Re=250 shedding benchmark reproduced within
   the monolithic's own band.

**GATE to declare Track B the engine:** stable Re=250 2-D shedding + 3-D
thin-plate Cd_reaction within 10% of monolithic + a demonstration solve at
≥ 10M DOF using gpu_cg/NCCL-PPE + iterative momentum (no direct solves).

---

## Shared rules

- One implementer at a time (tracks interleave at task granularity); GPU runs
  serialize on the box; nova available for SPD/NCCL legs.
- Every stability/iteration claim measured, ledgered, and runbook-recorded —
  negative results first-class.
- Monolithic oracle comparisons at every overlapping scale.
- Neither track modifies default solver paths; everything behind knobs.

## Out of scope (both tracks)

Multi-node beyond the proven 2-GPU NCCL PPE; adaptive-dt for NS; the
streaming mesh build (independent thread); 3-D hero production runs
(the engine decision precedes them).
