# M3 — Differentiable Dynamic Adaptivity (RATIFIED 2026-07-07: M2 -> M3, before THB)

RATIFIED by Baskar 2026-07-07 (placement: after M2, before THB/M7).
The adjoint-readiness MEMO below is HELD for Baskar's review before it
goes to the cuFEM team. Proposed 2026-07-06 night (conversation: cuFEM performs dynamic
adaptivity pure-device but has NO differentiability — differentiating
THROUGH adaptivity is DiffSim's contribution). Recommended slot: after
the physics M2, before THB (M7). M2's optimization demos will generate
the demand (any shape optimization that moves geometry beyond one
epoch's trust region re-carves); THB multiplies mesh-structure
complexity and must not be designed before this capability exists.

## Empirical foundation (already measured, M1c)

- Cross-epoch objective jumps are REAL and large (J=183 barrier measured
  mid-recovery-path); within-epoch objectives are smooth.
- Epoch trust region: |geo.d|_max > 0.35 h => re-carve (installed).
- Foot continuation (anchored per-epoch warm starts) = branch selection
  made deterministic + continuous; the IFT differentiates the selected
  branch. The mesh-level analogue is the core of rung 2.
- 4f: probe objectives over wrinkly INRs carry steep sub-edit-scale
  structure (landscape dataset in progress) — relevant to rung 3's
  landscape-smoothing motivation.

## The three rungs

### Rung 1 — Transfer-operator adjoints (engineering)
At every adaptivity event, u_new = P(theta) u_old with P an EXPLICIT
sparse interpolation operator. Reverse sweep: P^T; shape gradients:
dP/dtheta via the geometry dependence of interpolation weights (same
face/GP machinery as the SBM shape gradients). Checkpointing
generalizes: store (mesh epoch, field) at event boundaries; reverse
replay reconstructs epochs in reverse order.
Gates: transient adjoint across a SCHEDULED remesh (fixed event times,
fixed new meshes) exact vs FD at the H2 tolerance class (1e-8).

### Rung 2 — Branch differentiation across events (engineering + rules)
Adaptivity decisions (refine/coarsen/retain) are discrete; the gradient
exists on each branch. Machinery: event trust regions (the epoch rule,
generalized), deterministic tie-breaks, branch-stability preconditions
for FD verification (the 4e lesson: skip verification directions that
cross a branch, require >= 2 verified). Optimization protocol:
epoch-homotopy (re-carve, re-anchor, continue — the tutorial's DRIFT
management, promoted to the optimizer loop).
Gates: recovery problem whose alpha* lies BEYOND one trust region,
solved by homotopy; gradient exactness on each visited branch.

### Rung 3 — Relaxed classification (research fork = N6)
Smooth the discrete decisions themselves (smoothed-Heaviside
classification thresholds; blended transfer operators) so optimization
landscapes crossing many events become C^0/C^1. Ties to the 4f
landscape question. Deliverable: a study, not a production default.

## Adjoint-readiness requirements for cuFEM device adaptivity (MEMO)

To be adopted by cuFEM BEFORE its adaptivity design freezes;
zero-performance-cost provisions that keep differentiability possible:

1. **Explicit transfer operators.** Every remesh/rebalance/repartition
   produces P as a materialized sparse object (or slot map + weights),
   never fused-and-discarded interpolation. P^T must be applicable.
2. **Event logs with deterministic replay.** Record per event: marked
   elements, decision outcome, tie-break seed, resulting mesh hash. A
   reverse sweep must reconstruct the exact epoch sequence.
3. **Deterministic, seedable tie-breaking** in refinement marking and
   partitioning (no atomics-order-dependent decisions feeding marks).
4. **Classification/marking thresholds as runtime PARAMETERS** (not
   compile-time constants) — rung 3's relaxation handle, and free.
5. **Field transfer at events routed through one API** (the P apply) —
   one instrumentation point for the tape.

## Cost estimate

Rung 1: one milestone-sized engineering push (transfer ops exist in
h-adaptive FEM; the adjoint is mechanical given requirement 1).
Rung 2: mostly protocol + gates on existing machinery. Rung 3: research
(student-scale, N6).
