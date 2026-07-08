# Adjoint-Readiness Requirements for cuFEM Device Adaptivity
## (Standalone memo for the cuFEM team — extracted from the M3 spec; now with working proof)

DiffSim's differentiable-adaptivity milestone (M3) requires five
provisions from cuFEM's device-side adaptivity design. All are
zero-performance-cost if adopted BEFORE the design freezes; all are
nearly impossible to retrofit. Each is now backed by a WORKING,
GATED prototype in DiffSim (July 2026).

1. **Explicit transfer operators.** Every remesh/rebalance/repartition
   must produce the field-transfer P as a materialized sparse object
   (or slot map + weights) — never fused-and-discarded interpolation.
   P^T must be applicable.
   PROOF: DiffSim transfer_operator — the adjoint carried EXACTLY
   (1e-7-class vs FD) across a scheduled re-carve; and for NESTED
   refinement P is exact-conservative (measured ZERO mass drift in
   adaptive Cahn-Hilliard) — conservation comes free if P is explicit.

2. **Event logs with deterministic replay.** Record per adaptivity
   event: marked elements, decision outcome, tie-break seed, resulting
   mesh hash. A reverse (adjoint) sweep must reconstruct the exact
   epoch sequence.

3. **Deterministic, seedable tie-breaking** in refinement marking and
   partitioning (no atomics-order-dependent decisions feeding marks).

4. **Classification/marking thresholds as runtime parameters** (not
   compile-time constants) — the handle for relaxed/smoothed
   classification studies (research fork N6), and free.

5. **One API for field transfer at events** (the P apply) — a single
   instrumentation point for the differentiation tape.

Context for 1-2: optimization through moving geometry requires
epoch-continuation (optimize within a trust region, re-mesh at epoch
boundaries, transfer state via P, differentiate the selected branch).
DiffSim has this gated end-to-end: continuation recovers exactly where
plain Gauss-Newton stalls at 42% (beyond-trust-region gate), and the
mechanism has run production field-tests (neural-SDF shape recovery;
adaptive phase-field). The requirements above are exactly what made
those results possible in the prototype.
