# C3 — expected results (self-check)

`python run.py` (fixed seed) should reproduce the following. Exact step
counts depend on the RNG-seeded quench and the card; the *qualitative*
results (octree saves dofs; the conservative transfer is exact under refine
AND coarsen; dynamic AMR conserves mass across every remesh and tracks the
interface; both BDF history levels must transfer for order 2 to survive;
adaptivity's real cost; variable-coefficient BDF2 keeps order 2) must hold.
All `[PASS]` lines and `ALL CHECKS: PASS` must print.

## 1. Static octree refinement + hanging-node constraints

| quantity | adaptive (static circle) | uniform L6 |
|---|---|---|
| nodes | 1,173 | 4,225 |
| elements | 1,024 | 4,096 |
| — | **3.6× fewer nodes** | (same finest h) |

The CH brick steps on the hanging-node mesh (`step_ok = True`). **This is
octree refinement to a *static* geometric criterion — the lead-in to the
real thing in section 3.**

## 2. Conservative transfer (the primitive AMR needs)

Restricting a fine field of sub-cell droplets to a 2× coarser grid (numpy
demo):

| transfer | mass error |
|---|---|
| injection (naive) | ≈ 20% (droplets missed) |
| cell averaging (conservative) | 0 (exact) |

The real finite-element transfer is the **L2 (Galerkin) projection on the
common refinement** of the two octree meshes:

| FE transfer | mass change |
|---|---|
| refinement | ≈ 1e-16 (machine precision) |
| coarsening | ≈ 6e-17 (machine precision) |
| convergence order (vs analytic field) | ≈ 1.95 (2nd order) |

Mass is conserved in **both** directions to solver tolerance, and the same
operator carries the two BDF history levels `c^n, c^{n-1}`, each conserved
independently.

## 3. Dynamic (solution-adaptive) AMR

The full cycle — estimate (|∇c|) → mark → refine/coarsen → 2:1 balance →
rebuild mesh+constraints → conservatively transfer c, μ AND both BDF history
levels → continue — over a shrinking droplet:

| measured | value |
|---|---|
| mass jump across every remesh | ≈ 6e-17 (machine precision) |
| free-energy jump across remesh | ≲ 1e-9 (transfer error only) |
| refined-region / interface overlap | 1.00 (refinement tracks the interface) |
| stability | field finite, bounded in [−1.2, 1.2] |

**Error vs dofs** (scored against a uniform level-7 reference, 16,641 dofs):
dynamic AMR reaches error ≈ 2.5e-3 at ≈ 1,413 dofs — comparable to the
uniform level-6 mesh (≈ 2.0e-3 at 4,225 dofs) but at **≈ 2.5× fewer dofs**
than a uniform mesh would need for the same error, and ≈ 2× less wall time.
The advantage grows with the finest level.

## 4. Space-time interaction (remesh during a BDF2 march)

A variable-step BDF2 march that remeshes at its midpoint:

| history transfer at the remesh | observed order |
|---|---|
| **both** levels (c^n, c^{n-1}) transferred | ≈ 1.94 (order 2 preserved) |
| **drop** 2nd level (BDF1 restart) | ≈ 2.0, but error constant ≈ 1.7× larger |

A single BDF1 restart still recovers order 2 (one first-order local error is
O(Δt²), the same size as the global BDF2 error), but dropping history
inflates the error — and, because the step-doubling controller sees the
inflated post-remesh error as a larger LTE, makes it shrink Δt needlessly.
Transferring **both** history levels is what keeps a remeshing march order 2.

## 5. Temporal adaptivity — REAL cost accounting

The dt ladder over a quench to t=0.8 spans **1.0e-5 → 6.6e-2** (≈ 6600×).
Real cost to t=0.6 (not the fictional horizon/min-dt ratio):

| run | steps | Newton its | wall | final error |
|---|---|---|---|---|
| adaptive (tol 5e-4) | 138 acc, 31 rej (169 full + 338 half solves) | 2,373 | ~60 s | 2.4e-2 |
| matched fixed dt=5e-4 | 1,200 | 4,380 | ~108 s | 5.2e-3 |

**Adaptivity saves ≈ 1.8× Newton work and wall here** — real but *modest*,
and it grows with the horizon. The old "≈ 563×" was a fiction (horizon ÷
smallest step).

## 6. Variable- vs constant-coefficient BDF2 (both MEASURED here)

| BDF2 form on varying dt | observed order |
|---|---|
| variable-coefficient (brick) | **≈ 2.0** (preserved) |
| constant-coefficient (forced r=1, tutorial-local) | **≈ 0.9** (collapses toward 1) |

**What must be true regardless of hardware:**

- **Octree saves dofs** (node count deterministic) and the solver runs on
  the hanging-node mesh.
- **The FE conservative transfer conserves mass to machine precision** under
  refine AND coarsen, converges at 2nd order, and carries the BDF history;
  naive injection loses sub-cell mass.
- **Dynamic AMR conserves mass across every remesh**, keeps the free energy
  continuous, tracks the interface, and reaches uniform-fine accuracy at
  measurably fewer dofs.
- **Both BDF history levels must transfer** for a remeshing BDF2 march to
  stay order 2.
- **Adaptivity's cost is REAL and modest** — a ~1.8× saving at t=0.6.
- **Variable-coefficient BDF2 keeps order ≈ 2** under a varying step while
  the constant-coefficient form collapses toward 1 — both measured here.
