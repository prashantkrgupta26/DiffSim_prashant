# C3 — expected results (self-check)

`python run.py` (fixed seed) should reproduce the following. Exact step
counts depend on the RNG-seeded quench and the card; the *qualitative*
results (octree saves dofs, conservative transfer is exact, adaptivity's
real cost, variable-coefficient BDF2 keeps order 2) must hold. Eight
`[PASS]` lines and `ALL CHECKS: PASS` must print.

## 1. Octree refinement + hanging-node constraints

| quantity | adaptive (static circle) | uniform L6 |
|---|---|---|
| nodes | 1,173 | 4,225 |
| elements | 1,024 | 4,096 |
| — | **3.6× fewer nodes** | (same finest h) |

The CH brick steps on the hanging-node mesh (`step_ok = True`). **This is
octree refinement to a *static* geometric criterion — NOT solution-adaptive
AMR** (a Phase-3 deliverable).

## 2. Conservative-transfer error (a piece of true AMR)

Restricting a fine field of sub-cell droplets to a 2× coarser grid:

| transfer | mass error |
|---|---|
| injection (naive) | ≈ 20% (droplets missed) |
| cell averaging (conservative) | 0 (exact) |

Dynamic AMR must use a conservative restriction or mass leaks every time a
region coarsens.

## 3. Temporal adaptivity — REAL cost accounting

The dt ladder over a quench to t=0.8 spans **1.0e-5 → 6.6e-2** (≈ 6600×).
Real cost to t=0.6 (not the fictional horizon/min-dt ratio):

| run | steps | Newton its | wall | final error |
|---|---|---|---|---|
| adaptive (tol 5e-4) | 138 acc, 31 rej (169 full + 338 half solves) | 2,373 | ~60 s | 2.4e-2 |
| matched fixed dt=5e-4 | 1,200 | 4,380 | ~108 s | 5.2e-3 |

**Adaptivity saves ≈ 1.85× Newton work and 1.79× wall here** — real but
*modest*, and it grows with the horizon (the step-doubling overhead of 3
solves per accepted step is amortized only once the coarsening tail
dominates). The old "≈ 563×" was a fiction (horizon ÷ smallest step).

## 4. Variable- vs constant-coefficient BDF2 (both MEASURED here)

| BDF2 form on varying dt | observed order |
|---|---|
| variable-coefficient (brick) | **2.01** (preserved) |
| constant-coefficient (forced r=1, tutorial-local) | **0.93** (collapses toward 1) |

The constant-coefficient degradation is measured locally by a
`_const_march` that forces the coefficient ratio r=1 on a varying history —
no dependence on an inaccessible dev note.

**What must be true regardless of hardware:**

- **Octree saves dofs** (node count deterministic, matches exactly) and the
  solver runs on the hanging-node mesh.
- **Conservative transfer is exact**; naive injection loses sub-cell mass.
- **Adaptivity's cost is REAL and modest** — a ~1.8× saving at t=0.6, not a
  500× fantasy. Accounting = accepted/rejected steps, full+half solves,
  Newton iterations, wall.
- **Variable-coefficient BDF2 keeps order ≈ 2** under a varying step, while
  the constant-coefficient form collapses toward 1 — both measured here.
