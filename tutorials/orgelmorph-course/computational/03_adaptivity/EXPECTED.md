# C3 — expected results (self-check)

Running `python run.py` (fixed seed) should reproduce the following.
Exact step counts depend on the RNG-seeded quench and the card; the
*qualitative* results (octree saves dofs, dt grows through coarsening,
variable-coefficient BDF2 keeps order 2) must hold.

## 1. Spatial adaptivity (octree refinement)

| quantity | adaptive (refined at interface) | uniform L6 |
|---|---|---|
| nodes | 1,173 | 4,225 |
| elements | 1,024 | 4,096 |
| — | **3.6× fewer nodes** | (same finest $h$) |

The CH brick steps on the hanging-node adaptive mesh (`step_ok = True`):
`build_constraints` ties the hanging nodes, and the same stepper runs
unchanged.

## 2. Temporal adaptivity (LTE step ladder over a quench)

| quantity | value |
|---|---|
| adaptive steps to $t=0.8$ | 142 |
| $\Delta t$ range | 1.0e-5 (onset floor) → 6.6e-2 (coarsening), ≈ 6600× |
| equivalent fixed-$\Delta t$ steps | ≈ 80,000 |
| **step savings** | **≈ 563×** |

$\Delta t$ collapses to the floor during the violent spinodal onset,
then grows more than four orders of magnitude as the domains coarsen —
the property that makes long phase-field horizons affordable.

## 3. Why variable-coefficient BDF2

| BDF2 form on varying $\Delta t$ | observed order |
|---|---|
| variable-coefficient (current brick) | **2.01** (order preserved) |
| constant-coefficient (cited G3 baseline) | 0.90 / 0.95 (collapses toward 1) |

**What must be true regardless of hardware:**

- **Octree saves dofs** — the adaptive mesh has several times fewer nodes
  than the uniform mesh at the same finest resolution, and the solver
  runs on it. (Node count is deterministic; it will match exactly.)
- **$\Delta t$ grows through coarsening** — the shipping gate asserts
  > 4× growth; you should see far more (the onset floor is orders below
  the coarsening step). The march stays physical ($c\in[-1,1]$).
- **Variable-coefficient BDF2 keeps order ≈ 2** under a varying step.
  This is the load-bearing correctness result: the constant-coefficient
  form *measured* 0.90/0.95 on the same alternating-$\Delta t$ sequence
  (audit doc, G3). If your measured order drops toward 1, the scheme is
  using the wrong (constant-step) coefficients.
- **The `ALL CHECKS: PASS` line prints.**

If the octree does not save dofs, or the adaptive step does not grow, or
the variable-dt order is near 1, re-read the walkthrough.
