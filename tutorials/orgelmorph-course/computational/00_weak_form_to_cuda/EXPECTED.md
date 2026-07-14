# C0 — expected results (self-check)

Running `run.py --mode reference` should reproduce the following to a couple of
significant figures. Small differences from a different card or BLAS are normal;
the **observed convergence orders** and the **Jacobian-consistency contrast**
are the load-bearing quantities. Six `[PASS]` gate lines and `ALL GATES: PASS`
must print, and the baseline check must pass.

## 1. Spatial MMS convergence (error ~ h^(p+1))

Manufactured solution `u* = sin(πx) sin(πy)`, source `f = 2π²u* + α u*³`,
`α = 12`. Newton converges in 6 iterations at every level.

| degree | levels | L2 error coarse → fine | observed order | expected |
|---|---|---|---|---|
| p=1 | 3,4,5,6 | 3.3e-3 → 5.1e-5 | 2.01 | 2 |
| p=2 | 2,3,4,5 | 1.6e-3 → 3.2e-6 | 2.98 | 3 |

## 2. Newton convergence (correct analytic Jacobian)

Residual norm `4.2e-1 → 8.9e-13` in 6 iterations — quadratic (the residual is
squared each step once the tangent is exact).

## 3. Jacobian verification (the hands-on red→green gate)

Analytic Jacobian vs a finite difference of the residual, relative error:

| module | reaction Jacobian | analytic-vs-FD | test |
|---|---|---|---|
| `scalar_reaction` (completed) | present (`3αu²`) | **7.9e-9** | PASS |
| `scalar_reaction_starter` | **missing** | **4.9e-3** | FAIL |

The threshold is `1e-4`; the two states are separated by ~6 orders of
magnitude, so the gate is deterministic and hardware-independent.

## 4. Production cross-check

The transparent NumPy stiffness equals `assemble_brick_csr(dm, PoissonBrick())`
(the production Warp-kernel assembly through `Tᵀ K T`) to **`max|ΔK| = 0.0`** —
the readable path *is* the device math.

## What must be true regardless of hardware

- p=1 order in `[1.7, 2.3]`, p=2 order in `[2.6, 3.4]`.
- Completed Jacobian matches FD (`< 1e-6`); starter does not (`> 1e-4`).
- Transparent stiffness equals the production kernel (`< 1e-10`).
- Newton converges (`≤ 10` iterations).
