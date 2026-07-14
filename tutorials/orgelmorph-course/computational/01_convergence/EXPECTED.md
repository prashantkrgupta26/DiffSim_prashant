# C1 — expected results (self-check)

Running `python run.py` should reproduce the following to a couple of
significant figures. Small differences from a different card or BLAS are
normal; the **observed orders** are the load-bearing quantities and must
land in the stated bands. Eight `[PASS]` lines and `ALL GATES: PASS`
must print.

## 1. Spatial convergence (steady manufactured solution)

Two error sources are **separated**: the manufactured field is *steady*,
so the measured error is pure spatial error, independent of `dt`. We
report `L2` and `H1` for **both** fields `c` and `mu`.

| degree | levels | L2(c) coarse → fine | order L2(c) / H1(c) / L2(μ) | expected |
|---|---|---|---|---|
| p=1 | 3,4,5,6 | 1.8e-2 → 2.8e-4 | 2.01 / 1.02 / 2.00 | 2 / 1 / 2 |
| p=2 | 2,3,4,5 | 1.7e-3 → 3.2e-6 | 3.03 / 2.00 / 2.99 | 3 / 2 / 3 |

`L2 ~ h^(p+1)`, `H1 seminorm ~ h^p`, for each field. The discrete mass
error converges at the same L2 rate.

## 2. Algebraic-error control (p=1, level 5)

Tightening the Newton tolerance from `1e-4` to `1e-12` must **not** move
the discretization error (the linear solve is direct `splu`, exact to
round-off):

| Newton tol | L2(c) | Newton its |
|---|---|---|
| 1e-4 | 1.114226e-3 | 1 |
| 1e-8 | 1.114226e-3 | 2 |
| 1e-12 | 1.114226e-3 | 2 |

The error moves by ≈ **3e-13 of itself** — the plot measures
discretization, not solver, error. A study that fails this check must be
fixed before any order is quoted.

## 3. Temporal convergence (fixed over-resolved mesh, verified reference)

| scheme | dt = 1.6e-2, 8e-3, 4e-3, 2e-3 | observed order | expected |
|---|---|---|---|
| BDF1 | 1.4e-3 … 1.8e-4 | 0.97 | dt^1 |
| BDF2 | 1.3e-4 … 1.3e-6 | 2.23 | dt^2 |

**Reference verified:** the BDF2 order is stable when the reference `dt`
is halved (2.06 → 2.04, Δ 0.02), and a Richardson estimate bounds the
reference's own error at ≈ 3e-9 — well below the coarsest sampled error.

## 4. Deliberate failures (diagnose these)

| failure | measured order | why |
|---|---|---|
| wrong BC (natural, not pinned) | 0.00 | error flat & huge; boundary dominates |
| wrong source (κ term dropped) | 0.00 | converges to the *wrong* steady state |
| under-resolved feature (k=6) | 1.18 | pre-asymptotic; mesh can't resolve it |
| under-resolved reference | −0.01 | error saturates at the reference's error |

**What must be true regardless of hardware:**

- **The measured order matches the theory** — L2 order `p+1`, H1 order
  `p`, for both fields; BDF1 ≈ 1, BDF2 ≈ 2. A low order means a bug
  (missing weak-form term, wrong BC, mismatched source), *not* a mesh
  that is too coarse.
- **Error sources are separated.** Steady MMS removes the time error from
  the spatial study; self-convergence on a fixed mesh removes the spatial
  error from the temporal study.
- **Algebraic error is controlled**, not assumed — the invariance check
  proves the plot is not solver-limited.
- **The reference is verified** before it is trusted.

If an order comes out low, do **not** just take a finer mesh — a
first-order bug does not become second order by refining. Re-derive the
source, re-check the boundary treatment, and confirm the algebraic and
reference checks pass first.
