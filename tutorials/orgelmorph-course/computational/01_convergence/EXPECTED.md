# C1 — expected results (self-check)

Running `python run.py` (fixed problem, seeds where relevant) should
reproduce the following to a couple of significant figures. Small
differences from a different card or BLAS are normal; the **observed
orders** are the load-bearing quantities and must land in the stated
bands.

## Spatial convergence (manufactured solution)

| degree | levels | $L^2$ errors (coarse → fine) | observed order | expected |
|---|---|---|---|---|
| $p=1$ | 4, 5, 6 | 2.9e-3, 7.1e-4, 1.8e-4 | 2.00, 2.01 | $h^{2}$ |
| $p=2$ | 3, 4 | 2.0e-4, 2.5e-5 | 3.00 | $h^{3}$ |

The error falls like $h^{p+1}$: doubling the resolution cuts the $p=1$
error by $\approx 4$ and the $p=2$ error by $\approx 8$.

## Temporal convergence (fixed mesh, shrinking $\Delta t$)

| scheme | $\Delta t$ = 8e-3, 4e-3, 2e-3 | observed order | expected |
|---|---|---|---|
| BDF1 (backward Euler) | 1.1e-3, 5.7e-4, 2.9e-4 | 0.97, 0.99 | $\Delta t^{1}$ |
| BDF2 | 4.5e-5, 9.8e-6, 2.3e-6 | 2.18, 2.07 | $\Delta t^{2}$ |

At the same $\Delta t$, BDF2 is roughly **25× more accurate** than BDF1
here, and the gap widens as $\Delta t$ shrinks (one extra order).

**What must be true regardless of hardware:**

- **The measured order matches the theory.** $p=1$ gives order $\approx
  2$, $p=2$ gives $\approx 3$ (order $= p+1$). BDF1 gives $\approx 1$,
  BDF2 gives $\approx 2$. A measured order well below the expected value
  means a bug (missing term in the weak form, a boundary applied wrong,
  or a source that does not match the manufactured field).
- **MMS error is discretization error only.** Because the source terms
  are built from the exact field, everything except the discretization
  itself cancels; the error you see is the method's, nothing else.
- **The `ALL GATES: PASS` line prints.** It asserts p1 order > 1.8, p2
  order > 2.6, BDF1 in [0.8, 1.3], BDF2 > 1.7 — the same thresholds the
  shipping gates in `tests/test_cahn_hilliard.py` use.

If an order comes out low, do **not** just take a finer mesh — a
first-order bug does not become second order by refining. Re-derive the
source term and re-check the boundary treatment.
