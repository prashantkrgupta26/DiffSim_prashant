# Selected full solutions — C2 verification exercises

*Full worked solutions for the two **verification** exercises only (the
`g=0` mass-conservation coincidence, and the penalty `beta` conditioning
crossover). Hints for all questions are in `hints.md`. Instructor-only —
do not distribute before the deadline.*

---

## V1 — The `g=0` Dirichlet wall nearly conserves mass, and it is a
coincidence of the initial condition, not the boundary condition (Q1)

**Claim.** In the BC test matrix, a Dirichlet wall pinned at `c=0.0`
shows a mass drift almost an order of magnitude smaller than walls
pinned at `c=+0.9` or `c=-0.9`, but this is *not* evidence that
Dirichlet BCs can conserve mass — it is a coincidence of the reference
run's initial condition, and the flux balance shows why.

**Procedure.**
1. Run `bc_test_matrix()` (or `python run.py`, section 3) and read off
   the mass drift for the three prescribed-composition rows: `c=+0.9`,
   `c=-0.9`, `c=0.0` (all-edges), from `EXPECTED.md` §3.
2. Note the initial condition used by every run in this chapter
   (`bc.py`, `run_bc`/`_run_generic`): `st.set_initial(lambda x: 0.05 *
   rng.standard_normal(len(x)), ...)` — a small-amplitude perturbation
   around **mean zero**.
3. Compare each wall value `g` to that IC mean (`≈0`): the smaller
   `|g - mean|`, the less material must flow through the boundary to
   pull the interior toward `g`, so the *measured* drift at fixed step
   count is smaller — not because the BC conserves, but because less
   time has been given to a slower process.
4. Confirm with the flux balance (`flux_balance`, EXPECTED.md §2):
   under Dirichlet the influx is large early and only *decays* as the
   interior equilibrates — it does not go to exactly zero within the 60
   steps of the reference run for any `g != 0`-adjacent case; for `g`
   close to the IC mean the influx is simply smaller throughout.

**Expected result.** From `EXPECTED.md` §3 (BC test matrix, 128 pinned
nodes, 60 steps): `c=+0.9` → drift `0.954`; `c=-0.9` → drift `0.943`;
`c=0.0` → drift `0.094`. The `c=0.0` drift is about 10x smaller than the
`|g|=0.9` cases — but it is not zero, and it is not qualitatively
different in kind: all three are the *same* non-conserving mechanism (a
reservoir wall), just with different pumping rates because the wall
value is different distances from the interior's starting mean. The
natural (no-flux) row, by contrast, gives `1.3e-16` — sixteen orders of
magnitude smaller, at floating-point noise, which is the qualitatively
different (structurally conserving) case.

**Common wrong conclusion.** "The `c=0.0` Dirichlet run conserves mass
(drift only 0.094, an order of magnitude down), so a well-chosen
Dirichlet value can conserve mass just as well as natural BCs." No — a
drift of `0.094` is still ten orders of magnitude larger than the
no-flux drift of `1.3e-16`; the two are not in the same regime. The
`g=0` case is smaller only because the reference run's initial mean
composition happens to sit near zero, so there is little pumping to do
in 60 steps — change the IC's mean composition (or the wall value, or
run longer) and the `c=0.0` drift grows just like the others. Mass
conservation under Dirichlet is never exact; only natural and wall-energy
BCs are exact by construction (the boundary integral is omitted / made
to vanish identically), which is a *structural* property, not a
numerically-small one.

---

## V2 — The penalty method's exactness-vs-conditioning crossover (Q4)

**Claim.** The weakly-imposed (penalty) Dirichlet value converges to the
strong (row-replace) value as `1/beta` as `beta -> infinity` — but only
up to a point. Beyond some `beta`, the diagonal penalty term dominates
the O(1/h) stiffness entries so completely that floating-point roundoff
degrades the solve, and the boundary error stops improving (or grows).

**Procedure.**
1. `strong_vs_weak_dirichlet()` sweeps `betas=(1e0, 1e1, ..., 1e6)` by
   default (`bc.py`) and reports `bc_err` (the boundary-value error) and
   `sol_err` (the max error against the exact linear solution) at each.
2. Read the trend already measured in the reference run
   (`EXPECTED.md` §4): boundary error falls from `6.7e-1` at `beta=1e0`
   to `2.0e-6` at `beta=1e6` — a clean `1/beta` line over six decades,
   consistent with `dirichlet_penalty`'s construction
   (`A[i,i] += beta; r[i] += beta*g`, so the imposed value approaches
   `g` as `1/beta` while the rest of the row is unaffected).
3. To find the crossover the chapter's Explore-on-your-own question asks
   for, extend the `betas` tuple well past `1e6` — e.g. `1e7` through
   `1e14`/`1e16` in decade steps — and re-run `strong_vs_weak_dirichlet`
   with the extended sweep. Track `bc_err` at each `beta`, and (for a
   sharper diagnostic) `np.linalg.cond(A_p)` on the assembled penalty
   matrix.
4. Watch for the point where `bc_err` stops decreasing (the `1/beta`
   line flattens or turns upward) rather than continuing to fall.

**Expected result.** The `1/beta` trend continues cleanly through the
range the reference run reports (`beta = 1e0` to `1e6`, error `6.7e-1`
down to `2.0e-6`, `EXPECTED.md` §4) — no sign of a problem yet at `1e6`.
The mechanism for the eventual breakdown is a standard floating-point
one, not a chapter-specific number: `dirichlet_penalty` adds `beta` to a
single diagonal entry of a matrix whose other entries are O(1/h) (order
one on this tiny `n=8` mesh). In IEEE double precision (~16 significant
decimal digits), once `beta` is large enough that the *other* row/column
entries become smaller than the diagonal entry by more than about
`1e16`, they are no longer resolved in the assembled matrix at working
precision, and `np.linalg.solve` starts to see round-off noise instead of
the true coupling — the boundary error stops shrinking as `1/beta` and
the solution error away from the boundary can start to grow. Students
extending the sweep themselves should look for this flattening/growth
somewhere in the very-large-`beta` regime (many decades past `1e6`) and
report the `beta` at which it happens on their own hardware/BLAS,
since the reference run does not itself probe past `1e6` — the crossover
value is not a fixed chapter number to quote but a floating-point-limited
threshold to *measure*.

**Common wrong conclusion.** "Larger `beta` is always better, so use the
largest `beta` you can." No — past the conditioning crossover, a larger
`beta` degrades rather than improves the solve; the practically useful
range is bounded above as well as below (too small and the BC is barely
imposed; too large and roundoff swamps the row). The chapter's point is
the *trade*: penalty buys you a symmetric matrix and no destroyed row
(useful on hanging-node / immersed boundaries) at the cost of both
approximate imposition *and* an upper bound on how well-conditioned the
resulting system can be — it is never a free substitute for strong
imposition when a clean row is available.
