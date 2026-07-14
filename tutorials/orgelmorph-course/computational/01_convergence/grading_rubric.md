# Grading rubric — C1 Convergence: basis order and time integration

*The course rubric (`../../ASSESSMENT.md`) specialised to this chapter's
headline result. Weights are the course standard; the "C1 specifics"
column says what to look for.*

| Component | Weight | C1 specifics — full marks |
|---|---:|---|
| **Scientific correctness & interpretation** | 30% | Correctly separates spatial/temporal/algebraic error and states which is measured where; reports `L2 ~ h^(p+1)`, `H1 ~ h^p` for **both** `p=1`/`p=2` and **both** fields `c`/`mu`; does **not** treat a tight Newton tolerance as evidence of correct discretization; explains why the manufactured field must be *steady*. |
| **Numerical verification** | 20% | Algebraic-control sweep shown flat (error moves ≈3e-13 of itself, `1e-4`→`1e-12`); temporal orders (BDF1≈1, BDF2≈2) reported *with* the reference itself verified (order stable across halved reference `dt`, Richardson bound on the reference's own error) — a temporal order without a verified reference is incomplete. |
| **Reproducibility** | 15% | `python run.py` reproduces the eight `[PASS]` lines (`ALL GATES: PASS`) within `EXPECTED.md`'s bands; all three figures regenerate from the same core via `gen_figures.py`, never hand-copied; device/command used is stated (this chapter has no `config.resolved.yaml`/`metadata.json` harness — the run's console output stands in for it). |
| **Software & CUDA fluency** | 10% | `splu` used on the indefinite `(c,mu)` saddle (not swapped for an unsuitable solver); sensible level/mode choice for the exercise; no silently stalled Newton (`dx_inf` checked, not just assumed converged). |
| **Failure diagnosis** | 10% | At least one deliberate failure reproduced and diagnosed by *mechanism*, not just symptom: wrong BC (boundary mismatch dominates, order≈0), wrong source (converges to the wrong steady state, order≈0), under-resolved feature (pre-asymptotic, order 1.18, recovers on refinement), under-resolved reference (error saturates at the reference's own error, order≈−0.01). |
| **Exploration & research bridge** | 10% | One "Explore on your own" question answered with evidence (a sweep, plot, or fitted trend) — e.g. the level-7 round-off-floor prediction, or the `k=2` intercept-vs-slope check; a paragraph connecting the verification chain here to a real research workflow (e.g. gating a new physics term before trusting its science). |
| **Communication** | 5% | Axes/units labeled; orders quoted with the levels/`dt`'s they were fit over; honest that "it ran and printed a number" is not evidence of correctness on its own. |

## Automatic zero-credit triggers (flag, don't fail silently)

- An observed order reported without stating the levels (spatial) or
  `dt`'s (temporal) it was fitted over.
- Claiming a valid discretization-error measurement from a run where the
  algebraic-control (Newton-tolerance) sweep was never performed — no
  evidence the plot isn't solver-limited.
- A temporal order quoted from a self-convergence reference that was
  never verified (no order-stability check, no Richardson bound).
- Hand-copied numbers that do not match the run's own printed output or
  regenerated figures.
- Treating any of the four deliberate failures' finite, non-crashing
  order as evidence of a working discretization.

## Partial-credit guidance

- Correct `L2(c)` order reported but `H1(c)` and/or `L2(mu)`/`H1(mu)`
  omitted → cap Scientific correctness at half; this chapter's whole
  point is measuring both norms and both fields.
- Right temporal orders quoted but the reference verification
  (order-stability / Richardson bound) skipped → half of the Numerical
  verification weight — this is exactly the self-convergence pitfall the
  chapter is built to catch.
- A deliberate failure reproduced and described only as "the order came
  out wrong" without identifying which mechanism (BC mismatch vs. wrong
  source vs. pre-asymptotic vs. unverified reference) → half of the
  Failure-diagnosis weight.
- A beautiful log-log figure with no fitted order/level range stated in
  the report text → Communication credit only; the science credit is in
  the numbers, not the plot.
