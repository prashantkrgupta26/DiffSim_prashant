# Instructor companion — P2 Adaptive time stepping

*Teaching notes for the instructor. Student-facing material is in
`README.md` and the course document chapter `p2.tex`. Grading is in
`grading_rubric.md`; hints and selected solutions in `solutions/`.*

## What this chapter is really teaching

The Cahn–Hilliard model and free energy are unchanged from Chapter P1;
this chapter is entirely about the *time* axis. Three ideas do the
teaching:

1. **Verify before you trust.** The variable-step BDF2 of
   Eq.~(eq:vbdf2) is only second order if the coefficients are rebuilt
   from the true step ratio `r` every step. Before using it on the
   chaotic quench, the chapter measures its observed order on a smooth
   problem (BDF1 ≈ 1, BDF2 ≈ 2) — prediction, then measurement, the
   same habit as P1's dispersion check.
2. **Accuracy is a distance to a reference, not a property of the
   energy trace.** Two integrators (or two step sequences) can trace
   visually indistinguishable `F(t)` curves while one is much further
   from the tight-`Δt` reference in `c(T)`. This chapter is where
   students learn to stop reading accuracy off "the energy went down."
3. **Honest cost accounting.** Step-doubling costs one full *and* two
   half solves per attempt, so the speed-up in accepted steps is always
   larger than the speed-up in total solves. The naive worst-case bound
   (a fixed step pinned at the quench's smallest adaptive step for the
   *whole* horizon) is an upper bound on a badly-chosen fixed step, not
   the speed-up — citing it as such is the chapter's most-quoted trap.

## Common student misconceptions & typical incorrect conclusions

- **"Fewer accepted steps means the run is more accurate."** No —
  accepted-step count is a *cost* number, not an accuracy number.
  Accuracy is always the error vs. the reference; a short adaptive run
  at a loose tolerance can be less accurate than a longer fixed run at
  a tight one. Push students to report accuracy and cost as two
  separate axes (Fig.~p2conv, right panel).
- **"The step-count speed-up and the solve-count speed-up are the same
  number."** They are not, because step-doubling means every *attempt*
  costs three solves, not one. A student who reports only
  `PtwoSpeedupSteps` without `PtwoSpeedupSolves` has not accounted for
  the real cost (Table~p2cost).
- **"Tighter tolerance is always worth it."** The tolerance sweep has a
  knee: below it the pixelwise error stops falling (it sits on the
  same trajectory-sensitivity floor as the seed-sensitivity in P1)
  while the solve count keeps climbing. Quoting a tolerance far below
  the knee as "more accurate" is wishful.
- **"Lower free energy = more accurate."** Carried over from P1, and
  sharper here: two different `Δt` sequences reach two different
  (both plausible-looking) energy curves. The chapter's second keybox
  exists specifically to head this off.
- **"The naive worst-case bound is the honest speed-up."** A student
  who quotes the fixed step pinned at the quench's smallest step for
  the whole horizon (`PtwoWorstImplied`, ~10⁵–10⁶ steps) as "the"
  speed-up has reported an upper bound on a bad baseline, not a
  measurement — `EXPECTED.md` explicitly disallows this.
- **"A rejected step is wasted and shouldn't be counted."** Rejected
  steps still cost a full + two half solve before being discarded; the
  real cost table counts them. Students who report only accepted steps
  understate the adaptive run's actual cost.

## Expected runtime ranges (RTX 6000 Ada; scale by card)

| mode | mesh / horizon | wall | notes |
|---|---|---|---|
| quick | 16² / short horizon | comparable to P1's ≈ 75 s (measured) | CI smoke; tight numeric gate is reference-mode only |
| reference | 32² / `t_end=0.6` | ≈ 10–20 min | order study + reference + fixed sweep + adaptive + tolerance sweep |
| research | 64² / longer horizon, 5 tolerances | tens of minutes | — |

First run of a session pays a one-time Warp kernel-compile cost;
subsequent runs in the same environment are faster.

## Common CUDA / solver errors students hit

- **`cudss` selected and the march blows up.** Same reason as P1: the
  CH `(c,μ)` block is indefinite and cuDSS does no pivoting. This
  chapter uses `splu` (`run_harness.py` wraps `scipy.sparse.linalg.splu`
  to count every Newton/linear solve, including the step-doubling
  half-solves).
- **`dt_min` reached / step floor hit.** If a student tightens
  `work_tol` far below the knee or narrows `dt_min` too aggressively,
  the controller can hammer the floor during the stiff onset; check
  `exit_reason` and the rejected-step count before assuming a bug.
- **Confusing `fixed_dt` runs above the stiffness cliff with a solver
  bug.** The morphology looks "wrong" (destroyed) once a fixed step
  crosses the cliff — that is the expected failure mode (`Explore on
  your own`, last question), not a NaN or a crash.
- **Comparing fixed and adaptive runs at different mesh/kappa.** The
  chapter only ever compares *same-mesh* runs so spatial error cancels
  and the study isolates the time integration — a student sweeping
  `level` alongside `dt` has confounded the comparison.

## Discussion prompts

- Step-doubling triples the solve cost per attempt. Under what
  circumstances (horizon length, stiffness contrast between the quench
  and the coarsening tail) does that overhead still pay for itself?
- If a lab reports a huge step-count speed-up but a much smaller
  solve-count speed-up, have they disagreed with each other, or just
  reported two honest facts about the same run?
- The tolerance-knee sits on a trajectory-sensitivity floor, similar to
  seed-sensitivity in P1. Is that floor a property of the *integrator*,
  the *problem* (chaotic coarsening), or both?
