# Instructor companion — P1 Binary Cahn–Hilliard

*Teaching notes for the instructor. Student-facing material is in
`README.md` and the course document chapter `p1.tex`. Grading is in
`grading_rubric.md`; hints and selected solutions in `solutions/`.*

## What this chapter is really teaching

The mechanics of Cahn–Hilliard are a vehicle for the course's central
habit: **measure the claim, report the uncertainty**. Three ideas do the
teaching:

1. **Dimensional reasoning is a check.** `ℓ ~ √(κ/W)` is not a formula to
   memorise — it is the answer to "why can't `κ` alone set a length?".
   Students who write `√κ` have not looked at the units.
2. **Prediction then measurement.** The dispersion relation gives `k*`
   *before* the run; the `S(q)` probe measures it. Agreement to within a
   FFT shell is the point, not the pretty morphology.
3. **"The energy decreases" is three different statements.** Separating
   continuous Lyapunov / discrete energy-stability / one monotone run is
   the single most transferable idea here — it recurs in every later
   chapter.

## Common student misconceptions & typical incorrect conclusions

- **"Interface width scales as √κ."** The most common error. Push them to
  the units of `κ` (energy·length²/volume) — it cannot be a length on its
  own.
- **"Lower energy = more accurate / more physical."** No — this chapter
  only claims the energy *decreases*; accuracy is C1/P2's job. Students
  conflate the Lyapunov property with correctness of the discrete
  trajectory.
- **"The morphology is reproducible because the seed is fixed."** The
  *figures* are reproducible on one machine; the morphology is chaotic
  across mesh/GPU/FP-order. The invariants (energy, `λ*`, exponent, phase
  values) are what reproduce. Watch for students quoting a single
  coarsening slope as "1/3" with no uncertainty — that is the trap.
- **"Flory–Huggins conserves mass, so the projection is harmless."** True
  on the reference run (projection fires only on *transient* Newton
  iterates), false on a coarse mesh or deep quench (the accepted step gets
  clipped and mass drifts ~2e-4). The lesson is that a safeguard trades
  conservation for robustness — visibly.
- **Reading the coarsening exponent off the peak-wavelength definition.**
  It is quantization-limited on a small mesh (near-zero, poorly-fit
  slope). The interfacial-area length is the headline; the peak and
  first-moment definitions are there precisely to show they disagree.

## Expected runtime ranges (RTX 6000 Ada; scale by card)

| mode | mesh / steps | wall | notes |
|---|---|---|---|
| quick | 32² / 60 | **≈ 75 s** (measured) | most of it Warp compile + Python start-up |
| reference | 64² / 250 | ~3–8 min | the EXPECTED numbers |
| research | 128² / 600, 5 seeds | ~20–40 min | ensemble mean ± sd |

First run of a session pays a one-time Warp kernel-compile cost (~30–60 s);
subsequent runs in the same environment are faster.

## Common CUDA / solver errors students hit

- **`cudss` selected and the march blows up.** The CH `(c,μ)` block is
  indefinite; cuDSS does no pivoting. P1 uses `splu` for a reason (C4 is
  the full story). If a student forces `--solver cudss` here, expect
  `c.min/max` to explode — that is *expected*, and a good teachable moment.
- **`out of memory` at level 7+.** They asked for research mesh on a small
  card. 2-D P1 is tiny; OOM means a much larger mesh than intended — check
  `level` in the resolved config.
- **FP32 gives a wrong energy budget.** The chapter is `precision: fp64`.
  A student who switched to fp32 will see the interfacial energy noisy and
  the mass drift inflated.
- **`nan` in Flory–Huggins.** Almost always the log hitting the wall with
  the regularization disabled, or a `c` initialised outside `(0,1)`.

## Discussion prompts

- Why is a *conserved* order parameter (Model B) fourth-order in space
  while Allen–Cahn (P6) is second-order? What does conservation cost you?
- If two labs on two GPUs get different final morphologies, have they
  disagreed? Which numbers *should* agree?
- The coarsening exponent drifts toward 1/3 only on a large box over a
  long horizon. When is it honest to quote "1/3", and when is it
  wishful?
