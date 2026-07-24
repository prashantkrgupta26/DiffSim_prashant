# Instructor companion — 03 Weak Dirichlet / Nitsche

*Teaching notes. Student-facing material is in `README.md` and course
document Chapter 3. Grading is in `grading_rubric.md`; hints in
`solutions/`.*

## What this chapter is really teaching

1. **Weak imposition is the door to immersed boundaries.** Strong row
   replacement only works on grid-aligned Dirichlet nodes. Nitsche imposes
   $u=g$ through consistent boundary integrals, which generalizes to a
   boundary that cuts through elements — the exact setting the SBM
   (Chapter 04) needs. Framing $d=0$ Nitsche as "SBM with the shift off" is
   the key mental model.
2. **The penalty is the only tunable term.** Consistency and
   adjoint-consistency are exact; only $\alpha$ is free. $\alpha=0$ →
   the obstacle vanishes; $\alpha$ too large → conditioning degrades. The
   anti-vacuity run makes this concrete.
3. **Faithfulness with a body.** The immersed obstacle is the first place
   the projection-vs-monolithic comparison could plausibly fail; it does
   not (rel 3% drag) — the Nitsche layer preserves it.

## Common student misconceptions

- **"The absolute $C_d$ is wrong (literature says less)."** Confinement.
  The unit box inflates $C_d$; both engines see it, so compare them, not the
  book.
- **"Strong and weak should give the same $C_d$."** On a coarse mesh they
  differ (different obstacle-trace representations). Refinement narrows it;
  neither is "the" answer at level 4.
- **"$\alpha=0$ is just a smaller penalty."** No — it *removes* the
  Dirichlet enforcement; the obstacle becomes invisible ($C_d\to0$). That is
  the point of the anti-vacuity check.

## Expected runtime ranges (RTX 6000 Ada; scale by card)

| mode | level / steps | wall | notes |
|---|---|---|---|
| quick | 4 / 300 | ~2–4 min | the projection + monolithic marches to steady |
| reference | 4 / 600 | ~5–10 min | the EXPECTED numbers |
| research | 5 / 600 | ~20–40 min | finer mesh |

The marchers stop early on a rate tolerance (projection ~184 steps,
monolithic ~109), so wall time is often below the nominal step budget.

## Common CUDA / solver errors students hit

- **`blew_up: true` if they force the base strong split on the open
  outflow.** Expected rung-A finding; the weak `consistent_projection` path
  is the robust one. Good teachable moment about the outflow instability.
- **`--solver cudss`** on the predictor saddle — indefinite, same failure as
  Chapters 01–02.

## Discussion prompts

- Why does weak imposition generalize to a cut boundary while strong row
  replacement does not?
- If strong and weak give different $C_d$ at level 4, which do you trust,
  and how do you decide? (Refine; watch them converge.)
- What single change turns this $d=0$ Nitsche run into the genuine SBM of
  Chapter 04? (A sub-cell offset — the shift becomes nonzero.)
