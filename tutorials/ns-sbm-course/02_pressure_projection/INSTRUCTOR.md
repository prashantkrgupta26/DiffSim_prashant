# Instructor companion — 02 Pressure projection

*Teaching notes. Student-facing material is in `README.md` and course
document Chapter 2. Grading is in `grading_rubric.md`; hints in
`solutions/`.*

## What this chapter is really teaching

1. **Faithfulness is the validation, not the benchmark.** The projection is
   *right* because it reproduces the same-mesh monolithic — a
   mesh-independent claim. The Ghia agreement is a bonus, and on a coarse
   mesh it can even look "better" than the oracle by coincidence; do not let
   students read that as superiority.
2. **SPD is the load-bearing property.** The projection exists because its
   pressure step is SPD and therefore AMG/CG-friendly at 100M DOF. The
   indefinite saddle cannot be factorized at the hero scale (the cuDSS
   capacity wall). This chapter plants the *why* that Chapter 06 cashes in.
3. **The right divergence gate.** The pointwise $\|\nabla\!\cdot u\|$ is not
   the projection's target; the PPE-space solenoidality identity is. This is
   the single subtlest point of the two-engine story.

## Common student misconceptions

- **"The projection is more accurate than the monolithic (closer to
  Ghia)."** Coincidence of the coarse mesh. The oracle is the reference; the
  projection's job is to *match* it, not beat it.
- **"`consistent_projection` should change the cavity result."** For an
  *enclosed* flow the pin already stabilises pressure, so the base split is
  faithful. The consistency machinery matters at an *open outflow*
  (Chapter 03). Students expect a visible effect here and find none — that
  is the correct finding.
- **"The projection `||div u||` = 2 means it is less incompressible than the
  monolithic (0.19)."** No — different engines, different pointwise weak-div;
  both control the weak/PPE-space divergence. The identity, not the
  pointwise norm, is the gate.

## Expected runtime ranges

As Chapter 01 (the fixtures are shared). The projection predictor is a small
saddle solve plus a tiny SPD PPE at these sizes; on `splu` it is comparable
to the monolithic.

## Common CUDA / solver errors students hit

- **Forcing `cudss` on the *predictor* saddle** — same indefinite-operator
  failure as Chapter 01. Note the contrast: cuDSS/AMG *is* appropriate for
  the PPE (SPD), which is Chapter 06's whole point.
- **Expecting a blow-up from dropping consistency here** — the enclosed pin
  prevents it. Point them to Chapter 03's open-outflow rung-A finding.

## Discussion prompts

- Why can the SPD PPE use AMG/CG when the saddle cannot?
- What does `consistent_projection` make true that the naïve split does not?
  (The monolithic steady state becomes a fixed point of the split.)
- Which divergence — pointwise or PPE-space — would you report to argue the
  correction is a true projection, and why?
