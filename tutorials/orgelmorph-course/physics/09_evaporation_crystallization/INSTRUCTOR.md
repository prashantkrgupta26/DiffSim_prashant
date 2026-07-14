# Instructor companion — P9 Evaporation-conditioned embryo growth

*Teaching notes for the instructor. Student-facing material is in
`README.md` and the course document chapter `p9.tex`. Grading is in
`grading_rubric.md`; hints and selected solutions in `solutions/`.*

## What this chapter is really teaching

1. **A derived quantity should be validated, not just asserted.**
   `φ* = 1 − |drive|/χ_ca` is derived analytically, then independently
   validated by the no-drying embryo composition sweep. The measured
   crossover sits *below* the derived value — not agreement, but a
   correctly-explained gap (self-enrichment via the P7 crystal-bulk
   channel makes `φ*` an upper bound). This is the chapter's central
   scientific-honesty lesson: a derivation that doesn't exactly match
   measurement is not a failure if the direction and mechanism of the
   gap are understood.
2. **Naming matters and shapes correct science communication.** These
   runs *implant* a deterministic supercritical embryo — they are not
   spontaneous nucleation. Calling this "nucleation" (as earlier phase-1
   material apparently did, per the "Phase-1 correction" framing) would
   mislead a reader into thinking thermal fluctuations produced the
   crystal, when in fact P8's FDT-noise machinery is the only place that
   happens in this course.
3. **The arc is the whole course's thesis in miniature.** Wet film →
   phase separation → embryo growth → crystalline film integrates P1–P8
   into one falsifiable claim: the *same* embryo's fate (dissolve vs
   grow) is set by the drying-conditioned local composition relative to
   `φ*`, not by time or evaporation per se — precisely what the
   no-drying composition sweep isolates as a control.

## Common student misconceptions & typical incorrect conclusions

- **"The default run shows spontaneous nucleation."** It does not — the
  embryo is deterministically *implanted* at a chosen radius and time.
  A student describing this chapter's headline result as "nucleation"
  has missed the explicit Phase-1 naming correction; genuine nucleation
  is P8's `noise_psi` mode.
- **"φ* should exactly match the composition-sweep crossover."** It
  should not, and matching exactly would actually be a red flag (it
  would mean the self-enrichment mechanism isn't operating). The
  expected, correct result is a crossover *below* `φ*`, with the gap
  explained by the embryo raising its own local `φ_0` as it orders (the
  P7 crystal-bulk channel) — so the effective threshold for a real,
  self-enriching embryo is lower than the homogeneous derivation.
- **"A `time_horizon` stop means the film finished drying."** It does
  not — it means the march ran out of requested simulated time,
  independent of how dry the film actually is. Read the `TERMINATION`
  enum (`time_horizon`/`solvent_target`/`height_floor`/`min_dt`/
  `step_budget`) and report the actual status, not an inferred "drying
  time."
- **"The embryo grows because the composition is favorable — radius
  doesn't matter."** It does — at a fixed super-solubility composition a
  small embryo (r0=0.05) still redissolves (Gibbs–Thomson), while a
  larger one (r0=0.20) grows. Composition sets whether growth is
  thermodynamically *possible*; radius (relative to `r*`) sets whether a
  given embryo actually clears the barrier.
- **"Constant Onsager mobility is the production model."** It is a
  labelled tutorial simplification — the production path uses the
  Vignes composition-singular mobility (`tests/test_multiphase_s3.py`).
  The qualitative arc survives the simplification, but the drying-front
  sharpness is softened; students should name this when interpreting
  fine spatial detail.

## Expected runtime ranges (RTX 6000 Ada; scale by card)

| mode | mesh / battery | wall | notes |
|---|---|---|---|
| quick | level 5 / reduced steps | comparable to P1's **≈ 75 s** (measured) | mostly Warp compile + start-up |
| reference | level 5 = 32², solubility + sweep + sub/supercritical control + wet/dry arc | a few minutes | the EXPECTED numbers |
| research | `--level 6` = 64² | tens of minutes | finer drying-front resolution |

First run of a session pays the one-time Warp kernel-compile cost;
subsequent runs are faster.

## Common CUDA / solver errors students hit

- **The missing-splat guard (`Tm` default).** `make_stepper` asserts the
  crystal energetics actually reached the stepper — a dropped keyword
  historically defaulted `Tm=1`, producing a huge spurious *melting*
  force (the S3b campaign bug this chapter's code walk-through
  references). If a student sees an embryo melting instantly regardless
  of composition, check that `Tm` is the intended PCBM-class value, not
  a silent default.
- **cuDSS re-plan on level changes.** As in P6/P7, switching `--level`
  forces a symbolic re-factorization; `splu` fallback is documented.
- **Misreading `TERMINATION`.** A `min_dt` stop signals Newton/stiffness
  trouble (often from an under-resolved drying front given the constant-
  mobility simplification), not a converged dry film — check the status
  before interpreting the terminal state as "done."
- **Reading a mid-growth transient as the terminal state.** The chapter
  is explicit: read the *terminal* crystalline state, not an
  intermediate snapshot — an embryo that is still visibly growing when
  the march stops has not necessarily reached its final fate.

## Discussion prompts

- Why is `φ*` correctly described as an "upper bound" rather than an
  approximation error? What would it mean scientifically if the
  composition-sweep crossover had instead come out *above* `φ*`?
- The wet-vs-dry arc holds the embryo's implant radius fixed and varies
  only *when* it is implanted. What does this control for, and what
  would change if radius were varied instead?
- Given the constant-Onsager-mobility simplification, which of this
  chapter's numbers would you trust quantitatively, and which only
  qualitatively? How would the answer change if a student swapped in the
  production Vignes mobility?
