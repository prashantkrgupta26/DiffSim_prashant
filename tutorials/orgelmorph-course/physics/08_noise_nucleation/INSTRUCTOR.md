# Instructor companion — P8 Thermal noise and nucleation

*Teaching notes for the instructor. Student-facing material is in
`README.md` and the course document chapter `p8.tex`. Grading is in
`grading_rubric.md`; hints and selected solutions in `solutions/`.*

## What this chapter is really teaching

1. **A discrete noise normalization is a derivation to be verified, not
   assumed.** The `√(2L/(dt·wJ))` per-Gauss-point scaling is not a
   convention — it is the unique normalization that makes the assembled
   nodal-force variance independent of both `dt` and mesh resolution.
   Getting either factor wrong (Exercise 2) breaks equipartition in a
   way that is only visible by *measuring* the equilibrium variance
   across a `dt`/mesh sweep — this is the central, load-bearing result
   of the chapter, not a footnote before the "interesting" nucleation
   part.
2. **Nucleation is a rare, stochastic, barrier-crossing process, so a
   single run is one sample.** The ensemble machinery
   (`diffsim.diagnostics.stochastic`) exists because the *right* answer
   to "does it nucleate" is a probability and a distribution, not a
   yes/no from one seed.
3. **Isolating one physical knob at a time.** This chapter varies
   *only* `noise_psi` at fixed undercooling — a deliberate restriction so
   students can attribute changes in nucleation statistics to the noise
   amplitude alone, not to a confounded change in driving force or
   barrier height that a real temperature change would also produce.

## Common student misconceptions & typical incorrect conclusions

- **"This is a temperature sweep."** It is explicitly **not**. Only
  `noise_psi` (≡ √(k_BT) in the *forcing*) varies; the undercooling
  `T/Tm` — and therefore the driving force and barrier — is held fixed.
  A genuine temperature sweep would move all three simultaneously and
  confound the noise-amplitude effect with a barrier-height effect. Push
  students to state this distinction explicitly in their report.
- **"A single seed's nucleation (or lack of it) is the result."** One
  realization is one sample of a rare stochastic event. The chapter's
  entire ensemble apparatus exists to prevent exactly this — a report
  that shows one X(t) trajectory per amplitude with no probability, no
  distribution, and no CI has skipped the central methodological point.
- **"The FDT verification is a warm-up before the 'real' nucleation
  physics."** It is the reverse — the equipartition check IS the central
  verification of this chapter (stated explicitly, "not a footnote").
  Students should be able to say what a wrong normalization would look
  like (variance drifting with `dt` or with mesh) before they trust any
  nucleation number downstream.
- **"BDF2 would just be more accurate here."** The brick *asserts* noise
  off under BDF2 — BDF2 with white noise forcing is not simply "more
  accurate," it changes the noise's effective statistics in a way the
  production code refuses to allow. This is a hard constraint, not a
  performance trade-off a student can opt out of.
- **"Clipping doesn't matter since we report ψ∈[0,1] anyway."** The
  ψ=0 floor *rectifies* sub-barrier fluctuations (turns two-sided noise
  into an effectively one-sided perturbation), which is exactly why the
  quantitative FDT test uses a separate clip-free well — conflating the
  two wells invalidates the equipartition measurement.

## Expected runtime ranges (RTX 6000 Ada; scale by card)

| mode | mesh / battery | wall | notes |
|---|---|---|---|
| quick | coarse FDT well (CPU) + reduced ensemble | comparable to P1's **≈ 75 s** (measured) | mostly Warp compile + start-up |
| reference | FDT `dt`/mesh sweep (CPU/splu) + level-5 ensemble (GPU, 4 seeds/amplitude) | a few minutes | the EXPECTED numbers; **ensemble multiplies wall by realization count** |
| research | `--level 6` nucleation mesh, more seeds/amplitude | tens of minutes to ~1 hr | scales linearly with total realizations |

First run of a session pays the one-time Warp kernel-compile cost;
subsequent runs are faster. Note the FDT well deliberately runs on CPU —
do not "helpfully" move it to GPU/cuDSS, that reintroduces the re-plan
cost this chapter is designed to avoid for a tiny problem.

## Common CUDA / solver errors students hit

- **BDF2 + noise assertion failure.** Passing `tstep="bdf2"` with
  `noise_psi != 0` raises — this is intentional (the brick asserts noise
  off under BDF2), not a bug to work around. Use BDF1.
- **Forcing the FDT well onto cuDSS/GPU.** The FDT-well verification
  intentionally uses a coarse CPU/splu mesh so the `dt`/mesh sweep runs
  fast without paying a cuDSS symbolic re-plan cost on every mesh change
  — moving it to GPU is not wrong, exactly, but slower and misses the
  point of why the split exists.
- **Zero-noise run shows nonzero X.** With `noise_psi=0`, X should stay
  ≈0 (ψ=0 is metastable behind the barrier). A nonzero X here is a bug
  signal (e.g. an unintended seed elsewhere in the initial condition),
  not evidence of "spontaneous" nucleation.
- **Ensemble too small to resolve the CI.** With only 1–2 seeds per
  amplitude, the reported CI on probability/induction-time is
  meaningless — insist on the full 4-seed-per-amplitude minimum before
  quoting a CI.

## Discussion prompts

- Why must the FDT-normalization verification use a *stable* well
  (clip-free, two-sided fluctuations) rather than the undercooled,
  clipped nucleation setup? What would clipping do to the measured
  variance?
- If a student's Exercise-2 mesh-blind noise experiment shows the
  variance now *does* depend on mesh, what does that tell them about
  what "mesh convergence" means for a stochastic PDE, versus a
  deterministic one (P1–P7)?
- The chapter's ensemble is only 4 seeds per amplitude. Is that enough
  to resolve the reported probabilities (P=1.00 at both nonzero
  amplitudes)? What would change if P were, say, 0.5?
