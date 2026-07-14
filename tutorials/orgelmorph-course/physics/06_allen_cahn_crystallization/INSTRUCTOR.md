# Instructor companion — P6 Allen–Cahn crystallization

*Teaching notes for the instructor. Student-facing material is in
`README.md` and the course document chapter `p6.tex`. Grading is in
`grading_rubric.md`; hints and selected solutions in `solutions/`.*

## What this chapter is really teaching

1. **Non-conserved vs conserved order parameters.** Crystallinity can be
   *created* — unlike composition in P1 — so it obeys a second-order
   Allen–Cahn flow, not the fourth-order Cahn–Hilliard. The sign of
   `drive = Δh(T/Tm − 1)` at the melting point is the whole model in one
   line: negative below `Tm` (grows), positive above (melts).
2. **Interface velocity and the critical radius are two faces of the same
   energy balance.** `v(ΔT)` is a kinetic statement (how fast the front
   moves once it exists); `r*` is a thermodynamic statement (whether a
   given seed exists at all). Both come from the same `drive` and
   interface energy `σ`.
3. **Avrami exponents are diagnostic, not decorative.** An exponent below
   the ideal `n=2/3` is not a bug — it tells you the nucleation mode
   (finite pre-placed seeds vs athermal point nuclei appearing over
   time). Reading `n` correctly requires knowing what seeded the run.

## Common student misconceptions & typical incorrect conclusions

- **"Orientation θ evolves with the crystal."** It does not here —
  `theta_mode="frozen"` makes θ a fixed grain *label* with **no**
  grain-boundary energy or dynamics. Students who see two crystals merge
  cleanly into one blob (rather than forming a grain boundary) are seeing
  exactly this simplification. Evolving θ is P7's Kobayashi–Warren–Carter
  concept — do not let students describe P6's θ as "evolved."
- **"Thresholded area and quadrature ⟨ψ⟩ are the same measurement."**
  They agree in the third digit here but are conceptually different: the
  quadrature ⟨ψ⟩ is the primary, honest measure of partial (diffuse-
  interface) crystallinity; the ψ>0.5 threshold discards the interface
  and is cut-sensitive (0.4/0.5/0.6). A student who reports only the
  threshold and calls it "the crystallinity" has skipped the primary
  number.
- **"T=250K, L_psi=11 are the real material parameters."** These Avrami
  run values are explicitly *pedagogical accelerants* (not the physical
  PCBM rate) — chosen only so X sweeps the full range inside a short
  run. A student quoting them as physical PCBM kinetics has missed the
  labelled caveat.
- **"v(ΔT) should have a maximum like real crystallization."** It does
  not, because the mobility `L_psi` here is *constant* — there is no
  thermally-activated transport freezing out at deep undercooling. A
  monotone-rising v is *correct* for this model; a maximum only appears
  once a student implements an activated `L_psi(T)` (Exercise 1).
- **"The measured r\* ratio (2.11) should exactly match Gibbs–Thomson
  (1.42)."** It should not exactly match — finite-time classification, a
  weakly T-dependent effective σ, and the φ-coupling all shift it. The
  correct claim is agreement in *sense and order of magnitude*, not
  numerical equality.

## Expected runtime ranges (RTX 6000 Ada; scale by card)

| mode | mesh / battery | wall | notes |
|---|---|---|---|
| quick | level 5 / reduced steps | comparable to P1's **≈ 75 s** (measured) | mostly Warp compile + start-up |
| reference | level 5 = 32², four experiments incl. 4-seed Avrami | a few minutes | the EXPECTED numbers |
| research | `--level 6` = 64² | tens of minutes | finer interface resolution |

First run of a session pays the one-time Warp kernel-compile cost;
subsequent runs are faster.

## Common CUDA / solver errors students hit

- **cuDSS re-plan stalls on level changes.** Switching `--level` between
  runs in the same process forces a symbolic re-factorization; a
  documented `splu` fallback exists if cuDSS is unavailable on the card.
- **`nan` or runaway ψ near T=Tm.** `drive` is small near the melting
  point; a student sweeping T close to `Tm` with too large a `dt` can see
  slow, noise-dominated dynamics that look like a bug but is just a weak
  driving force — check `drive` before suspecting the solver.
- **r\* classification flips with too-short a horizon.** The grow/shrink
  classification in `run_critical_radius` needs enough march time to
  resolve the trend; a student who cuts the horizon short may
  misclassify a slowly-growing seed as static.
- **Avrami fit window too narrow.** If `X*` never reaches the
  `0.05 < X* < 0.9` window (e.g. too few steps), the least-squares fit
  is ill-conditioned — the CI on `n` blows up. Check the number of
  points in the fit window before trusting the exponent.

## Discussion prompts

- Why is Allen–Cahn (P6) second-order in space while Cahn–Hilliard (P1) is
  fourth-order? What physical property of the order parameter causes the
  difference?
- If a student's activated-mobility experiment (Exercise 1) produces a
  velocity maximum, at what undercooling does it peak, and how does that
  compare to where the constant-mobility curve is still rising?
- The Avrami exponent here (≈1.3) is below the ideal `n=2`. Is this a
  failure of the model, or an honest diagnostic of the seeding protocol?
  What would you need to change to measure a genuine `n≈3`?
