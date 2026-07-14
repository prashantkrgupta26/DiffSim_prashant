# Instructor companion — P7 Coupled Cahn–Hilliard + Allen–Cahn

*Teaching notes for the instructor. Student-facing material is in
`README.md` and the course document chapter `p7.tex`. Grading is in
`grading_rubric.md`; hints and selected solutions in `solutions/`.*

## What this chapter is really teaching

1. **A single physical parameter (`χ_ca`) can mean two different numbers**
   depending on the bulk free-energy convention (p1 absolute vs r14
   increment). This is not a stylistic choice — feeding a p1-style
   absolute value into an r14-convention model silently changes the
   physics. `test_chi_limits.py` exists precisely so this mistake is
   caught mechanically, not by inspection.
2. **Commensurate controls are how you isolate a coupling honestly.**
   Comparing a crystal run to a *differently masked* no-crystal run
   amplifies whatever the (near-zero) denominator happens to be. Scoring
   `ch_only` / `coupled_nochi` / `full` on the *same* mask and *same*
   metric is the fix — and it decomposes the effect into physically
   meaningful channels instead of a single opaque ratio.
3. **Correlation is not cause — perturb the coupling.** The frozen-
   geometry and fixed-coupling/varied-kinetics experiments exist to rule
   out "crystallization happening at all causes demixing" in favor of
   "the χ coupling causes demixing; kinetics only sets the timing."

## Common student misconceptions & typical incorrect conclusions

- **"χ_ca means the same thing in p1 and r14."** The classic confusion.
  In **p1**, χ_eff at the pure limits *is* χ_aa/ac/ca/cc directly
  (absolute pair interactions). In **r14**, the same input χ_ca produces
  an *increment* χ_aa + χ_ca at the pure limit. A student who ports a p1
  χ_ca value into an r14 run without subtracting χ_aa has silently
  changed the physics — `test_chi_limits.py` is the mechanical check
  that catches this, and every student should run it before trusting
  any χ-dependent number.
- **"Full contrast ÷ ch_only contrast gives the amplification."** No —
  `ch_only` is ≈0 on the crystal footprint (ordinary CH does essentially
  nothing there), so dividing by it is a divide-by-floor. The correct
  reporting is *absolute* contrasts and *channel increments*
  (`coupled_nochi − ch_only`, `full − coupled_nochi`); a relative ratio
  is only meaningful against the resolved `coupled_nochi` denominator
  (≈1.12×, not some inflated number against ch_only).
- **"The χ-expulsion channel is what drives demixing."** It is the
  *smaller* channel (≈0.044 vs ≈0.376 for crystal-bulk here). Students
  often assume the χ-contrast term dominates because it is the more
  "chemistry-flavored" explanation; the arithmetic says otherwise — the
  crystal-bulk term (`φ_k² W`, present even at χ_ca=χ_aa) is the
  dominant channel in this regime.
- **"The energy decreasing proves the coupling is 'correct.'"** As in P1,
  this only shows the scheme is a gradient flow of a well-formed free
  energy — it is a discrete-energy-stability statement, not a
  correctness certificate for the physics.

## Expected runtime ranges (RTX 6000 Ada; scale by card)

| mode | mesh / battery | wall | notes |
|---|---|---|---|
| quick | level 5 / reduced steps | comparable to P1's **≈ 75 s** (measured) | mostly Warp compile + start-up |
| reference | level 5 = 32², controls + seed sweep + budget + causality | a few minutes | the EXPECTED numbers |
| research | `--level 6` = 64² | tens of minutes | finer contrast resolution |

First run of a session pays the one-time Warp kernel-compile cost;
subsequent runs are faster.

## Common CUDA / solver errors students hit

- **Confusing the χ convention crashes nothing — it just gives a wrong
  number silently.** This is the most dangerous failure mode precisely
  *because* there's no exception; insist students run
  `test_chi_limits.py` first, every time they touch χ.
- **cuDSS re-plan on a level change.** As in P6, switching `--level`
  between runs forces a symbolic re-factorization; `splu` fallback is
  documented if cuDSS is unavailable.
- **Denominator warnings ignored.** If a student modifies the mask logic
  and `ch_only`'s denominator stops being ≈0, the "no divide-by-floor"
  guard should raise a flag — a silently large relative ratio against
  ch_only is a sign the commensurate-controls invariant was violated.
- **Frozen-geometry run (`L_psi=0`) mistaken for a bug.** A frozen crystal
  is intentional for the causality control — `ψ` not evolving is
  correct, not a stalled solver.

## Discussion prompts

- Why does the crystal-bulk channel (`φ_k² W`) alone produce demixing
  even when χ_ca = χ_aa? What does this say about whether demixing
  *requires* an explicit contact-energy contrast?
- The chapter reports a relative amplification (1.12×) only against
  `coupled_nochi`, never against `ch_only`. Why is that the only
  defensible denominator here?
- The frozen-geometry experiment shows the χ coupling still demixes at
  fixed geometry. What would it mean scientifically if it had *not* —
  i.e., if demixing required active crystal growth?
