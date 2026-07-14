# Hints — P9 exploratory questions

*Hints for every "Explore on your own" question (course document `p9.tex`,
§"Explore on your own"). These are nudges, not answers — full worked
solutions for the two verification exercises are in
`verification_solutions.md`. Do not distribute the solutions file to
students before the deadline.*

**Q1 — Sweep implant time from wet to dry; sharp crossover in φ_s? Does
LOCAL φ_0 match derived φ*?** Hint: run `run_arc` (or `run_static_embryo`)
at several implant times spanning the drying curve, recording `φ_s` (the
solvent fraction, a proxy for global dryness) and the *local* `φ_0` at
the implant site at that moment. Plot terminal fate (grow/dissolve)
against implant time — expect a fairly sharp transition. Then compare
the *local* φ_0 at the crossover implant time to the derived `φ*`: it
should sit close to (at or slightly above) the composition-sweep
crossover from the main run, not necessarily exactly at the homogeneous
`φ*` — remember the self-enrichment caveat.

**Q2 — Vary χ_ca; re-derive φ* and re-check the sweep. Where does
φ* hit 0?** Hint: `φ* = 1 − |drive|/χ_ca` is monotone in χ_ca — larger
χ_ca (crystal dislikes its surroundings more) raises `φ*` (needs *more*
small-molecule fraction to be worth crystallizing against that penalty),
while decreasing χ_ca lowers it. `φ* = 0` (no solubility barrier at all,
grows even wet) occurs when `χ_ca = |drive|` exactly — solve for that
χ_ca given the tutorial's `drive` and check it against a plausible
materials range in `materials.yaml` before calling it physical.

**Q3 — Vary implant radius r0 around the critical radius at
super-solubility composition.** Hint: at a fixed φ_f above φ* (definitely
super-solubility), sweep r0 downward from the reference supercritical
value (0.20) toward the subcritical one (0.05) used in the main
sub/supercritical control. Find the r0 where the fate flips from grow to
dissolve — that is the Gibbs–Thomson r* at this composition/undercooling
(connect back to P6's `run_critical_radius` machinery and its ratio
comparison to the classical prediction).

**Q4 — Turn on FDT noise (P8's advanced mode); does drying-conditioned
timing survive?** Hint: `make_stepper(noise_psi=...)` lets the film
nucleate on its own instead of via a deterministic implant. Check whether
nucleation events still cluster after the composition crosses toward
`φ*`-favorable regions, or whether noise can produce (rare) early
nucleation even in wetter regions. Remember BDF1-only and the noise
normalization from P8 apply here too — this question is explicitly a
bridge back to that chapter.

**Q5 — Design problem: which knobs give fine, pure, bicontinuous
domains?** Hint: this is a discussion/design question for the Research
bridge deliverable, not a single numeric answer. Frame it around the
three levers this course has built up: crystallization kinetics
(`k_e`/`L_psi`, P6/P8), the χ-expulsion vs crystal-bulk decomposition
(P7), and the drying rate / blend ratio that sets *when* the local
composition crosses `φ*` (this chapter). Note the tension: faster drying
locks in structure sooner (less time to coarsen) but a too-fast drying
front (constant-mobility simplification) may not have time to reach a
percolating morphology at all — connect to P7's closing purity-vs-
network discussion question.
