# Hints — C1 exploratory questions

*Hints for every "Explore on your own" question (course document `c1.tex`,
§"Explore on your own"). These are nudges, not answers — full worked
solutions for the two verification exercises are in
`verification_solutions.md`. Do not distribute the solutions file to
students before the deadline.*

**Q1 — Extend the `p=1` spatial sweep to level 7 (128×128). Does the
order stay at 2? Predict where round-off floors it.**
Hint: keep using the same least-squares fit across all levels, but also
look at the *raw* error values, not just the fitted slope — a round-off
floor shows up as the error stopping to shrink by ≈4× per level (the
`p=1` signature) while a slope fit that blindly includes the flattened
point will silently understate the order. Use the levels-3-to-6 numbers
in `EXPECTED.md` (`1.8e-2 → 2.8e-4`, order 2.01) to extrapolate what
level 7 "should" give if the ≈4×-per-level trend continues, then compare
against what you actually measure.

**Q2 — Change the manufactured wavenumber to `k=2`. Confirm the
intercept jumps but the order stays `p+1`.**
Hint: pass `k=2.0` through `spatial_mms`'s `k` argument (it threads
through to `c_star_k`/`f_c_k`/`f_m_k`); use the same `p=1` levels as the
default study so you have a like-for-like comparison. Plot both `k=1`
and `k=2` on the same log-log axes: two parallel lines with the same
slope but a higher line for `k=2`. Think about *why* the constant grows
with `k` before you look at `verification_solutions.md` — it's an
interpolation-error-constant argument, not a resolution failure (contrast
with the `k=6` under-resolved-feature failure, which genuinely does
change the slope).

**Q3 — Reproduce the wrong-source failure. Why is a wrong-but-nonzero
order more dangerous than a crash?**
Hint: drop the `kappa*lap(c)` term from `f_m` (see
`fail_wrong_source` in `convergence.py`) and rerun `spatial_mms`. Compare
what a rushed reader watching only "did the run finish and print a
number" would see for this failure versus a `NaN` or an exception — one
halts a pipeline immediately and demands attention, the other looks like
data. Connect this to the algebraic-control lesson: a printed number is
not evidence of correctness by itself.

**Q4 — Push the `k=6` under-resolved-feature failure to levels 5,6,7.
At what resolution does the order recover to 2?**
Hint: call `spatial_mms(p=1, levels=(5,6,7), k=6.0)` directly (the
`fail_underresolved_feature` helper is hardcoded to levels `2,3,4`).
Estimate how many mesh cells per half-wavelength you have at each level
for `k=6` (six half-waves across the unit box) and compare against the
same "resolve the feature with enough cells" reasoning used for
interface widths in physics P1 — the order should climb back toward 2
once there are enough cells per feature.

**Q5 — Compare DOF cost: `p=2` level 4 vs. `p=1` level 6.**
Hint: level `L` means `2^L` cells per side; count degrees of freedom for
each combination (the free-node count is available directly as
`len(cons.free_nodes)` from `build_dm`, or estimate it from cells ×
nodes-per-element for the given `p`). Divide each configuration's
`L2(c)` error (from the `p=1`/`p=2` spatial studies) by its DOF count to
compare cost-normalized accuracy — this is the classic "higher order vs.
more elements" trade-off question.
