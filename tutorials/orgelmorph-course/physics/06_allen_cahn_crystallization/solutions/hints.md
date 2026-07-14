# Hints — P6 exploratory questions

*Hints for every "Explore on your own" question (course document `p6.tex`,
§"Explore on your own"). These are nudges, not answers — full worked
solutions for the two verification exercises are in
`verification_solutions.md`. Do not distribute the solutions file to
students before the deadline.*

**Q1 — Activated mobility L_psi(T) = L0·exp(−Ea/T); does v(ΔT) develop a
maximum?** Hint: the constant-mobility curve only ever rises because
nothing in the model slows molecular attachment at deep undercooling.
An activated `L_psi(T)` competes two effects: `|drive|` growing with
`ΔT` (speeds v up) against `exp(−Ea/T)` falling as `T` drops (slows v
down). Plot `v` against `ΔT` for a couple of `Ea` values — the peak
should sit at smaller `ΔT` for larger `Ea` (transport freezes out
sooner). Compare directly against Figure `p6_kinetics` (left panel).

**Q2 — Vary the number of pre-placed nuclei; does n move toward 2/3?**
Hint: more, smaller nuclei approach the "point nuclei with area growing
as `t²`" limit that gives the ideal exponents; fewer, larger pre-placed
seeds start with area already and foreshorten the early `t²` regime
(exactly why the reference run's `n≈1.3` sits below 2). Re-run
`run_avrami` with a different seed count and refit; track how the CI
shrinks or grows with the number of grains resolved.

**Q3 — Sweep ε²; confirm the crystal-interface width scales as √ε².**
Hint: same logic as P1's `√(κ/W)` — do not vary `ε²` alone and declare
victory; also vary the barrier height `Δσ` and check the *ratio*
`ε²/Δσ` is what sets the width, not `ε²` alone. Count mesh cells the
same way P1 does (width × N / box); watch for `r*` becoming biased once
the crystal interface itself is under-resolved (~3 cells is the usual
floor).

**Q4 — Two seeds with different θ; do they show a grain boundary?**
Hint: with `theta_mode="frozen"` they cannot — there is no
grain-boundary energy term, so contact just merges the two ψ blobs into
one. This question is designed to make that limitation concrete before
P7 introduces evolving θ (Kobayashi–Warren–Carter) and grain boundaries
become a real physical object with an energy cost.

**Q5 — Measure composition inside vs outside the crystal.** Hint: this
is the φ-expulsion effect that P7 makes central. In P6 there is no χ_ca
> χ_aa contrast built into the demo run, so any contrast you measure
here is the "crystal-bulk channel" alone (P7's decomposition). Compare
your number qualitatively to P7's `EXPECTED.md` crystal-bulk contrast
(≈0.376) once you get there — do not expect an exact match, the setups
differ.
