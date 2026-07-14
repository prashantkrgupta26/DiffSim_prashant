# Hints — P4 exploratory questions

*Hints for every "Explore on your own" question (course document `p4.tex`,
§"Explore on your own"). These are nudges, not answers — full worked
solutions for the two verification exercises are in
`verification_solutions.md`. Do not distribute the solutions file to
students before the deadline.*

**Q1 — Sweep χ₁₂ past and below χ₁₂\* = 2.84; confirm det H changes sign
and the noise decays below it.**
Hint: call `spinodal_hessian` directly at the initial blend for several
`χ₁₂` values bracketing 2.84 and watch `det H` cross zero at the value the
closed form predicts. Then actually march a couple of sub-critical values
and confirm the spread (`std φ₁`) stays flat instead of growing — the
sub-critical run is the control that shows growth is not automatic just
because you perturbed the field.

**Q2 — Make the solvent selective (χ₁ₛ ≠ χ₂ₛ); which solute segregates,
and does the binodal tilt / does the GMM sensitivity grow?**
Hint: the solute with the larger `χ_is` (stronger solvent aversion) is
pushed toward the solute-poor/solvent-rich side less than the other —
work through which sign of `χ₁ₛ − χ₂ₛ` favors solute 1 in the `H_12` term
of Eq. (spinodal Hessian). Plot the predicted binodal tie-line for a
symmetric vs. an asymmetric-solvent case on the same triangle to see the
tilt directly, and re-run the interface-excluded refit to see whether the
sensitivity grows (a tilted, less-symmetric cloud is a harder clustering
problem).

**Q3 — Push N₁ to 5 or 10; watch the Newton iteration count and the
reject ladder.**
Hint: this is the direct P4 analogue of P1's Flory–Huggins log-barrier
cost. A larger `N₁` weakens the `1/N₁` entropic term that keeps the
Newton iterate away from `φ₁ = 0`, so the barrier is weaker exactly where
the quench is pushing hardest. Plot Newton-iterations-per-step and
reject-ladder activations against `N₁` ∈ {1, 2, 5, 10} — this is why the
tutorial demonstrates `N₁ = 2` on a small mesh (`asym_level: 4`) rather
than a large one.

**Q4 — Turn off the off-diagonal mobility M₁₂; does the tie-line change,
or only the coarsening rate?**
Hint: the tie-line endpoints come from equating chemical potentials at
equilibrium (`predict_binodal`), which is independent of the mobility
matrix — mobility only sets the *rate* of approach to that equilibrium,
not the equilibrium itself. Confirm by comparing the GMM endpoints (not
just the spread-vs-time curve) with `M₁₂ = 0` against the reference run;
the *rate* at which the spread opens should change, the *endpoints*
should not (within sensitivity).

**Q5 — Map a real blend (e.g. `P3HT_PCBM`) from
`materials/ternary_p4.yaml`; predict from its spinodal whether it demixes,
then check against the GMM endpoints.**
Hint: load the blend with `load_blend("P3HT_PCBM")`, evaluate
`spinodal_hessian` at the composition you choose, and read the sign of
`det H` *before* running anything — that is the falsifiable prediction.
Then run it and compare: does the sign match, and do the GMM endpoints
land inside/near the predicted binodal the way the `synthetic_demix`
reference case does?
