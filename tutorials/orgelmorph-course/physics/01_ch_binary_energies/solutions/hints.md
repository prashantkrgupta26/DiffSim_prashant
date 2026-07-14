# Hints — P1 exploratory questions

*Hints for every "Explore on your own" question (course document `p1.tex`,
§"Explore on your own"). These are nudges, not answers — full worked
solutions for the two verification exercises are in
`verification_solutions.md`. Do not distribute the solutions file to
students before the deadline.*

**Q1 — Vary κ ∈ {2,5,10}×10⁻⁴; confirm width ~ √(κ/W), count cells.**
Hint: plot the measured interface width against `√(κ/W)`, not `√κ`, on
log–log; the slope is ½ against the *ratio*. Cell count = width / (box /
N). The interface falls below ~3 cells around the smallest κ — at that
point the tanh profile is aliased and the energy budget is mesh-limited.

**Q2 — Verify the dispersion relation; find where growth stops.**
Hint: the probe measures `σ(k) = ½ ln[S(k,τ)/S(k,0)]/τ`. Move `c̄` (FH)
until `f''(c̄) > 0`: outside the spinodal band every mode decays, so the
probe peak vanishes. The band edge is `f''(c̄) = 0`.

**Q3 — 5-seed ensemble: spread vs the between-energy difference.**
Hint: run `--mode research`. Compare the seed-to-seed sd of the final
energy/length to the poly-vs-FH gap. The exponent is "resolved" only if
its uncertainty is smaller than the effect you are claiming.

**Q4 — Extend the horizon: does n → 1/3?**
Hint: plot the fitted `n` against the *start* of the fit window. On a
larger box (research mode) the late-window slope climbs toward the
Lifshitz–Slyozov 1/3; on the small reference box, finite-size saturation
caps `L` at ~box/3 and biases the fit low.

**Q5 — Drive FH into the box projection.**
Hint: coarsen the mesh (lower `level`) or raise `B`. Watch `proj_dofs` and
`|Δm|` rise together once the *accepted* (converged) state is itself
clipped — this is the safeguard trading conservation for robustness. See
`verification_solutions.md` for the full worked version.
