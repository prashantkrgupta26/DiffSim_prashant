# Hints — P3 exploratory questions

*Hints for every "Explore on your own" question (course document `p3.tex`,
§"Explore on your own"). These are nudges, not answers — full worked
solutions for two verification exercises are in
`verification_solutions.md`. Do not distribute the solutions file to
students before the deadline.*

**Q1 — Move `φ*` toward 0 or 1; verify `projected_dofs=0` even at
`φ*=0.99`; then quench from a violating IC and watch the projection fire
without moving the converged-state mass; then try a linear wall (`h=0`)
and observe the step collapse.**
Hint: raise `|g|` at fixed `h>0` and re-check `projected_dofs` after each
run — it should stay 0 as long as `φ*=-g/(2h)` is strictly interior
(`EXPECTED.md` reports this checked up to `φ*=0.9975`). To fire the
projection *honestly*, raise the initial fluctuation amplitude so `φ(0)`
itself overshoots `(0,1)`, and confirm the *converged*-state quadrature
mass is still unchanged (the projection guards Newton iterates, not the
answer). Then set `h=0`: the wall energy `f_w=gφ` has no interior
minimum, so the "equilibrium" wants to run to a boundary — watch the FH
log-barrier stiffen and the time step collapse (non-termination), and
connect this to Chapter~P2's stiffness discussion.

**Q2 — Put a wall on the air side with the opposite preference (the
opposing-wall case); does the morphology invert top-to-bottom?**
Hint: compare the `p3_fields`/`p3_profiles` plots for the opposing case
against the single-wall attracting case — the opposing case should show
enrichment at the substrate *and* depletion at the air face
simultaneously (see the `opposing.air_phi ≈ 0.25` reference number).
Frame the answer for a device: whichever face a given wall preference
sits on determines which component reaches that electrode.

**Q3 — Raise χ above the spinodal (the demixing case): does the
attracting wall bias which phase wets the substrate? Compare `F_wall`
with the single-phase cases.**
Hint: run the demix case (χ=2.7) and compare its substrate `φ` and
`F_wall` against the single-phase attracting case (`φ*=0.75` in both).
`EXPECTED.md` states `F_wall` falls most strongly in the demixing case
("as the phases sharpen toward 0 and 1") — a whole phase domain coats
the substrate rather than a smooth gradient, so the wall energy is more
negative even though the surface-composition target `φ*` is the same
number.

**Q4 — Verify conservation explicitly: track the quadrature mass `∫φ dV`
over the run for each BC; why does the wall energy leave mass exactly
fixed, and why is the nodal mean misleading near the wall?**
Hint: log `quadrature_mass` (not a plain array mean) at every accepted
step for all six cases and plot the drift — it should sit at machine
precision throughout. The mechanism: the wall term only appears on the μ
rows (it modifies the natural condition on `grad φ · n`), so the φ-row
mass-balance residual is untouched and integrates to zero net flux
regardless of how the wall enriches or depletes. The nodal mean is
misleading because it weights boundary nodes differently from the
quadrature rule the assembly actually integrates with — see
`verification_solutions.md` for the full worked version.

**Q5 — Measure the boundary-layer thickness at three more κ values and
confirm `δ ~ √κ`; why did the scaling study use a sub-spinodal bulk
rather than the demixing χ used elsewhere?**
Hint: add κ points to the existing four-point sweep and refit both the
1/e-decay length and the exponential fit on log–log axes against κ — the
slope should stay near the predicted ½ within the same few-percent band
reported for the original four points. A sub-spinodal bulk is used
because it has no *interior* structure of its own (no spinodal domains
competing with the wall layer); demixing would confound the measured
decay length with the interior interface width from Chapter~P1, since
both would be present in the same profile.
