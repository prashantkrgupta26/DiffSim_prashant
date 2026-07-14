# Selected full solutions — P3 verification exercises

*Full worked solutions for two **verification** exercises only (the
explicit mass-conservation check and the boundary-layer scaling). Hints
for all five questions are in `hints.md`. Instructor-only — do not
distribute before the deadline.*

---

## V1 — Quadrature mass is conserved exactly by the wall condition (Q4)

**Claim.** The wall energy is a natural (Neumann-type) condition on the μ
equation, not a mass flux on the φ equation, so the discrete quadrature
mass `m = ∫_Ω φ dV` is conserved to machine precision in every boundary
condition tested, while the *nodal* mean against the nominal 0.5 shows a
spurious drift that is a boundary-weighting artifact, not a leak.

**Procedure.**
1. Run the harness reference mode (`configs/p3.yaml --mode reference`),
   which marches all six cases (neutral, attracting, repelling, opposing,
   confined, demixing).
2. At every accepted step, log both (a) the *quadrature* mass via
   `diffsim.diagnostics.conservation.quadrature_mass` — the same Gauss
   points the assembly integrates with — and (b) a naive nodal mean of
   the φ DOF vector.
3. Plot `|m(t) − m(0)|` for both measures, for all six cases, on a log
   axis.
4. Separately, confirm from `substrate.py`'s `_assemble_host` that the
   wall term (`-∫_w N_a (g_i + 2h_i φ_i) dS`) is added only to the μ rows,
   never to the φ rows.

**Expected result.** From `baseline.yaml`/`EXPECTED.md`: the quadrature
mass drift is `< 10⁻¹³` (`{max: 1.0e-12}` in the tolerance gate) in every
one of the six cases, including the demixing case where the projection
transiently fires. The nodal-mean check, by contrast, shows a spurious
~10⁻³ "drift" against the nominal 0.5 — this is explicitly called out in
`EXPECTED.md` as a boundary-weighting artifact, not a physical leak,
because nodal averaging does not match the quadrature rule the assembly
actually uses near a boundary face. The mechanism is structural, not
numerical luck: since the mass-flux condition (a) is *never* touched by
adding the wall term to the μ residual, the discrete φ-balance integrates
to `d(∫φ dV)/dt = 0` regardless of how strongly the wall enriches or
depletes — this holds even in the demixing case where `F_wall` is most
negative.

**Common wrong conclusion.** "The morphology visibly changed (enrichment
at the wall), so some mass must have moved in from outside." No — the
wall condition redistributes existing mass toward or away from the
boundary; it does not add or remove any. The correct way to see this is
the quadrature integral holding fixed while the *spatial distribution* of
φ changes; conflating "the field looks different near the wall" with
"mass was not conserved" is exactly the error this exercise is designed
to catch.

---

## V2 — Boundary-layer thickness scales as δ ~ √κ (Q5)

**Claim.** Linearizing Cahn–Hilliard about a stable (sub-spinodal) bulk
gives a wall-layer decay length `δ ~ √(κ/f'')`, so at fixed `f''` the
layer thickness scales as `δ ∝ √κ` — a slope of ½ on a log–log plot of
`δ` vs `κ`.

**Procedure.**
1. Choose a sub-spinodal bulk composition (stable, so the only structure
   in `φ(y)` is the wall-induced boundary layer — no competing interior
   spinodal domains).
2. Run the boundary-layer study over the four documented κ values,
   measuring the near-wall excess `φ(y) − φ_bulk` and extracting (a) the
   `1/e` decay depth and (b) an exponential-fit decay length for each κ.
3. Plot both length definitions vs κ on log–log axes and fit the slope.
4. Compare the fitted slopes to the predicted ½.

**Expected result.** From `EXPECTED.md` and `doc_numbers.yaml`
(`PthreeDeltaSlope`, `bdlayer.slope_1e`/`slope_fit` in `baseline.yaml`,
tolerance `atol: 0.10`–`0.15` around a reference of 0.5): the measured
`1/e` slope and the independent exponential-fit slope both come out
within a few percent of the predicted ½, and the layer visibly thickens
from `δ_1/e` at the lowest κ to a larger value at the highest κ (the
actual endpoints are `PthreeBdDeltaLo`/`PthreeBdDeltaHi` at
`PthreeBdKappaLo`/`PthreeBdKappaHi` in the rendered document — read the
current run's `results.json` for the numeric values, since these are
generated, not hand-set). The qualitative, falsifiable claim to report is
"lower κ ⇒ thinner substrate boundary layer, with an observed exponent
compatible with ½ inside its fit uncertainty" — not a bare "δ ~ √κ" with
no fit shown.

**Common wrong conclusion.** "The two slope estimates (1/e depth vs
exponential fit) disagree slightly, so the √κ scaling is only
approximate/wrong." No — both estimates agree with ½ *within their
stated tolerance* (`atol` in `baseline.yaml`); minor differences between
two different length definitions on a finite κ range is expected
estimation noise, not evidence against the scaling law. The honest
statement, as with P1's dispersion check, is "both estimators land within
their fit uncertainty of the predicted exponent," not "the fit was
exact." Also watch for students who ran this sweep on the demixing bulk
instead of the sub-spinodal one — that conflates this wall-layer length
with the interior interface width of Chapter~P1 and will not reproduce
the documented slope.
