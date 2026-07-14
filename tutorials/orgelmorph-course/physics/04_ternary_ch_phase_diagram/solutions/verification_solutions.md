# Selected full solutions — P4 verification exercises

*Full worked solutions for two **verification** exercises (the spinodal
sign/critical-interaction check, and the N-shift of the spinodal under
unequal degrees of polymerization). Hints for all five "Explore on your
own" questions are in `hints.md`. Instructor-only — do not distribute
before the deadline.*

---

## V1 — The quench is spinodal-unstable, and χ₁₂\* is the boundary (Q1)

**Claim.** At the initial blend `(φ₁, φ₂) = (0.35, 0.35)`, the 2×2
exchange Hessian `H` of the Flory–Huggins free energy has `det H < 0`
(spinodal-unstable, noise grows), and the boundary of instability at this
composition is the critical interaction `χ₁₂* = √(H₁₁H₂₂) − (1/(N_s φ_s) −
χ₁ₛ − χ₂ₛ)`. Below `χ₁₂*` the same composition is stable and noise
decays instead of growing.

**Procedure.**
1. Call `spinodal_hessian` at the initial blend for the reference
   `synthetic_demix` coefficients (`χ = (3.5, 1.0, 0.6)`, `N = (1,1,1)`)
   and read off `det H` and the smaller eigenvalue `λ_min`.
2. Solve `det H = 0` for `χ₁₂` at the same `(φ₁, φ₂)`, using that `H₁₂` is
   the only Hessian entry carrying `χ₁₂` — this gives the closed-form
   `χ₁₂*` without a search.
3. Sweep `χ₁₂` across `χ₁₂*` (e.g. 2.0, 2.84, 3.5) holding `χ₁ₛ, χ₂ₛ, N`
   fixed, re-evaluating `det H` at each value.
4. For at least one sub-critical `χ₁₂` (< `χ₁₂*`), actually march the
   field for the same horizon and confirm the composition spread (`std
   φ₁`) stays flat rather than opening up.

**Expected result.** From `EXPECTED.md`/`baseline.yaml`: `det H = −6.475`
at the reference `χ₁₂ = 3.5`, and `χ₁₂* = 2.840` — since `3.5 > 2.840`,
the quench is above the critical interaction and `det H < 0`, consistent.
The sweep should show `det H` crossing zero exactly at `χ₁₂ = 2.840` (to
the closed-form value, not a fit), and the sub-critical run's spread
should stay near its start value (`0.0201`, the reference start-spread)
rather than growing to the reference end-spread (`0.208`).

**Common wrong conclusion.** "The field didn't separate at
`χ₁₂ = 2.0`, so the code/solver is broken." No — 2.0 is below the
analytic `χ₁₂* = 2.840` at this composition, so the correct, predicted
outcome is that noise decays. The verification is that decay happens
*exactly* below the closed-form boundary and growth happens above it, not
that every `χ₁₂` value must demix.

---

## V2 — Unequal `N_i` enlarge the spinodal by a predictable amount (§"The `N_i` shift", extends Q3)

**Claim.** Because the entropic terms of the exchange Hessian
(Eq. for `H₁₁`, `H₁₂`) scale as `1/N_i`, raising the degree of
polymerization of solute 1 from `N₁ = 1` to `N₁ = 2` weakens the entropy
that resists demixing and *lowers* the critical interaction `χ₁₂*` (i.e.
enlarges the unstable region) at the same composition, deepening `det H`
and driving a more complete phase separation.

**Procedure.**
1. Evaluate `spinodal_hessian` at the reference composition with the
   symmetric `N = (1,1,1)` and record `det H` and `χ₁₂*`.
2. Re-evaluate with the asymmetric `N = (2,1,1)` (same `χ`, same
   composition) and record the new `det H` and `χ₁₂*` — this is a pure
   analytic prediction, no simulation needed yet.
3. Run the asymmetric-N companion blend (`configs/p4.yaml`'s
   `asym_level: 4`, `asym_t_end: 0.1`, the deliberately small/stiff-mesh
   demonstration case) and read the GMM coexisting-phase endpoints.
4. Compare: does the asymmetric blend's simulated separation go *further*
   toward the triangle edges than the symmetric reference, consistent
   with the deeper `det H`?

**Expected result.** From `EXPECTED.md`: symmetric `N=(1,1,1)` gives
`det H = −6.475`, `χ₁₂* = 2.840`; asymmetric `N=(2,1,1)` gives
`χ₁₂* = 1.979` (a shift `Δχ₁₂* = −0.860`) and `det H = −13.605` — both
predicted analytically before running anything. The simulated asymmetric
phases land at `A = (0.096, 0.902)` and `B = (0.954, 0.045)`, near the
triangle edges, versus the symmetric reference's `A = (0.162, 0.523)`,
`B = (0.529, 0.186)` — a visibly more complete separation, consistent
with the deeper instability.

**Common wrong conclusion.** "Doubling `N₁` should roughly double the
effect on `χ₁₂*`." No — the Hessian entries depend on `1/(N₁φ₁)` etc., not
linearly on `N₁`, and `χ₁₂*` also depends on `H₂₂` and the solvent term,
none of which change; the actual shift (`−0.860` on a base of `2.840`, a
~30% drop) has to be read off the closed form, not guessed from a naive
proportionality. The stiffer log barrier this asymmetry also produces
(more Newton iterations, reject-ladder activity — Q3) is a *separate*
numerical-cost consequence, not part of the spinodal prediction itself;
keep the two effects apart.
