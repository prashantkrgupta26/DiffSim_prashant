# Selected full solutions — P2 verification exercises

*Full worked solutions for two **verification-flavored** exercises: the
integrator's observed temporal order (chapter Step 1, the mandatory
verification before the quench is trusted) and the stiffness cliff vs.
the predicted fastest spinodal growth rate (Q2, "Explore on your own").
Hints for all five exploratory questions are in `hints.md`.
Instructor-only — do not distribute before the deadline.*

---

## V1 — Observed temporal order: BDF1 ≈ 1, BDF2 ≈ 2 (Step 1)

**Claim.** A BDF of order `p` has local truncation error `O(Δt^{p+1})`
and global error `O(Δt^p)`; for BDF1 (`p=1`) that is a global error
`O(Δt)`, for BDF2 (`p=2`) it is `O(Δt^2)` — *provided* the variable-step
coefficients of the variable-step BDF2 formula are rebuilt from the true step ratio `r`
every step, not only at `r=1`.

**Procedure.**
1. Run the smooth single-mode relaxation (`c = a cos(2πx)`, a Neumann
   eigenmode with `κk² > 1` so the mode decays smoothly) at each
   `Δt` in `order_dts`, on the *same* mesh, against a fine-`Δt`
   reference (`order_ref_dt`) — spatial error cancels because the mesh
   is unchanged, isolating the temporal error.
2. Compute the relative L2 error of `c(T)` at each `Δt` for BDF1 and
   for BDF2 separately.
3. Fit the log–log slope with the shared `observed_order` convergence
   diagnostic.

**Expected result.** From `EXPECTED.md` and `baseline.yaml`: BDF1's
observed order is gated to `[0.80, 1.25]` (documented as "≈ 1.0"); BDF2's
is gated to `[1.70, 2.30]` (documented as "≈ 2.0") — i.e. first- and
second-order as designed. This is the evidence needed *before* trusting
the same variable-step BDF2 machinery on the chaotic quench. As a
second confirmation, the fixed-`dt` quench sweep itself is gated to
converge at order `[1.60, 2.40]` ("order ≈ 2") in its convergent
regime, before the stiffness cliff is reached.

**Common wrong conclusion.** "BDF2 achieved order 2 on the smooth
problem, so it is automatically order 2 on the quench too." No — the
smooth-problem check only verifies the *mechanism* (the coefficient
rebuild in the variable-step BDF2 formula is implemented correctly); the quench sweep is a
*separate* measurement, and it only holds "in its convergent regime" —
i.e. below the stiffness cliff. Past the cliff the fixed step no longer
resolves the quench at all, and the order-2 behavior is not observed
(nor claimed).

---

## V2 — The stiffness cliff tracks the predicted fastest growth rate (Q2)

**Claim.** The fastest spinodal growth rate is `λ_max ~ M/(4κ)`
(the same dispersion relation as Chapter P1, evaluated at its peak).
The time step needed to resolve that growth scales like `1/λ_max ~
4κ/M`: a *sharper* interface (smaller `κ`) gives a *faster* growth rate
and hence forces a *smaller* safe fixed step — the stiffness cliff
should move to a smaller `Δt` as `κ` shrinks, and to a larger `Δt` as
`κ` grows.

**Procedure.**
1. At the reference `κ = 2×10⁻³` (`EXPECTED.md`'s ground truth), run the
   fixed-`dt` sweep (`fixed_dts`) and locate the cliff: the `Δt` below
   which the relative L2 error vs. the reference falls at the expected
   order, and above which it jumps discontinuously.
2. Re-run the same sweep at a larger `κ` (e.g. the quick-mode value,
   `κ = 4×10⁻³`, which halves the sharpness) and re-locate the cliff.
3. Confirm the cliff moved to a larger safe `Δt` as `κ` increased, in
   the direction `λ_max ~ M/(4κ)` predicts.

**Expected result.** At the reference `κ`, the chapter reports (Step 2)
that pushing the fixed step past its safe value lands *above* the
cliff and the error "jumps ... more than an order of magnitude worse"
than the safe choice — the qualitative signature to look for at each
`κ`. The safe step and the cliff step both come out of your own
reference-mode run (`PtwoSafedt`/`PtwoCliffdt` in the generated
`numbers/p2.tex`); this exercise asks you to show that *both* move
outward together when `κ` is doubled, tracking `1/λ_max ~ 4κ/M`, not
that they hit a specific new value.

**Common wrong conclusion.** "The cliff is a fixed property of the
mesh/solver, independent of `κ`." It is not — it is set by the
*physics* (how fast the fastest unstable mode grows), which is exactly
why the chapter insists you must know the quench's stiffness *in
advance* to choose a fixed step safely: a step safe for one `κ` is not
safe for a sharper one.
