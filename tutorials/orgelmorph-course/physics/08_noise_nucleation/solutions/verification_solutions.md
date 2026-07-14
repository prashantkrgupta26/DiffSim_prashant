# Selected full solutions — P8 verification exercises

*Full worked solutions for the two **verification** exercises this
chapter's headline draws on: the discrete-FDT equipartition check, and
estimating the nucleation barrier slope from the ensemble sweep. Hints
for all "Explore on your own" questions are in `hints.md`.
Instructor-only — do not distribute before the deadline.*

---

## V1 — Discrete FDT equipartition: variance is dt-independent and ∝1/V_cell

**Claim.** The per-Gauss-point noise standard deviation
`std(ξ_gp) = noise_psi·√(2·L_psi/(dt·wJ_gp))` is the unique normalization
that makes the assembled nodal-force variance `Var(b_a) = 2·noise_psi²
·L_psi·∫N_a²dV / dt` — independent of both the time step `dt` and the
mesh resolution. Equipartition then predicts the equilibrium spatial
variance of ψ in a stable well is set by `kBT` and the well/mesh alone,
*not* by `dt`, and scales as `1/V_cell` (finer mesh → each cell samples a
smaller physical volume → larger fluctuation per cell, at fixed total
noise power).

**Procedure.**
1. Build a clip-free, stable *quadratic* r14 well (superheated so ψ
   fluctuates two-sidedly around a single minimum, not against the
   ψ∈[0,1] floor) on a coarse CPU/splu mesh.
2. Fix `kBT` (i.e. `noise_psi`), sweep `dt` over a few decades
   (reference: `3.0e-3, 1.5e-3, 7.5e-4`), and measure the converged
   equilibrium `Var(ψ)` at each.
3. Separately, fix `dt` and sweep the mesh resolution, and measure
   `Var(ψ)·V_cell` at each mesh.
4. Report the coefficient of variation (CoV) of `Var(ψ)` across the `dt`
   sweep, and of `Var(ψ)·V_cell` across the mesh sweep.

**Expected result.** From `EXPECTED.md`: `Var(ψ) = 0.0479, 0.0460,
0.0461` at `dt = 3.0e-3, 1.5e-3, 7.5e-4` — CoV = **0.022**, i.e.
`dt`-independent to within noise. Across meshes, `Var(ψ)·V_cell` is
constant to CoV = **0.11**, confirming `Var(ψ) ∝ 1/V_cell` — the
equipartition prediction. Both CoVs are small compared to the physical
signal (varying the amplitude/undercooling), so the normalization is
verified, not just "close enough."

**Common wrong conclusion.** "The FDT verification is a sanity check I
can skip since the nucleation ensemble is the 'real' result." No — this
IS the central, load-bearing verification: if the `1/dt` or `1/wJ`
factor were wrong, the equilibrium variance would drift systematically
with `dt` or mesh (a wrong `1/dt` normalization would make it scale
*with* `dt`, per the chapter text), and every downstream nucleation
number (which depends on the same noise machinery) would be
un-calibrated. A student who reports nucleation statistics without first
confirming the CoVs above has skipped the chapter's actual physics
content.

---

## V2 — Estimating the nucleation barrier slope from the ensemble

**Claim.** Classical nucleation theory predicts a rate `~exp(−ΔF*/kBT)`.
Since `noise_psi² = kBT` in this model, plotting `ln(nuclei density)`
against `1/noise_psi²` over a range of amplitudes spanning the onset
(where the nucleation probability first drops meaningfully below 1)
should give an approximately linear trend whose slope is (up to a
prefactor) `−ΔF*`.

**Procedure.**
1. Extend the amplitude sweep below the reference `noise_psi ∈
   {0.03, 0.06}` (both already give P=1.00, "essentially certain") down
   toward where `P(nucleate)` starts dropping below 1 — you need several
   points spanning that transition, not just the two saturated ones.
2. Run the 4-seed-per-amplitude ensemble at each point; record the mean
   nuclei density (and its ensemble spread).
3. Plot `ln(nuclei density)` vs `1/noise_psi²` (proxy for `1/kBT`); fit a
   line over the region where nucleation is not yet saturated at P=1.
4. Read the barrier estimate off the slope, and report its uncertainty
   from the fit together with the ensemble spread at each point.

**Expected result.** At the reference amplitudes `0.03 → 0.06`, nuclei
density goes `16.5 → 19.2` — both already at `P=1.00`, so this pair
alone is in the saturated regime and *not* where the Arrhenius slope is
cleanly resolved (the exercise explicitly directs students to sweep
*below* 0.03, toward the onset, to get a meaningful fit). The correct
report states: the reference two-point sweep demonstrates "larger noise
nucleates more" qualitatively, but a quantitative barrier slope requires
extending the sweep to lower amplitudes where P(nucleate) is not
saturated, and reports the fit's `R²`/CI honestly — exactly the P1-style
"measure the claim, report the uncertainty" habit this course repeats.

**Common wrong conclusion.** "I can fit the barrier slope from just the
two reference amplitudes (0.03, 0.06)." Both of these already sit at
P=1.00 — deep in the saturated regime, far from the barrier-limited
onset where the Arrhenius form is expected to hold linearly. A two-point
"fit" here is not a verification; the honest approach documents that the
reference sweep alone is insufficient and additional lower-amplitude
runs are required (Exercise 1 explicitly asks for this extension).
