# Selected full solutions — P6 verification exercises

*Full worked solutions for the two **verification** exercises this
chapter's headline draws on: the interface-velocity-vs-undercooling
measurement, and the Avrami exponent fit. Hints for all "Explore on your
own" questions are in `hints.md`. Instructor-only — do not distribute
before the deadline.*

---

## V1 — Interface velocity v rises monotonically with undercooling ΔT

**Claim.** With a constant mobility `L_psi`, the growth-front velocity
`v = dr_eff/dt` (with `r_eff = √(⟨ψ⟩_area/π)`) increases monotonically as
the undercooling `ΔT = Tm − T` deepens — there is no thermal-transport
maximum, because nothing in the model slows molecular attachment at low
`T`.

**Procedure.**
1. Run `run_interface_velocity` at two (or more) undercoolings, e.g.
   `ΔT = 138` (`T` near `Tm`) and `ΔT = 258` (deep quench).
2. For each, fit `r_eff(t)` over the pre-fill window (before the front
   reaches the domain boundary) and read the slope `v`.
3. Compare `v` across the undercoolings and confirm it increases.

**Expected result.** From `EXPECTED.md`: `v = 0.007 → 0.122` as
`ΔT = 138 → 258` — a roughly 17× increase, monotone. Because `drive =
Δh(T/Tm − 1)` grows in magnitude linearly with undercooling and the
mobility `L_psi` is *constant*, the front simply moves faster the deeper
the quench: there is no competing transport-limited slowdown to produce
a maximum. A real crystallizing material would show `v(ΔT)` rise, peak,
then fall as molecular diffusion freezes out at deep undercooling — that
requires an activated `L_psi(T)` (Exercise 1), not implemented here.

**Common wrong conclusion.** "v should have a maximum like real
crystallization — this model is wrong." No: the model is *labelled* as
using constant mobility precisely so students can see what changes when
you add the missing physics (Exercise 1). A monotone-rising v is the
*correct*, expected output of the constant-mobility model — the absence
of the maximum is a stated simplification, not a bug.

---

## V2 — Avrami/JMAK exponent n from the baseline-corrected fit

**Claim.** The crystalline-fraction sigmoid `X(t) = 1 − exp[−(kt)ⁿ]`
linearizes as `ln[−ln(1−X*)] = n ln t + n ln k` after baseline-correcting
`X* = (X − X₀)/(1 − X₀)` to remove the pre-placed seeds' initial area;
fitting over the pre-impingement window `0.05 < X* < 0.9` gives a finite
exponent `n` with a CI and `R²`, and because the nuclei here are *finite
pre-placed* seeds (not point nuclei appearing over time), `n` sits below
the ideal `n≈2` (pre-existing 2-D nuclei) or `n≈3` (nucleation over
time).

**Procedure.**
1. Run the multi-nucleus Avrami battery (`T=250K`, `L_psi=11`,
   accelerated pedagogical parameters) with several pre-placed seeds.
2. Record `X(t)` (both quadrature ⟨ψ⟩ and thresholded ψ>0.5) and the
   baseline `X₀` (crystalline fraction at t=0, from the seeds' finite
   area).
3. Baseline-correct to `X*`, restrict to the fit window `0.05 < X* <
   0.9`, and least-squares fit `ln[−ln(1−X*)]` vs `ln t`; report `n`, its
   95% CI, `R²`, and the number of points in the window.

**Expected result.** From `EXPECTED.md`: `n = 1.30 ± 0.07` with `R² =
0.976`, seed baseline `X₀ = 0.024`, and `4 → 3` nuclei-to-grains
resolved (one merge/overlap). The exponent is *intermediate*, below the
ideal `n=2` for pre-existing 2-D nuclei — this is expected, not an
error: the seeds start with finite area (they are supercritical
pre-placed discs, not point nuclei), so the early `t²` growth regime
that would establish the ideal exponent is foreshortened. A tight `R²`
and a narrow CI mean the intermediate value is a *resolved*
measurement, not fitting noise.

**Common wrong conclusion.** "n should equal 2 (or 3); getting 1.3 means
the fit or the model is broken." No — n=1.3 is the *correct*, physically
meaningful output for finite pre-placed seeds. The ideal exponents (2
for pre-existing point nuclei, 3 for nuclei appearing continuously over
time) are asymptotic limits that require the nuclei to start
infinitesimally small. Reducing the seed size toward point nuclei (Q2)
should push the fitted `n` upward toward those limits — that experiment,
not asserting the ideal value by fiat, is how you'd verify the
interpretation.
