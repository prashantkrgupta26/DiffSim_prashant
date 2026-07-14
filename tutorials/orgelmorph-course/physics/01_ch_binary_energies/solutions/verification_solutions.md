# Selected full solutions — P1 verification exercises

*Full worked solutions for the two **verification** exercises only (the
interface-width scaling and the dispersion check). Hints for all questions
are in `hints.md`. Instructor-only — do not distribute before the
deadline.*

---

## V1 — Interface width scales as √(κ/W), not √κ (Q1)

**Claim.** The equilibrium interface between the two phases is a `tanh`
profile whose width is set by the *ratio* `κ/W`, so `ℓ = √(2κ/W)` for the
symmetric polynomial well (`W = 1/4` in the barrier-height convention of
the chapter; the exact prefactor depends on how `W` is defined).

**Procedure.**
1. Run reference mode at `κ ∈ {2, 5, 10}×10⁻⁴` (edit `kappa` in the
   config or pass an override).
2. Measure the interface width from the converged field — either fit the
   `tanh` half-width, or read it from `√(κ/W)` and compare to the
   diagnostics' reported width.
3. Plot measured width vs `√(κ/W)` **and** vs `√κ` on log–log.

**Expected result.** Against `√(κ/W)` the points lie on a slope-1 line
(width ∝ √(κ/W)); against `√κ` they also happen to line up here *because W
is constant* — the discriminating test is to also vary the quench depth
(FH `B`, which changes the effective `W`): the width tracks `√(κ/W)` and
*not* `√κ`. Cell count = width·N/box; at the reference `κ = 5×10⁻⁴`,
`ℓ ≈ 0.032` ≈ 2.0 cells (from `EXPECTED.md`). At `κ = 2×10⁻⁴` the width
falls to ~1.3 cells — below the ~3-cell resolution floor, so the energy
budget there is mesh-limited and should not be trusted.

**Common wrong conclusion.** "Both `√κ` and `√(κ/W)` fit, so they are
equivalent." They are not — the coincidence is that `W` was held fixed.
Vary `W` and only `√(κ/W)` survives.

---

## V2 — Dispersion: measured fastest mode tracks k* (Q2)

**Claim.** Linearising CH about the mean gives `σ(k) = −M k²(f''(c̄) +
κk²)`, maximised at `k* = √(−f''(c̄)/2κ)`, `λ* = 2π/k*`.

**Procedure.**
1. The harness already runs `linear_probe`: a few tiny-`dt` steps from the
   small-amplitude initial field, reading each Fourier shell's growth
   `σ_meas(k) = ½ ln[S(k,τ)/S(k,0)]/τ` (the ½ because `S ∼ amplitude²`).
2. Overlay `σ_meas(k)` (points) on the analytic `σ(k)` (line); mark `k*`.
3. Change `κ` or `c̄` and confirm the peak moves as `k* = √(−f''/2κ)`.

**Expected result.** From `EXPECTED.md`: predicted `λ* = 0.199`, probe-
measured `λ* = 0.182` — agreement within one FFT shell (the shell spacing
on a 64² box is coarse). Both free energies share the same `f''` at their
unstable mean here, so both give the same `k*`. Increasing `κ` lowers
`k*` (broader domains, larger `λ*`); moving `c̄` toward the spinodal edge
(`f'' → 0`) sends `k* → 0` and the growth vanishes.

**Common wrong conclusion.** "The measured `λ*` disagrees with the
prediction (0.182 ≠ 0.199), so the theory is wrong." No — 0.182 and 0.199
are the same to within the FFT shell resolution; the honest statement is
"agreement within one `q`-shell", which is what the chapter reports. The
fix for a tighter test is a larger box (finer `q`-grid), i.e. research
mode.
