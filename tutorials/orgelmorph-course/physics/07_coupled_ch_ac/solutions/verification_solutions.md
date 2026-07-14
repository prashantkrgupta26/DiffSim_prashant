# Selected full solutions — P7 verification exercises

*Full worked solutions for the two **verification** exercises this
chapter's headline draws on: the four-fold χ pure-limit test
(`test_chi_limits.py`) and the commensurate-controls demixing
decomposition. Hints for all "Explore on your own" questions are in
`hints.md`. Instructor-only — do not distribute before the deadline.*

---

## V1 — The four-fold χ pure limits (p1 vs r14), verified

**Claim.** With crystallinity, a single Flory χ becomes four matrices
`χ_aa, χ_ac, χ_ca, χ_cc`. In the **p1** bulk, χ_eff at the four pure
limits (ψ_i, ψ_j ∈ {0,1}) collapses to exactly those four inputs
(*absolute* pair interactions): `χ_eff(0,0)=χ_aa`, `(1,0)=χ_ca`,
`(0,1)=χ_ac`, `(1,1)=χ_cc`. In the **r14** bulk, the same physical input
`χ_ca` produces an *increment* on top of `χ_aa`: `χ_eff(1,0) = χ_aa +
χ_ca`, not `χ_ca` alone.

**Procedure.**
1. Run `test_chi_limits.py`, which evaluates `chi_eff_limits` for both
   `bulk="p1"` and `bulk="r14"` at the four `(ψ_i,ψ_j)` corners and
   compares against the production evaluator `_np_chi_eff` directly (not
   a reimplementation).
2. Read off the p1 pure limits and the r14 pure limit for the same input
   `χ_ca`.
3. Confirm the r14 pure limit equals `χ_aa + χ_ca`, not `χ_ca`.

**Expected result.** From `EXPECTED.md`: p1 pure limits
`aa/ca/ac/cc = 1.20/2.60/2.10/3.00` (with `χ_aa=1.2`, `χ_ca=2.6` as
inputs — the p1 pure limit for the crystal-amorphous contact *is* the
input value, 2.60). The same `χ_ca=2.6` input under r14 gives a pure
limit of `3.80 = χ_aa(1.2) + χ_ca(2.6)` — an increment, not the raw
input. Both are verified against the production evaluator, so this is
not a convention mismatch bug — it is the documented difference between
the two bulk models.

**Common wrong conclusion.** "χ_ca=2.6 means the same physical
interaction strength in both bulks, so I can port the number directly
between them." No — doing so silently changes the crystal-amorphous
contact energy by `χ_aa` (here, a ~46% difference: 2.6 vs 3.8). The fix
(Exercise 2) is to subtract `χ_aa` when converting a p1 absolute value
into an r14 input, and to *always* re-verify with `test_chi_limits.py`
after switching bulk conventions.

---

## V2 — Commensurate-controls decomposition of crystallization-driven demixing

**Claim.** Scoring three controls — `ch_only` (K=0, no crystal),
`coupled_nochi` (crystal present, χ_ca=χ_aa), and `full` (χ_ca>χ_aa) — on
the *same* mask (the full run's ψ>0.5 footprint) and the *same* metric
(species-0 composition contrast inside vs outside the mask) decomposes
crystallization-driven demixing into a **crystal-bulk channel**
(`coupled_nochi − ch_only`) and a **χ-expulsion channel**
(`full − coupled_nochi`), avoiding the divide-by-floor of comparing
against a near-zero `ch_only` denominator.

**Procedure.**
1. Run all three controls with the same seed, mesh, and mask (the
   `full` run's crystal footprint, fixed across all three so `ch_only`
   — which never forms a crystal — is scored on a mask *defined by
   another run*).
2. Compute the contrast for each: `⟨φ_0⟩_in − ⟨φ_0⟩_out`.
3. Compute the two channel increments and, only against the *resolved*
   `coupled_nochi` denominator, a relative amplification.
4. Repeat over ≥3 seeds to report the spread of the `full` contrast.

**Expected result.** From `EXPECTED.md`: `ch_only ≈ −0.000` (ordinary CH
produces no contrast on a mask it never causes to form), `coupled_nochi
= +0.376` (crystal-bulk channel alone), `full = +0.420`. Crystal-bulk
channel = `0.376 − (−0.000) ≈ 0.376`; χ-expulsion channel = `0.420 −
0.376 = 0.044`. The relative amplification of `full` over
`coupled_nochi` is `1.12×` (denominator resolved, meaningful). Seed
sensitivity of the `full` contrast is `0.420 ± 0.001` (3 seeds) — small
relative to the channel sizes, so the decomposition is a *reproducible*
result, not noise. The crystal-bulk channel dominates (~8.5× the
χ-expulsion channel) in this regime.

**Common wrong conclusion.** "The full/ch_only ratio (undefined or huge,
since ch_only≈0) tells us the coupling amplifies demixing by some large
factor." This is the divide-by-floor trap the chapter is built to avoid
— dividing by a ≈0 denominator produces a number that reflects the
floor, not the physics. The honest statement is the *absolute* channel
increments (0.376 and 0.044) and, separately, the resolved relative
ratio against `coupled_nochi` (1.12×) — never a ratio computed against
`ch_only`.
