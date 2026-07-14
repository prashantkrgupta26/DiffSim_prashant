# Selected full solutions — P5 verification exercises

*Full worked solutions for two **verification**-flavored exercises: the
Biot-collapse check (implicit in the chapter's headline, exercised via
Q1) and the resolution/conservation check (Q5). Hints for all five
questions are in `hints.md`. Instructor-only — do not distribute before
the deadline.*

---

## V1 — Matched-dryness wavelength collapses onto the Biot number (Q1)

**Claim.** The matched-dryness lateral wavelength depends on `(k_e, D_s)`
only through `Bi = k_e·h0/D_s`: runs with different rate/mobility pairs
that share a `Bi` should dry to the same morphology, to within the
sweep's stated collapse spread.

**Procedure.**
1. Run the reference-mode harness (`run_harness.py --config
   configs/p5.yaml --mode reference --output outputs/p5 --overwrite`),
   which sweeps `ke_grid × ds_base` (the matched-state rows) *and*
   `ke_regime × ds_grid` (the 2-D Biot regime map) — two sweeps sharing
   one mesh, per the config's own header.
2. From `results.json`, read the matched-dryness wavelength (cells) at
   `φ_s = 0.20` for every `(k_e, D_s)` pair in the regime map, and compute
   each pair's `Bi = k_e·h0/D_s`.
3. Group runs by (nearly) equal `Bi` reached via *different* `(k_e, D_s)`
   combinations, and compare their matched-dryness wavelengths.
4. Read the harness's own `checks.bi_collapse_spread` scalar rather than
   eyeballing the regime-map figure.

**Expected result.** From `EXPECTED.md`/`baseline.yaml`: the sweep spans
`Bi ∈ [0.75, 6.0]`, straddling the `Bi = 1` drying/diffusion crossover.
Runs sharing a `Bi` agree in matched-dryness wavelength to a spread of
~6% (the checked gate is `bi_collapse_spread ≤ 0.15`). Concretely, at the
matched-state baseline row (`D_s = 0.2`): `k_e = 0.30` (`Bi = 1.5`) gives
33.4 cells and `k_e = 0.60` (`Bi = 3.0`) gives 26.1 cells at `φ_s = 0.20` —
different rates, different sizes, but the regime-map runs confirm it is
`Bi`, not the raw `(k_e, D_s)` pair, that organizes the outcome. The
wavelength itself is noisy (single seed, only ~2 lateral domains across
the box), so the honest claim is "the collapse holds to ~6% over the
tested range", not a sharp size-versus-Bi power law.

**Common wrong conclusion.** "The wavelength changed with `k_e`, so rate
matters more than Bi." Rate changes `Bi` (holding `D_s` fixed), so of
course the wavelength moves with `k_e` alone in the matched-state row —
the actual test of the Bi-hypothesis is whether *different* `(k_e, D_s)`
pairs that land on the *same* `Bi` (the regime-map sweep) also land on the
same wavelength. Citing the matched-state row alone does not test the
hypothesis; the regime map does.

---

## V2 — Conservation is resolution-limited: level 4 leaks, level 5–6 does
not (Q5)

**Claim.** Only the solvent leaves the moving frame, so each solute's
content `h(t)·∫φᵢ` (quadrature on the fixed reference domain, multiplied
by the current height) should be exactly constant over the run. This
holds only once the demixed interfaces and the surface-enrichment layer
are resolved.

**Procedure.**
1. Run the reference-mode harness at its default `level: 6` and record
   `checks.balance_polymer_rel_max` / `checks.balance_fullerene_rel_max`
   / `checks.budget_closure_max` from `results.json`.
2. Re-run with `level: 4` (16×16) — everything else held fixed — and
   record the same three drift diagnostics.
3. If time allows, also try `level: 5` (32×32) to find the resolution at
   which conservation recovers.
4. Relate the recovery level to the interface width `ℓ ~ √(κ/W)`
   (Chapter P1, with this chapter's `κ = 3.0×10⁻⁴` and the Flory–Huggins
   `χ = (1.5, 0.3, 0.3)`) versus the cell size `Δx = 1/n_x` at each level.

**Expected result.** From `EXPECTED.md`/`baseline.yaml`: at level 5–6 the
per-solute drift is ~10⁻¹⁵ (machine precision) and the height/solvent
budget closes to the same order (both gated at `max: 1.0e-9` in
`baseline.yaml`, i.e. comfortably inside machine-precision territory for
this run). At level 4 (16×16) the *same* run leaks ~10% of a solute — the
demixed interfaces and the surface-enrichment layer are under-resolved at
that mesh. This is exactly why the `quick` tier is pinned to level 5 (the
coarsest mesh that still conserves), never level 4.

**Common wrong conclusion.** "The level-4 leak means the moving-frame
scheme doesn't conserve solute." The scheme *does* conserve solute
exactly in the continuous formulation and does so numerically once the
interfaces are resolved (level 5–6); the level-4 leak is a resolution
failure of the *discretization*, not evidence against the conservation
property of the model. The correct diagnosis names the under-resolved
interface/enrichment layer as the mechanism, not "the conservation law is
wrong" or "there's a bug".
