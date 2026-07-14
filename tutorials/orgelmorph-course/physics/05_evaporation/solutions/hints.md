# Hints — P5 exploratory questions

*Hints for every "Explore on your own" question (course document `p5.tex`,
§"Explore on your own"). These are nudges, not answers — full worked
solutions for two verification-flavored exercises are in
`verification_solutions.md`. Do not distribute this file to students
before the deadline.*

**Q1 — Widen the Biot range (push `D_s` down and `k_e` up). Does the
matched-dryness wavelength ever break its collapse onto Bi, or is Bi the
only group that matters here?**
Hint: the checked sweep spans `Bi ∈ [0.75, 6.0]`, straddling the `Bi=1`
crossover — that is the *tested* range, not a proof the collapse holds
everywhere. Push `D_s` an order of magnitude lower or `k_e` several times
higher and recompute the matched-dryness wavelength for the new pairs;
watch whether points sharing the new, more extreme `Bi` still land
together, or whether a second group (e.g. how close the film gets to
`h_min`, or how few `nchecks` resolve the trajectory near cutoff) starts to
matter. Remember the wavelength is noisy with only ~2 lateral domains
across the box — don't over-read a single seed's wiggle as a broken
collapse.

**Q2 — Change the solvent selectivity (χ₁ₛ vs χ₂ₛ). Does a solvent that
prefers one solute change which component enriches at the drying surface
— and does that show up more in the vertical profile than the lateral
wavelength?**
Hint: make `chi` asymmetric between the two solute–solvent interactions
and look at the *vertical* (physical-frame) composition profile near the
top surface (Fig. `p5_morphology`, physical-frame panel) rather than the
lateral wavelength — surface enrichment is a through-thickness effect, so
it should show up as a skewed near-surface profile between the two solute
fields before it shows up as a lateral length-scale change.

**Q3 — The drying curve `φ_s(t)` is roughly linear early (constant-rate
period) then bends. Explain the bend from the surface composition
feeding back into `K = Σᵢ k_{e,i}·φ̄ᵢ^top`.**
Hint: `K` depends on the *current* top-surface solute fractions, not the
bulk mean — early on the surface looks like the bulk (roughly
constant-rate drying), but as demixing proceeds and one solute enriches
at the top, `φ̄ᵢ^top` moves away from its initial value and `K` changes
with it, bending the `φ_s(t)` curve. Plot `φ̄ᵢ^top(t)` alongside `φ_s(t)`
and look for the bend to line up with when the surface composition starts
to diverge from the bulk mean.

**Q4 — Combine with Chapter P3: add a substrate wall energy and dry the
film. Does evaporation-driven enrichment at the top compete with
substrate enrichment at the bottom? Which wins, and at what Bi?**
Hint: at low Bi (diffusion-limited) the bulk has time to respond to the
substrate's preferential wetting even while the top is also enriching —
expect the substrate term to dominate the near-bottom profile. At high Bi
(drying-limited) the surface enrichment happens fast relative to bulk
diffusion, so the top layer may "lock in" before the substrate signal
propagates far from the wall. Sweep Bi with the wall energy fixed and
look at which boundary's enrichment layer is thicker/stronger as Bi
crosses 1.

**Q5 — Re-run at level 4 and watch the per-solute drift blow up. At what
mesh does conservation recover? Relate the answer to the interface width
`ℓ ~ √(κ/W)` of Chapter P1.**
Hint: level 4 (16×16) leaks ~10% of a solute; level 5 (32×32) is the
coarsest mesh that still conserves to machine precision (this is exactly
why `quick` mode is pinned to level 5, not level 4). The mechanism is
resolution of the demixed interfaces and the surface-enrichment layer —
compare the interface width `ℓ ~ √(κ/W)` (with this chapter's `κ` and
Flory–Huggins `χ`) in physical units against the level-4 and level-5 cell
size `Δx` and see at which level `ℓ` first spans more than a couple of
cells. See `verification_solutions.md` for the full worked version.
