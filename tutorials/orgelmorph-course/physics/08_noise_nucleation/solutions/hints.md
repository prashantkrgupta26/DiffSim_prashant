# Hints — P8 exploratory questions

*Hints for every "Explore on your own" question (course document `p8.tex`,
§"Explore on your own"). These are nudges, not answers — full worked
solutions for the two verification exercises are in
`verification_solutions.md`. Do not distribute the solutions file to
students before the deadline.*

**Q1 — Estimate the barrier from ln(nuclei density) vs 1/noise_psi².**
Hint: classical nucleation theory predicts rate `~exp(−ΔF*/kBT)`, and
`noise_psi² = kBT`, so `1/noise_psi²` plays the role of `1/kBT`. Lower
the amplitude toward where the nucleation probability first drops below
1 (below the reference 0.03/0.06 range) — you need several amplitudes
spanning that onset to get a meaningful slope, not just two points.
Plot on a semi-log axis; the slope of the line is (up to a prefactor)
`−ΔF*`.

**Q2 — Remove the 1/wJ factor; does variance now depend on mesh?**
Hint: this is the mesh-blind-noise experiment — modify `verify_fdt`'s
noise-std expression to drop the `1/w^J_gp` term and re-run the mesh
sweep (same clip-free well). Compare `Var(ψ)·V_cell` across meshes: with
the correct normalization it's constant (CoV ≈0.11); without the `1/wJ`
factor it should now trend with resolution because finer meshes give
each Gauss point a smaller effective volume without compensating for it.
This IS the point of Eq. for `std(ξ_gp)` — you're demonstrating the
normalization is load-bearing, not cosmetic.

**Q3 — Deepen the undercooling at fixed noise; does induction time
shift earlier?** Hint: lowering T increases `|drive|`, which lowers the
nucleation barrier `ΔF*`. A lower barrier at the same noise amplitude
should cross the `X=0.02` first-passage threshold sooner on average —
plot the induction-time distribution (not just the mean) at two
undercoolings and see if it shifts left. Remember to flag explicitly:
changing T here IS a genuine temperature change (unlike the main
noise-amplitude sweep) — it moves the barrier, not just an amplitude.

**Q4 — Change only noise_seed; what's reproducible?** Hint: individual
grain positions will differ completely (nucleation sites are set by
where the random field happens to cross the barrier first) — but the
*ensemble* statistics (probability, mean X, nuclei density) should
reproduce within their reported CI across different seeds. This is
directly analogous to P1's discussion of morphology vs invariants, but
here the seed is a physical fluctuation realization, not a numerical
initial-condition artifact — a different seed is a genuinely different,
equally valid experimental replicate, not noise to be averaged away
apologetically.

**Q5 — Quantify clipping vs amplitude; when does it bias statistics?**
Hint: track the saturated-ψ (>0.99) fraction reported alongside each
amplitude (reference values ≈0.15 at noise_psi=0.03, ≈0.21 at 0.06).
Increasing the amplitude further should increase this fraction. To see
where clipping starts to bias *pre-nucleation* statistics (not just the
terminal state), look at the early-time variance in the clipped runs and
compare it to the clip-free well's prediction at the same nominal
amplitude — divergence between the two marks where the ψ=0 floor is
rectifying enough sub-barrier fluctuations to matter.
