# Hints — P7 exploratory questions

*Hints for every "Explore on your own" question (course document `p7.tex`,
§"Explore on your own"). These are nudges, not answers — full worked
solutions for the two verification exercises are in
`verification_solutions.md`. Do not distribute the solutions file to
students before the deadline.*

**Q1 — Sweep χ_ca upward; does the χ-expulsion channel ever overtake the
crystal-bulk channel?** Hint: the crystal-bulk channel (`coupled_nochi −
ch_only`) does not depend on χ_ca at all — it comes from the `φ_k² W`
term alone. The χ-expulsion channel (`full − coupled_nochi`) grows with
`χ_ca − χ_aa`. Plot both against the sweep on the same axes; at the
reference point (χ_ca=2.6, χ_aa=1.2) the crystal-bulk channel is ~8.5×
larger — find how far you must push χ_ca to close that gap, and whether
that value is still physically reasonable (check `materials.yaml`).

**Q2 — Switch the bulk to r14; re-derive the matching pure-limit χ.**
Hint: p1's absolute pure limit is `χ_eff(1,0) = χ_ca`; r14's pure limit
is `χ_eff(1,0) = χ_aa + χ_ca^{r14}`. To match the *same* absolute
interaction, you need `χ_ca^{r14} = χ_ca^{p1} − χ_aa`. Verify by feeding
your derived value into `chi_eff_limits(bulk="r14")` and comparing
against `test_chi_limits.py`'s p1 reference numbers — they should agree
to solver tolerance.

**Q3 — Delay the crystal seed until after partial demixing.** Hint: run
the CH-only blend for some steps first, then implant the crystal and
turn on K. Compare the resulting composition pattern's characteristic
length scale to a simultaneous run's. If the crystal snaps to an
existing domain, the crystal-bulk channel "templates" onto pre-existing
structure rather than creating new structure — measure this with the
same contrast-on-mask machinery, just at different implant times.

**Q4 — Two crystallizers, (M,K)=(3,2): compete or segregate?** Hint:
re-run the commensurate-controls battery once per crystallizing species
and check whether their `ch_only`/`coupled_nochi`/`full` masks overlap.
If the two species' crystal footprints overlap heavily, you likely have
competition for the same amorphous "b" reservoir; if they're spatially
separated, look at whether the initial noise seed alone explains the
segregation or whether the crystal-bulk channel is actively pushing them
apart.

**Q5 — Relate the purity contrast to solar-cell transport needs.** Hint:
this is a discussion/design question, not a numerical one — pair it with
the "Research bridge" deliverable item. A pure domain helps carrier
transport within a phase but a *fully* segregated, non-bicontinuous
morphology kills charge separation at the interface. Use the
crystal-bulk vs χ-expulsion channel split to argue which knob (crystal
energetics vs χ) an experimentalist would tune to get "pure but still
percolating" domains, and connect to P9's arc (drying-conditioned
composition sets when/where this template forms).
