# P10 — hints

**Q1 (swap to P3HT_PCBM).** Set `system: P3HT_PCBM` — but note its polymer N is
recorded as *unknown* (value `None`, range [100, 1000]), so you must also set a
concrete `N` (e.g. `[100, 1.0, 1.0]`) and *flag it as an assumed-within-range
choice*. You will find the phase contrast at matched dryness is an order of
magnitude smaller than for PDPP5T:PCBM — weak χ_pf ≈ 0.86 barely demixes at
tutorial scale. That is the honest negative: contrast is not a usable screening
metric for a blend that does not phase-separate on the accessible time/length
scale.

**Q2 (widen the k_e window).** Increase `k_e_fast` (e.g. to 1.0). The Biot span
grows, so the wavelength *ordering* has more room — but at a single seed the
run-to-run scatter usually still swamps it. This is the setup for Q3.

**Q3 (seed ensemble).** Loop `--seed` over ~8–16 values at one rung, collect the
matched-φ_s wavelength, and report mean ± std. Compare the slow-vs-fast
difference to the ensemble std: if the difference is smaller than the spread,
the "faster = finer" claim does not clear its own error bar. This is exactly the
campaign machinery of chapter C9.

**Q4 (level-7 polymer conservation).** Run `--mode research` (or set
`conv_levels: [7]`). The polymer per-solute balance should shrink toward the
fullerene's machine-precision value as the surface layer is resolved —
confirming the residual is a resolution artefact, not a conservation bug.
