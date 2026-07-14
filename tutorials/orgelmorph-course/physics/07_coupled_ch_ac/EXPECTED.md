# P7 — expected results (self-check)

Running `python run.py` (defaults: **level 5 = 32×32**, p1 bulk,
χ_aa = 1.2, χ_ca = 2.6, seeded crystal) runs the commensurate-controls
battery and checks `outputs/results.json` against `baseline.yaml`
(tolerance-based, **not** bit-identical). Representative values (level 5,
cuDSS):

| quantity | value |
|---|---|
| four-fold χ (p1, **absolute** limits) aa/ca/ac/cc | 1.20 / 2.60 / 2.10 / 3.00 |
| same χ_ca in **r14** (increment χ_aa+χ_ca) | 3.80 |
| contrast — ch_only (shared mask) | −0.000 |
| contrast — coupled_nochi (crystal-bulk channel) | +0.376 |
| contrast — full (adds χ-expulsion) | +0.420 |
| crystal-bulk channel / χ-expulsion channel | +0.376 / +0.044 |
| relative full / coupled_nochi (denominator resolved) | 1.12× |
| seed sensitivity of full contrast (3 seeds) | 0.420 ± 0.001 |
| coupled free energy F (start → end) | −0.6592 → −0.6708 |
| largest positive energy step | 0 (monotone) |
| causality: frozen-geometry χ-channel | +0.190 |
| causality: kinetics fast / slow contrast | 0.420 / 0.371 |

**What must be true regardless of hardware** (the `baseline.yaml`
invariants):

- **The four-fold χ convention is explicit.** In **p1** (this chapter) the
  four χ are the *absolute* pair interactions at the pure limits; in
  **r14** the non-amorphous χ are *increments* over χ_aa. Verified by
  `test_chi_limits.py` against the production evaluator.
- **The comparison is commensurate.** All three controls (ch_only /
  coupled_nochi / full) are scored on the **same** mask (the full crystal
  footprint) with the **same** contrast metric. No divide-by-floor: the
  ch_only denominator is ≈0, so we report *absolute* contrasts and the
  channel increments, and a relative ratio only against the resolved
  coupled_nochi denominator.
- **Demixing decomposes into two channels.** The **crystal-bulk** term
  (φ_k² W) enriches the crystallizing species even with χ_ca = χ_aa — this
  is the dominant channel (≈0.38). The **χ-expulsion** (χ_ca > χ_aa) adds
  a smaller increment (≈0.04). Ordinary Cahn–Hilliard gives ≈0 on the
  crystal footprint.
- **The coupled free energy decreases monotonically** (largest positive
  step ≈ 0) — the coupling redistributes energy but cannot increase the
  total.
- **Causality runs in the right direction.** With the crystal frozen
  (L_psi = 0) the χ coupling still demixes at fixed geometry; fast and slow
  crystallization both reach a large contrast — the coupling sets the
  magnitude, the kinetics only the timing.

Notes: the crystal stays modest in size (area ≈ 0.02–0.06); the metric is
the *purity contrast*, not the crystal area. Third-significant-figure
differences from a different card are normal.
