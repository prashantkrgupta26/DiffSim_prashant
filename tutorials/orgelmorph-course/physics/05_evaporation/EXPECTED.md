# P5 — expected results (self-check)

Running `python run.py` (defaults: 64×64, χ = (1.5, 0.3, 0.3), initial
blend φ = (0.22, 0.22) — solvent 0.56, quench to *t* = 6, fixed seed)
should reproduce the following, for the evaporation-rate sweep
k_e ∈ {0.15, 0.3, 0.6}.

| quantity | slow (k_e = 0.15) | fast (k_e = 0.6) |
|---|---|---|
| drying: solvent φ_s final | 0.296 | 0.099 |
| film height h final | 0.624 | 0.488 |
| morphology domain scale | 0.600 | 0.398 |

**What must be true regardless of hardware:**

- **Every film dries** — the mean solvent fraction φ_s falls and the
  height h shrinks; a larger k_e dries faster (the fast film reaches the
  dryness cutoff, the slow one is still drying at the horizon).
- **Faster evaporation freezes a finer morphology.** The domain scale is
  *smaller* for the fast film (≈0.40) than the slow film (≈0.60): faster
  drying quenches the blend before it can coarsen. If your fast film came
  out coarser, re-check that both films actually demixed (contrast > 0).
- **The blend must demix as it concentrates.** With χ = 1.5 the solutes
  are miscible while dilute and demix only once the solvent has largely
  left — that is the evaporation-*induced* separation. A film that never
  loses enough solvent stays mixed and its "domain scale" is just noise.

Note: the slow run marches the full horizon (drying is slow) and takes
longer than the fast run (which stops early at the dryness cutoff).
Third-significant-figure differences from a different card are normal;
the qualitative story (all dry, faster = finer) must hold.
