# P9 — expected results (self-check)

Running `python run.py` (defaults: 64×64, M=2/K=1 film mode, r14 with the
crystal-contact solubility, fixed seed) implants the same crystal seeds
in the film either early (wet) or mid-drying, and reports the drying
state at implant and the terminal crystalline area for each.

Measured (32×32, level 5, k_e = 0.1, fixed seed):

| quantity | wet implant | dry implant |
|---|---|---|
| φ_s at implant | 0.836 | 0.633 |
| crystalline area: start → end | 0.312 → 0.000 | 0.312 → 1.000 |
| ψ_max (terminal) | 0.00 | 0.99 |

The wet seed **dissolves completely** (area → 0, ψ_max → 0); the same
seed implanted mid-drying **grows to a fully crystalline film**
(area → 1, ψ_max → 0.99). The fate is set entirely by the solvent
fraction at implant.

**What must be true regardless of hardware:**

- **Seeds implanted in the wet film dissolve.** In the solvent-rich film
  the crystallizable species sits below the r14 solubility φ\*, so the
  crystal-contact penalty beats the undercooling and the seed
  redissolves (terminal crystalline area ≈ 0, ψ_max small).
- **Seeds implanted mid-drying grow.** Once drying has concentrated the
  species above φ\*, and continuing solvent loss keeps deepening the
  quench, the same seed grows to a crystalline film (terminal area large,
  ψ_max near 1).
- **Crystallization strictly follows solvent loss.** The φ_s at the two
  implant times differ (wet ≫ dry); the fate is set entirely by *when*
  the seed is introduced relative to the drying — the evaporation-induced
  ordering.
- **Seeding has margin.** The implant radius clears the Gibbs–Thomson
  critical radius and the amplitude is ≈0.95; a sub-critical embryo
  (too small, or half-amplitude) redissolves even mid-drying. Read the
  **terminal** state, not a mid-growth transient.

**Recorded simplification.** This tutorial uses a constant Onsager
mobility (with the cuDSS solver) instead of the production Vignes
composition-singular mobility of `tests/test_multiphase_s3.py`; the
qualitative arc (dissolve-when-wet vs grow-when-dry) is unchanged, the
drying-front sharpness is softened.

If a mid-drying seed dissolves, increase the implant radius/amplitude or
the implant time; if a wet seed grows, the solubility (χ_ca) is too low.
