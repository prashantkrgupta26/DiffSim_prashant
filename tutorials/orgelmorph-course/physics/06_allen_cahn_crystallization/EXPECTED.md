# P6 — expected results (self-check)

Running `python run.py` (defaults: 64×64, PCBM-class r14 energetics,
Tm = 558 K, fixed seed) should reproduce the following.

| quantity | value |
|---|---|
| grow (T = 333 K < Tm): area start → end | 0.0693 → 0.1101 (1.59×) |
| melt (T = 700 K > Tm): area start → end | 0.0693 → 0.0000 |
| Avrami exponent n (4 seeds, T = 250 K) | 1.47 |
| crystalline fraction X_end | 0.858 |
| nuclei → grains resolved | 4 → 3 |

**What must be true regardless of hardware:**

- **Below Tm the seeded crystal grows; above Tm it melts.** The driving
  force `drive = Δh(T/Tm − 1)` changes sign at the melting point — this
  sign flip is the whole point and must hold exactly.
- **The multi-nucleus run gives a JMAK sigmoid** X(t) rising from ≈0 to
  near 1, and the double-log fit returns a finite Avrami exponent.
- **The exponent is intermediate (≈1.5), below the ideal n = 2.** This is
  expected and honest: the nuclei are *finite pre-placed supercritical
  seeds* (point seeds are sub-critical here and redissolve — the
  Gibbs–Thomson floor), so the crystals start with area and the early
  t² regime is foreshortened. The exercises explore how the nucleation
  mode moves n.
- **Grain count ≤ seed count** (some seeds merge or a seed lands where
  another already grew).

Notes: the Avrami run uses a deeper undercooling (T = 250 K) and a larger
kinetic prefactor than the grow/melt demo so the crystalline fraction
actually sweeps the full range — a too-slow run pins X near 0 and the
exponent is meaningless. Third-significant-figure differences from a
different card are normal.
