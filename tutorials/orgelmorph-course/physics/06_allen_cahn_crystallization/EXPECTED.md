# P6 — expected results (self-check)

Running `python run.py` (defaults: **level 5 = 32×32**, PCBM-class r14
energetics, Tm = 558 K, fixed seed) runs the four-experiment battery and
checks `outputs/results.json` against `baseline.yaml` (tolerance-based,
**not** bit-identical). Representative measured values (level 5, cuDSS):

| quantity | value |
|---|---|
| grow (T = 333 K < Tm, drive −0.527): ⟨ψ⟩ start → end | 0.0681 → 0.1168 (1.72×) |
| melt (T = 700 K > Tm, drive +0.333): ⟨ψ⟩ end | 0.0034 (melts away) |
| interface velocity v at ΔT = 138 → 258 | +0.007 → +0.122 (monotone ↑) |
| critical radius r\* (T = 333 K, deep) | ≈ 0.045 |
| critical radius r\* (T = 400 K, mild) | ≈ 0.095 |
| r\*(mild)/r\*(deep) measured vs Gibbs–Thomson 1/\|drive\| | 2.11 vs 1.42 |
| Avrami exponent n (baseline-corrected, 4 seeds, T = 250 K) | 1.30 ± 0.07, R² = 0.976 |
| seed baseline X₀ / X_end (ψ>0.5) / ⟨ψ⟩_end | 0.024 / 0.856 / 0.820 |
| nuclei → grains resolved | 4 → 3 |

**What must be true regardless of hardware** (these are the `baseline.yaml`
invariants):

- **The driving force `drive = Δh(T/Tm − 1)` changes sign at Tm.** Below
  Tm (drive < 0) the seeded crystal grows; above Tm (drive > 0) it melts
  away. This sign flip is the whole point.
- **Crystallinity is measured by QUADRATURE**, ⟨ψ⟩ = ∫ψ dV / ∫dV (the
  primary number); the thresholded ψ>0.5 area is reported alongside for
  sensitivity (0.4/0.5/0.6 cuts differ only in the third digit here).
- **The growth-front interface velocity v increases monotonically with
  undercooling.** The model uses a *constant* mobility, so there is no
  thermal-transport maximum — a labelled simplification, not physics.
- **A critical radius separates growing from redissolving seeds, and it
  shrinks with undercooling** (r\* ∝ 1/\|drive\| qualitatively). The
  measured ratio (2.1) and the classical prediction (1.4) agree in sense
  and order of magnitude; the gap reflects finite-time classification, a
  weakly T-dependent effective interface energy, and the φ-coupling.
- **The multi-nucleus run gives a JMAK sigmoid**; the baseline-corrected
  double-log fit `ln[−ln(1−X*)] = n ln t + const` with X* = (X−X₀)/(1−X₀)
  returns a finite exponent with a clean R² and a tight CI.
- **The exponent is intermediate (≈1.3), below the ideal n = 2** — honest
  and expected: the nuclei are *finite pre-placed supercritical seeds*
  (point seeds are sub-critical here and redissolve — the very
  Gibbs–Thomson floor experiment [3] measures), so they start with area
  and the early t² regime is foreshortened.
- **Grain count ≤ seed count** (some seeds merge or land where another
  already grew).

**Accelerated parameters.** The Avrami run uses T = 250 K and L_psi = 11,
which are *pedagogical* accelerated values (not the physical PCBM rate)
chosen so X sweeps the full range in a short run. **Orientation θ is a
fixed grain LABEL, not an evolved field** (`theta_mode = "frozen"`, no
grain-boundary energy); evolving θ is the coupled concept of P7.

Third-significant-figure differences from a different card are normal.
