# P5 — expected results (self-check)

Reference mode (`run_harness.py --mode reference`: level 6 = 64×64,
χ = (1.5, 0.3, 0.3), initial blend φ = (0.22, 0.22) — solvent 0.56, fixed
seed) sweeps evaporation rate `k_e` and solute mobility `D_s`, drying
**every** rate past φ_s = 0.10 and comparing morphology at a **matched
dryness** φ_s ∈ {0.30, 0.20, 0.10}. Ground truth on RTX 6000 Ada
(CUDA 12.9, cuDSS).

## Matched-dryness rate comparison (baseline row, D_s = 0.2)

| k_e (Bi) | exit | t_dry | wavelength (cells) @ φ_s=0.20 | contrast @ φ_s=0.20 |
|---|---|---|---|---|
| 0.30 (Bi 1.5) | `phis_stop` | 6.05 | 33.4 | 0.367 |
| 0.45 (Bi 2.25) | `phis_stop` | 4.1 | 28.6 | 0.366 |
| 0.60 (Bi 3.0) | `phis_stop` | 3.06 | 26.1 | 0.365 |

## What must be true regardless of hardware

- **Every film actually dries.** All runs exit `phis_stop` (dried to the
  cutoff), *not* `t_end`; φ_s falls monotonically and the height shrinks
  (h → 0.478 at φ_s ≈ 0.08). A t_end-truncated run is never called
  "dried".
- **Only the solvent leaves — solute is conserved exactly.** The
  per-solute moving-frame content h(t)·∫φ_i drifts by ~10⁻¹⁵ (machine
  precision) at level 5–6, and the height/solvent budget closes to the
  same order. **This is resolution-limited**: at level 4 (16×16) the same
  run leaks ~10 % because the demixed interfaces and the surface enrichment
  layer are under-resolved. That is why `quick` mode uses level 5.
- **Dryness sets the *degree* of demixing.** The phase contrast rises
  monotonically as the film dries (0.316 → 0.367 → 0.418 at φ_s = 0.30,
  0.20, 0.10) and does so **rate-independently**: at matched φ_s = 0.20 the
  contrast agrees across all seven runs to ~0.9 %.
- **The domain *size* collapses onto the Biot number** Bi = k_e·h0/D_s.
  Runs that share a Bi from *different* (k_e, D_s) give the same
  matched-dryness wavelength (spread ~6 %). The sweep spans
  Bi ∈ [0.75, 6.0], straddling the Bi = 1 drying/diffusion crossover. The
  wavelength itself is statistically noisy (single seed, ~2 lateral
  domains across the box), so trust the collapse, not a sharp size-vs-Bi
  law.
- **The old "faster = finer" was a confound.** It compared a *dried* fast
  film against a *still-wet* slow one — a dryness difference read as a rate
  effect. The units are also fixed: the wavelength is a real length
  (fraction of box width, and in cells = fraction × n_x > 1), never a bare
  "cells" number below one.

Third-significant-figure differences from another card are normal; the
qualitative story (all dry; solute conserved; dryness sets demixing
degree; size collapses onto Bi) must hold. `run.py` prints the
matched-dryness table; `run_harness.py` runs the full workflow and checks
`baseline.yaml`.
