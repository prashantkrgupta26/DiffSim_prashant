# Physics P6 — Allen–Cahn crystallization

Crystallinity is a **non-conserved** order parameter (crystal can be
created), so it obeys Allen–Cahn rather than Cahn–Hilliard. This chapter
isolates crystallization through four experiments:

1. **grow / melt** — a seeded crystal grows below Tm and melts above it,
   measured by the **quadrature** crystallinity ⟨ψ⟩ = ∫ψ dV / ∫dV (the
   thresholded ψ>0.5 area is a secondary check);
2. **interface velocity** v as a function of undercooling ΔT = Tm − T
   (rises monotonically — constant mobility, no thermal maximum);
3. **critical radius** r\* separating growing from redissolving seeds,
   at two undercoolings (r\* ∝ 1/|drive|, shrinks with undercooling);
4. **Avrami/JMAK** kinetics — the crystalline fraction fit to a
   baseline-corrected exponent n ± CI with R².

**Read** the course document, Chapter *"Allen–Cahn crystallization"*
(start with `crystallization.py`).

**Run:**
```bash
python run.py                 # full battery + baseline checks, 32x32
python run.py --level 6       # finer mesh (slower)
```

`run.py` prints the four experiment tables, writes `outputs/results.json`,
and checks it against `baseline.yaml` (tolerance-based, not bit-identical);
compare with [`EXPECTED.md`](EXPECTED.md). Figures + numbers via
`python gen_figures.py`.

| file | role |
|------|------|
| `crystallization.py` | the core: `run_grow_melt`, `run_interface_velocity`, `run_critical_radius`, `run_avrami`, `quad_mean_psi` |
| `run.py` | driver; prints the battery, writes + checks results |
| `gen_figures.py` | regenerates `p6_*.png` + `numbers/p6.tex` |
| `baseline.yaml` | tolerance-based scientific-invariant checks |
| `EXPECTED.md` | reference numbers |

Uses `diffsim.physics.multiphase.MultiPhaseStepper` (M=1, K=1) with the
r14 crystallization energetics (PCBM-class, materials.yaml) and
`diffsim.diagnostics.conservation` for the quadrature crystallinity.

**Accelerated parameters:** the Avrami run uses T = 250 K and L_psi = 11,
pedagogical values (not the physical PCBM rate) chosen so X sweeps the
full range in a short run. **Orientation θ is a fixed grain label**, not
an evolved field — evolving θ is the coupled concept of P7.

## Learning objectives

By the end you can:

- Write the non-conserved (Allen–Cahn) gradient flow for crystallinity ψ
  and explain why crystal — unlike composition — does not need to be
  conserved, so the equation is second-order, not fourth-order.
- Explain why `drive = Δh(T/Tm − 1)` changes sign at the melting point,
  and predict (before running) whether a seed grows or melts at a given T.
- Measure interface velocity `v(ΔT)` and explain why it rises
  monotonically here (constant mobility) rather than showing the
  transport-limited maximum of a real material.
- Derive the critical-radius scaling `r* ∝ σ/|drive|` from the
  interface-vs-bulk energy balance, and bracket it numerically by
  classifying a radius sweep into grow/shrink at two undercoolings.
- Fit the Avrami/JMAK exponent `n ± CI` from the baseline-corrected
  double-log linearization, and explain why finite pre-placed seeds give
  `n` below the ideal 2/3 value.
- Distinguish the quadrature crystallinity ⟨ψ⟩ (primary) from the
  thresholded ψ>0.5 area (secondary, threshold-sensitive) measure.

## Prerequisites

- **Concepts:** non-conserved vs conserved order parameters, gradient
  flows, the Gibbs–Thomson (interface-vs-bulk energy) balance, and
  least-squares fitting with an uncertainty/CI (as in P1's coarsening
  exponent).
- **Chapters:** Chapter 00 (environment green). Chapter **P1**
  (`01_ch_binary_energies`) — the free-energy/gradient-flow language and
  the "measure the claim, report the uncertainty" habit this chapter
  reuses for interface velocity, `r*`, and the Avrami exponent.

## Expected cost

- **Device:** any CUDA GPU, 2-D, well under 1 GB of device memory — an
  8 GB card is ample. **Solver:** the multiphase brick with cuDSS (a
  documented `splu` fallback if cuDSS is unavailable).
- **Quick mode:** comparable to **P1's measured ≈ 75 s wall** on an RTX
  6000 Ada (mostly Warp kernel compile + Python start-up); target < 2 min.
  First run of a session pays the one-time compile cost.
- **Reference mode** (level 5 = 32×32, the four-experiment battery incl.
  the 4-seed Avrami fit): a few minutes; target 5–30 min.
- **Research mode** (finer mesh, `--level 6`): tens of minutes.

## Required deliverable

Submit the eight-item report of `../../ASSESSMENT.md`, specialised:

1. The `config.resolved.yaml` + `metadata.json` from a reference run.
2. The harness `baseline.yaml` check green (or a documented deviation).
3. The three figures (`p6_grow_melt`, `p6_kinetics`, `p6_avrami`)
   regenerated from your saved run.
4. **Headline:** the interface velocity `v(ΔT)` shown monotone
   increasing, the critical radius `r*` bracketed at two undercoolings,
   and the Avrami exponent `n ± CI` with its `R²` and fit window — all
   three together are the headline (the quadrature crystallinity
   ⟨ψ⟩ grow/melt trajectories and the thresholded area are secondary).
5. **Verification:** report both quadrature ⟨ψ⟩ and the thresholded
   ψ>0.5 area for the grow/melt runs and show they agree qualitatively;
   state the threshold sensitivity (0.4/0.5/0.6 cuts).
6. **Failure:** seed a sub-critical radius at a given undercooling and
   show it redissolves instead of growing (the Gibbs–Thomson floor) —
   diagnose which number (r vs r*) predicted this.
7. **Exploration:** answer one "Explore on your own" question with a
   plot (the activated-mobility `v(ΔT)` maximum or the `ε²` interface-
   width scaling are recommended).
8. **Research bridge:** one paragraph on what `v(ΔT)`, `r*`, and the
   Avrami exponent predict for real PCBM-class crystallization kinetics
   in a cast film.
