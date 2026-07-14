# Physics P8 — Thermal noise and nucleation

Where do crystal seeds come from? From thermal **fluctuations**. An
undercooled melt is metastable; the crystal cannot form until a
fluctuation pushes a region over the nucleation barrier. Here the noise is
**physics** (its amplitude is kᵦT), unlike the numerical seed of Chapter 1.

This chapter (Phase-1 corrected) does two things:

1. **DERIVES and VERIFIES the discrete FDT normalization.** The per-Gauss-
   point noise std carries √(2L/(dt·wJ)) so the assembled nodal-force
   variance is mesh/dt-independent (the discrete fluctuation-dissipation
   theorem). The verification fixes kᵦT, puts ψ in a stable quadratic well,
   and shows the equilibrium variance converges to a **dt-independent**
   plateau and scales as 1/V_cell across meshes (equipartition). This is
   the central result.
2. **Reports nucleation as ENSEMBLES.** A noise-amplitude (kᵦT calibration)
   sweep runs several realizations per amplitude and reports the nucleation
   probability, crystalline-fraction distribution, nuclei density and
   induction time (mean + 95% interval) via `diffsim.diagnostics.stochastic`
   — not a single seed. Clipping is quantified.

**Read** the course document, Chapter *"Noise and nucleation"* (start with
`nucleation.py`).

**Run:**
```bash
python run.py                 # FDT verification + ensemble sweep + checks
python run.py --level 6       # finer nucleation mesh (slower)
```

`run.py` runs the FDT verification and the ensemble sweep, writes
`outputs/results.json`, and checks it against `baseline.yaml`; compare with
[`EXPECTED.md`](EXPECTED.md). Figures + numbers via `python gen_figures.py`.

| file | role |
|------|------|
| `nucleation.py` | core: `run_one`, `run_ensemble`, `sweep`, `fdt_well`, `verify_fdt` |
| `run.py` | driver; FDT verification + ensemble sweep + checks |
| `gen_figures.py` | regenerates `p8_*.png` + `numbers/p8.tex` |
| `baseline.yaml` | tolerance-based scientific-invariant checks |
| `EXPECTED.md` | reference numbers |

Uses `diffsim.physics.multiphase.MultiPhaseStepper` (M=1, K=1) with FDT
noise (`noise_psi`) and `diffsim.diagnostics.stochastic` for the ensemble
aggregation. **This is a noise-amplitude sweep, not a temperature sweep**
(only `noise_psi` varies). BDF1 only (the brick asserts noise off under
BDF2). The FDT well runs on a coarse CPU/splu mesh (fast, no cuDSS
re-plan); the nucleation ensemble runs at level 5 on the GPU.

## Learning objectives

By the end you can:

- Derive the discrete fluctuation–dissipation (FDT) noise normalization
  for a per-Gauss-point white-noise forcing — why the `√(2L/(dt·wJ))`
  scaling makes the assembled nodal-force variance mesh/dt-independent —
  and explain what would go wrong (variance drifting with mesh or `dt`)
  if the `1/wJ` or `1/dt` factor were missing.
- Design and run the equipartition verification: a clip-free, stable
  quadratic well, showing the equilibrium variance is `dt`-independent
  and scales as `1/V_cell` across meshes.
- Explain why nucleation must be reported as an **ensemble** (probability,
  crystalline-fraction distribution, nuclei density, induction time with
  a 95% CI) rather than a single seed's trajectory, and compute those
  statistics with `diffsim.diagnostics.stochastic`.
- State precisely why this is a **noise-amplitude** sweep and not a
  temperature sweep — and what a genuine temperature change would also
  move (driving force, barrier, mobility) that this sweep holds fixed.
- Quantify clipping (the ψ∈[0,1] projection) and explain why the
  quantitative FDT test must avoid it (a clip-free well) while the
  nucleation runs must quantify it (saturated-ψ fraction).

## Prerequisites

- **Concepts:** the Allen–Cahn crystallization model and driving-force
  sign (P6); fluctuation–dissipation theorems and equipartition; ensemble
  statistics (mean, sd, confidence intervals) as introduced in P1's
  5-seed ensemble discussion.
- **Chapters:** Chapter 00 (environment green). **P6**
  (`06_allen_cahn_crystallization`) for the crystallization model that
  the noise perturbs; **P1**'s seed discussion (why a single trajectory
  is one sample, and the seed here is now *physical* rather than
  numerical).

## Expected cost

- **Device:** any CUDA GPU, 2-D, well under 1 GB of device memory — an
  8 GB card is ample. **Solver split (chapter-specific):** the FDT-well
  verification runs on a **coarse CPU/splu mesh** (fast, deliberately
  avoiding a cuDSS symbolic re-plan for a tiny problem); the nucleation
  **ensemble** runs at level 5 on the **GPU**. **BDF1 only** — the brick
  asserts noise off under BDF2.
- **Quick mode:** comparable to **P1's measured ≈ 75 s wall** on an RTX
  6000 Ada (mostly Warp kernel compile + Python start-up); target < 2 min.
  First run of a session pays the one-time compile cost.
- **Reference mode** (FDT `dt`/mesh sweep + 4-seed-per-amplitude ensemble
  sweep): a few minutes; target 5–30 min. **The ensemble multiplies wall
  time by the realization count** — a 4-seed sweep at 3 amplitudes is
  ~12 realizations, not 1.
- **Research mode** (finer nucleation mesh, `--level 6`, more seeds):
  tens of minutes to an hour, scaling with realization count.

## Required deliverable

Submit the eight-item report of `../../ASSESSMENT.md`, specialised:

1. The `config.resolved.yaml` + `metadata.json` from a reference run.
2. The harness `baseline.yaml` check green (or a documented deviation).
3. The three figures (`p8_fdt`, `p8_sweep`, `p8_fields`) regenerated from
   your saved run.
4. **Headline (two parts, both required):** (a) the discrete FDT
   normalization verified — equilibrium variance shown `dt`-independent
   (CoV ≈0.02) and scaling as `1/V_cell` across meshes (CoV ≈0.11,
   equipartition); (b) the nucleation **ensemble** statistics — nucleation
   probability, induction-time distribution, and mean ± 95% CI of the
   crystalline fraction, at ≥2 noise amplitudes.
5. **Verification:** the zero-noise control showing no nucleation
   (X≈0, P=0) as the negative control for the ensemble statistics.
6. **Failure:** remove the `1/wJ` mesh-quadrature factor from the noise
   normalization (Exercise 2) and show the equilibrium variance now
   depends on the mesh — diagnose why this breaks equipartition.
7. **Exploration:** answer one "Explore on your own" question with a
   plot (the barrier-slope-from-ensemble question is recommended).
8. **Research bridge:** one paragraph on what the FDT-verified noise
   amplitude and the ensemble nucleation statistics predict for a real
   undercooled-melt nucleation rate.
