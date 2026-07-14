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
