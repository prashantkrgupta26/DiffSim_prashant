# Physics P8 — Thermal noise and nucleation

Where do crystal seeds come from? From thermal **fluctuations**. An
undercooled melt is metastable; the crystal cannot form until a
fluctuation pushes a region over the nucleation barrier. This concept
turns on the physical FDT noise and shows nucleation happening, plus a
noise-amplitude sweep. Here the noise is **physics** (its amplitude is
kᵦT), unlike the numerical seed of Chapter 1.

**Read** the course document, Chapter *"Noise and nucleation"* (start
with `nucleation.py`).

**Run:**
```bash
python run.py                 # noise-amplitude sweep, 64x64
python run.py --level 6
```

`run.py` prints the final crystalline fraction, grain count, and
induction time vs noise amplitude; compare with
[`EXPECTED.md`](EXPECTED.md). Figures via `python gen_figures.py`.

| file | role |
|------|------|
| `nucleation.py` | the core: `run_one`, `sweep` |
| `run.py` | driver; prints the noise sweep |
| `gen_figures.py` | regenerates `p8_*.png` + `numbers/p8.tex` |
| `EXPECTED.md` | reference numbers |

Uses `diffsim.physics.multiphase.MultiPhaseStepper` (M=1, K=1) with FDT
noise (`noise_psi`); BDF1 only (the brick asserts noise off under BDF2).
