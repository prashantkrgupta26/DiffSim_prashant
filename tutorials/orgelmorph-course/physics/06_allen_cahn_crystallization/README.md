# Physics P6 — Allen–Cahn crystallization

Crystallinity is a **non-conserved** order parameter (crystal can be
created), so it obeys Allen–Cahn rather than Cahn–Hilliard. This concept
isolates crystallization: a seeded crystal that **grows** below the
melting point and **melts** above it, the growth **kinetics** (the
Avrami/JMAK law), and **orientation** markers that distinguish grains.

**Read** the course document, Chapter *"Allen–Cahn crystallization"*
(start with `crystallization.py`).

**Run:**
```bash
python run.py                 # grow/melt + Avrami fit, 64x64
python run.py --level 6
```

`run.py` prints the grow/melt areas and the fitted Avrami exponent;
compare with [`EXPECTED.md`](EXPECTED.md). Figures via
`python gen_figures.py`.

| file | role |
|------|------|
| `crystallization.py` | the core: `run_grow_melt`, `run_avrami` |
| `run.py` | driver; prints the kinetics table |
| `gen_figures.py` | regenerates `p6_*.png` + `numbers/p6.tex` |
| `EXPECTED.md` | reference numbers |

Uses `diffsim.physics.multiphase.MultiPhaseStepper` (M=1, K=1) with the
r14 crystallization energetics (PCBM-class, materials.yaml).
