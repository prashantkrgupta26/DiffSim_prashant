# Physics P7 — Coupled Cahn–Hilliard + Allen–Cahn

The real morphology problem is **coupled**: composition and crystallinity
evolve together. The signature of the coupling is **crystallization-driven
demixing** — as a species crystallizes it expels the others, sharpening
the composition pattern. This concept turns the coupling on/off, measures
the feedback, and scales it up the (M,K) ladder: (2,1) → (3,1) → (3,2).

**Read** the course document, Chapter *"Coupled CH + AC"* (start with
`coupled.py`).

**Run:**
```bash
python run.py                 # ON vs OFF + the ladder, 64x64
python run.py --level 6
```

`run.py` prints the demixing purity contrast with and without
crystallization; compare with [`EXPECTED.md`](EXPECTED.md). Figures via
`python gen_figures.py`.

| file | role |
|------|------|
| `coupled.py` | the core: generic `(M,K)` coupled `run` |
| `run.py` | driver; prints the demixing contrast table |
| `gen_figures.py` | regenerates `p7_*.png` + `numbers/p7.tex` |
| `EXPECTED.md` | reference numbers |

Uses `diffsim.physics.multiphase.MultiPhaseStepper` with the four-fold χ
(χ_aa/ac/ca/cc) that couples composition and crystallinity.
