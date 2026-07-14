# Physics P9 — Evaporation-induced crystallization (the hero concept)

The full arc, at tutorial scale: a **wet** ternary film **dries**, the
concentrating blend **phase-separates**, and the crystallizable species
**nucleates and grows** into a **crystalline film** — how a real
solution-cast organic solar cell forms. The mechanism: crystallization
is forbidden below a solubility set by the drying, so seeds implanted in
the wet film **dissolve** while the same seeds implanted mid-drying
**grow**.

**Read** the course document, Chapter *"Evaporation-induced
crystallization"* (start with `arc.py`).

**Run:**
```bash
python run.py                 # wet-implant vs dry-implant, 64x64
python run.py --level 6
```

`run.py` prints the drying state at implant and the terminal crystalline
area for both cases; compare with [`EXPECTED.md`](EXPECTED.md). The stage
strip + wet-vs-dry figures come from `python gen_figures.py`.

| file | role |
|------|------|
| `arc.py` | the core: `run_arc`, `implant`, `sites` (S3b patterns) |
| `run.py` | driver; prints the mechanism table |
| `gen_figures.py` | regenerates `p9_*.png` + `numbers/p9.tex` |
| `EXPECTED.md` | reference numbers |

Uses `diffsim.physics.multiphase.MultiPhaseStepper` (M=2, K=1) in film
mode with r14 crystallization. Tutorial simplification (recorded in
(recorded in `arc.py`): constant mobility instead of the production Vignes
mobility of `test_multiphase_s3.py`.
