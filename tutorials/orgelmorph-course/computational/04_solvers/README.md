# Computational C4 — The solver ecosystem

Every implicit step ends in a linear solve, and there is no single best
solver. This tutorial teaches how to *choose*, using a small live 2-D
benchmark plus the measured 3-D scaling tables from the dev notes.

- **Direct (splu / cuDSS)** — LU factorization. Robust, exact, no
  tuning; cheap in 2-D, brutal fill-in in 3-D. Measured live: cuDSS
  (GPU) overtakes splu (CPU) as the 2-D problem grows past ~10⁴ dofs.
- **Block preconditioner (blockch / blockch_dev)** — iterative outer +
  block factorization; latency-bound (loses in 2-D by 1–2 orders) but
  wins in 3-D past cuDSS's memory wall (**8.4×** at slab64; marches at
  811k dofs where cuDSS ceilings).
- **Matrix-free** — never stores the matrix; the only survivor beyond
  int32 nnz / card memory (128×128×64 = 6.4M dofs on one 48 GB card).

**Read** the course document, Computational Chapter *"The solver
ecosystem"* (start with `solvers.py`).

**Run:**
```bash
python run.py                 # live 2-D benchmark + cited 3-D + decision rule
```
`run.py` prints the crossover table, the cited 3-D story, and the
`choose_solver` rule; compare with [`EXPECTED.md`](EXPECTED.md).

**Regenerate the document's figures + numbers** (optional):
```bash
python gen_figures.py         # writes ../../latex/figures/c4_*.png + numbers/c4.tex
```

| file | role |
|------|------|
| `solvers.py` | the core: `benchmark_solvers_2d`, `capture_ch_system`, `choose_solver`, `CITED_3D` — read this first |
| `run.py` | the driver you run; prints the benchmark + decision rule |
| `gen_figures.py` | regenerates the crossover + cited-3D figures and `numbers/c4.tex` |
| `EXPECTED.md` | reference numbers your run should reproduce (shape, not exact ms) |

The **2-D numbers are live** (the real Cahn–Hilliard Jacobian, captured
from the production brick). The **3-D numbers are cited** from
`docs/dev/2026-07-13-m5-device-assembly.md` (D3 table) and
`docs/dev/2026-07-13-blockch-mpf.md` (B2–B5) — they take minutes per step
to reproduce, so we do not re-run them here.
