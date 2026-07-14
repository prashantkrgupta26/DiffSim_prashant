# Differentiable D1 — What differentiable simulation is

The first differentiable concept. A classical solver answers *"what
happens?"*; a differentiable solver also answers *"what should I
change?"* — it returns the gradient $dJ/dp$ of any scalar output with
respect to any input. This concept computes that gradient **three
independent ways** through a short Cahn–Hilliard march (hand adjoint,
autograd twin, finite differences) and shows they agree, because a
gradient you cannot trust is worse than none.

**Read** the course document, Chapter *"What differentiable simulation
is"* (start with `three_way.py` — the importable core it walks through).

**Run:**
```bash
python run.py                 # 9x9 nodes, 3 steps, BDF1, ~seconds
python run.py --steps 4 --order 2   # variable-coefficient BDF2 march
```

`run.py` prints the three-way table and self-checks that the methods
agree; compare it with [`EXPECTED.md`](EXPECTED.md).

**Regenerate the document's figures + numbers** (optional):
```bash
python gen_figures.py         # writes ../../latex/figures/d1_*.png + numbers/d1.tex
```

| file | role |
|------|------|
| `three_way.py` | the core: `three_way`, `grad_adjoint/twin/finite_diff` — read this first |
| `run.py` | the driver you run; prints the self-check table |
| `gen_figures.py` | regenerates `d1_agreement.png`, `d1_fd_valley.png`, `numbers/d1.tex` |
| `EXPECTED.md` | reference gradients your run should reproduce |

The gradient is taken through the production phase-field brick: the
adjoint (`src/diffsim/adjoint/phasefield.py`) is bit-parity with
`CahnHilliardStepper` — no toy solver.
