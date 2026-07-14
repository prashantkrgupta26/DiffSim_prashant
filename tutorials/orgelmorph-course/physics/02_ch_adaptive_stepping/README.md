# Physics P2 — Cahn–Hilliard with adaptive time stepping

The same binary quench as P1, integrated two ways: a **fixed** time step
and an **error-controlled adaptive** step (step-doubling LTE control).
You see the time step grow by orders of magnitude through the coarsening
tail, the resulting step-count saving, and that the free-energy
trajectory and final morphology agree with the fixed-dt reference.

**Read** the course document, Chapter *"Adaptive time stepping"* (start
with `adaptive.py` — the importable core).

**Run:**
```bash
python run.py                 # fixed vs adaptive, 64x64
python run.py --tol 3e-4      # tighter tolerance -> smaller steps
python run.py --level 6
```

`run.py` prints a cost/accuracy self-check table; compare it with
[`EXPECTED.md`](EXPECTED.md).

**Regenerate the document's figures + numbers:**
```bash
python gen_figures.py         # writes ../../latex/figures/p2_*.png + numbers/p2.tex
```

| file | role |
|------|------|
| `adaptive.py` | the core: `run_fixed`, `run_adaptive`, `compare` — read first |
| `run.py` | the driver you run; prints the self-check table |
| `gen_figures.py` | regenerates the figures and `numbers/p2.tex` |
| `EXPECTED.md` | reference numbers your run should reproduce |

Uses the production brick `diffsim.physics.cahn_hilliard`
(`CahnHilliardStepper` + `adaptive_march`) directly. The variable-
coefficient BDF2 that keeps the adaptive step second-order is discussed
in the Computational track.
