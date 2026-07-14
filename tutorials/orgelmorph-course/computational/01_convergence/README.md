# Computational C1 — Convergence: basis order and time integration

The flagship computational concept. A discretization is only trustworthy
if it converges at the rate the theory predicts, and the only honest way
to know is to *measure* it. This tutorial promotes the two Cahn–Hilliard
convergence gates that ship with the solver into a taught study:

- **Spatial** — the method of manufactured solutions (MMS): plug a known
  field into the equations, add its residual as a source, and check the
  $L^2$ error falls like $h^{p+1}$ for $p=1$ and $p=2$.
- **Temporal** — self-convergence against a fine-$\Delta t$ reference:
  BDF1 (backward Euler) is first order in $\Delta t$, BDF2 is second.

**Read** the course document, Computational Chapter *"Convergence"*
(start with `convergence.py` — the importable core it walks through).

**Run:**
```bash
python run.py                 # both studies, ~1-2 min on the GPU
```
`run.py` prints the observed orders and a `PASS/FAIL` self-check;
compare with [`EXPECTED.md`](EXPECTED.md).

**Regenerate the document's figures + numbers** (optional):
```bash
python gen_figures.py         # writes ../../latex/figures/c1_*.png + numbers/c1.tex
```

| file | role |
|------|------|
| `convergence.py` | the core: `spatial_mms`, `temporal_convergence`, the manufactured solution — read this first |
| `run.py` | the driver you run; prints the order tables + gate check |
| `gen_figures.py` | regenerates the log-log figures and `numbers/c1.tex` |
| `EXPECTED.md` | reference orders your run should reproduce |

The tutorial uses the production brick
`src/diffsim/physics/cahn_hilliard.py` and the manufactured sources from
`tests/test_cahn_hilliard.py` (`test_ch_mms_orders`,
`test_ch_bdf2_variable_dt_order`) — no toy solver.
