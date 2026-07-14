# Computational C1 — Convergence: basis order and time integration

The flagship computational concept. A discretization is only trustworthy
if it converges at the rate the theory predicts — and only if you measure
the *right* error. This tutorial promotes the Cahn–Hilliard convergence
gates that ship with the solver into a taught study that **separates the
error sources** and **controls the algebraic error**:

- **Spatial** — a *steady* manufactured solution (MMS), so the time error
  drops out and the measured error is pure spatial error. We report `L2`
  and `H1` for **both** fields `c` and `mu` over ≥4 mesh levels: `L2 ~
  h^(p+1)`, `H1 ~ h^p` for `p=1` and `p=2`.
- **Algebraic control** — tighten the Newton tolerance and show the
  discretization error does not move (the plot is not solver-limited).
- **Temporal** — self-convergence against a *verified* fine-`dt`
  reference (halve its `dt`, order stable, Richardson bound): BDF1 is
  first order, BDF2 second, over 4 `dt` levels.
- **Deliberate failures** — wrong BC, wrong source, under-resolved
  feature, under-resolved reference: four ways to get a wrong order.

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
