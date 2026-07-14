# Differentiable D6 — Learning the free energy from snapshots

D3 recovered scalar *parameters* of a known free energy. This concept
recovers the free-energy *functional* itself: given a time series of
morphology snapshots, learn the bulk energy $f(c)$ that produced them.
This is the Wodo–Ganapathysubramanian *learn the thermodynamics from the
morphology* programme, at tutorial scale. We parametrize $f'(c)=\sum_i
a_i c^{p_i}$ on a monomial basis and recover the ground truth $f'(c)=c^3-c$
by L-BFGS on the adjoint gradient.

**Read** the course document, Chapter *"Learning the free energy from
snapshots"* (start with `learn_energy.py`).

**Run:**
```bash
python run.py                 # learn f'(c)=c^3-c from 10 snapshots
```

`run.py` FD-verifies the trajectory-loss gradient, recovers the
coefficients from clean data, and shows the noisy recovery sharpen with
more snapshots; compare with [`EXPECTED.md`](EXPECTED.md).

**Regenerate the document's figures + numbers** (optional):
```bash
python gen_figures.py         # writes ../../latex/figures/d6_recovery.png + numbers/d6.tex
```

| file | role |
|------|------|
| `learn_energy.py` | the core: `learn`, `loss_and_grad`, `fd_check` — read this first |
| `run.py` | the driver; gradient self-check + clean & noisy recovery |
| `gen_figures.py` | regenerates `d6_recovery.png`, `numbers/d6.tex` (seed-averaged) |
| `EXPECTED.md` | reference recovery your run should reproduce |

Uses `PolyBasisEnergy` + the production adjoint
(`src/diffsim/adjoint/phasefield.py`); `df'/da_i = c^{p_i}` feeds
`CHAdjoint.gradient` directly. An MLP free energy drops into the same
loop (the autograd twin already differentiates an arbitrary $f'$) — a
documented next step.
