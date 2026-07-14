# Physics P7 — Coupled Cahn–Hilliard + Allen–Cahn

The real morphology problem is **coupled**: composition (Cahn–Hilliard)
and crystallinity (Allen–Cahn) evolve together. The signature is
**crystallization-driven demixing** — a crystallizing species enriches its
own phase and expels the others, sharpening the composition pattern.

This chapter (Phase-1 corrected):

1. **resolves the four-fold χ convention** — in **p1** (the mode run here)
   χ_aa/ac/ca/cc are *absolute* pair interactions at the pure limits; in
   **r14** the non-amorphous χ are *increments* over χ_aa. Verified by
   `test_chi_limits.py` against the production evaluator;
2. **measures the coupling with 3 commensurate controls** (ch_only /
   coupled_nochi / full) on the **same** mask and **same** metric — no
   divide-by-floor. The demixing decomposes into a dominant *crystal-bulk*
   channel and a smaller *χ-expulsion* channel;
3. **reports the coupled free-energy budget** (entropy + χ + cryst +
   gradients; total decreases — Lyapunov);
4. **runs causality experiments** — frozen geometry (coupling demixes at
   fixed geometry) and kinetics at fixed coupling (rate sets timing,
   coupling sets magnitude).

**Read** the course document, Chapter *"Coupled CH + AC"* (start with
`coupled.py`).

**Run:**
```bash
python test_chi_limits.py     # the four-fold chi convention (p1 & r14)
python run.py                 # controls + budget + causality + checks, 32x32
python run.py --level 6       # finer mesh (slower)
```

`run.py` prints the four experiment blocks, writes `outputs/results.json`,
and checks it against `baseline.yaml` (tolerance-based); compare with
[`EXPECTED.md`](EXPECTED.md). Figures + numbers via `python gen_figures.py`.

| file | role |
|------|------|
| `coupled.py` | core: `run(mode=...)`, `chi_eff_limits`, `coupled_energy`, `contrast_on_mask` |
| `run.py` | driver; controls + seed sweep + budget + causality + checks |
| `test_chi_limits.py` | unit test of the four pure χ limits (p1 & r14) |
| `gen_figures.py` | regenerates `p7_*.png` + `numbers/p7.tex` |
| `baseline.yaml` | tolerance-based scientific-invariant checks |
| `EXPECTED.md` | reference numbers |

Uses `diffsim.physics.multiphase.MultiPhaseStepper` with the four-fold χ
and `diffsim.diagnostics` (conservation/energy) for the quadrature
free-energy budget.
