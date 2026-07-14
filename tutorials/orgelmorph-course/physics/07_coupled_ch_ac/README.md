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

## Learning objectives

By the end you can:

- Explain why crystallinity introduces a *four-fold* χ (aa/ac/ca/cc)
  instead of a single Flory parameter, and correctly convert between the
  **p1** (absolute pair interactions at the pure limits) and **r14**
  (increments over χ_aa) conventions — and verify both against the
  production evaluator with `test_chi_limits.py`.
- Design a **commensurate-controls** experiment (same mask, same metric,
  no divide-by-floor) to isolate a coupling's effect, rather than
  comparing differently-masked runs.
- Decompose crystallization-driven demixing into its two physical
  channels — the crystal-bulk term (`φ_k² W`, present even with
  χ_ca = χ_aa) and the χ-expulsion term (χ_ca > χ_aa) — and state which
  dominates.
- Assemble and verify a coupled free-energy budget
  (entropy + χ + crystal + gradients) and confirm the total decreases
  monotonically.
- Run causality controls (frozen geometry; fixed coupling at different
  kinetic rates) to distinguish "the coupling causes demixing" from
  "crystallization happening at all causes demixing."

## Prerequisites

- **Concepts:** the Allen–Cahn crystallization model and its driving
  force sign (P6); the Flory–Huggins χ parameter and the ternary/χ
  extension (P4); the "measure the claim, report the uncertainty" habit
  (P1).
- **Chapters:** Chapter 00 (environment green). **P6**
  (`06_allen_cahn_crystallization`) for the crystallinity model and
  non-conserved gradient flow; **P4** (ternary blends / χ) for the
  Flory–Huggins interaction parameter this chapter promotes to a
  four-fold matrix; **P1** for the free-energy/gradient-flow language
  and the discrete-energy-stability habit.

## Expected cost

- **Device:** any CUDA GPU, 2-D, well under 1 GB of device memory — an
  8 GB card is ample. **Solver:** the multiphase brick with cuDSS (a
  documented `splu` fallback if cuDSS is unavailable).
- **Quick mode:** comparable to **P1's measured ≈ 75 s wall** on an RTX
  6000 Ada (mostly Warp kernel compile + Python start-up); target < 2 min.
  First run of a session pays the one-time compile cost.
- **Reference mode** (level 5 = 32×32, the commensurate-controls battery
  + budget + causality experiments): a few minutes; target 5–30 min.
- **Research mode** (finer mesh, `--level 6`): tens of minutes.

## Required deliverable

Submit the eight-item report of `../../ASSESSMENT.md`, specialised:

1. The `config.resolved.yaml` + `metadata.json` from a reference run.
2. The harness `baseline.yaml` check green (or a documented deviation).
3. The two figures (`p7_demixing`, `p7_ladder`) regenerated from your
   saved run.
4. **Headline:** the demixing decomposed into the crystal-bulk channel
   vs the χ-expulsion channel via the three commensurate controls
   (ch_only / coupled_nochi / full) on the same mask and metric, **and**
   the four-fold χ limits verified against `test_chi_limits.py` (p1
   absolute vs r14 increment), **and** the coupled energy budget shown
   decreasing monotonically.
5. **Verification:** the `test_chi_limits.py` pure-limit check passing
   for both p1 and r14 conventions; the commensurate-controls contrast
   reproduced across ≥3 seeds with its spread reported.
6. **Failure:** confuse the two χ conventions (feed an r14 increment
   value where p1 expects an absolute) and show the resulting χ_eff pure
   limits disagree with `test_chi_limits.py` — diagnose which number
   revealed the mistake.
7. **Exploration:** answer one "Explore on your own" question with a
   plot (the χ_ca sweep vs the χ-expulsion channel, or the p1→r14
   conversion exercise, are recommended).
8. **Research bridge:** one paragraph on what the crystal-bulk vs
   χ-expulsion decomposition predicts for purity/network trade-offs in a
   real solution-cast blend (P7's closing "explore on your own" framing).
