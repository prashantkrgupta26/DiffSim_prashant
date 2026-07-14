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

## Learning objectives

By the end you can:

- Separate the three error sources in a time-dependent FE computation —
  spatial, temporal, algebraic — and design a study that isolates each
  one at a time instead of measuring a tangled mixture.
- Measure the observed **spatial** order by a *steady* manufactured
  solution: `L2 ~ h^(p+1)` and `H1 ~ h^p`, for **both** `p=1` and `p=2`,
  and for **both** fields `c` and `mu`.
- Show a convergence plot is not solver-contaminated: tighten the Newton
  tolerance and confirm the discretization error is invariant (algebraic
  control), the load-bearing check that must pass before any order is
  quoted.
- Measure the observed **temporal** order (BDF1 first order, BDF2 second)
  by self-convergence against a fine-`dt` reference on a fixed
  over-resolved mesh, and **verify** that reference itself (halve its
  `dt`, check the order is stable, Richardson-bound its own residual
  error) before trusting it.
- Recognize the four deliberate-failure modes this chapter demonstrates —
  wrong BC, wrong source, an under-resolved feature, an under-resolved
  reference — and read the signature that diagnoses each: a
  plausible-looking but wrong order is not the same as a crash.
- Read a log-log convergence plot correctly: the observed order is the
  fitted *slope* across all levels, not any single error value or a
  two-point ratio.

## Prerequisites

- **Concepts:** the weak (finite-element) form and basis functions,
  Gauss quadrature, the method of manufactured solutions, the order of a
  BDF time scheme, and Richardson extrapolation. Python + NumPy.
- **Chapters:** Physics P1 (`01_ch_binary_energies`) — this chapter
  discretizes the *same* Cahn–Hilliard brick P1 introduces; read P1
  first if you have not. Chapter 00 (`00_setup_and_smoke_test`,
  environment green).

## Expected cost

- **Device:** any CUDA GPU; the meshes here are 2-D only and use well
  under 1 GB of device memory, so an 8 GB card is ample.
- **Solver:** direct sparse LU (`splu`) on the mixed `(c, mu)` saddle,
  exact to round-off — this is why the algebraic-control check can
  isolate Newton as the only algebraic knob (see `EXPECTED.md` §2).
- **Run target:** `python run.py` executes all four studies (spatial,
  algebraic, temporal, failures) in **~1-2 min on the GPU** (this file's
  own quickstart note above); comparable to the physics-track
  measurement of physics P1's quick mode, **≈ 75 s wall on an RTX 6000
  Ada** (mostly Warp kernel compile + Python start-up) — expect the same
  first-run compile overhead here since it drives the same brick.

## Required deliverable

Submit the eight-item report of `../../ASSESSMENT.md`, with the
chapter-specific content:

1. The full console output of your `python run.py` run (this chapter has
   no `config.resolved.yaml`/`metadata.json` harness like the physics
   track; report the device used and the eight printed `[PASS/FAIL]`
   lines instead).
2. The eight-item self-check summary green (`ALL GATES: PASS`), or a
   documented deviation.
3. The three figures (`c1_spatial.png`, `c1_temporal.png`,
   `c1_diagnostics.png`) regenerated from your own run via
   `gen_figures.py`.
4. **Headline:** the observed spatial orders — `L2 ~ h^(p+1)` (`p1`
   $\to$ 2, `p2` $\to$ 3) and `H1 ~ h^p` (`p1` $\to$ 1, `p2` $\to$ 2) —
   for **both** fields `c` and `mu`; and the observed temporal orders,
   BDF1 $\to$ 1 and BDF2 $\to$ 2, against a Richardson-verified reference.
5. **Verification:** the algebraic-control sweep (discretization error
   moves by only a few parts in `1e-13` of itself as the Newton tolerance
   tightens `1e-4` → `1e-12`) *and* the reference-verification check
   (BDF2 order stable across two reference `dt`'s, Richardson bound on
   the reference's own error).
6. **Failure:** reproduce at least one of the four deliberate failures
   (wrong BC, wrong source, under-resolved feature, under-resolved
   reference) and diagnose it from the printed order and error trend, not
   just "it looked wrong."
7. **Exploration:** answer one "Explore on your own" question from
   `c1.tex` with evidence (a sweep, a plot, a fitted trend) — e.g. the
   level-7 round-off floor or the `k=2` intercept-vs-slope check.
8. **Research bridge:** one paragraph on why this verification chain
   (separate the errors, control the algebraic one, verify the reference)
   is the gate you would run before trusting any new physics term added
   to this brick.
