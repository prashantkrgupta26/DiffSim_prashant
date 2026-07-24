# 02 — The pressure-projection engine (the scalable one)

Predictor → **SPD pressure-Poisson (PPE)** → L2 velocity correction. This
chapter re-runs the **same** lid-driven cavity as Chapter 01 through the
projection engine and shows it reproduces the same-mesh monolithic oracle
(faithfulness) and Ghia. The payoff: the pressure step is a symmetric
positive-definite Poisson problem — the operator that admits AMG/CG at 100M
DOF (Chapter 06), unlike the indefinite saddle.

**Read** the course document, Chapter 2 (*The pressure-projection engine*).
Start with `cavity.py` — the importable core, **shared with Chapter 01**, so
the two engines run on identical fixtures.

**Run:**
```bash
python run.py                                       # level 4, Re=100, both engines
python run.py --config configs/ldc.yaml --mode reference --output outputs/ldc
```

`run.py` prints the centerline self-check table; compare with
[`EXPECTED.md`](EXPECTED.md).

| file | role |
|------|------|
| `cavity.py` | the shared core (read Chapter 01's copy first) — `march_projection`, `run_cavity` |
| `run.py` | the driver you run; harness + centerline self-check table |
| `gen_figures.py` | regenerates the centerline figure + `numbers/c2.tex` |
| `EXPECTED.md` | reference numbers your run should reproduce |
| `baseline.yaml` | the tolerance baseline the harness checks |

The chapter uses the production brick `src/diffsim/steppers/leray.py`
(`LerayProjectionStepper`) directly — no toy solver.

## Learning objectives

By the end of this chapter you can:

- State the Helmholtz–Leray split (predictor → PPE → correction) and where
  the extrapolated pressure $p^\*$ enters.
- Explain why the PPE is **SPD** and why that — not the indefinite saddle —
  is the operator that scales to 100M DOF via AMG/CG.
- Explain the six items of `consistent_projection` (collocated
  PSPG-consistent divergence, fine-scale in both source and correction,
  disjoint outflow BCs, rotational-incremental pressure, boundary-vorticity,
  backflow) and why the naïve split breaks faithfulness at an open outflow.
- Read the right divergence gate: the PPE-space solenoidality identity
  $\|\sigma B^\top\hat u - K_p\phi\|\approx0$, not the pointwise
  $\|\nabla\!\cdot u\|$.

## Prerequisites

- **Concepts:** projection/fractional-step methods, the Helmholtz
  decomposition, SPD systems and multigrid intuition. Chapter 01's VMS weak
  form.
- **Chapters:** 00 (environment), 01 (the monolithic oracle and the
  faithfulness idea).

## Expected cost

- **Device:** any CPU runs level 4 in a couple of minutes (`splu` for the
  small saddle-predictor; the PPE is tiny here). A GPU is optional.
- **Quick / reference / research:** as Chapter 01.

## Required deliverable

Submit the eight-item report of `../ASSESSMENT.md`, specialised:

1. The `config.resolved.yaml` + `metadata.json` from a reference run.
2. The `baseline.yaml` check green.
3. The centerline figure (proj / mono / Ghia).
4. **Headline:** `max|proj − mono|` (faithfulness) with the station of the
   max, and the projection $\|\nabla\!\cdot u\|$.
5. **Verification:** the projection reproduces the same-mesh monolithic
   (`d_proj_mono`), stated against the coarse-mesh `d_proj_ghia`.
6. **Failure:** argue (from Chapter 03's open-outflow finding) what breaks
   without `consistent_projection` at an outflow — the enclosed cavity is
   pin-stabilised, so name the mechanism rather than reproducing a blow-up
   here.
7. **Exploration:** compare the projection and monolithic $\|\nabla\!\cdot
   u\|$ and explain why they legitimately differ.
8. **Research bridge:** one paragraph on why the SPD PPE is the 100M-DOF
   path (Chapter 06) and the saddle is not.
