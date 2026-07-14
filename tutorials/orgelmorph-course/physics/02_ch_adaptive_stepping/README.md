# Physics P2 — Cahn–Hilliard with adaptive time stepping

The same binary quench as P1, but the question is **how to choose the
time step**. This chapter is a *repeatable scientific workflow*, not a
demonstration: it **verifies** the integrator's temporal order, builds a
tight-`dt` **reference**, and reports the **real cost** and the **honest**
matched-accuracy speed-up of error-controlled (step-doubling) adaptivity.

**Accuracy is measured against the reference, never read off the energy.**
A lower free energy does *not* mean a more accurate solution.

## Run it (the harness workflow)

```bash
cd physics/02_ch_adaptive_stepping
export PYTHONPATH=<repo>/src

# <2 min smoke (coarse mesh / short horizon)
python run_harness.py --config configs/p2.yaml --mode quick \
    --output outputs/p2 --overwrite

# the cited numbers (32x32, sharp interface, quench to t=0.6)
python run_harness.py --config configs/p2.yaml --mode reference \
    --output outputs/p2 --overwrite

# render every figure + latex/numbers/p2.tex from the SAVED run
python gen_figures.py --run outputs/p2
```

The reference run's `results.json` is checked against `baseline.yaml`
(tolerance-based scientific invariants, not bit-identity); the harness
exits non-zero on a required-check failure.

## What the workflow produces

1. **Temporal-order verification** on a *smooth* single-mode relaxation:
   BDF1 → order ≈ 1, BDF2 → order ≈ 2 (evidence the variable-step BDF2 is
   correct before it is used on the chaotic quench).
2. **Reference-based accuracy**: relative L2 of `c(T)` *plus* errors in
   free energy, structure-factor length scale and phase fraction, vs a
   tight small-`dt` reference. The **stiffness cliff** (largest fixed
   `dt` that still resolves the quench) is measured.
3. **Real cost**: accepted **and rejected** steps, full + half solves
   (step-doubling = 1 full + 2 half per attempt), total Newton / linear
   solves, wall time, peak device memory.
4. **Matched-accuracy speed-up**: the largest fixed `dt` whose `c(T)`
   error matches the adaptive run, and the honest cost ratio (plus the
   accuracy–cost knee from the tolerance sweep).

| file | role |
|------|------|
| `run_harness.py` | the workflow: order study, reference, sweeps, cost, matched speed-up |
| `adaptive.py` | importable core (`build_mesh_dm`, `free_energy`, fixed/adaptive marches) |
| `configs/p2.yaml` | canonical run record (quick / reference / research modes) |
| `baseline.yaml` | tolerance-based verification gate (reference mode) |
| `doc_numbers.yaml` | maps document macros → results.json paths (staleness gate) |
| `gen_figures.py` | renders `../../latex/figures/p2_*.png` + `numbers/p2.tex` from a saved run |
| `run.py` | thin student driver (prints the cost/accuracy self-check) |
| `EXPECTED.md` | what the numbers must satisfy (tolerance-based) |

Uses the production brick `diffsim.physics.cahn_hilliard`
(`CahnHilliardStepper` + `adaptive_march`) directly; cost counters are
obtained by wrapping `stepper.step` (no source edits).

## Learning objectives

By the end of this chapter you can:

- Verify a time integrator's *observed* order (BDF1 ≈ 1, BDF2 ≈ 2) on a
  smooth single-mode relaxation before trusting it on a chaotic quench,
  and explain why the variable-step BDF2 coefficients must be rebuilt
  every step from the actual step ratio `r = Δt_n/Δt_{n-1}`.
- State why accuracy is measured against a tight-`Δt` reference and
  *never* read off the free energy — two trajectories can reach
  different (even both "lower") energies while one is far less accurate.
- Locate the quench's "stiffness cliff" — the largest fixed step that
  still resolves the spinodal onset — and explain why a step chosen
  safely below it is wastefully small for the coarsening tail.
- Read the step-doubling accept/reject ladder and account for its
  *real* cost: one full **plus two half** solves per attempt, not one.
- Find the accuracy–cost knee from a tolerance sweep, and report a
  **matched-accuracy speed-up** (adaptive vs. fixed, same `c(T)` error)
  honestly in both steps *and* solves — distinguishing it from the
  naive worst-case bound, which is not the speed-up.
- Separate the *robust* physics statistics (free energy, structure-
  factor length scale) from the pixelwise field, which carries a
  trajectory-sensitivity (step-sequence / seed-like) floor.

## Prerequisites

- **Concepts:** implicit multistep (BDF) time integration and Newton
  solves; local-truncation-error estimation and step-size control
  (step-doubling, PI-style controllers); reading an observed
  convergence order off a log–log slope (the same diagnostic used
  elsewhere in the course). Python + NumPy.
- **Chapters:** Chapter 00 (`00_setup_and_smoke_test`, environment
  green) **and** Chapter P1 (`01_ch_binary_energies`) — P2 reuses P1's
  model and free energy *unchanged* (the conserved Cahn–Hilliard flow of
  the Ginzburg–Landau energy) and only changes the time integration, so
  P1's derivation, resolution, and "energy decreasing is not accuracy"
  discussion are assumed rather than re-derived here.

## Expected cost

- **Device:** any CUDA GPU; the meshes are 2-D (16×16 quick, 32×32
  reference, 64×64 research) and use well under 1 GB of device memory.
  Solver `splu` (the CH saddle is indefinite, the same reason as P1's
  C4; cuDSS is *not* used here).
- **Quick mode** (`--mode quick`, 16×16 / short horizon): comparable to
  P1's measured ≈ 75 s wall on an RTX 6000 Ada for a similarly small
  2-D mesh (most of it Warp kernel compile + Python start-up); target
  < 2 min.
- **Reference mode** (32×32, quench to `t=0.6`, the `EXPECTED.md`
  numbers): this run does more than P1's single march — order study,
  reference march, fixed-`dt` sweep, the adaptive run, and a tolerance
  sweep — so budget more time; `EXPECTED.md` records ≈ 10–20 min.
  Target 5–30 min.
- **Research mode** (64×64, longer horizon, 5 tolerances): tens of
  minutes.
- The **first run** of a session pays a one-time Warp kernel-compile
  cost; subsequent runs in the same environment are faster.

## Required deliverable

Submit the eight-item report of `../../ASSESSMENT.md`, with the
chapter-specific content:

1. The `config.resolved.yaml` + `metadata.json` from a reference run.
2. The harness `baseline.yaml` check green (or a documented deviation).
3. The five figures (`p2_order`, `p2_convergence`, `p2_dt_newton`,
   `p2_morphology`, `p2_energy`) regenerated from your saved run via
   `gen_figures.py`.
4. **Headline:** the matched-accuracy speed-up — the largest fixed step
   whose `c(T)` error matches the adaptive run, reported as a step-count
   ratio *and* a solve-count ratio (the step-doubling overhead of three
   solves per attempt is why the solve saving is smaller than the step
   saving) — plus the accepted step spanning `dt_min` to `dt_max` across
   orders of magnitude.
5. **Verification:** the observed temporal order on the smooth
   single-mode relaxation (BDF1 ≈ 1, BDF2 ≈ 2) *and* the fixed-`dt`
   quench sweep converging near order ≈ 2 up to the stiffness cliff;
   the reference self-check (halving `ref_dt` changes `c(T)` by < 5×10⁻³
   relative L2).
6. **Failure:** push a fixed step across the stiffness cliff (or turn
   the controller off using the largest fixed step above the cliff —
   see "Explore on your own") and diagnose which reported number flags
   it and by how much.
7. **Exploration:** answer one "Explore on your own" question with
   evidence (a sweep or plot) — the tolerance-knee sweep (`work_tol`)
   is the recommended one.
8. **Research bridge:** one paragraph on what the matched-accuracy
   speed-up and the `dt` span predict for the cost of a real, longer,
   more scale-separated coarsening campaign.
