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
