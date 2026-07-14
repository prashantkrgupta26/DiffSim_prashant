# P2 — expected results (tolerance-based self-check)

Running the harness at `--mode reference` should satisfy the scientific
invariants in `baseline.yaml` (the harness runs the check automatically and
exits non-zero on a required failure). These are **documented tolerances on
scientific invariants, not bit-identical GPU output** — a different card or
BLAS moves the last digits, and the *coarsened pattern* is step-sequence
sensitive (as it is seed sensitive in Chapter 1), so only the *robust*
statistics are gated tightly.

Ground truth captured on RTX 6000 Ada (CUDA 12.9 / driver 13.2, splu),
reference mode (32×32, κ = 2×10⁻³, quench to *t* = 0.6):

## What must be true (gated)

- **The integrator is correct.** Observed temporal order on the smooth
  single-mode relaxation: **BDF1 ≈ 1.0**, **BDF2 ≈ 2.0**; the fixed-`dt`
  quench sweep converges at **order ≈ 2** in its convergent regime.
- **The reference is reproducible and self-consistent.** Reference free
  energy *F* ≈ 0.10 and structure-factor length scale ≈ 20 cells
  (within ~10 %); halving the reference `dt` changes *c(T)* by
  < 5×10⁻³ (relative L2).
- **The adaptive run reaches the reference physics.** Free-energy error
  and length-scale error vs the reference are small (|Δ*F*| ≲ 10⁻³,
  length scale within ~1 %) — regardless of the larger *pixelwise* L2,
  which carries a trajectory-sensitivity floor.
- **The adaptive step spans orders of magnitude.** It collapses to
  *dt*min ≲ 10⁻⁶ through the spinodal quench and grows to the ceiling
  *dt*max ≈ 10⁻² through coarsening. This shrink-then-grow profile is the
  whole point.
- **Adaptivity saves work at matched accuracy.** To match the adaptive
  run's *c(T)* accuracy, a fixed step needs several times more steps; the
  reported step-saving is > 1.5× (and larger in steps than in solves,
  because step-doubling costs three solves per step).

## What must NOT be claimed

- **Do not equate lower energy with accuracy.** Accuracy is the error vs
  the reference, not "whose energy is lower".
- **Do not cite the naive worst-case** (a fixed step pinned at the
  quench's smallest step for the whole horizon, ~10⁵–10⁶ steps) **as the
  speed-up.** It is an upper bound on a badly-chosen fixed step, recorded
  only for contrast.

## Cost tiers

- `--mode quick` — coarse 16×16 / short horizon, 3 tolerances, < 2 min
  (CI smoke; the tight numeric gate is reference-mode only).
- `--mode reference` — the numbers above (≈ 10–20 min).
- `--mode research` — 64×64, longer horizon, 5 tolerances.

Compare with the figures rendered by `gen_figures.py`; every cited number
in the chapter comes from `results.json`.
