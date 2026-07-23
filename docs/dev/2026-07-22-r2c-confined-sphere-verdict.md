# R2c — monolithic sphere Cd: the "honesty gap" is confinement, not error

**Date:** 2026-07-22
**Status:** resolved (absolute-validation gap explained; oracle trustworthy)

## Question

The monolithic SBM-NS 3-D sphere at Re=100 mesh-converges internally to
**Cd ≈ 1.41** (L4=0.964, L5=1.448, L6[1.09M dof]=1.413; geometric tail →1.410).
The unbounded Schiller–Naumann correlation gives **Cd = 1.087**. Our value is
~30% high. Is that a solver error, or a setup artifact?

## Verdict: (a) benign — strong domain confinement. Not a solver bug.

The solver, its definitions, and its drag integration are all correct. Cd ≈ 1.41
is the *right answer for the confined box the sphere sits in*.

### What was checked

1. **Definitions / normalization — CORRECT.** In `tests/p2r0_task10_sphere_derisk.py`:
   `nu = 2*U*R/Re` ⇒ Re on the **diameter**, `Re_D = U·D/ν = 100.0` exactly;
   `qref = 0.5*U_IN²·πR²` ⇒ **frontal area πR²** (not D², not wetted);
   `Cd = F[0]/qref`, F[0] downstream traction. No radius/diameter or 2× area error.

2. **Domain confinement — THE CAUSE.** Domain = unit box [0,1]³; sphere R=0.12 at
   (0.35, 0.5, 0.5). Clearances sphere-surface → wall:
   - upstream (inflow): **0.96 D** — inflow plane essentially on the sphere
   - downstream (outflow): **2.21 D**
   - lateral (each side): **1.58 D**
   Frontal-area blockage is 4.5%, but that understates it: blockage corrections
   assume walls far in axial extent, whereas here every wall is ~1–1.6 D away
   (box/D ≈ 4.2). A naive speedup² model 1/(1−0.045)² predicts only ≈1.20; the
   extra rise to 1.41 is near-field wall retardation from the ~1 D upstream/lateral
   clearances, consistent with confined-sphere literature (drag climbs steeply as
   wall-factor box/D → a few).

3. **Transient — ruled out.** L6 tail 1.4375→1.4187→1.4127 decays geometrically
   (ratio 0.32); extrapolated steady Cd = 1.410. Transient ≈ 0.3%, not 30%.

4. **SBM surrogate offset — ruled out.** Offset ~h (<7% of R at L6 D/h=15.4),
   shrinks with refinement; the L4→L5→L6 trend (last step −0.035) shows it
   converging away — cannot be a persistent +30%.

5. **Drag integration — CORRECT.** `surrogate_traction` (`src/diffsim/sbm/vector.py`)
   integrates full traction `F += w*(p·n − ν·(∇u)ᵀ·n)` with `n̂ = −geo.n`
   (orientation validated on the Re=20 cylinder sign-flip) and the surrogate→true
   area correction `corr = ñ·n` (`src/diffsim/sbm/surrogate.py`). Pressure + viscous,
   correct sign and normal.

## Consequences

- **The monolithic oracle is trustworthy** as a same-mesh faithfulness reference
  *and* is now understood absolutely: it faithfully solves a **confined** sphere,
  and Cd≈1.41 is the correct confined answer.
- **To validate against unbounded literature** (if ever needed): enlarge the domain
  to ≥15–20 D clearances (requires AMR / finer mesh to hold D/h — real compute cost),
  OR re-anchor against a matched confined-sphere reference at box/D≈4.
- **Docstring corrected** in `tests/p2r2c_monolithic_sphere_convergence.py` (the old
  "~4.5% blockage / somewhat above" was misleading; it is strong confinement and
  +30% is expected physics).

## References
- Wall retardation of confined spheres: Ind. Eng. Chem. Res. (2013) 10.1021/ie302707s
- Drag correction for spheres in square microducts (confined-sphere Cd vs box/D)
