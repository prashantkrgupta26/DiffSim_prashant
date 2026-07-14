# Physics P5 — Solvent evaporation (the drying film)

An organic film is cast from solution and **dries**: solvent leaves the
top surface, the film thins, and the solutes concentrate until they
demix. Evaporation is the **clock** of morphology formation. This chapter
adds the moving, evaporating top surface (film mode) and asks the honest
question: does the drying **rate** reshape the morphology, or does it just
decide *when* you look?

**The corrected story (matched terminal state).** Comparing rates at a
fixed *time* confounds drying rate with final state (a slow film only
reaching φ_s=0.30 vs a fast one dried to 0.10). We instead compare at a
**matched dryness** — the same mean solvent fraction φ_s ∈ {0.30, 0.20,
0.10}, interpolated along each run's φ_s(t) trajectory. Then:

- **Dryness sets the domain size** (wavelength shrinks, contrast rises as
  the film dries);
- **Rate is a weak, secondary knob** — at matched dryness the faster
  (higher-Bi) film is if anything slightly *coarser* (less time to develop
  lateral structure), the opposite of the naive "faster = finer";
- **Morphology collapses onto the Biot number** Bi = k_e·h0/D_s: different
  (k_e, D_s) with the same Bi dry to the same morphology.

**Run (full workflow):**
```bash
PYTHONPATH=<repo>/src python run_harness.py --config configs/p5.yaml \
    --mode reference --output outputs/p5 --overwrite
PYTHONPATH=<repo>/src python gen_figures.py --run-dir outputs/p5
```
`run_harness.py` sweeps (k_e, D_s), records the metric-vs-φ_s trajectory,
interpolates to the matched dryness levels, verifies per-solute
conservation + the height/solvent budget, computes Bi, records the march
exit reason, and writes a tolerance-checked `results.json`. `run.py` is a
lighter student driver of the same core. Compare with
[`EXPECTED.md`](EXPECTED.md).

Modes: `quick` (level 5, 3 runs, <2 min smoke — level 5 is the coarsest
mesh that still conserves solute; level 4 leaks ~10%), `reference`
(level 6, 7 runs, the checked numbers), `research` (level 7, finer grid).
The film stepper uses cuDSS; `--solver splu` is a documented fallback.

| file | role |
|------|------|
| `evaporation.py` | core: `run_film` (trajectory-aware) + `morphology_metrics` + `lateral_wavelength` |
| `run_harness.py` | the workflow: config, provenance, matched-state + regime, baseline check |
| `run.py` | lighter student driver; prints the matched-dryness table |
| `gen_figures.py` | renders `p5_*.png` + `numbers/p5.tex` from a saved run dir |
| `configs/p5.yaml` | canonical run record (quick/reference/research) |
| `baseline.yaml` | tolerance-based reference invariants (gate) |
| `doc_numbers.yaml` | doc-macro → results.json mapping (staleness gate) |
| `EXPECTED.md` | reference numbers + what must hold on any card |

Uses `diffsim.physics.multiphase.MultiPhaseStepper` (M=2, K=0) in film
mode (the Landau-mapped moving frame of the wodo_film formulation) and the
shared `diffsim.diagnostics` (morphology, conservation).

## Learning objectives

By the end of this chapter you can:

- Write the moving-frame film equations (evaporative flux `K =
  Σᵢ k_{e,i}·φ̄ᵢ^top`, height ODE `dh/dt = -K`) that sit atop the ternary
  Cahn–Hilliard bulk dynamics of Chapter P4, and explain why only the
  solvent — never a solute — leaves the frame.
- Explain, in one sentence, why comparing morphologies at a fixed *time*
  across different evaporation rates confounds the rate with the dryness
  reached, and design the fix: read every metric at a **matched** mean
  solvent fraction `φ_s`, interpolated along each run's own `φ_s(t)`.
- Define the drying Biot number `Bi = k_e·h0/D_s`, predict which regime
  (`Bi > 1` drying-limited vs `Bi < 1` diffusion-limited) a given
  `(k_e, D_s)` pair sits in, and **verify** that different `(k_e, D_s)`
  sharing a `Bi` dry to the same matched-dryness morphology.
- Keep apart two different claims — "dryness sets the demixing *degree*
  (contrast)" vs "Bi sets the domain *size* (wavelength)" — and report
  which invariant is rate-independent and which is not.
- Report the lateral wavelength as a genuine length (a fraction of the
  box width, and in cells `> 1`), and explain why a bare "cells" number
  below one is a units bug, not a result.
- Verify per-solute conservation in the moving frame (`h(t)·∫φᵢ`
  constant) and the height/solvent budget closure, and explain why both
  are resolution-limited — recovered only once the demixed interfaces and
  the surface-enrichment layer are resolved.

## Prerequisites

- **Concepts:** the ternary Cahn–Hilliard free energy, spinodal, and
  Onsager mobility machinery of Chapter P4 (Flory–Huggins χ/N, two
  conserved solute fields); dimensionless-group reasoning (a Biot/Péclet
  ratio comparing two rates); the interface-width and resolution
  intuition of Chapter P1 (`ℓ ~ √(κ/W)`, cell counts). Python + NumPy.
- **Chapters:** Chapter 00 (`00_setup_and_smoke_test`, environment
  green); Chapter P1 (`01_ch_binary_energies`, free-energy and resolution
  foundations); **Chapter P4** (`04_ternary_ch_phase_diagram`) is a hard
  prerequisite — P5 does not re-derive the bulk dynamics, it *reuses*
  P4's ternary machinery verbatim (the same `MultiPhaseStepper(M=2, K=0)`
  case, the same χ/N Flory–Huggins free energy, the same two solute
  fields φ₁, φ₂) and adds only the moving, evaporating top boundary
  (film mode) on top of it.

## Expected cost

- **Device:** any CUDA GPU. The film stepper uses cuDSS; `--solver
  splu` is a documented fallback for small runs (same convention as
  Chapter P4). Meshes are 2-D (level 5–7, i.e. 32×32 to 128×128) so an
  8 GB laptop card is ample.
- **Quick mode** (`--mode quick`, level 5 = 32×32, 3 runs): target
  **< 2 min**, comparable to Chapter P1's measured **≈ 75 s wall** on an
  RTX 6000 Ada for a similarly-sized 2-D mesh (mostly Warp kernel compile
  + Python start-up — pays once per session).
- **Reference mode** (level 6 = 64×64, 7-run `(k_e, D_s)` sweep, the
  `EXPECTED.md` numbers): target **5–30 min**.
- **Research mode** (level 7 = 128×128): a finer regime grid — per
  `configs/p5.yaml` this widens the matched-state sweep to 4 `k_e` rows
  and the 2-D Biot regime map to a 3×3 `(k_e, D_s)` grid; correspondingly
  longer wall time than reference.

## Required deliverable

Submit the eight-item report of `../../ASSESSMENT.md`, with the
chapter-specific content:

1. The `config.resolved.yaml` + `metadata.json` from a reference run.
2. The harness `baseline.yaml` check green (or a documented deviation).
3. The four figures (`p5_drying`, `p5_matched`, `p5_morphology`,
   `p5_regime`) regenerated by `gen_figures.py` from your saved run.
4. **Headline:** the matched-dryness domain size versus evaporation rate,
   reported as the Biot-number collapse — e.g. wavelength at `φ_s=0.20`
   of 33.4 cells (Bi=1.5, slow) vs 26.1 cells (Bi=3.0, fast), and that
   runs sharing a Bi from different `(k_e, D_s)` agree to within the
   collapse spread in `EXPECTED.md`.
5. **Verification:** per-solute conservation (`h(t)·∫φᵢ` drift at machine
   precision, ~10⁻¹⁵ at level 6) and the height/solvent budget closure to
   the same order — an independent check that only the solvent left the
   frame.
6. **Failure:** re-run at level 4 (16×16) and report the per-solute drift
   rising to ~10% — the demixed interfaces and surface-enrichment layer
   are under-resolved; state why `quick` mode is pinned to level 5, not
   level 4.
7. **Exploration:** answer one "Explore on your own" question with
   evidence (e.g. widen the Biot range and check whether the
   matched-dryness collapse ever breaks).
8. **Research bridge:** one paragraph on what the Bi-collapse predicts
   for choosing a solvent/anneal schedule for a real cast blend — i.e.
   which knob (solvent volatility `k_e` or blend mobility `D_s`) an
   experimentalist should turn to hit a target domain size.
