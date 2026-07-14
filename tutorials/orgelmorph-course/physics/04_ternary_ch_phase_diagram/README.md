# Physics P4 — Ternary Cahn–Hilliard and the Gibbs phase diagram

Real cast blends are (at least) three components: two solutes plus a
solvent. With two independent conserved compositions the free energy is
a surface over the **Gibbs triangle**, and demixing follows **tie-lines**
on that triangle. This chapter derives the **spinodal** from the
free-energy Hessian, predicts the **binodal** by a common-tangent
construction, and measures the two coexisting phases from the simulation
by **clustering** the composition cloud (a 2-component Gaussian mixture) —
theory and simulation compared on one triangle.

**Read** the course document, Chapter *"Ternary Cahn–Hilliard"* (start
with `ternary.py`).

**Run (student driver):**
```bash
python run.py                 # ternary quench, prints spinodal + GMM phases
python run.py --blend P3HT_PCBM
```

**Run (full checked workflow):**
```bash
PYTHONPATH=<repo>/src python run_harness.py \
    --config configs/p4.yaml --mode reference --output outputs/p4 --overwrite
python gen_figures.py --run outputs/p4      # figures + numbers from the run dir
```

`run_harness.py` writes `results.json` (clustered phase means / covariances /
populations + sensitivity, admissibility, per-solute quadrature content
drift, lever-rule residual, spinodal det at IC, predicted binodal, and the
N-shift) and checks it against `baseline.yaml`. Compare with
[`EXPECTED.md`](EXPECTED.md).

| file | role |
|------|------|
| `ternary.py` | core: `run` + analysis (`spinodal_hessian`, `spinodal_maps`, `gmm2`, `cluster_phases`, `predict_binodal`, `bary_to_xy`) |
| `run.py` | student driver; prints spinodal + GMM coexisting phases |
| `run_harness.py` | harness workflow: config → run → diagnostics → `results.json` (baseline-gated) |
| `gen_figures.py` | renders `p4_*.png` + `numbers/p4.tex` **from a saved run dir** |
| `configs/p4.yaml` | canonical config (quick/reference/research modes) |
| `baseline.yaml` | tolerance baseline (measured-then-locked, reference mode) |
| `doc_numbers.yaml` | doc-macro ↔ results.json staleness mapping |
| `EXPECTED.md` | reference numbers |

Coefficients (χ, N) are read from `materials/ternary_p4.yaml` through the
validated loader `materials/loader.py` — no hand-copied numbers. Uses the
`(M=2, K=0)` case of the production `diffsim.physics.multiphase.MultiPhaseStepper`
(fields φ₁, μ₁, φ₂, μ₂; solvent eliminated), which exposes the fast cuDSS
solver (documented `splu` fallback for small runs without cuDSS).

**Figures:** `p4_landscape.png` (the signature theory-vs-sim triangle),
`p4_spinodal.png` (Hessian eigenvalue sign map), `p4_gibbs.png` (cloud
evolution), `p4_fields.png` (the two solute fields), `p4_nshift.png`
(symmetric vs. asymmetric-N).

## Learning objectives

By the end of this chapter you can:

- Write the ternary Flory–Huggins free energy with the solvent eliminated
  and derive the 2×2 exchange chemical potentials `μ_i`.
- Derive the spinodal from the sign of the 2×2 Hessian determinant `det H`
  and solve it for the critical interaction `χ₁₂*` at a given composition.
- Predict the binodal by a common-tangent construction (equal `μ_i`, equal
  grand potential `ω`) pinned by the lever rule, and distinguish it from
  the spinodal.
- Explain why a 2-component Gaussian-mixture clustering of the composition
  cloud replaces a median split, and read the interface-excluded refit as
  a sensitivity/uncertainty on the endpoints.
- Verify three independent honesty checks that must agree: Gibbs-simplex
  admissibility, per-solute quadrature-content conservation, and the
  lever-rule residual connecting the clustering to the conserved mean.
- Explain how unequal degrees of polymerization `N_i` shift the spinodal
  (via the entropic `1/N_i` Hessian terms) and predict the direction of
  the shift before running the asymmetric blend.

## Prerequisites

- **Concepts:** the ternary Flory–Huggins free energy (extends Chapter
  P1's binary free energy to two composition fields with solvent
  elimination), the 2×2 Hessian / eigenvalue sign test for concavity,
  Gaussian-mixture (EM) clustering, and common-tangent constructions.
  Python + NumPy.
- **Chapters:** Chapter 00 (`00_setup_and_smoke_test`, environment green)
  **and** Chapter P1 (`01_ch_binary_energies`). This chapter is Chapter P1
  "promoted to two composition fields": the bulk free-energy term is
  literally P1's Flory–Huggins term (`bulk="p1"`), the `(M=2, K=0)`
  `MultiPhaseStepper` generalizes P1's stepper, and the log-regularization
  cost P1 introduces recurs here (see Q3 below). Without P1's binary
  spinodal derivation the ternary Hessian in §"The spinodal" will look
  unmotivated.

## Expected cost

- **Device:** any CUDA GPU. The chapter uses the `(M=2, K=0)` case of the
  production `MultiPhaseStepper`, which exposes the fast **cuDSS** solver,
  with a documented `splu` fallback for small runs without cuDSS (the
  coupled multiphase block is well-conditioned for cuDSS, unlike P1's
  binary `(c, μ)` saddle — see `run_harness.py`).
- **Quick mode** (`--mode quick`, 33×33, `t_end 0.3`): comparable to
  Chapter P1's measured **≈75 s wall** for a similarly-sized 2-D mesh (P1's
  32×32/60-step quick mode on an RTX 6000 Ada, mostly Warp kernel compile
  + Python start-up); no separate P4 quick-mode measurement is recorded
  here, so treat 75 s as the anchor and expect the same order of
  magnitude, target < 2 min.
- **Reference mode** (65×65, `t_end 0.6`, the `EXPECTED.md`/`baseline.yaml`
  numbers, plus the small asymmetric-N companion run): target 5–30 min
  per the course-wide cost spec (`configs/p4.yaml`).
- **Research mode** (129×129, `t_end 1.0`, "sharper phases" per
  `configs/p4.yaml`): a single finer, longer-horizon run — no fixed sweep
  or ensemble count is given in `configs/p4.yaml`/`EXPECTED.md` (unlike
  P1's 5-seed research mode); tens of minutes expected, scaling with the
  129×129 mesh.
- First run of a session pays a one-time Warp kernel-compile cost, as in
  every other chapter.

## Required deliverable

Submit the eight-item report of `../../ASSESSMENT.md`, with the
chapter-specific content:

1. The `config.resolved.yaml` + `metadata.json` from a reference run
   (`configs/p4.yaml`, `--mode reference`).
2. The harness `baseline.yaml` check green (or a documented deviation).
3. The five figures (`p4_landscape`, `p4_spinodal`, `p4_gibbs`,
   `p4_fields`, `p4_nshift`) regenerated from your saved run via
   `gen_figures.py`.
4. **Headline:** the two coexisting-phase endpoints `(φ₁, φ₂)` and
   populations from the GMM clustering, contrasted with the spinodal
   determined by the sign of `det H` at the initial blend; report the
   interface-excluded sensitivity as the endpoints' uncertainty.
5. **Verification:** the lever-rule residual (conserved mean vs
   population-weighted phase average) together with the admissibility
   and per-solute quadrature-conservation checks — an independent
   cross-check that the clustering and the conservation agree.
6. **Failure:** push the asymmetric blend to `N₁ = 5` or `10` (Q3) and
   report the rising Newton iteration count / reject-ladder activity as
   the log barrier weakens — diagnose *why* the stiffer quench costs more
   iterations rather than just observing that it does.
7. **Exploration:** answer one "Explore on your own" question with
   evidence (the recommended one is Q1: sweep `χ₁₂` across `χ₁₂*` and
   confirm `det H` changes sign and the spread grows only above it).
8. **Research bridge:** one paragraph on what the spinodal/binodal
   comparison predicts for a real cast blend (e.g. mapping
   `materials/ternary_p4.yaml`'s `P3HT_PCBM` blend, per Q5).
