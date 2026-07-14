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
