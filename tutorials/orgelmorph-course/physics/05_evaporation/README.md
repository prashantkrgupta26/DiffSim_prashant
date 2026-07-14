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
