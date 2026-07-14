# P10 — Material-system reproduction (physics capstone)

The "put it all together on a real system" capstone. Pick ONE documented blend,
load **every** parameter through the validated loader, reproduce a published
morphology trend, then stress the conclusion with mesh + time convergence and a
parameter-uncertainty sweep — and state honestly which conclusions are robust.

**System:** `PDPP5T_PCBM` (a DPP donor : PCBM-class acceptor spin-coated from
chloroform, after Negi et al.). Chosen because it carries a concrete chi triple,
concrete degrees of polymerization, **and** an explicit spin-speed → Biot
ladder — "the real chi / N / Biot" — and because it demixes robustly at
tutorial scale. Every parameter is `accelerated_tutorial` / fitted
(representative, not measured table values); the run archives the full
provenance to `materials.resolved.json`.

**Reproduced trend:** the Negi spin-coating **rate → morphology** axis. The
recorded spin-speed Biot ladder is order-of-magnitude uncertain, so only its
*ordering* is reproducible; we map the rungs, in rank order, onto a moderate
resolvable drying-rate window and read morphology at **matched dryness** (same
mean solvent fraction φ_s for every rung, so rate is not confounded with state).

**Physics engine:** this capstone does not re-implement the drying film — it
imports the maintained P5 brick (`physics/05_evaporation/evaporation.py`,
`diffsim.physics.multiphase` in film mode), so it exercises the same production
code path the course teaches.

## Files

| file | what it is |
|------|-----------|
| `material_case.py` | core: loads the system, maps the Biot ladder, provenance audit, matched-state reduction (imports the P5 drying-film brick) |
| `run_harness.py` | full workflow: ladder + mesh/time convergence + chi sensitivity → `results.json`, gated vs `baseline.yaml` |
| `gen_figures.py` | figures → `../../latex/figures/p10_*.png` and numbers → `../../latex/numbers/p10.tex` |
| `configs/p10.yaml` | canonical run record with `quick` / `reference` / `research` modes |
| `baseline.yaml` | reference-mode tolerance gate (measured invariants) |
| `doc_numbers.yaml` | doc-macro ↔ results.json staleness map |
| `EXPECTED.md` | reference numbers + what must be true regardless of hardware |

## Run it

```bash
cd physics/10_material_case_study
PYTHONPATH=<repo>/src python run_harness.py --config configs/p10.yaml \
    --mode reference --output outputs/p10 --overwrite
PYTHONPATH=<repo>/src python gen_figures.py --run-dir outputs/p10
```

`--mode quick` is a <2 min two-rung smoke on a 16×16 mesh (not gated).

## The verdict (spoiler)

**Robust** (survives mesh + time refinement and the order-of-magnitude χ_pf
uncertainty): the *degree* of demixing (phase contrast) is set by **dryness**,
rises as the film dries, is nearly rate-independent at matched φ_s, and
increases monotonically with χ_pf; morphology collapses onto the Biot number.

**Not robust at tutorial scale:** the "faster spin = finer morphology"
*wavelength* ordering. Single seed, few lateral domains → noisy, moves under
refinement. Reproducing it quantitatively needs seed ensembles and a bigger box
(a research-mode campaign, chapter C9), not a tutorial run.
