# Track F — Phase field — microstructure evolution (M4)

Part of the [DiffSim curriculum](../README.md). Every chapter is a runnable,
self-contained script: *Learning outcome → Background → documented code →
Expected results (measured numbers you must reproduce) → Explore.*

| ch | script | outcome |
|---|---|---|
| F1 | `F1_allen_cahn.py` | free energies, non-conserved gradient flow; shrinking-circle law at 0.4% |
| F2 | `F2_cahn_hilliard.py` | mixed (c,μ) form; conservation as structure (1.9e-15); spinodal decomposition |
| F3 | `F3_adaptivity.py` | interface-band refinement, transfer operators, conservation by nestedness |

Run a chapter from the repo root, e.g.:

```bash
python tutorials/F_phasefield/F1_allen_cahn.py
```

Related solver project page: [`docs/projects/phase-field.md`](../../docs/projects/phase-field.md).
