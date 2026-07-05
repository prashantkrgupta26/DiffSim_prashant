# DiffSim benchmarks

Literature-anchored validation drivers. Philosophy:

- **CI-coarse variants live in `tests/`** (fast, tolerance-locked, run on
  every commit: `tests/test_cavity.py`, `tests/test_cylinder.py`).
- **These scripts are the full/parametric versions** — refinement studies,
  higher Re, longer horizons — for reports, papers, and student runs. They
  print tables against the reference values and never assert; you read them.

| Script | Case | Reference | Current status (measured, level-5 CI variant) |
|---|---|---|---|
| `cavity_ghia.py` | lid-driven cavity, Re = 100/400/1000 | Ghia, Ghia & Shin (1982) | Re=100: max profile diff 0.035 (monolithic), 0.0035 (projection) |
| `cylinder_forces.py` | immersed cylinder in channel, Re = 20 | confined-cylinder literature band | C_d = 2.847, C_l = -3e-5 at 14% blockage |

Roadmap additions (M1b completion): cylinder Re=100 (Strouhal + force
history), sphere Re=300 (3D), and the `m1b_baselines.json` lock of every
number in this table.

Both steppers (monolithic linearized, Leray projection) run every benchmark —
a standing project discipline (spec §17): two time-discretizations agreeing
on the same physics is a cheap, powerful cross-check.
