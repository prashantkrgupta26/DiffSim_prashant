# DiffSim benchmarks

Literature-anchored validation drivers, grouped by physics. Philosophy:

- **CI-coarse variants live in `tests/`** — fast, tolerance-locked, run on
  every commit (`tests/test_cavity.py`, `tests/test_cylinder.py`, …).
- **These scripts are the full / parametric versions** — refinement studies,
  higher Re, longer horizons, cluster campaigns — for reports, papers, and
  student runs. Most print tables against the reference and never assert; you
  read them.

All drivers are run from the repo root, e.g.

```bash
python benchmarks/navier-stokes/cavity_ghia.py
python benchmarks/phase-field/wodo_nova.py --list
```

Each script self-registers its sibling imports via `_bench_bootstrap.py`, so
grouping into subfolders is transparent — no `PYTHONPATH` needed.

## The groups

| Folder | Physics | Headline result |
|---|---|---|
| [`poisson-sbm/`](poisson-sbm/README.md) | shifted-boundary Poisson convergence | 3-D band asymptotic 2nd order; the dyadic-halo rule |
| [`navier-stokes/`](navier-stokes/README.md) | incompressible flow, p1 & p2 | cavity vs Ghia; cylinder C_d 1.352 (p1) / 1.334 (p2) |
| [`heat-mass/`](heat-mass/README.md) | SBM-thermal + Boussinesq | de Vahl Davis Nu to 0.05 % / 0.02 % |
| [`phase-field/`](phase-field/README.md) | Cahn–Hilliard / Allen–Cahn / evaporating films | Wodo CMS-2012, all 14 cases, ~11 min on one A100 |
| [`inverse-heroes/`](inverse-heroes/README.md) | differentiable inverse design | sphere & **bunny INR recovery, headline 1.23e-3** |
| [`performance/`](performance/README.md) | device-migration profiling | the 300× assembly finding; H1 epoch 44.3× |

## Solver backends

All flow/thermal drivers accept `--solver {splu, fused, cudss, amgx}`:

- `splu` — host direct (scipy). Fine for small systems; refactorizes every
  matrix.
- `fused` — device single-sync Krylov (ours). Wins on SPD systems.
- `cudss` — NVIDIA cuDSS (`pip install nvmath-python[cu12]`). GPU direct: the
  drop-in `splu` replacement and the measured default for stepping (cavity L8:
  2.4 s/step vs splu 16.9 s — 7.1×).
- `amgx` — NVIDIA AMGX algebraic multigrid: the large-scale path for the SPD
  subsystems. Block preconditioning of the coupled (u,p) system is the open
  production item (see `docs/dev/m1b-deferred-findings.md` §8).

Baselines lock in `tests/baselines/*.json`; the findings logs
(`docs/dev/*findings*.md`) carry every number's provenance.
