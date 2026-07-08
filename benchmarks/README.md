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

## Solver backends

All drivers accept `--solver {splu, fused, cudss, amgx}`:

- `splu` — host direct (scipy). Fine small; refactorizes every new matrix.
- `fused` — device single-sync Krylov (ours). Wins on SPD systems; Jacobi
  preconditioning is not competitive on the monolithic (u,p) block.
- `cudss` — NVIDIA cuDSS via `pip install nvmath-python[cu12]`. GPU direct:
  the drop-in splu replacement and the MEASURED default for stepping
  (cavity L8: 2.4 s/step vs splu 16.9 s — 7.1x).
- `amgx` — NVIDIA AMGX (algebraic multigrid): the large-scale path for the
  SPD subsystems (Leray PPE); classical AMG does not converge on the
  coupled (u,p) block (findings 8e) — block preconditioning is the open
  production item. Build once:
      git clone --recursive https://github.com/NVIDIA/AMGX
      cmake -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc \
            -DCMAKE_CUDA_ARCHITECTURES=<arch> AMGX
      make -C build -j amgxsh
      cp build/libamgxsh.so <repo>/extern/
      pip install --no-build-isolation <path-to-pyamgx-clone>  # AMGX_DIR set
  Configs: AMGX's own validated JSONs, packaged under
  src/diffsim/solvers/amgx_configs (see m1b findings 8 for the sharp edges).

Measured per-step timings: see the table in m1b findings 8 / the commit that
introduced this section.

## Current inventory (2026-07-08)

| Script | What it demonstrates | Status |
|---|---|---|
| `band_study.py` | p2-band Neumann sphere ladder (`--solver`, `--radius`) | 3-D asymptotic 2nd order (Nova, L7) |
| `cavity_ghia.py`, `cavity_p2.py` | cavity vs Ghia at p1/p2 | p2-L4 beats p1-L5 |
| `cylinder_forces.py`, `cylinder_p2.py` | cylinder C_d | 1.352 (p1) / 1.334 (p2, α~p²) |
| `coupled_nu.py`, `heated_cylinder.py`, `devahl_davis.py` | SBM-thermal + Boussinesq | Nu 0.05%/0.02% (de Vahl Davis) |
| `b1_dqdkappa_wip.py`, `b2_closure_recovery.py`, `b3_retrain_demo.py` | field-κ adjoints; closures-in-loop | dQ/dκ 3e-9; S2 demo RMSE 0.049 |
| `hero_h1/h2/h3/h4`, `hero_d_bunny_drag.py`, `hero_m3_bunny_continuation.py` | the inverse-design hero ladder | sphere ✅✅; **bunny headline 1.23e-3** |
| `adaptive_ch_opener.py` | re-mesh + transfer (zero drift) | M4 |
| `wodo_fig3.py`, `wodo_fig67.py` | evaporating-film replication (CMS-2012) | Fig-3 ✅ first-run; Fig 6/7 campaign |
| `wodo_nova.py` | Wodo figs 3-7 Nova runner (cluster/wodo_campaign/) | device-bound 56×/8×; 250×100 full res |
| `profile_stages.py` | the M1d migration profiler | the 300× finding |
| `p2band_ns_wip.py`, `shape_opt_cylinder.py` | recorded WIP / research items | honest markers |

Baselines lock in `tests/baselines/*.json`; the findings logs
(`docs/superpowers/*findings*.md`) carry every number's provenance.
