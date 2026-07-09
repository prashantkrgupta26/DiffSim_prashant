# navier-stokes — incompressible flow

Steady and transient incompressible Navier–Stokes on immersed geometry, at both
p1 and p2, validated against the classical literature benchmarks. Two
time-discretizations (monolithic linearized, Leray projection) run the same
physics as a standing cross-check.

| Script | Case | Reference | Measured |
|---|---|---|---|
| `cavity_ghia.py` | lid-driven cavity, Re 100/400/1000 | Ghia, Ghia & Shin (1982) | Re=100 max profile diff 0.035 (monolithic), 0.0035 (projection) |
| `cavity_p2.py` | cavity at p2 | Ghia (1982) | p2-L4 beats p1-L5 |
| `cylinder_forces.py` | confined cylinder, Re 20 | Schäfer–Turek band | C_d = 1.352 (p1) |
| `cylinder_p2.py` | confined cylinder at p2 | Schäfer–Turek band | C_d = 1.334 (p2, α = p²·10) |
| `shape_opt_cylinder.py` | adjoint shape-gradient demo | — | research driver (WIP marker) |
| `p2band_ns_wip.py` | p2-band NS study | — | research driver (WIP marker) |

## Run

```bash
python benchmarks/navier-stokes/cavity_ghia.py
python benchmarks/navier-stokes/cylinder_p2.py --solver cudss
```

## The unified penalty law

The Nitsche/SBM Dirichlet penalty follows a single measured law across
formulations: **α ≈ (1 + Pe/4) · p²**. Under-penalization at high level blows the
solution up (T → −15.9 at L7 with a fixed α=20); the Péclet- and order-aware α
is what takes the p2 cylinder from C_d 3.087 back to the correct 1.334. See
`docs/dev/m2-milestone-report.md`.
