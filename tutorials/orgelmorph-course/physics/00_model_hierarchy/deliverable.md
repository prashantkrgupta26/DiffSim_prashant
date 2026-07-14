# P0 deliverable — model map + parameter table

Fill this in after reading the chapter. Check your numbers against
`outputs/p0/results.json` (run `run.py` first).

## Part 1 — one-page model map

Redraw the model hierarchy from memory. For **each** of the eight concepts
(P1, P3, P4, P5, P6, P7, P8, P9) write:

| concept | order parameter(s) | conserved? | ingredient added | free-energy term added |
|---|---|---|---|---|
| P1 Binary CH | φ | conserved | — (base case) | bulk f(φ) + (κ/2)|∇φ|² |
| P3 + substrate | φ | conserved | wetting/no-flux boundary | wall / contact energy |
| P4 Ternary CH | | | | |
| P5 + evaporation | | | | |
| P6 Allen–Cahn | | | | |
| P7 Coupled | | | | |
| P8 + nucleation | | | | |
| P9 Evap-conditioned | | | | |

Mark which additions change the **dynamics type** (conserved ↔ non-conserved)
and which only add a term to the **same** free energy F.

## Part 2 — completed parameter table (pick one real system)

Choose a system from `materials/materials.yaml` (e.g. `P3HT_PCBM`,
`PDPP5T_PCBM`, or the `PCBM_class` crystallization set) and fill all four
parameter *kinds* the chapter separates:

### (1) Dimensional material parameters (value + units + source)

| parameter | value | units | source (materials.yaml provenance) |
|---|---|---|---|
| χ (…:…) | | – | |
| N (…) | | – | |
| T_m | | K | |
| Δh | | – | |

### (2) Nondimensional groups (what actually governs the solution)

| group | formula | value |
|---|---|---|
| Cahn number κ̃ | (λ/L)² | |
| FH A, B | ~1/N, ~χ | |
| reduced undercooling | T/T_m − 1 | |

### (3) Numerical parameters (chosen to resolve the physics)

| parameter | value | why |
|---|---|---|
| mesh level | | ≥ 3 cells across interface |
| Δt | | |
| Newton / linear tol | | |

Minimum mesh level for ≥ 3 interface cells (from `run.py`): **___**

### (4) Accelerated-tutorial parameters (label each as accelerated!)

| parameter | physical | accelerated | fidelity traded |
|---|---|---|---|
| e.g. quench depth B | | | |
| e.g. crystallization T | 500 K | 250 K | overstates drive ~5× |

## Self-check

`run.py` reproduces: Cahn number `5.0e-4`, interface `2.02` (poly) / `1.43`
(FH) cells at level 6 (these match the P1 tutorial exactly), minimum level
`7` for 3 cells, crystallization drive `-0.72` (accelerated) vs `-0.14`
(physical), ratio `5.3×`.
