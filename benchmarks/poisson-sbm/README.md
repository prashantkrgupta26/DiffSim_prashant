# poisson-sbm — shifted-boundary convergence

The foundational verification of the surrogate-boundary method (SBM): a Poisson
problem on an immersed sphere/circle, solved on the surrogate (grid-aligned)
boundary with the Taylor-shift correction, refined level by level to measure the
observed spatial order.

| Script | What it validates | Reference / gate |
|---|---|---|
| `band_study.py` | p2-band Neumann sphere ladder; observed convergence order across octree levels | asymptotic 2nd order in 3-D |

## Run

```bash
python benchmarks/poisson-sbm/band_study.py 3 4 5      # dim=3, levels 4→5 (minutes)
python benchmarks/poisson-sbm/band_study.py 3 6        # level 6 (~20–30 min)
python benchmarks/poisson-sbm/band_study.py --radius 0.19 --solver cudss 3 4 5
```

Flags: `--radius <float>` (immersed radius; default 0.19), `--solver
{auto,direct,cudss,krylov}` (override the auto-picked backend). The script
prints the solver used, residual, iteration count, and tolerance per level.

## What it taught us (the dyadic-halo rule)

Benchmark radii must **avoid dyadic multiples of `h`** at every level. A radius
like `r = 0.25` lands the surrogate boundary exactly on cell faces at multiple
levels and produces catastrophic error halos (L7 error 400× the clean value).
The canonical clean radius is `r = 0.19` in 2-D; the 3-D band ladder uses
`r = 0.15` (levels 4→7, observed orders 1.97 / 1.94 / 2.03 — asymptotically
second order, confirmed on Nova). Full write-up:
`docs/theory/p2_band_study_for_students.md` and
`docs/dev/nova-mirrors/band_sweep/`.

Regression anchors (byte-exact on the default path): L4 err `1.9471e-03`,
L5 err `5.6772e-04`.
