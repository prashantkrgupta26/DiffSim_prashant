# C10 — Expected results (self-check)

Ground truth on RTX 6000 Ada (CUDA 12.9 / driver 13.2), `splu`, material
case **P3HT:PCBM** (`materials.yaml`), χ = 0.86 ± 0.5 (order-of-magnitude),
`fh_A = 0.15`, nominal level 5. The *dominant source* and the *ordering* of the
budget are load-bearing; last digits vary across cards (documented rtol in
`baseline.yaml`).

## The six facets

| # | facet | measured |
|---|-------|----------|
| 1 | code verification | mass drift `2.6e-4`, energy monotone (largest +inc `0`) — plus C1's MMS orders |
| 2 | solution verification | L_area by level: 0.428 (16²) → 0.423 (32²) → 0.403 (64²); discretization unc `0.020` |
| 3 | model validation | λ\* predicted `0.1877` vs probe-measured `0.1818` — `3.2%` discrepancy (onset wavelength) |
| 4 | parameter uncertainty | dL/dχ ≈ `0.111`; σ_param = 0.111 × 0.5 = `0.0557` |
| 5 | stochastic variability | 5 seeds: σ_stoch = `0.0337` |
| 6 | model–experiment | not quantified (no lab data); structure in place via facet 3 |

## The uncertainty budget (`outputs/budget.json`)

| source | σ in L_area | % of variance |
|--------|-------------|---------------|
| **parameter (χ)** | **0.0557** | **67** |
| stochastic (seed) | 0.0337 | 24 |
| numerical (mesh) | 0.0201 | 9 |
| **combined (quadrature)** | **0.0681** | 16.7% of the observable |

**Dominant source: parameter (χ)**, 1.7× the next largest. Because P3HT:PCBM's
χ is known only to order-of-magnitude, the morphology's uncertainty is set by
the *material input*, not the seed noise or the mesh. To sharpen the prediction,
measure χ better — don't buy more GPU seeds.

## Gate

```
python ../../common/check_results.py \
    --results outputs/budget.json --baseline baseline.yaml
# -> PASS: 9 checks, 0 required-failure(s), 0 warn.
```
