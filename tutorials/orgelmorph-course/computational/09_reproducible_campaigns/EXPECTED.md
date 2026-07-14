# C9 — Expected results (self-check)

Ground truth captured on RTX 6000 Ada (CUDA 12.9 / driver 13.2), `splu`,
`fh_A = 0.15`, level 5 (32×32), 40 steps, `dt = 0.02`, `kappa = 5e-4`.
Observed *trends* and *bookkeeping invariants* are load-bearing; the last
digits of `L_area` vary across cards/BLAS (documented rtol in `baseline.yaml`).

## Campaign aggregate (`summary.json`)

```
runs: 9 total, 8 completed, 1 failed (0 real, 1 injected).
campaign_complete (no unexpected failure): True
```

| sweep | intent | χ (fh_B) | mean L_area | sd | 95% CI | n |
|-------|--------|----------|-------------|-----|--------|---|
| chi_scan | exploratory | 0.36 | 0.261 | 0.037 | [0.235, 0.287] | 2/2 |
| chi_scan | exploratory | 0.86 | 0.416 | 0.010 | [0.409, 0.423] | 2/2 |
| chi_scan | exploratory | 1.36 | 0.517 | 0.004 | [0.515, 0.520] | 2/2 |
| chi_confirm | confirmatory | 0.86 | 0.374 | 0.011 | — | 2/2 |

Parametric sensitivity `dL_area/dχ ≈ 0.256` (least-squares over the scan).
The confirmatory replicate (seeds 11, 12) lands at 0.374 vs the exploratory
0.416 at the same χ — the seed-set difference IS the stochastic variability
C10 quantifies.

## Failures surfaced (never dropped)

- **Injected** (`fault.bad_solver_typo`): `ConfigError: unknown solver 'nope'`
  — caught, campaign continues, `n_failed_injected = 1`.
- **Real** (tightened-gate demo, `gen_figures.py` teeth pass): χ = 1.36 exceeds
  a `1e-3` mass gate with drift `3.17e-2` → validation failure →
  `campaign_complete = false`, exit code `3`.

## Gate

```
python ../../common/check_results.py \
    --results outputs/demo/summary.json --baseline baseline.yaml
# -> PASS: 10 checks, 0 required-failure(s), 0 warn.
```
