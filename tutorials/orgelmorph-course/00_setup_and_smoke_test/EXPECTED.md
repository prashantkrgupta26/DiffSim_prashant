# 00 — expected results (self-check)

## `doctor.py`

`python doctor.py` prints an environment report and ends in **exactly one**
of these lines:

| final line | meaning | exit |
|---|---|---|
| `READY: full CUDA` | CUDA + Warp + cuDSS all work — every chapter runs as written | 0 |
| `READY: reduced (no cuDSS)` | CUDA + Warp work but cuDSS is absent — cuDSS chapters fall back to `splu`/matrix-free (correct, slower) | 0 |
| `NOT READY: <fix>` | a blocking problem, with a one-line remedy | 1 |

The report includes: python, diffsim commit + dirty flag, numpy/scipy,
Warp, torch + CUDA runtime, GPU name + memory + count, FP64 device tensor,
nvmath/cuDSS, pyamgx (optional), a **live Warp kernel compile+run**, and a
**live 8×8 Cahn–Hilliard assemble+solve**. "Imports succeed" is not enough —
the last two prove the device toolchain actually executes.

On the reference box (RTX 6000 Ada, driver 595.71, CUDA 13.0, Warp 1.14,
torch 2.12, nvmath 0.9) the verdict is `READY: full CUDA`.

## `run.py` (smoke)

```bash
python run.py --config configs/smoke_cuda.yaml --output outputs/smoke
```

Writes the standard `outputs/smoke/{config.resolved.yaml, metadata.json,
results.json, history.npz, run.log, checkpoints/, figures/}` and runs the
tolerance check against `baseline.yaml`. It must print `baseline check
PASSED` and exit 0. The scientific invariants checked (NOT bitwise identity):

- `all_finite == true` — no NaN/Inf leaked from the solve.
- the field **separates**: `c_max` rises above the initial ±0.05 band toward
  the polynomial wells (near ±1), `c_min` falls symmetrically.
- `n_steps >= 1`.

A different card will differ in the last digits — that is expected and fine.
If `all_finite` is false or the field does not move, re-run `doctor.py`.
