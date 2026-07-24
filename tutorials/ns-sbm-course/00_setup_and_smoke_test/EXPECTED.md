# 00 — expected results (self-check)

## `doctor.py`

`python doctor.py` prints an environment report and ends in **exactly one**
of these lines:

| final line | meaning | exit |
|---|---|---|
| `READY: full CUDA` | CUDA + Warp + cuDSS all work — every chapter runs as written | 0 |
| `READY: reduced (no cuDSS)` | CUDA + Warp work but cuDSS is absent — the 2-D chapters use `splu`, the 3-D chapter is slower | 0 |
| `NOT READY: <fix>` | a blocking problem, with a one-line remedy | 1 |

The report includes: python, diffsim commit + dirty flag, numpy/scipy,
the **NS steppers** (linearized + leray), Warp, torch + CUDA, GPU name +
memory, FP64, nvmath/cuDSS, pyamgx (optional), a **live Warp kernel
compile+run**, and a **live 8×8 monolithic NS assemble+solve**. "Imports
succeed" is not enough — the last two prove the device toolchain executes.

## `run.py` (smoke)

```bash
python run.py --config configs/smoke_cpu.yaml --output outputs/smoke --device cpu
```

Writes the standard `outputs/smoke/{config.resolved.yaml, metadata.json,
results.json, history.npz, run.log, checkpoints/, figures/}` and runs the
tolerance check against `baseline.yaml`. It must print `baseline check
PASSED` and exit 0.

Reference values (gpubox, CPU/splu, level 3 = 8×8, Re=100, dt=0.05, 20
steps):

| quantity | monolithic | projection |
|---|---|---|
| `|u|max` | 1.000 | 1.000 |
| `||div u||` | 0.117 | 1.33 |
| `all_finite` | true | true |
| `max|proj − mono|` | \multicolumn{2}{c}{0.870 (transient at 20 steps)} |

**The scientific invariants checked (NOT bitwise identity):**

- `all_finite == true` — no NaN/Inf leaked from either engine.
- `|u|max ≈ 1` — the lid trace ($U=1$) is respected; the field is bounded.
- `||div u||` is **finite and bounded**, *not* zero. The equal-order VMS
  scheme controls the *weak* divergence; a pointwise `||div u||` of order 1
  is correct here, and it is the wrong steady-state gate (see Chapter 01).
  The two engines legitimately show *different* `||div u||` — that is not an
  error.
- `max|proj − mono| = 0.870` at just 20 steps is a **transient**: on this
  coarse mesh the two engines have not converged to their common steady
  state yet. Chapters 01–02 march to steady and show the agreement tighten
  to `0.049`.

A different card will differ in the last digits — expected and fine. If
`all_finite` is false or `|u|max` blows past 2, re-run `doctor.py`.
