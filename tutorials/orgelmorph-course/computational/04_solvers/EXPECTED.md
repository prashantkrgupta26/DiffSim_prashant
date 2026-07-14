# C4 — expected results (self-check)

`python run.py` measures correctness and speed live and echoes the cited
3-D story. **Residuals are near machine precision and reproducible;
timings vary run-to-run and by card.** Six `[PASS]` lines and
`ALL CHECKS: PASS` must print.

## 1. Solver correctness on the CH saddle (residual matters)

The captured mixed (c,μ) Jacobian (~8,450 dofs) solved three ways:

| solver | relative residual ‖Ax−b‖/‖b‖ | verdict |
|---|---|---|
| splu (pivoted CPU direct) | ~1e-12 | accurate |
| cuDSS (GPU direct) | ~5e-13 | **accurate** |
| blockch (CH block precond) | ~4e-10 | accurate |

**All three verify accurate** — including cuDSS, which handles the
indefinite 2-D saddle fine. This *overturns* the folklore that "cuDSS
diverges on the CH saddle"; on the systems measured here it does not.
Always check the residual rather than assume. In 2-D, correctness does not
pick the solver — speed does.

## 2. Live 2-D benchmark (factorize + solve, cold + warm, CUDA-synced)

| level | dofs | splu (CPU) | cuDSS (GPU) | speedup | cuDSS residual |
|---|---|---|---|---|---|
| 5 | 2,178 | ~10 ms | ~25 ms | **0.4×** (splu wins) | ~1e-14 |
| 6 | 8,450 | ~120 ms | ~40 ms | ~3× | ~5e-13 |
| 7 | 33,282 | ~500 ms | ~80 ms | **3–7×** (cuDSS wins) | ~7e-14 |

Crossover near 10⁴ dofs: below it, cuDSS's launch/transfer overhead
dominates; above it, GPU direct pulls away — and stays accurate.

## 3. Cited 3-D scaling — the wall is MEMORY, not accuracy

| case | dofs | cuDSS s/call | blockch_dev s/call |
|---|---|---|---|
| 3d_l5 (32³) | 202,752 | 6.95 | 1.44 |
| 3d_slab64 | 417,792 | 17.7 | **2.10 (8.4×)** |
| 3d_slab64z32 | 811,008 | **CEILING** (48 GB, >16 min) | 2.16 |

- cuDSS ceilings at ~811k dofs because the **factors do not fit** — a
  memory limit, not an accuracy one.
- blockch_dev beats cuDSS **8.4×** at slab64 and marches where cuDSS
  ceilings (it never forms the full LU).
- block-masked cuDSS cuts the fill (17.7 → 3.70 s/call), viable below the
  ceiling.
- **AMGX: NO** (mass-dominated inners at production Δt).
- matrix-free/blockch reach 6,389,760 dofs on one 48 GB card.

## 4. Decision support (`recommend_solver`) — a rationale, not a name

small 2-D → **splu**; large 2-D → **cuDSS** (verified); small 3-D → masked
cuDSS / blockch_dev; large 3-D → **blockch_dev** (memory); extreme →
matrix-free. Each recommendation carries the caveat to *verify the
residual* and names *why*.

**What must be true regardless of hardware:**

- **Every solver verifies accurate on the 2-D CH saddle** (residual < 1e-6).
- **splu wins tiny, cuDSS wins large** in 2-D (the crossover).
- **The 3-D limit is memory** (cuDSS fill ceiling), so blockch_dev /
  matrix-free take over — not because cuDSS is inaccurate.
- **The capture uses a supported API** (`capture_system=True`), not a
  `solve_linear` monkeypatch.

A "cuDSS always faster" or "never faster" result means the timing is
swamped (hot cache, contended GPU) — re-run on a quiet card.
