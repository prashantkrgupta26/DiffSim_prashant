# C4 — expected results (self-check)

Running `python run.py` benchmarks the linear solve live and echoes the
cited 3-D story. **Timings vary run-to-run and by card** (clock
governor, contention); the *shape* — splu wins tiny, cuDSS wins as the
2-D problem grows — is the load-bearing result, not any single
millisecond value.

## Live 2-D linear-solve benchmark (real CH Jacobian)

| level | dofs | nnz | splu (CPU) | cuDSS (GPU) | speedup |
|---|---|---|---|---|---|
| 5 | 2,178 | 37,636 | ~10 ms | ~95 ms | **0.1×** (splu wins) |
| 6 | 8,450 | 148,996 | ~100 ms | ~115 ms | ~0.9× (even) |
| 7 | 33,282 | 592,900 | ~470 ms | ~160 ms | **2–3×** (cuDSS wins) |

The crossover sits around $10^4$ dofs: below it, cuDSS's kernel-launch
and transfer overhead dominate; above it, GPU direct factorization pulls
away and keeps scaling better.

## Cited 3-D scaling (dev notes — NOT re-run)

| case | dofs | cuDSS s/call | blockch_dev s/call |
|---|---|---|---|
| 3d_l5 (32³) | 202,752 | 6.95 | 1.44 |
| 3d_slab64 | 417,792 | 17.7 | **2.10 (8.4×)** |
| 3d_slab64z32 | 811,008 | **CEILING** (48 GB) | 2.16 |

- cuDSS's factorization fill hits the 48 GB card at ~811k dofs.
- blockch_dev beats cuDSS **8.4×** on the slab64 solve and *marches*
  where cuDSS ceilings.
- block-masked cuDSS cuts the 3-D fill (17.7 → 3.70 s/call) — strong
  under ~5×10⁵ dofs.
- **AMGX verdict: NO** — the inners are mass-dominated at production
  $\Delta t$, so algebraic multigrid adds nothing.
- matrix-free / blockch: 128×128×64 = 6,389,760 dofs runs on **one**
  48 GB card; cuDSS cannot.

**What must be true regardless of hardware:**

- **splu wins at the smallest size and cuDSS wins at the largest** — the
  `CHECK ... PASS` line asserts exactly this crossover.
- **No solver is universally best.** The concept is the decision rule
  (`choose_solver`), not a winner: 2-D → direct; small 3-D → cuDSS;
  large 3-D → blockch; beyond the matrix itself → matrix-free.

If cuDSS is *always* faster (even at level 5) or *never* faster, the
timing is being swamped by something else (a hot cache, a contended
GPU) — re-run on a quiet card.
