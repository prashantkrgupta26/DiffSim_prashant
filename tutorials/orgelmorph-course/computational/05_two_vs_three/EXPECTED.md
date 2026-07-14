# C5 — expected results (self-check)

`python run.py` measures the 2-D-vs-3-D scaling, sparsity, complete memory,
and physics live, then echoes the cited ladder. Counts are
**deterministic** and match exactly; step times and the exact morphology
metrics vary by card/seed. Five `[PASS]` lines and `ALL CHECKS: PASS` must
print.

## 1. Live scaling (real Cahn–Hilliard systems)

- **dofs per level:** 2-D ≈ ×3.9, 3-D ≈ ×7.0 (theory ×4 vs ×8).
- **nnz per dof:** 2-D ≈ 18, 3-D ≈ 50.

## 2. Sparsity, decomposed to its FE origins

nnz/dof = coupled nodes × fields = (2p+1)^dim × n_fields:

| dim | coupled nodes | × fields | = nnz/dof | a "3^d−1 stencil" would say |
|---|---|---|---|---|
| 2-D | 9 | 2 | **18** | 8 (wrong — undercounts) |
| 3-D | 27 | 2 | **54** | 26 (wrong) |

The density is set by element **connectivity**, basis **order**, **fields**,
the **block** structure, **constraints**, and **dimension** — not a
finite-difference stencil.

## 3. Complete memory accounting (`estimate_capacity.py`)

Every buffer, not just CSR values (the solver workspace usually dominates):

| case | dofs | total | biggest term | fits 48 GB? |
|---|---|---|---|---|
| 2-D n=256 (splu) | 132,098 | ~0.2 GB | direct fill-in | yes |
| 3-D n=128 (blockch) | 4,293,378 | **~8.0 GB** | solver workspace | yes |
| 3-D n=256 (blockch) | 33,949,186 | **~63.4 GB** | solver workspace | **NO → matrix-free/multi-GPU** |

The int32 CSR ceiling (nnz > 2³¹) is an **implementation choice**; int64 or
a distributed/block-masked pattern lifts it.

## 4. Physics at equal resolution (2-D is not cheap 3-D)

Same spinodal quench, same h. The 2-D and 3-D morphologies differ in
**interfacial-area density** and **S(q) wavelength**, and only 3-D can be
bicontinuous (both phases percolating) — impossible for two phases in 2-D.
The exact metrics are card/seed-dependent; the *difference* between 2-D and
3-D is the load-bearing result.

## 5. Cited device-scale ladder (dev notes — NOT re-run)

| case | dofs | nnz | int32 |
|---|---|---|---|
| 3d_slab64 | 417,792 | 6.5×10⁷ | 3% |
| 3d_film128 | 6,389,760 | 1.02×10⁹ | 47% |
| mk32 128×128×48 (M=3,K=2) | 8,028,160 | 2.17×10⁹ | **OVERFLOW** |
| mk32 256×256×128 (Nova) | 84,541,440 | 2.28×10¹⁰ | **OVERFLOW** |

**What must be true regardless of hardware:**

- **3-D grows faster in both dofs and nnz/dof** — bigger *and* denser.
- **The sparsity matches the FE model** (54 nnz/dof in 3-D), not a 3^d−1
  stencil.
- **The complete memory estimate flags 256³ as over 48 GB.**
- **The physics differs by dimension** at equal resolution.
