# AMGX and the 2^31 nnz wall — build recipe & verdict

**Task:** rebuild AMGX with "64-bit indexing" to clear the 100M-DOF blocker
(single-rank AMGX overflows at nnz ≈ 2^31, i.e. ~79.5M DOF at a 27-pt stencil,
BELOW the 100M target).

**Verdict (2026-07-24): there is NO 64-bit build flag that widens single-rank
`nnz`/`row_ptrs`. The 32-bit limit is a hard property of the AMGX C ABI, not a
compile-time option. Rebuilding is a no-op for this wall.** The real path is
distributed / multi-rank AMGX. Details below.

---

## Why "rebuild with 64-bit indexing" does not exist for this wall

The AMGX single-rank upload API is hard-wired to 32-bit `int` for the row count,
the nnz count, and the CSR row-pointer array. From `include/amgx_c.h`:

```c
AMGX_RC AMGX_matrix_upload_all
(AMGX_matrix_handle mtx,
 int  n,               // 32-bit
 int  nnz,             // 32-bit  <-- overflows at 2^31
 int  block_dimx, int block_dimy,
 const int  *row_ptrs, // 32-bit CSR row offsets; last entry == nnz
 const int  *col_indices,
 const void *data, const void *diag_data);
```

Every upload entry point (`AMGX_matrix_upload_all`,
`AMGX_matrix_upload_all_global`, `AMGX_matrix_upload_distributed`) takes
`int n, int nnz` and `const int *row_ptrs`. The only 64-bit widening AMGX offers
is for **global column indices** in the *distributed* path
(`col_indices_global` may be `int64_t`; `AMGX_distribution_set_32bit_colindices`
selects int vs int64). That widens the *global column address space across
ranks* — it does NOT widen a single rank's local `nnz` or `row_offsets`.

Internally this is confirmed too. `include/matrix.h`:

```c
typedef typename IndPrecisionMap<AMGX_indInt>::Type INDEX_TYPE;  // == int (32-bit)
...
index_type num_rows, num_cols, num_nz;   // index_type == TConfig::IndPrec
```

`AMGX_indInt64` (→ `int64_t`) exists in `IndPrecisionMap`, but **no instantiated
solver mode uses it** — every `AMGX_mode_*` in `amgx_config.h` ends in `I` =
`AMGX_indInt` (32-bit): `dDDI`, `dDFI`, `dFFI`, … There is no `dDDI64`. There
are **no `option(...)` entries in `CMakeLists.txt`** for index width. So no
`-D...` flag changes any of the above.

Upstream `NVIDIA/AMGX/main` header confirms the same signatures (checked
2026-07-24): only column indices have a 64-bit variant; `n`, `nnz`, `row_ptrs`
are `int` throughout.

## Empirical confirmation (against the existing lib, box, 2026-07-24)

`/tmp/amgx_int_probe.py` (run with the box venv) reproduces the exact numbers:

```
int32 max                = 2,147,483,647
DOF at which nnz>2^31     = 79,536,431  (~79.5M DOF)   <-- overflow BELOW 100M
nnz for 100M DOF (27-pt) = 2,700,000,000  = 1.26 x int32-max
c_int(2,700,000,000).value = -1,594,967,296   -> OVERFLOW (wraps negative)
int32 indptr stores 100M-DOF nnz -> -1,594,967,296  -> OVERFLOW
```

The 100M-DOF PPE (2.7 billion nnz) wraps to a negative int32 at the ABI
boundary — in *both* the `int nnz` argument and the `int32` `indptr[-1]`. Our
pipeline hits this via `pyamgx.Matrix.upload_CSR` → `AMGX_matrix_upload_all`
(see `src/diffsim/solvers/amgx.py:162`), and pyamgx's `Matrix.pyx` declares
`nnz` as C `int`.

## The stock build recipe (CUDA 12.4, sm_89) — unchanged, still 32-bit

For reference, this is how AMGX is (and was) built on the box. It is correct for
what it does; it simply cannot be made 64-bit-nnz by any flag.

```bash
export CUDA_HOME=/usr/local/cuda-12.4          # box /usr/bin/nvcc is 11.5 (no sm_89)
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \              # RTX 6000 Ada; GH200 -> 90
  -DCMAKE_NO_MPI=1 \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
cmake --build build -j --target amgxsh          # -> build/libamgxsh.so
# run: LD_LIBRARY_PATH=/usr/lib/wsl/lib:<build>
```

There is no `-DAMGX_*64*` or index-width option to add. (For the GH200 hero,
change only `-DCMAKE_CUDA_ARCHITECTURES=90` and drop `/usr/lib/wsl/lib` — it is
WSL-specific. That build is still 32-bit-nnz.)

## The real path to > 2^31 nnz (100M DOF): distributed / multi-rank AMGX

Because the limit is per-rank, the supported way past 2^31 nnz is **MPI-multi-rank
AMGX**, where each rank owns a partition whose *local* nnz stays < 2^31 and the
global assembly uses the 64-bit `col_indices_global` path
(`AMGX_matrix_upload_all_global` / `AMGX_matrix_upload_distributed` with
`AMGX_distribution_*`). This needs:

1. AMGX rebuilt **with MPI** (drop `-DCMAKE_NO_MPI=1`; link an MPI).
2. Domain partitioning of the PPE (≥2 ranks so each partition's nnz < 2^31; for
   100M DOF / 2.7B nnz, ≥2 ranks suffices numerically, but use ≥4 for headroom).
3. Our `pyamgx` upload path replaced with the distributed upload
   (`upload_all_global` + partition vector / `AMGX_distribution_handle`), passing
   `col_indices_global` as `int64_t`. This is a non-trivial pipeline change, not
   a rebuild.

This is a materially larger task than a flag flip and should be scoped
separately (multi-GPU on the GH200 node, or MPI over the two RTX 6000 Ada GPUs
as a functional prototype). Note the L8 de-risk showed memory is fine (100M PPE
~33 GB) — the wall is purely the 32-bit index ABI, so distributed AMGX (not more
memory) is the fix.

### Alternative if staying single-rank

If a single-rank solve of 100M DOF is required, AMGX cannot do it at 27-pt
stencil. Options: (a) a solver whose API is 64-bit-nnz (e.g. cuDSS handles
64-bit but hit `ALLOC_FAILED` at L6 1.09M here; hypre with `HYPRE_BigInt`;
Ginkgo with int64 indices), or (b) reduce nnz/row (lower-order stencil) so
nnz < 2^31 — only viable up to ~79.5M DOF.
