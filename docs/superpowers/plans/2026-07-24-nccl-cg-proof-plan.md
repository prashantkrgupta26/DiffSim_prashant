# NCCL-CG Standalone Proof — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
> Design spec: `docs/superpowers/specs/2026-07-24-nccl-cg-proof-design.md` (read it — it
> holds the locked decisions and rationale).

**Goal:** A standalone single-node multi-GPU preconditioned CG for an SPD operator
(torch.distributed/NCCL, per-GPU AMGX block preconditioner), proving correctness +
weak scaling + block-Jacobi iteration counts, before any stepper integration.

**Architecture:** One process per GPU (torchrun, no MPI). Distributed PCG reuses the
math of `krylov_dev.py` with two comm points/iter: `ncclAllReduce` on the two dots +
halo exchange before each local SpMV. Row-block slab partition (== geometric slab given
the mesh fast-path's lexicographic axis-0-slowest numbering). Per-GPU AMGX (`amgx.py`)
as the block preconditioner (block < 79.5M ⇒ ABI cap moot).

**Tech:** torch 2.12 / NCCL 2.29.7 (gpubox, 2× RTX 6000 Ada, PCIe), warp, `amgx.py`,
`DeviceScalarPoissonAssembler` / `assemble_csr_device` (operators.py:475/585), the CG
kernels in `krylov_dev.py`.

## Global Constraints
- **No MPI.** Single node. torchrun launch (`--standalone --nproc_per_node=N`).
- **Compute on gpubox**, not the Mac (standing preference). Tasks 1–2 are CPU-testable
  and developed/tested on the Mac venv; Tasks 3–5's real multi-GPU runs go to gpubox
  via `scripts/remote/gpubox-run.sh` (respect the run-lock; never `rm` on the box).
- **fp64 CG, rtol 1e-10.** AMGX preconditioner fp64 for the proof.
- **The N=1 degenerate path must equal serial CG bit-for-bit** and run on CPU (always-on
  suite guard).
- **Do NOT modify** `krylov_dev.py` / `amgx.py` / `operators.py` — consume them. If a
  genuine reuse gap appears, note it; don't expand scope.
- **Comm + preconditioner + SpMV are injected** (protocol objects), so the CG core is
  testable with a SerialComm + Jacobi precond on CPU, and swapped for TorchDistComm +
  AMGX on GPU without touching the CG loop.

---

### Task 1: Slab partition + halo maps — `src/diffsim/mesh/partition.py`  (CPU-testable)
**Files:** Create `src/diffsim/mesh/partition.py`; Create `tests/test_partition.py`.
**Produces:**
- `slab_partition(dims, n_ranks) -> list[SlabPart]` where `dims=(nx,ny,nz)` is the
  uniform grid node count per axis (axis-0 = slowest, matching the mesh fast-path).
  Each `SlabPart` (dataclass) has: `rank`, `owned: slice/np.ndarray` (global node ids of
  the owned axis-0 slab), `ghost: np.ndarray` (global ids of the 1-plane halo this rank
  reads), `send: dict[nbr_rank -> np.ndarray]` (owned global ids each neighbor needs),
  `recv: dict[nbr_rank -> np.ndarray]` (ghost global ids received), and a
  `local_index(global_ids) -> np.ndarray` map into the rank's local (owned+ghost) vector.
- Partition is contiguous along axis-0; halo width = 1 plane (7-point stencil).

**Tests (CPU, deterministic):**
- Owned slabs over all ranks partition `[0, prod(dims))` disjointly and completely.
- For interior ranks, `ghost` = exactly the neighbors' adjacent axis-0 planes; boundary
  ranks have a half-width halo.
- `send`/`recv` are symmetric between neighbor pairs (rank r's send to r+1 == rank r+1's
  recv from r).
- `local_index` round-trips owned+ghost global ids to a contiguous local range.
- Cover n_ranks ∈ {1,2,4}, dims small (e.g. 4×4×4, 8×4×4).
- Run: `.venv/bin/python -m pytest tests/test_partition.py -q` → PASS. Commit.

### Task 2: Distributed PCG core + comm/precond protocols — `src/diffsim/solvers/dist_cg.py`  (N=1 CPU-testable)
**Files:** Create `src/diffsim/solvers/dist_cg.py`; Create `tests/test_dist_cg_serial.py`.
**Consumes:** Task 1 (`SlabPart`).
**Produces:**
- `class Comm(Protocol)`: `allreduce_sum(x: float) -> float`; `exchange_halo(local_vec) -> None`
  (fills ghost entries in place).
- `class SerialComm(Comm)`: N=1 — `allreduce_sum` returns x; `exchange_halo` is a no-op
  (no ghosts). CPU.
- `def pcg(spmv, precond, b_local, comm, *, rtol=1e-10, maxit, dot=None) -> (x_local, info)`:
  distributed PCG. `spmv(p_local) -> Ap_owned` (caller does halo exchange via `comm`
  inside spmv, then local matvec on owned rows). Two `comm.allreduce_sum` per iter
  (`⟨r,z⟩`, `⟨p,Ap⟩`). `precond(r_owned) -> z_owned`. `info` = dict(iters, resid_history).
- Reuse the recurrence/vector-op structure of `krylov_dev.py` (import its dot/axpy
  kernels where they apply; for the CPU serial test a numpy fallback is acceptable if
  krylov_dev requires warp-CUDA).

**Tests (CPU):**
- With `SerialComm`, a Jacobi precond, and a small SPD matrix (e.g. 1-D/2-D 5-point
  Laplacian as dense/scipy), `pcg` converges to `‖x-x_ref‖/‖x_ref‖ < 1e-8` vs
  `scipy.sparse.linalg.spsolve`, and iteration count matches a reference scipy CG within
  ±1.
- Determinism: two runs give identical iters + residual history.
- Run: `.venv/bin/python -m pytest tests/test_dist_cg_serial.py -q` → PASS. Commit.

### Task 3: Stage-1 synthetic Poisson + torchrun driver — `scripts/nccl_cg_proof.py`  (GPU: gpubox)
**Files:** Create `scripts/nccl_cg_proof.py`; Create `src/diffsim/solvers/dist_cg_torch.py`
(the `TorchDistComm(Comm)` + AMGX block-precond wiring); extend `tests/test_dist_cg_serial.py`
or add `tests/test_nccl_cg_stage1_serial.py` for the CPU-runnable N=1 synthetic case.
**Consumes:** Tasks 1–2.
**Produces:**
- `TorchDistComm`: `allreduce_sum` via `torch.distributed.all_reduce(SUM)`;
  `exchange_halo` via `batch_isend_irecv` using the Task-1 send/recv maps. Each rank pins
  its device from `LOCAL_RANK`.
- Synthetic 7-point Poisson on a partitioned uniform cube with a manufactured solution
  `u`, `f=-Δu`, Dirichlet BC from `u`. Local SpMV on owned rows reading ghost values.
- AMGX block preconditioner: one `amgx.py` solver per rank on the owned block (fp64).
- Driver `scripts/nccl_cg_proof.py --stage synthetic --dims ... --ranks N`: builds the
  partitioned system, runs `pcg` with `TorchDistComm`+AMGX, compares to a single-GPU
  AMGX/`splu` reference, prints `‖x_dist-x_ref‖∞/‖x_ref‖∞` + iters.

**Validation:**
- **CPU (always-on):** N=1 synthetic via `SerialComm`+Jacobi == serial reference (in the
  test). 
- **gpubox (via gpubox-run.sh):** `torchrun --standalone --nproc_per_node=2 scripts/nccl_cg_proof.py
  --stage synthetic ...` → rel-err < 1e-8 vs single-GPU reference; record iters for N=1,2.
  (4-GPU deferred to a Nova/Horizon follow-on.)
- Commit after the box run reports rel-err < 1e-8.

### Task 4: Stage-2 real device-K_p operator  (GPU: gpubox)
**Files:** Modify `scripts/nccl_cg_proof.py` (add `--stage real`); reuse
`DeviceScalarPoissonAssembler`/`assemble_csr_device` (operators.py) to build the real
octree scalar-Poisson per slab.
**Consumes:** Tasks 1–3.
**Validation (gpubox):** 2-GPU `--stage real` rel-err < 1e-8 vs single-GPU reference on
the same operator. Commit after the box confirms.

### Task 5: Scaling/iteration harness + writeup — `docs/dev/nccl-cg-proof.md`  (GPU: gpubox)
**Files:** Modify `scripts/nccl_cg_proof.py` (scaling mode: fixed DOF/GPU weak-scaling,
iteration-count table for N=1/2); Create `docs/dev/nccl-cg-proof.md`.
**Validation (gpubox):** produce the weak-scaling (time/iter) + iteration-count tables;
write the verdict on block-Jacobi vs the two-level upgrade. Append a ledger line to
`.superpowers/sdd/progress.md`. Commit.

---

## Self-review checklist (per task)
- No edits to krylov_dev/amgx/operators; comm+precond+spmv injected.
- N=1 serial path is bit-for-bit serial CG and runs on CPU.
- GPU tasks validated on gpubox via the remote toolkit (run-lock respected), rel-err<1e-8.
- fp64 throughout; iteration counts recorded for the block-Jacobi verdict.
