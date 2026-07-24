# Distributed NCCL-CG Proof — Design Spec

**Date:** 2026-07-24
**Status:** APPROVED (design). Ready for `writing-plans` → subagent-driven build.
**Author context:** Brainstormed with Baskar; this is the first milestone on the
100M-DOF PPE solver path (the "custom NCCL-CG pathway").

---

## Goal

Build and validate a self-contained, single-node multi-GPU **preconditioned
Conjugate Gradient** for an SPD operator, using **torch.distributed / NCCL** (no
MPI) with a **per-GPU AMGX block preconditioner**. This proves the mechanic that
unblocks the 100M-DOF pressure-Poisson (PPE) solve, in isolation, before any
stepper integration.

This is a **proof harness**, not a product: it answers four questions.
1. **Correctness** — does distributed CG reproduce the single-GPU solution?
2. **Convergence** — how many CG iterations does the block-Jacobi preconditioner
   cost at 1 / 2 / 4 GPUs (i.e. how much global coupling do we lose)?
3. **Weak scaling** — is time/iteration flat as we grow DOF and GPUs together?
4. **Substrate** — do torch.distributed NCCL + per-rank AMGX + warp/device SpMV
   compose cleanly on one node?

## Why this exists (context)

The 100M PPE cannot be solved by:
- **Single-GPU AMGX** — hard int32 C ABI cap (`AMGX_matrix_upload_all(int nnz)`)
  ⇒ ~79.5M DOF/rank ceiling.
- **cuDSS direct (incl. MG)** — fill-in is 2–7 TB at 100M; exceeds all Horizon
  configs. (Probed, verdict recorded: not viable.)

⇒ The sole path is a **custom distributed iterative solver**: partition the SPD
Laplacian across GPUs, run distributed CG with NCCL, and precondition each local
block with AMGX (each block < 79.5M ⇒ the ABI cap is a non-issue). Baskar's hard
constraints: **no MPI**, **NCCL/NVLink**, **single node** (max ROI on TACC
Horizon B200 / GB200 NVL nodes).

---

## Locked design decisions (from the brainstorm)

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | **Preconditioner** | **Block-Jacobi first** (per-GPU AMGX on the owned block, no cross-rank coupling). Two-level (block-Jacobi + global coarse-grid correction) is the known upgrade if iteration counts degrade. | Horizon is 2–4 GPU/node where the coupling loss is mild; fastest to a working solver; the proof *measures* the loss to decide if the upgrade is needed. |
| 2 | **First milestone** | **Standalone distributed-CG proof** (2→4 GPU) on a partitioned SPD operator. NOT integrated into the stepper. | The distributed CG is the one genuinely unproven piece; de-risk it in isolation, correctness on gpubox's 2 GPUs immediately. |
| 3 | **Comm substrate** | **torch.distributed (NCCL backend), one process per GPU** (torchrun, no MPI). | Already installed on box (torch 2.12.1+cu130, NCCL 2.29.7, 2 GPUs). Composes cleanly with per-GPU AMGX (1 context/process). NCCL auto-selects transport: PCIe on gpubox (correctness), NVLink-5 on Horizon (perf) — same code. |

**Note on "single process":** Baskar's hard constraint was *no MPI + NCCL +
single node*. torch.distributed satisfies all three — it launches N processes via
its own rendezvous (torchrun), not `mpirun`. The one-process-per-GPU model was
chosen because it maps 1:1 onto per-GPU AMGX contexts (a single-process/N-GPU
model would juggle N AMGX contexts with `cudaSetDevice`, more fragile).

---

## Architecture

One process per GPU (rank). Each rank owns a **subdomain slab** of the SPD system
+ its own AMGX context. The distributed PCG loop reuses the *math* of
`krylov_dev.py` with exactly two communication points per iteration:

1. **Halo exchange** (`dist.batch_isend_irecv`): before each local SpMV, fill
   ghost `p`-values from neighbor slabs.
2. **Local SpMV** on owned rows (device K_p / warp matvec) → `Ap` on owned rows.
3. **Two `ncclAllReduce(SUM)`**: `⟨r,z⟩` (preconditioned residual dot) and
   `⟨p,Ap⟩` — the *only* global reductions per iteration.
4. **Preconditioner apply**: `z = M⁻¹ r` = one AMGX V-cycle on the owned block
   (block-Jacobi; no cross-rank coupling — the measured quantity).

Standard PCG recurrences run on local (owned) vectors; global scalars come from
the two allreduces.

### Components (files to create)

| File | Responsibility |
|------|----------------|
| `src/diffsim/solvers/dist_cg.py` | Distributed PCG: halo exchange, local SpMV wiring, two allreduces, AMGX-block preconditioner apply, convergence loop + iteration/residual logging. |
| `src/diffsim/mesh/partition.py` | Row-block slab partition + halo index maps: owned/ghost node sets, per-neighbor send/recv index lists, local↔global renumber. |
| `scripts/nccl_cg_proof.py` | torchrun driver + correctness/scaling harness (Stage-1 synthetic + Stage-2 real operator; single-GPU reference; scaling tables). |
| `tests/test_dist_cg.py` | Gated correctness tests (see Success Criteria). |
| `docs/dev/nccl-cg-proof.md` | Results writeup: iteration-count + weak-scaling tables, verdict on block-Jacobi vs. two-level. |

### Reuse map (do NOT reinvent)

- **`src/diffsim/solvers/krylov_dev.py`** — the device CG/Krylov skeleton; the
  distributed loop is this math with halo + allreduce inserted. Reuse the vector
  ops and the PCG recurrence structure.
- **`src/diffsim/solvers/amgx.py`** — per-GPU AMGX with setup-reuse
  (`replace_coefficients`). Each rank builds one AMGX solver on its owned block
  and calls it as the preconditioner. `amgx.py:~162` `upload_CSR` is the int32
  ABI site — keep each block < 79.5M DOF.
- **`src/diffsim/assembly/operators.py::DeviceScalarPoissonAssembler`
  / `assemble_csr_device()`** — the device K_p assembler (bit-exact,
  ~0.18s/step @ L7). Stage-2 uses this to build the *real* octree Poisson
  operator per slab.
- **Mesh fast-path** (`src/diffsim/mesh/nodes.py::_build_uniform_fast`,
  `_uniform_complete_level`; `constraints.py` identity short-circuit) — numbers
  nodes **lexicographically axis-0-slowest**. This is WHY a contiguous row-block
  == a geometric slab (see Partition). L9/135M uniform build is tractable
  (~215s, ~50 GB RSS on the box).

---

## Partition & halo — row-block slabs

Partition along **axis-0 (slowest-varying)** into `N` contiguous slabs of
node-index ranges. Because the mesh fast-path numbers nodes lexicographically
axis-0-slowest, **a contiguous row-block IS a geometric slab** — so we get the
simplest possible partition (contiguous index ranges) *and* minimal
single-plane halos *and* symmetric nearest-neighbor comm, with **no reordering**.

- **Owned** rows: rank `r` owns global node indices `[r·B, (r+1)·B)` (last rank
  takes the remainder).
- **Ghost** nodes: the one boundary plane of each adjacent slab whose values the
  local SpMV stencil reads (for the 7-point stencil, exactly the neighbor's
  nearest plane).
- **Comm lists:** `partition.py` builds, per rank, the send-index list (owned
  nodes neighbors need) and recv-index list (ghosts this rank needs), symmetric
  between neighbors ⇒ `batch_isend_irecv` with matched send/recv pairs.

`partition.py` is operator-agnostic: it produces index maps; `dist_cg.py`
consumes them for both the synthetic and the real operator.

---

## Test operators & precision

Two stages share the same partition + solver:

- **Stage 1 — synthetic 7-point Poisson** on a partitioned uniform cube with a
  **manufactured solution** (choose `u(x,y,z)`, set `f = -Δu`, Dirichlet BCs
  from `u`). Analytic error check, trivially scalable, isolates the
  distributed-CG mechanic and the iteration-count question.
- **Stage 2 — real octree scalar-Poisson** from `DeviceScalarPoissonAssembler`,
  same slab partition. Proves the solver works on the *actual* PPE operator the
  projection stepper will hand it.

**Precision:** fp64 CG throughout, `rtol = 1e-10`. AMGX preconditioner also fp64
for the proof (mixed-precision fp32-preconditioner is a known later lever, kept
out here to isolate variables).

---

## Success criteria

- **Correctness (multi-rank):** `‖x_dist − x_ref‖∞ / ‖x_ref‖∞ < 1e-8` vs a
  single-GPU AMGX/`splu` reference, at N=2 and N=4 ranks, for both stages.
- **Correctness (degenerate, always-on):** N=1 rank must equal serial
  `krylov_dev` CG **bit-for-bit** — runnable on the Mac / 1-GPU, so it lives in
  the always-on CPU suite (guards the non-distributed code paths).
- **Iteration counts:** report CG iters at 1 / 2 / 4 GPU, fixed problem size ⇒
  quantifies block-Jacobi's coupling loss and decides the two-level upgrade.
- **Weak scaling:** fixed DOF/GPU (~8M/GPU), grow 1→2→4, report time/iteration
  (ideal = flat) and total solve time.
- **Distributed-dot consistency:** `ncclAllReduce(SUM)` dot-products match a
  serial reduction to fp64 round-off (~1e-12); note NCCL sum is not bitwise
  deterministic across rank counts — assert to tolerance, not bitwise.

## Gate strategy

`tests/test_dist_cg.py`:
- Multi-rank cases **skip** when `torch.cuda.device_count() < 2` (Mac-safe).
- The N=1 degenerate correctness test runs everywhere (CPU/Mac included).
- The 2-GPU correctness + scaling harness runs on gpubox via the remote toolkit
  (below). 4-GPU is a Horizon/Nova follow-on.

---

## HANDOFF — for a follow-on Claude Code session

**Start here. This section is the pickup brief.**

### 0. Where to run, and the base commit
- **Compute on gpubox, not the Mac** (standing preference). The Mac
  (`/Users/baskarg/DiffSim`) is orchestration + CPU file-level checks only. All
  GPU/NCCL work runs on gpubox via `scripts/remote/`.
- **git base caveat:** at spec-writing time the Mac checkout `master` was
  `dd62272` (a "beyond-FH" line) and **21 behind `origin/master` `cffb033`**.
  The projection-ladder / mesh-fast-path / device-K_p / AMGX-scaling work from
  the prior session landed via gpubox and may not be on this Mac checkout.
  **Before building: reconcile the base** — confirm which commit actually has
  `krylov_dev.py`, `amgx.py`, `DeviceScalarPoissonAssembler`, and the mesh
  fast-path (`git log --oneline --all | grep -iE 'mesh|amgx|device.*assembl'`
  on gpubox's canonical repo `GPUBOX_REPO_ABS`), and branch the NCCL-CG work off
  *that*. Do NOT assume the Mac `master` is the canonical base.

### 1. Box environment (verified 2026-07-24)
- **torch 2.12.1+cu130, NCCL 2.29.7**, `torch.cuda.device_count() == 2`.
- **warp 1.15.0** present (SpMV / kernels).
- **cupy NOT installed** (don't depend on it).
- GPUs: **2× RTX 6000 Ada, 49140 MiB each**. **PCIe, no NVLink / no peer
  access** on gpubox (Horizon has NVLink-5). Venv python:
  `/home/bglab/Baskar/DiffSim/.venv/bin/python` (`GPUBOX_VENV_PY`).

### 2. How to launch multi-GPU on the box
- torchrun, no MPI: `torchrun --standalone --nproc_per_node=2 scripts/nccl_cg_proof.py ...`
- Dispatch through the remote toolkit so it respects the run-lock:
  `scripts/remote/gpubox-run.sh '<cmd>' <label>` then poll with
  `gpubox-poll.sh <log>`. **Never** `rm` on the shared box without authorization.
- Nova: **≤4 concurrent jobs** (hard cap). **gpubox: unconstrained.**
  4-GPU/NVLink perf runs go to Nova (A100/GH200) or Horizon (B200) later —
  the Nova SLURM kit is wired (`scripts/remote/nova-sync-submit.sh`,
  `nova-poll.sh`; `NOVA_REPO_ABS=/work/mech-ai/baskarg/DiffSim`).

### 3. Build method
- Use `writing-plans` → `subagent-driven-development` (this repo's standard).
- The ledger is `.superpowers/sdd/progress.md` — append one line per milestone;
  it is the recovery map across compaction.
- Device auto-resolvers exist: use `default_device()` / `default_linsolver()`
  (from `diffsim`), not hardcoded `cuda:0` (Olga onboarding fix). In the
  distributed harness, each rank pins its device from `LOCAL_RANK`.

### 4. The two-level upgrade (only if the proof demands it)
If iteration counts blow up at 2–4 GPU, add a **global coarse-grid correction**:
a small coarse operator (aggressive coarsening of the full domain, replicated or
solved on rank 0), with restriction/prolongation across the partition, applied
additively with the block-Jacobi. Design it as a second preconditioner class
behind the same `M⁻¹` interface in `dist_cg.py` so the CG loop is unchanged.

### 5. After the proof
Integrate into the projection stepper's PPE solve as a `solver="nccl_cg"` option
(the stepper feeds it the device K_p + the slab partition). Then distribute the
rest of the pipeline (assembly, predictor, correction) for the full 100M NS hero
run. That is a *separate*, larger milestone (full SPMD stepper) — do not fold it
into this proof.

---

## Non-goals (YAGNI for this milestone)
- No stepper integration.
- No mixed-precision preconditioner.
- No two-level coarse grid (unless the proof's iteration counts force it).
- No dynamic/adaptive repartitioning — uniform slab only (matches the 100M
  uniform-grid hero).
- No >1 node / MPI.
