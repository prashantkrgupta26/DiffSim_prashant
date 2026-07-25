# NCCL-CG Standalone Proof — Results & Verdict

**Date:** 2026-07-24
**Branch:** `nccl-cg-proof`
**Spec/plan:** `docs/superpowers/specs/2026-07-24-nccl-cg-proof-design.md`,
`docs/superpowers/plans/2026-07-24-nccl-cg-proof-plan.md`

## What this proves

A standalone single-node multi-GPU **preconditioned Conjugate Gradient** for an
SPD operator — the mechanic that unblocks the 100M-DOF pressure-Poisson solve:

- **torch.distributed / NCCL**, one process per GPU, **no MPI**.
- **Row-block slab partition** (== geometric slab, given the mesh fast-path's
  lexicographic axis-0-slowest node numbering) with a 1-plane halo.
- **Per-GPU AMGX block-Jacobi preconditioner** (each rank solves its owned
  diagonal block; block < 79.5M ⇒ the AMGX int32 ABI cap is a non-issue).
- Validated on both a **synthetic 7-point Poisson** and the **real octree
  `DeviceScalarPoissonAssembler` K_p** operator.

Development method: the whole distributed loop was proven first on the **Mac via
gloo (2 CPU processes)** — partition + halo exchange + allreduce + iterative CG —
so the gpubox run only had to shake out genuinely GPU-specific issues. It did
(see "GPU-only bugs"), fast.

## Correctness (2× RTX 6000 Ada, gpubox)

All runs: `--rhs random` (non-eigenvector RHS, so CG runs many iterations and the
full β-recurrence / repeated-halo-exchange loop is exercised), fp64, rtol 1e-10.
`rel_err = ‖x_dist − x_ref‖∞ / ‖x_ref‖∞` vs a single-process scipy `spsolve` of
the identical system.

**Synthetic 7-point Poisson (2-GPU NCCL):**

| dims | DOF | Jacobi iters | AMGX-block iters | rel_err |
|------|-----|-------------|------------------|---------|
| 16×8×8 | 1,024 | 52 | 21 | 8.9e-11 |
| 24×12×12 | 3,456 | 79 | 26 | 1.3e-10 |

**Real octree K_p (2-GPU NCCL), level L → (2^L+1)³ nodes:**

| L | nodes | Jacobi iters (t) | AMGX-block iters (t) | rel_err |
|---|-------|------------------|----------------------|---------|
| 4 | 4,913  | 40 (0.29 s) | 21 (0.66 s) | 1.3e-10 |
| 5 | 35,937 | 80 (0.47 s) | 31 (1.71 s) | 1.6e-10 |

The NCCL/GPU iteration counts and rel_err are **identical to the gloo/CPU
reference** (e.g. real L4 = 40 iters on both, L5 = 80 on both) — the GPU path is
numerically the same as the CPU path. `--rhs mms` (single-eigenmode manufactured
solution) converges in 1 iteration on every backend, as expected.

L6 (274,625 nodes): the *distributed solve* runs fine, but the *serial scipy
reference* (`spsolve` on a 274k×274k matrix) exceeds a 500 s wall — direct
validation doesn't scale past ~L5. Larger-N validation needs an iterative or
multi-rank reference (future work); the correctness is already established at
L4/L5 and the mechanic is size-independent.

## Block-Jacobi verdict (decision-1)

The reason we measured iteration counts: **how much does block-Jacobi's loss of
cross-block coupling cost as the problem grows?**

- **Diagonal Jacobi**: iterations scale as expected for an unpreconditioned-ish
  Poisson — ~**double per ~8× DOF** (L4→L5: 40→80), i.e. `iters ∝ h⁻¹ ∝ N^{1/3}`.
- **AMGX block-Jacobi**: far flatter — **21→31** over the same 7× DOF jump — but
  it **still grows**. That growth is exactly block-Jacobi's cross-block coupling
  loss (at 2 blocks it is mild; the boundary between the two slabs is not
  preconditioned across).

**Verdict:** block-Jacobi is the right *first* preconditioner (fast to a working
solver, big iteration win over diagonal Jacobi), but its iteration count is **not
mesh-independent** and the growth will compound as GPU count and N rise. The
**two-level coarse-grid correction** (block-Jacobi + a small global coarse solve,
the parked decision-1 upgrade) is the indicated next lever for larger runs.

**Caveat on wall-time:** at these sizes AMGX-block has *higher* wall-time than
Jacobi despite fewer iterations (L4: 0.66 s vs 0.29 s), because each apply solves
the owned block to a tight tolerance (near-exact block inverse — chosen so M⁻¹ is
a *fixed* linear operator, required for standard CG). The cheaper **fixed
single-V-cycle apply** (fewer flops per apply, slightly more outer iterations) is
the natural perf follow-up, and would also relax the fixed-operator constraint
into flexible-CG territory if needed.

## GPU-only bugs the CPU/gloo path could not catch

1. **Device pinning** — every NCCL rank must `torch.cuda.set_device(local_rank)`;
   NCCL binds internal buffers to the *current* device, so without it both ranks
   collide on cuda:0 (CUDA error 999 in `batch_isend_irecv`).
2. **Device-resident collectives** — the post-solve gather used CPU tensors +
   `dist.gather`; NCCL rejects CPU tensors and lacks robust gather/scatter.
   Replaced with a device-resident **padded `all_gather`** (handles unequal slab
   sizes; works on both nccl and gloo).

## Runtime environment (gpubox / WSL2)

gpubox is WSL2; NCCL 2.29 multi-GPU there needs (see memory
`gpubox-wsl2-nccl-env`):

```
NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_CUMEM_ENABLE=0   # WSL2 cuMem error 999
LD_LIBRARY_PATH=/home/bglab/AMGX/build:$LD_LIBRARY_PATH      # libamgxsh.so (for --precond amgx)
```

These are a **gpubox/WSL2 correctness-dev** workaround only — they disable NCCL
perf features and must NOT be set on a real NVLink box (Nova A100/H200, Horizon
B200), which run NCCL with defaults.

Example run:

```
torchrun --standalone --nproc_per_node=2 scripts/nccl_cg_proof.py \
    --stage real --level 4 --backend nccl --precond amgx --rhs random
```

## Components (branch `nccl-cg-proof`)

| File | Role |
|------|------|
| `src/diffsim/mesh/partition.py` | slab partition + halo maps + `extract_local_block` (global CSR → local owned-rows × owned+ghost-cols) |
| `src/diffsim/solvers/dist_cg.py` | array-agnostic distributed PCG core + `Comm` protocol + `SerialComm` |
| `src/diffsim/solvers/dist_cg_torch.py` | `TorchDistComm` (NCCL/gloo allreduce + `batch_isend_irecv` halo) |
| `scripts/nccl_cg_proof.py` | driver: `--stage {synthetic,real}`, `--backend {gloo,nccl}`, `--precond {jacobi,amgx}`, `--rhs {mms,random}` |
| `tests/test_{partition,dist_cg_serial,extract_local_block}.py` | 107 CPU tests |

## Next steps

1. **Real NVLink scaling numbers** — gpubox is PCIe/WSL2 (correctness only).
   Weak-scaling *time*, 4/8-GPU, and iteration growth need Nova A100 (4–8/node) or
   H200 (4/node). (Nova's GH200 partition `nova-arm` is a *single* GPU node —
   good for a single-GPU *capacity* run, not multi-GPU scaling.)
2. **Two-level coarse correction** — per the verdict above.
3. **Fixed single-V-cycle AMGX apply** — the wall-time perf follow-up.
4. **Stepper integration** — expose `solver="nccl_cg"` in the projection PPE solve.
