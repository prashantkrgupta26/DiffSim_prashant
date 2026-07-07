# M1d — Full Device-Side Migration (M0/M0.5/M1 scope)

Status: DRAFT v1, scoped with Baskar 2026-07-06. Slots between M1c and
the spec's physics M2 (Heat/Mass/closures). Everything migrated here
doubles as measured blueprint evidence for the cuFEM C++ chapters.

## D0. Strategy: one codebase, two runtime profiles

- **pure-device**: all hot data resident in GPU memory; host
  orchestrates only. Target: PCIe-attached FP64-strong cards (Nova
  A100/H200, H100 systems). Built FIRST — it also runs on coherent
  machines.
- **coherent**: cold/large arrays (constraint maps, symbolic patterns,
  history checkpoints) live in CPU memory with coherent NVLink-C2C
  access; host-side stages remain cheap. Target: GH200 / GB200
  (TACC Horizon: 2,000 GB200-NVL4 nodes, 2x Blackwell 185GB + Grace
  120GB per node, 80 TF FP64/node). The Nova GH200 experiment
  (cluster/ kit, stage 6) prices this profile before investment.
- **Forward flag (Horizon)**: Blackwell de-emphasizes native FP64;
  carry a mixed-precision hook (FP32 factorization + FP64 iterative
  refinement; FP64-emulated GEMM) as a design provision. Our fused
  Krylov and cuDSS paths are the natural insertion points.

## D1. Migration ledger (priority = measured cost; tonight's profiling
##     pass replaces guesses with numbers)

| # | Stage (today: host) | Device design | Reference |
|---|---|---|---|
| 1 | NS numeric assembly (COO/LIL) | fixed-sparsity epochs: symbolic once -> slot-mapped warp fill; BOTH scatter variants prototyped (slot maps w/ atomics vs element coloring) and MEASURED | cuFEM anti-pattern recorded; overnight prototype |
| 2 | strong rows + pressure pin | masked rows folded into the assembly kernel (identity-row write) | cuFEM constraint elimination |
| 3 | T-projection / constraint application | phase 1: device SpMM of cached T; phase 2: cuFEM-style master-DOF elimination in-kernel (<=4 masters+weights) or Dendro-KT tensor-product matrix-free T | both recorded in conventions doc |
| 4 | GP field eval + interpolation transposes (steppers, adjoint chains) | trivial warp kernels (same loop shapes as the taped kernels) | — |
| 5 | face-block assembly + traction | same slot-map pattern on face sets | — |
| 6 | classification/carve (torch host) | keep host (epoch-rare) in M1d; device candidate for later | — |
| 7 | geometry oracle (torch) | keep torch; GPU torch eval already available; coherent profile benefits | — |

## D2. Gates (per migrated stage, the M1d ritual)

1. **Bit-or-tolerance consistency** vs the host implementation on the
   test-zoo meshes (like the constraints-vectorization gate: the host
   version stays in-tree as the oracle).
2. Full suite green.
3. **Measured speedup table** committed with the change (per level/dim);
   a migration that does not win gets reverted and recorded.
4. AD tier unaffected (the adjoint sweeps consume migrated outputs).

## D3. Exit criteria

- One NS transient step at 2-D L8 / 3-D L6 with NO host work besides
  orchestration and epoch setup (profile trace as evidence).
- H1-class steady epoch cost reduced >= 10x (assembly-bound today).
- The blueprint chapter "device assembly + constraints" written from
  the measured prototypes.

## D4. Tonight (approved)

Profiling pass (per-stage timing table across {2D L6-L8, 3D L4-L5} for
forward + adjoint paths) THEN the dual assembly prototypes (slot-map vs
coloring) gated per D2. Commit-per-green; findings entries for measured
surprises.
