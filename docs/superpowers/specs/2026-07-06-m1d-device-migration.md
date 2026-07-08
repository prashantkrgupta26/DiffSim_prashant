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

## D5. VERDICT (night one, 2026-07-06) + correction

- Per-step hot loop: DEVICE-ONLY ACHIEVED — assemble_device closes the
  zero-copy chain into cuDSS (agree 1e-15; the host round-trip was ~25%
  of solve time at 2D L7). Remaining host: GP-field einsums (ms),
  orchestration (negligible), epoch setup (once per epoch).
- Baskar's correction (2026-07-06 night): cuFEM performs DYNAMIC
  ADAPTIVITY PURE-DEVICE — so moving-geometry workloads do NOT motivate
  the coherent profile either. Coherent (GH200/GB200) interest reduces
  to CAPACITY ONLY (state > HBM); the Nova GH200 stage measures that.
  Pure-device is the profile for the entire roadmap otherwise.
- cuFEM has NO differentiability: differentiating THROUGH device
  adaptivity is DiffSim's contribution — proposed milestone M3
  (transfer-operator adjoints -> event-branch differentiation -> relaxed
  classification), plus an adjoint-readiness requirements memo to cuFEM
  (explicit transfer ops, event logs, deterministic tie-breaks,
  threshold parameters) BEFORE its adaptivity design freezes. Task #15.

## D6. FORMAL CLOSURE (2026-07-08)

Report: docs/superpowers/m1d-milestone-report.md. Three closing items:

1. **Device GP-field kernels (D1 item 4)** — the last per-step host
   compute (the stepper's velocity/scalar interp-to-GP einsums) moved
   to warp kernels: `src/diffsim/assembly/gp_field.py` (interp +
   consistent div, velocity/scalar gradients, fine-scale correction,
   GP/node axpbys, multi-component T spmv — house factory pattern,
   enable_backward=False, rolled loops for nbf>4/dim 3). Wired into
   LinearizedMonolithicStepper behind `use_device_assembly`: history
   node vectors up once per step, BDF extrapolation/history combination
   as device axpbys, GP fields feed DeviceNSAssembler as wp arrays (it
   now accepts device inputs directly), and solver="cudss" consumes the
   zero-copy device CSR (plan once, per-step refactorize+solve);
   solver="fused" runs the device BiCGStab over the same zero-copy CSR
   operator. GATES: gp-field kernels == host einsums at 1e-13 (2-D/3-D,
   hanging-T, p2); stepper trajectory parity < 1e-11 vs the host path
   (2-D + 3-D, splu and device-resident cudss); fused parity at the
   iterative tol; the extra_matrix/extra_rhs composition hook (cached
   face systems on the fixed pattern, added before strong rows) vs the
   host A+Af at 1e-12. tests/test_device_assembly.py 19/19.

2. **The D3 trace (benchmarks/m1d_closure_trace.py, 2026-07-08)** —
   one transient NS step, uniform cavity, use_device_assembly=True,
   f_fn=None, traced per stage (sync boundaries; host stages are pure
   numpy/python segments):

   | case | dofs | solver | step | per-step host compute | gate (<5%) |
   |---|---|---|---|---|---|
   | 2-D L8 | 198,147 | cudss (device CSR) | 54.1 ms | 0.44 ms | **0.81% PASS** |
   | 3-D L5 | 143,748 | cudss (device CSR) | 1.127 s | 0.47 ms | **0.04% PASS** |
   | 3-D L6 | 1,098,500 | fused BiCGStab (device CSR) | 10.0 s | 2.91 ms | **0.03% PASS** |

   Honest remainders (measured): g_fn strong-row callback + strong-value
   concat (0.16-1.95 ms), history rotate (0.17-0.85 ms O(N) numpy), and
   the f_fn body-force callback WHEN supplied (~2-7 ms at 2-D L8,
   allocation-dominated; f_fn=None removes it — zero body force needs
   no per-step eval). Per-step transfers: history node vectors up,
   solution down (the output/checkpoint transfers the GPU-only
   convention allows). Epoch setup (one-time): symbolic slot maps 1.5 s
   (2-D L8) / 23 s (3-D L6); first step (compile + solver plan) 2-31 s.
   **Capacity finding:** cuDSS factorization at 3-D L6 ALLOC_FAILED on
   the 48 GB card (default AND hybrid memory mode; plan() fine) — the
   direct-solver ceiling sits between L5 and L6 here; the fused device
   Krylov carries L6. This is the D5 coherent-profile capacity question
   made concrete (Nova GH200 probe = PENDING-EXTERNAL).

3. **H1-epoch >= 10x (benchmarks/m1d_h1_epoch.py)** — one GN epoch of
   the H1 hero (L4: one steady + adjoint + 4 FD steadies) on the
   current stack vs the documented M1c-era ~20 min/epoch (host-splu
   Picard; commit b0a7b73). The hero's Picard loop was ported to the
   M1d machinery as part of this closure (cProfile had shown its
   per-steady cost was 12.2 s host assemble_linear_ns + 10 s LIL
   surgery + 6.6 s PER-ITERATE cuDSS plan(): the un-migrated stack).
   Now: DeviceNSAssembler (face system via cached slots on the frozen
   pattern, strong rows in-kernel) + gp_field kernels + plan-once cuDSS
   refactorization. Measured 2026-07-08: **steady 31.1 s -> 3.1 s;
   epoch 181.7 s (6.6x, splu->cudss alone) -> 27.1 s = 44.3x — PASS**
   (>= 10x bar). Epoch-0 sanity identical to the host path to every
   printed digit (J=3.4196e+00, |GN step|=1.042e-02, adjoint-vs-Jac
   cos=-0.076665 — the low epoch-0 cos is historical hero behavior;
   the baseline records "cos up to 0.995" at CONVERGED epochs). The
   epoch is now ADJOINT-BOUND (9.3 s host cotangent fixed-point loop)
   — the next migration candidate if H1-class epochs ever dominate.
