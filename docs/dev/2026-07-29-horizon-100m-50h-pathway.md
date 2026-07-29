# Pathway: 100M-DOF NS-SBM in ≤50 h on one TACC Horizon GB200 NVL4 node (no MPI)

**Date:** 2026-07-29. **Directive:** Baskar — "clear pathway to 100M perhaps in
under 50 hours (on TACC Horizon) without MPI."
**Anchors:** every number here is measured unless marked (assumption); sources:
`docs/dev/2026-07-28-track-a2-campaign.md` (§10 GH200 hold session, §11 A3
probe round), `docs/dev/2026-07-27-r2b-saddle-campaign.md`,
`docs/dev/2026-07-18-horizon-readiness.md`, WP0 (job 11771926).

## 1. Target and budget

Production run: 100M DOF, 8,000 BDF2 steps ⇒ **≤22.5 s/step** for 50 h wall.
Platform: one GB200 NVL4 node — 4× B200 (~186 GB HBM3e each, ~768 GB
aggregate, ~8 TB/s each), 2× Grace, NVLink5 intra-node; **no MPI** —
single-process multi-GPU via NCCL (our 2-GPU NCCL CG is proven; the 4-GPU
extension is work item W3).

## 2. The measured engine (2026-07-29)

| Quantity | Value | Where measured |
|---|---|---|
| Solver | `fgmres_bdiag`, restart=120, warm-start x0=2xⁿ−xⁿ⁻¹ | A3 §11 |
| s/step @ 8.58M (1 GH200, device asm) | **40.76** (676.2 iters) | a3-p5-combo |
| vs campaign baseline | −26 % wall, −48 % iters (two opt-in knobs) | §11 |
| Device vs host assembly (same mesh, bit-identical iters) | 4.33× s/step, 12.8× RSS | §10.9 |
| Warm setup at ~9M | ~2 min (mesh 94 s sublinear + assembly) | §10 / WP0 |
| Iteration growth | ~2× per 1M→10M decade | R2b + A2 |
| GPU persistent @68M | 62.7 GiB (fits 95 GiB HBM) | leg 5 |
| Host RSS, device path | 14.9 GB @8.58M | leg 4/6 |

All PCD variants are measured-refuted at ≥9M (inner-solve economics); raw
fused BiCGStab converges on the uniform saddle but is slower (61.5 s/step) —
it re-enters via W5b (short recurrence kills basis memory).

## 3. Memory plan at 100M (fp64, on the NVL4 node)

| Component | Size @100M | Placement |
|---|---|---|
| Saddle CSR (int64 indices) | ~165 GB | sharded ~41 GB/GPU |
| Krylov basis, r120 (V+Z = (2m+1)·N·8) | 193 GB | sharded ~48 GB/GPU |
| Krylov basis, r60 alternative | 96.8 GB | (single-GH200 note below) |
| Solver workspace (~4N) + fields | ~10 GB | sharded |
| Mesh/constraints (host build) | ~40–90 GB | Grace |
| **Aggregate device total** | **~370–400 GB** | vs 768 GB — comfortable |

Single-GH200 contrast: 100M persistent ~92 GiB is at the 95 GiB edge and the
r60 basis alone is 96.8 GB — **the basis is a first-class design driver on one
GH200** (mitigations: fp32 basis, shorter restart, short recurrence). On the
NVL4 node the constraint dissolves in fp64; fp32 remains an accelerator, not
a necessity.

## 4. Step-time projection at 100M (assumptions labeled)

Per-iteration cost is bandwidth-bound (matrix pass + basis/vector traffic):
- Matrix pass: 165 GB ÷ ~20 TB/s effective aggregate (a1) ≈ **8 ms**.
- Vector/orthogonalization traffic at r120 averages ~60 basis vectors touched
  per iteration ≈ 48 GB ÷ 20 TB/s ≈ **2.4 ms** + NCCL dot allreduces (a2).
- Measured GH200 per-iteration at 8.58M (r120): 60 ms; BW-scaling to the node
  (a3: ×11.65 size, ÷~5–6 effective BW ratio incl. multi-GPU overheads) gives
  **12–20 ms/iteration @100M**.
- Iterations: 676 @8.58M ×~2 (one decade, measured growth; a4: warm-start
  benefit in a developed march is projected to offset part of this) ≈
  **~1,100–1,400 iters/step**.

**Projected: ~16–27 s/step ⇒ 8,000 steps in ~36–60 h.** The 50 h budget is
met in the central estimate in fp64; two accelerators buy margin:
fp32 operator+basis with iterative refinement (existing
`iterative_refinement_device` machinery) ≈ ÷1.7–2 on the BW terms → ~9–15
s/step; developed-march warm-start (validated in W4) reduces iterations
further. **Honest caveat: the single unproven component is W3 (4-GPU NCCL
saddle solve); everything else is measured or a direct BW scaling.**

## 5. Work plan (ordered; each item is small and testable)

- **W1 — bound the chunked dof-indices intermediate** (`device_assembly.py`):
  the 137.4 GB single allocation at 68M (measured, leg 5) becomes bounded
  chunks (≤ a few GB). Unlocks uniform device assembly ≥ ~30M. Small,
  localized; the ChunkTable machinery already exists for the scatter itself.
- **W2 — constrained-scatter ChunkTable extension** (`device_assembly.py`):
  brings the 4.33×/12.8× device-assembly win to ADAPTIVE meshes (currently
  Warp-2³¹-blocked; scipy half already fixed via coo_array). Required for the
  thin-shell hero; not required for the uniform volumetric 100M.
- **W3 — 4-GPU NCCL saddle solve** (largest new piece): row-block CSR shards +
  halo-exchange matvec + allreduce dots, generalizing the proven 2-GPU NCCL CG
  pattern to `fgmres_dev`/bdiag (and to fused if W5b lands). Gate: 4-GPU
  ladder point reproducing single-GPU iteration counts at 8.58M, then ~34M,
  then 100M.
- **W4 — production-march validation of warm-start + restart at scale**
  (cheap; the knobs exist): a 200-step developed march at ~9M measuring the
  settled iteration count (the 5-step probe understates warm-start).
- **W5 — accelerators (optional, ordered by leverage):** (a) fp32
  operator+basis with IR wrapper; (b) fused+bdiag `apply_dev` hook (short
  recurrence: no basis memory, no orthogonalization — the ticket from A3);
  (c) CUDA-graph the iteration loop; (d) node-block (4×4) Jacobi upgrade of
  bdiag.

Sequencing: W1 → W3 (the volumetric 100M hero path) with W4 in parallel;
W2 joins for the thin-shell hero; W5 as budget insurance.

## 6. Thin-shell variant

The thin-shell hero at production quality is an ADAPTIVE ~9–15M-DOF problem,
not 100M uniform — after W2 it runs at ~40 s/step-class on a single GH200
(today's engine), and far faster on the node. The 100M-uniform pathway above
is the volumetric NS-SBM (bluff-body) hero; both share W1/W3/W4/W5.

## 7. Risks

| Risk | Exposure | Mitigation |
|---|---|---|
| W3 NCCL FGMRES efficiency (halo+allreduce overheads) | central | measure at 8.58M first; fused short-recurrence fallback (fewer reductions) |
| Iteration growth ≠ 2× at the 10M→100M decade | moderate | W4's developed-march data; node-block Jacobi (W5d) |
| int64 nnz throughout the device stack | low (coo_array fix pattern proven) | audit pass in W1 |
| B200 effective-BW assumption (a1) | moderate | Horizon readiness doc benchmarks on arrival |
