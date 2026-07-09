# M1d Milestone Report — Full Device-Side Migration

*Formal closure 2026-07-08. Spec:
`docs/superpowers/specs/2026-07-06-m1d-device-migration.md` (D6 has the
closure evidence). Workstation: 2x RTX 6000 Ada (48 GB), CUDA 12.9,
warp 1.14, cuDSS via nvmath-python 0.9. Blueprint chapter:
`docs/superpowers/blueprint-device-assembly.md`.*

## 1. D1 ledger — done-list

| # | Stage | Status |
|---|---|---|
| 1 | NS numeric assembly | **DONE** — slot-map scatter (symbolic once/epoch, per-step device fill), constraint-aware; coloring variant prototyped, benchmarked, REJECTED (atomics win); ndof-generic (M4 films reuse it) |
| 2 | strong rows + pressure pin | **DONE** — folded into the device fill (span-zero + diag-one kernels; replaced 0.8-1.4 s/step host LIL surgery) |
| 3 | T-projection / constraints | **DONE** — cuFEM-style master-DOF expansion at slot-map build; hanging meshes gated at 1e-12 vs host T^T K T |
| 4 | GP field eval | **DONE (this closure)** — `assembly/gp_field.py` warp kernels: interp + consistent div, velocity/scalar gradients, fine-scale correction, BDF extrapolation/history axpbys, multi-component T spmv; stepper parity < 1e-11. The ADJOINT-chain interpolation transposes stay host (ms-class in every profile; migrate when they show) |
| 5 | face-block assembly + traction | **NOT migrated** — epoch-rare in current workloads (frozen-epoch heroes assemble the face system once and cache it); host cost never surfaced in a profile; recorded as the next candidate if a per-step face workload appears |
| 6 | classification/carve (torch host) | kept host per spec (epoch-rare) — M3 rung 3 (relaxed classification) is the successor |
| 7 | geometry oracle (torch) | kept torch per spec (GPU torch eval available) |

## 2. The measured numbers (the M1d ledger of record)

- **300x** — cuDSS vs host splu factorization: 686 s -> 2.26 s at 3-D
  L5 (16.6 s at 2-D L8); the profile headline that rewrote the D1
  priorities (commit b0a7b73).
- **4-40x** — device CSR assembly vs host scatter (40x 2-D L6, 30x 3-D
  L5, 14x 2-D L8); symbolic cost amortizes in ~20 assemblies.
- **56 s -> 2.9 s** — the symbolic slot-map builder vectorized via
  sparse fancy indexing at 3-D L5 (commit 5a5cb65); the M4 films then
  rode the same assembler to **56x** device-bound marches (36 ms/step,
  parity 1.45e-13).
- **2.2x end-to-end** — linearized stepper, cavity 2-D L6, 30 BDF2
  steps: 152 -> 69 ms/step (parity 2.6e-15); the step became
  SOLVER-BOUND, as designed (assembly alone was 40x).
- **zero-copy solve chain** — assemble_device: dlpack CSR straight into
  cuDSS, agreement 1e-15, 0.29 vs 0.40 s at 2-D L7 (the host round-trip
  was ~25% of solve time).
- **cuDSS mtlayer** — multithreaded host planning
  (libcudss_mtlayer_gomp): single-threaded plan() measured as a 3.5 h
  mostly-idle stall in per-step refactorization loops (M3 bunny v2);
  now the default via `solvers/linsolve.cudss_options()`.
- **THE CLOSURE TRACE (D3, 2026-07-08)** — one transient NS step,
  uniform cavity, device GP fields + device assembly + device-resident
  solve, f_fn=None (per-step host work = orchestration + the g_fn
  callback + history rotate):

  | case | dofs | solver | step | host compute | % |
  |---|---|---|---|---|---|
  | 2-D L8 | 198,147 | cudss | 54.1 ms | 0.44 ms | **0.81%** |
  | 3-D L5 | 143,748 | cudss | 1.13 s | 0.47 ms | **0.04%** |
  | 3-D L6 | 1,098,500 | fused BiCGStab | 10.0 s | 2.91 ms | **0.03%** |

  All under the 5% D3 bar. Honest remainders, measured: g_fn +
  strong-value concat 0.2-2 ms, history rotate 0.2-0.9 ms, f_fn
  callback ~2-7 ms at 2-D L8 when a body force is supplied.
- **H1-EPOCH 44.3x (D3 exit bar >= 10x)** — one GN epoch of the H1
  hero (L4, one steady + adjoint + 4 FD steadies), vs the documented
  M1c-era ~20 min/epoch (host-splu Picard, commit b0a7b73). Two
  measured rungs (benchmarks/m1d_h1_epoch.py, 2026-07-08): the
  splu->cudss switch alone = 181.7 s (6.6x, BELOW the bar — cProfile:
  12.2 s host assembly + 10 s LIL + 6.6 s per-iterate plan() per
  steady); porting the hero Picard loop onto the M1d machinery
  (DeviceNSAssembler + cached face-system slots + gp_field kernels +
  plan-once refactorization) = **27.1 s = 44.3x** (steady 31.1 -> 3.1 s;
  epoch-0 J/GN-step/cos reproduce the host path to every printed
  digit). The epoch is now ADJOINT-BOUND (9.3 s host cotangent loop) —
  the next candidate if it ever dominates.

## 3. Capacity finding + the GH200 item (PENDING-EXTERNAL)

cuDSS factorization at 3-D L6 (1.098M dofs, nnz 115M) ALLOC_FAILED on
the 48 GB card — default AND hybrid memory mode (plan() succeeds in
~5 s). The single-card direct-solver ceiling sits between 3-D L5 and
L6 on this workstation; the fused device BiCGStab over the same
zero-copy CSR operator carries L6 today (9.7 s/step at rtol 1e-10).

Architecture analysis (D5 verdict, Baskar's correction): **pure-device
is THE profile for the entire roadmap** — cuFEM performs dynamic
adaptivity pure-device, so moving geometry does not motivate coherence;
the coherent (GH200/GB200) interest reduces to CAPACITY ONLY (state >
HBM — exactly the L6 factorization above). The Nova GH200 coherent
profile measurement is **PENDING-EXTERNAL** (cluster kit stage 6); it
prices whether NVLink-C2C host-resident factors beat the in-HBM
iterative fallback before any Horizon investment.

## 4. Where this leaves the codebase

`use_device_assembly=True` on the linearized stepper = the pure-device
per-step loop: history node vectors up once, GP fields / assembly /
strong rows / factorize+solve on device, solution down once. The hero
steady driver (hero_h1_sphere_steady.steady) runs the same machinery
with the SBM face system composed via cached pattern slots. Gates:
19/19 in tests/test_device_assembly.py (consistency 1e-12-class vs the
in-tree host oracles at every stage, trajectory parity < 1e-11, both
scatter variants, hanging constraints, p2, 3-D). The blueprint chapter
records the designs cuFEM should copy and the one it should skip
(coloring).
