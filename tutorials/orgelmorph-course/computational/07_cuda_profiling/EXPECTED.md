# C7 — expected results (self-check)

`run.py --mode reference` (level 6) reproduces the following **ratios**; absolute
timings and bandwidths are hardware-dependent (RTX 6000 Ada here). Seven core
`[PASS]` gates and `ALL CORE GATES: PASS` must print, and the baseline check must
pass. The nsys gate is advisory.

## Measured (representative)

| quantity | value | invariant that must hold |
|---|---|---|
| one step | ~2.1 s, 12 Newton iters | step timed, iters in [1,20] |
| kernel-launch overhead | ~40 µs/launch | 0.1–1000 µs |
| device triad bandwidth | ~700 GB/s | — |
| triad arithmetic intensity | 0.083 flop/byte | < 0.5 (memory-bound) |
| PCIe copy bandwidth | ~26 GB/s | device BW > 10× PCIe |
| device / PCIe ratio | ~27× | > 10× |
| LU factor / solve | ~88 ms / ~3.6 ms | factor > 5× solve |
| D2H per step | ~35 MB | > 1 MB |
| device footprint (driver, isolated) | ~2.1 MB | > 1 MB |
| device footprint (analytical) | ~2.56 MB (Ae ~2.1 MB) | — |
| nsys capture | NVTX ranges captured | advisory |

## Key findings

- The step is **host-bound**: the sparse-LU factorization, repeated once per
  Newton iteration, dominates — the GPU is nearly idle on the `splu` path. The
  fix is factor reuse or a GPU solver (C4).
- **Data movement rules**: on-device bandwidth is ~27× the PCIe link;
  memory-bound kernels (assembly, triad) are limited by bytes moved, not flops.
- **Device memory** must be **accounted analytically** — the caching mempool
  hides the high-water mark from in-process queries; the dense element-Jacobian
  buffer `Ae` dominates.

## Nsight

On the course box `nsys profile` captures the `warmup`/`ch_step` NVTX ranges but
not the kernel-level rows (driver newer than the installed Nsight → no CUPTI
kernel tracing). Run `nsys stats --report cuda_gpu_kern_sum ...` on a matched
box for the per-kernel breakdown.
