# C7 — CUDA profiling and memory

**Why it matters.** "It's slow" is not a diagnosis; a profile is. This chapter
profiles ONE complete Cahn–Hilliard step with tools that work on any box,
teaches the Nsight Systems commands, and shows that on the tutorial's
direct-solver path the GPU is barely the bottleneck — the host LU factorization
is.

**Objectives.** Time one step with CUDA synchronization (and see why an
unsynchronized timer lies); measure kernel-launch overhead, device bandwidth,
and PCIe copy bandwidth, and place a kernel on a roofline; read a factor-vs-solve
split and argue for reuse; account for device memory analytically when the
mempool hides it; drive `nsys profile`/`nsys stats` with NVTX ranges.

**Prerequisites.** The Newton step and its assembly/solve stages (C0); solver
landscape + cost accounting (C4, C5); GPU execution model.

**Cost.** One CUDA GPU, 2-D, <1 GB problem memory, solver `splu`. ~30–40 s wall
including one nsys capture.

## What it does (all CUDA-synchronized / driver-measured)

1. **Step timing** — one complete synced warm step (~2.1 s at level 6, 12 Newton
   iters), JIT-compile first step excluded.
2. **Async pitfall + launch overhead** — a trivial kernel: ~40 µs/launch; an
   unsynced single-launch timer under-reports the work.
3. **Bandwidth vs arithmetic intensity** — a streaming triad `c=a+2b` sustains
   ~700 GB/s (AI 0.083 flop/byte, memory-bound); PCIe copies saturate ~26 GB/s
   (~27× slower). D2H volume ~35 MB/step.
4. **Plan/factor reuse** — LU factor (~88 ms) vs triangular solve (~3.6 ms),
   ~24×; the step refactors every Newton iteration → the case for reuse / a GPU
   solver (C4).
5. **Device-memory HWM** — driver footprint (isolated subprocess, mempool-
   quantized ~2.1 MB) + an analytical accounting (~2.56 MB, `Ae` buffer
   dominates).
6. **Nsight Systems** — profiles the one-step workload (`nsys_workload.py`), reads
   back the NVTX timeline (`warmup`, `ch_step`).

**Nsight note.** On this box nsys captures the NVTX ranges but not the
kernel/memcpy rows — the CUDA driver (13.2) is newer than the installed Nsight
(2023.4), so CUPTI kernel tracing is unavailable (`nsys stats` prints "does not
contain CUDA kernel data"). The commands are taught unchanged; on a matched
toolchain `cuda_gpu_kern_sum` prints the per-kernel table. All quantitative
numbers here come from synchronized timers + driver memory queries, so they
reproduce with or without nsys.

## Run it

```bash
export PYTHONPATH=<repo>/src
<repo>/.venv/bin/python run.py --config configs/c7.yaml --device cuda:0 \
    --solver splu --output outputs/c7 --overwrite --mode reference
<repo>/.venv/bin/python gen_figures.py --run-dir outputs/c7
```

Seven core gates (the nsys gate is advisory) check the invariants; `results.json`
is checked against `baseline.yaml`. Compare with `EXPECTED.md`.

## Files

- `profile_ch.py` — the profiling core (step timing, launch overhead, bandwidth,
  factor/solve, memory).
- `nsys_workload.py` — the minimal one-step workload nsys profiles (NVTX ranges).
- `mem_probe.py` — isolated fresh-process device-memory footprint probe.
- `run.py` — harness driver; `gen_figures.py` — figures `c7_{breakdown,bandwidth}.png`
  + `numbers/c7.tex`.
- `configs/c7.yaml`, `baseline.yaml`.
