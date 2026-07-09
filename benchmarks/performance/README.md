# performance — device-migration profiling

Drivers that measure the full-device (GPU-resident) migration of the assemble→
solve→adjoint loop: where time goes, and what the M1d slot-map + zero-copy cuDSS
path buys.

| Script | What it measures | Finding |
|---|---|---|
| `profile_stages.py` | per-stage timing of the migration | the 300× assembly finding (rolled loops 1.8 s vs 75 min) |
| `m1d_closure_trace.py` | host-compute residency in the device loop | D3 trace: 0.81 % / 0.04 % / 0.03 % host |
| `m1d_h1_epoch.py` | the H1 hero loop ported onto the M1d stack | epoch **44.3×** end-to-end |

## Run

```bash
python benchmarks/performance/profile_stages.py
python benchmarks/performance/m1d_h1_epoch.py
```

## Context

The device path is: slot-map device CSR scatter (4–40×), constraint-aware
expansion through the T-weights, zero-copy dlpack → cuDSS (parity 1e-15), and
cuDSS `mtlayer` multithreaded refactorization (40×+ on the epoch loop). The
honest note: the naive port was only 6.6× (below the bar) until the hero loop
itself was re-engineered onto the stack — see `docs/dev/m1d-milestone-report.md`.
