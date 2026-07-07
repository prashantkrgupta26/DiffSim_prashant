# Nova Arrival Kits (A100-80GB / H200-141GB / GH200-96GB)

One 24-48h allocation per card class; each runs the SAME staged campaign,
ordered so that early termination still yields a complete report.

## What works today (2026-07-06 night, commit ~685da9d)

Pull FRESH before running — this kit was updated the night of 07-06:

- **M1 hero bar MET**: H1 sphere-INR steady recovery converged
  (err 1.769e-3; baseline `tests/baselines/m1_hero_h1.json`); H2
  transient recovery converged (2.56e-4, `m2_hero_h2.json`). The
  hero-timing stage now measures a *converging* pipeline.
- **The L5 "dip" is RESOLVED** — it was a benchmark degeneracy
  (r=0.25 dyadic tangent-plane alignment, intrinsically 3-D). The
  canonical radius is now **r=0.27** (driver default). Your L7 run is
  the ASYMPTOTIC CONFIRMATION (expect order -> 2 from the monotone
  L4-L6 ladder 1.95e-3 / 5.68e-4 / 1.91e-4). See
  `cluster/samundra_band_study.md` (updated header).
- **Solvers**: cuDSS is the default heavy-solve path (300x vs host
  splu at 3-D L5); `solver="blocktri"` (FGMRES + exact-F block
  preconditioner, 2-3 iterations at sigma=0) is available for steady
  NS systems.
- **Device assembly**: `DeviceNSAssembler` (4-40x vs host; uniform +
  hanging-constrained meshes; strong rows folded in; zero-copy
  `assemble_device()` into cuDSS). Steppers accept
  `use_device_assembly=True`. Expect band-study L6 cases at ~25 s on
  the workstation baseline — if your per-case times are minutes,
  something regressed.
- Known model quirks (documented, sentinel-guarded): sphere .pt needs
  w0=1.0; window_center=0.03; the [-1,1]^3 -> [0,1]^3 window contract
  is in `ProvidedINROracle`.

## Quick start (on the allocated node)

```bash
git clone <repo> DiffSim && cd DiffSim
bash cluster/bootstrap.sh            # detects x86/ARM; makes .venv-nova
sbatch cluster/slurm/a100.sbatch     # or h200.sbatch / gh200.sbatch
# ...or run directly inside an interactive allocation:
.venv-nova/bin/python cluster/campaign.py --card a100 --hours 24
```

Results land in `cluster/results/<card>-<date>/`:
`report.md` (human summary) + `results.json` (machine-readable) +
per-stage logs.

## Campaign stages (in order; each independently valuable)

1. **sanity** — fast test tier; proves the environment.
2. **fp64-micro** — element-kernel + fused-Krylov FP64 microbenchmarks
   (the RTX 6000 Ada baseline is FP64-gimped ~1.4 TF; A100 ~9.7-19.5,
   H200 ~34 — expect 7-25x here).
3. **solver-table** — cavity L6/L7/L8 stepping timings
   {splu, fused, cudss}; reproduces the m1b findings-8e table on the
   new silicon.
4. **capacity** — empirical cuDSS ceiling search (2-D L9..L12,
   3-D L6..L8, try/except ALLOC) -> the new capacity table.
5. **band-study** — the P2-P1 Neumann SBM sphere at r=0.27, 3-D
   levels 4-7 (+ L8 STRETCH on H200 only). THE science payload: L6->L7
   order ~ 2 confirms asymptotic second order in 3-D (the old dip
   framing is resolved — benchmark degeneracy at r=0.25; see the
   runbook header + problem statement postscript).
6. **hero-timing** — one H1 epoch (steady 3-D INR flow + adjoint):
   measures the host/device split per machine. On GH200 this is the
   interesting number (coherent CPU-GPU vs our hybrid architecture).

## Card notes

- **A100**: plentiful on Nova (124 cards). The bread-and-butter tier.
- **H200**: 141 GB moves the measured 48-GB cuDSS wall (~2.8M dofs) to
  the ~8M class: L7 3-D Poisson direct becomes feasible; L8 is the
  recorded stretch goal (attempt, expect ALLOC, record where).
- **GH200 (ARM)**: bootstrap auto-detects aarch64. pyamgx/AMGX are
  SKIPPED there (source build; not needed for the campaign — cuDSS
  carries it). Stage 6 is the headline on this node.

## Partitions / GRES

Slurm scripts carry PLACEHOLDERS — fill from `sinfo -o "%P %G %N"` on
Nova (partition names vary by allocation). Everything else is portable.
