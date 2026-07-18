# Horizon readiness — DiffSim at TACC (NAIRR Deep Partnership)

**Status (2026-07-18):** NAIRR Pilot proposal reviewed favorably and recommended
for allocation; original target (Frontera) decommissioned; NAIRR has offered a
**TACC Horizon Deep Partnership** (available fall 2026), pending our
confirmation and TACC approval. This memo is the technical-readiness plan and
the sizing basis for the partnership discussions.

Machine facts from `docs.tacc.utexas.edu/hpc/horizon` (April 2026, docs in
progress — re-verify at onboarding):

- **GB (Grace-Blackwell) nodes: 2,000**, each **2× NVIDIA Blackwell GPUs
  (185 GiB HBM3 each)** + **1 Grace CPU (72 ARM cores, ~120 GiB LPDDR5X)**.
  ~80 TFlops FP64 per node. 4,000 GPUs total. (NVL4 platform branding; TACC
  schedules 2-GPU nodes.)
- **VV (Vera) CPU nodes: 4,752** (2×88 ARM cores, 500 GB).
- Quantum-X800 InfiniBand XDR 800 Gb/s, fat tree. ~400 PB storage at
  ~8 TB/s; **scratch purge: 10 days untouched**.
- Queues: `gb` ≤128 nodes / `gb-large` ≤512 nodes, **48 h max**, 1 SU;
  VV at 0.25 SU.
- **Everything is aarch64** (Grace hosts).

## 1. What one GB node buys OrgElMorph (measured-model projection)

Anchors (measured, `docs/dev/2026-07-09-blockch-preconditioner.md` G5 +
negi3d kit): ternary film `blockch_dev` — 230×230×70 = 15.15 M dofs,
1.61 B nnz (~106 nnz/dof), 33.4 GB GPU, 37.7 GB host, 236 s/step on RTX 6000
Ada (~0.96 TB/s). Memory model ≈ 2.2 GB/M dofs (int32 indices).

| Configuration | Max size | Film grid | Limiter |
|---|---|---|---|
| Today's code, one B200 | **~20 M dofs** (~44 GiB of 185 used) | ~265×265×75 | **int32 nnz ceiling** (negi3d already at 79%) |
| One B200, index-widened | **~60–70 M dofs** | **~430×430×85** (≈4× hero in-plane area) | HBM at ~2.6 GB/M dofs (wider indices) |
| One node, 2 independent solves | 2 × 60 M dofs | two campaign cases/node | none — day-1 mode |
| One node, multi-GPU solve | **~120–140 M dofs** | ~620×620×85 | needs partition/halo/NCCL (unbuilt) |

**Speed:** steps are bandwidth-bound; HBM3e ≈ 8× Ada ⇒ ~30 s/step at hero
size, ~2 min/step at 60 M dofs. Full dryness march (~200 accepted steps):
**~1.5–2 h per hero case per GPU; ~6–8 h per 60 M-dof case.**

**Campaign math (the data-collection pitch):** `gb` queue = 128 nodes × 2
GPUs × 48 h ⇒ **hundreds of concurrent forward solves; thousands of
hero-scale morphology trajectories per window** — training-set scale for the
learned-thermodynamics program (M4/beyond-FH, SP-1 R3).

Host-side note: Grace has ~120 GiB/node; our measured host peak
(2.5 GB/M dofs at assembly era) would bind near ~48 M dofs — `blockch_dev` is
device-resident so the true number is better; **measure-first item** on ARM.

## 2. Readiness ladder

**P0 — blockers (start immediately):**
1. **aarch64/Grace lane.** All Horizon hosts are ARM. On-ramp exists TODAY:
   Nova's `nova-arm` partition has **GH200** (Grace-Hopper — nearest proxy to
   Grace-Blackwell); `cluster/bootstrap.sh` already branches on aarch64; the
   single GH200 attempt in history (job 11447898) **failed at 00:00:01** —
   diagnose and make the GH200 kit green (warp/torch/cuDSS ARM wheels,
   end-to-end film case). TACC **Vista** (Grace-Hopper) is the TACC-side
   bridge; a startup allocation there is a natural Deep-Partnership ask.
2. **Index widening (kill the int32 nnz ceiling).** 2.1 B-slot CSR caps us at
   ~20 M dofs regardless of 185 GiB HBM. Index-width genericity in
   assembly/solver paths ⇒ 3× larger simulations immediately.

**P1 — campaign throughput (fall-ready):**
3. **Day-1 mode = independent solves per GPU** (2/node): zero new solver code;
   the film factory already drives config sweeps (14 cases / 11 min / A100).
4. **Campaign hardening:** Horizon SLURM kits (48 h wall ⇒ per-case
   checkpoint/restart at step granularity); **on-device reductions as primary
   output** (S(q,t) via `structure_factor_torch`, descriptors, h-milestone
   snapshots — not raw fields) to live with the 10-day purge and I/O scale;
   failure-tolerant array jobs; artifact-pull discipline
   (`fetch-artifacts.sh` pattern).

**P2 — capability growth:**
5. **Multi-GPU single-solve** (the original-spec M5-NVLink milestone:
   partition, halo overlap, NCCL, ≥80% node-scale efficiency). Film campaigns
   don't need it day-1; **maize-canopy CFD does**.
6. **Mixed-precision activation** — finding 8j (`m1a-deferred-findings.md`,
   literally labeled "Horizon prep"): fp32-factor + FP64-IR, 2–8 refinements
   measured; a net loss on Ada (structure-bound), profitable where
   FLOP-bound — Blackwell at scale is that regime. Spec D0 hook exists.
7. **Thin-shell CFD (maize) scoping memo** — immersed **sub-h thin features**
   = the recorded M7-tier verification gap (thin plate < h ⇒ diagnostic
   path; SBM5/6 gap tests) + plant STL ingestion at scale + NS multi-GPU.
   New application track on SBM strengths; scope separately.

**P3 — logistics:** NAIRR reply (below), TACC accounts for the group,
allocation-size conversion from the Frontera request, Deep-Partnership terms
(early access, liaison, acknowledgment expectations), Vista bridge request.

## 3. Timeline to fall 2026

| When | What |
|---|---|
| **Now (July)** | Reply YES to NAIRR; ask Deep-Partnership questions (§4). Start P0-1: GH200 kit green on Nova. |
| **August** | P0-2 index widening; P1 campaign kits + checkpoint/restart; Vista bridge runs (ARM at TACC). |
| **September** | Scale rehearsal on Vista/GH200: one full film campaign end-to-end under 48 h-wall discipline; measure ARM host peaks; lock Horizon kit headers. |
| **Oct– (early access)** | Single-node burn-in on GB nodes (memory/step-time model vs projection), then first `gb`-queue campaign. P2 items as capacity work continues. |

## 4. Questions for the NAIRR/TACC discussion

1. Deep Partnership scope: early-access window? named TACC liaison? expected
   deliverables (feedback, reports, acknowledgment text)?
2. SU sizing: conversion of the recommended Frontera allocation to Horizon
   SUs (`gb` node-hours); burst vs steady usage; `gb-large` eligibility for
   campaign windows.
3. **Vista startup/bridge allocation** for ARM porting ahead of Horizon
   availability.
4. Storage: project/work quotas (docs say TBD), purge-exempt space for
   campaign reductions, data-egress path (~TB-scale reduced outputs).
5. Software: CUDA/driver baselines at launch, Apptainer availability, Python
   wheel story on aarch64 (warp/torch/nvmath-cuDSS), queue policies for
   array-style campaign jobs.
6. Team accounts: PI + students (current collaborators + incoming), MFA,
   onboarding timeline.

## 5. Cross-references

- Sizing anchors: `docs/dev/2026-07-09-blockch-preconditioner.md` (G5),
  `cluster/wodo_campaign/a100_negi3d.sbatch` (negi3d memory forecast).
- Mixed precision: `docs/dev/m1a-deferred-findings.md` §8j ("Horizon prep").
- Multi-GPU milestone definition: `docs/dev/specs/2026-07-02-diffsim-design.md`
  §10 (M5-NVLink row).
- Thin-feature gap: `docs/dev/m1-verification-coverage.md` (M7 tier).
- The roadmap-reconciliation task (pending) should absorb P0/P2 items as
  numbered milestones with Horizon as their deadline.
