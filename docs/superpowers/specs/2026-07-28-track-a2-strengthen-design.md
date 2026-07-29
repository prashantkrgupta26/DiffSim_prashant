> **TRACK A2 VERDICT (2026-07-29, GH200 hold session job 11777138):** The WP4
> success gate as written (pcd-amgx ≤~2× growth AND ≤~60 s/step at 9.08M)
> **FAILS** — all three PCD inner variants are wall-clock intractable at 9.08M
> (jacobi >29 min/step, amgx-Ap >48 min/step despite 0.22 s Ap applies,
> amgx-F catastrophic; F-inner large-block matvec economics at outer-apply
> frequency are the wall). **The campaign GOAL is nonetheless achieved via
> `fgmres_bdiag` + device assembly: 54.8 s/step at 8.58M (uniform, controlled
> same-mesh A/B: 4.33× s/step and 12.8× RSS over host assembly at bit-identical
> 1300.2 iterations)** — under the ≤60 s bar. Adaptive 9.08M stands at 247.8
> s/step (host asm) pending the constrained-scatter ChunkTable extension; the
> chunked dof-indices intermediate needs a size bound for uniform ≥~30M
> (measured 137.4 GB single alloc at 68M; persistent 62.7 GiB fits HBM).
> Warp is device-strict on GH200 (no Grace spill). Warm setup ~2 min. AMGX
> aarch64+pyamgx delivered on nova. WP1 telemetry + WP3 wiring shipped; WP2
> refuted-with-mechanism. Full record: `docs/dev/2026-07-28-track-a2-campaign.md`.

# Track A2 — Strengthen the Monolithic Engine (A4 + telemetry + device assembly)

**Date:** 2026-07-28. **Approved:** Baskar ("Looks right - write the spec and plan").
**Context:** Track A closed VIABLE via `fgmres_bdiag` (kill-gate 1.985× across
1.097M→9.08M; GH200 job 11772638; 242 s/step at 9.08M) but production-marginal:
8000 steps ≈ 22 days. PCD holds near-flat outer counts (35 at 1.1M) but is
wall-time-infeasible at 9M with Jacobi-CG inners (≥86 min/step, DNF twice) and
diverges on the 2-D fine-graded mesh. Host-assembly transients MEASURED
243–245 GB at 9.08M ⇒ device assembly mandatory ≥10M. Full record:
`docs/dev/2026-07-27-r2b-saddle-campaign.md` §8. Track B is PAUSED
(2-D projection structurally divergent — B2 verdict); it resumes 3-D-focused
after this campaign.

**Goal:** production-rate monolithic engine at ~10M DOF — PCD-class outer
counts at ~10× lower step cost than bdiag, with the host-memory wall removed.

---

## WP1 — Inner-solve telemetry (prerequisite, binding)

Per-apply counters for the three PCD inner solves (F, Ap, Mp) in
`src/diffsim/solvers/saddle_precond.py::make_pcd_apply`: iterations used,
cap-hit flag, exit residual. Aggregated per outer solve and surfaced through
the existing `LinearSolveResult`/`solver_stats` path into the ladder npz.
Default-inert (no overhead when not requested is NOT required — counters are
O(1) per apply — but output appears only when asked). This is the recorded
prerequisite from the Track A final review: without it, A4 tuning re-debugs
the 2d-r11 divergence and the 9M wall-cost blind.

## WP2 — A4: AMGX V-cycle velocity inner for PCD

Replace the F-block inner Jacobi-CG in `make_pcd_apply` with AMGX on the
EXTRACTED velocity sub-CSR (ndof-strided rows/cols — valid AMG territory; the
raw saddle remains forbidden, documented). Reuse the existing wrapper
`src/diffsim/solvers/amgx.py::amgx_solve` (+ `amgx_configs/`, e.g.
PCG_CLASSICAL_V_JACOBI) and the `blockamgx` meta-via-cache plumbing pattern in
`linsolve.py`. New opt-in backend name: `fgmres_pcd_amgx` (or a
`pcd_inner="amgx"` knob on the existing branch — plan decides; either way
opt-in, defaults untouched). Ap/Mp inners stay Jacobi-CG (scalar SPD, cheap).
The F matrix changes per step (Picard convection): AMGX setup is re-done per
solve — measure setup vs solve split via WP1 telemetry; if setup dominates,
record honestly (a lagged-setup variant — refresh AMGX hierarchy every k steps
— is the fallback probe, flag-gated).

Constraints carried forward: AMGX int32 nnz ceiling ~79.5M (the 9.08M velocity
block nnz is far below; the 100M block is NOT — record the projection with
this stated ceiling).

## WP3 — Device assembly in the ladder path

Wire the drivers' existing `assembly="device"` through
`tests/gpu_saddle_ladder.py::run_ladder_point` (env `SADDLE_ASSEMBLY`,
default = today's host path). Gate: parity vs host assembly at CPU scale
(existing device-assembly tests already prove kernel parity — the wiring gate
is that the ladder passes the knob through faithfully), then the 9.08M GH200
leg builds WITHOUT the 243 GB host transient (target: setup RSS well under
100 GB; capture via the sbatch RSS/SMI sampling).

## WP4 — GPU validation ladder + verdict

Matrix: {bdiag, pcd-amgx} × {3d-L6, 3d-L7r9} with device assembly, 5 steps
(3d-L7r9 pcd-amgx may use 3 steps if wall demands — record honestly), plus a
2d-r11 pcd-amgx retest (does the AMGX inner fix the graded-mesh divergence?).
Record per leg: outer iters/step, inner iters + cap-hits (WP1), s/step split
(assembly / AMGX setup / solve), memory.

**Success gate (binding, time-boxed to this one plan):** at 9.08M,
pcd-amgx outer growth from L6 ≤ ~2× AND s/step ≤ ~60 s (≈4× better than
bdiag's 242 s). Partial outcomes recorded first-class: e.g. "AMGX setup
dominates ⇒ lagged-setup probe result ⇒ verdict". If pcd-amgx fails the gate,
bdiag remains the engine and this campaign records why + what it would take.

## Out of scope

Multi-GPU/NCCL beyond existing, Track B items (paused), adaptive-dt,
100M hero runs, streaming mesh build. Recording: campaign report
`docs/dev/2026-07-28-track-a2-campaign.md` + runbook section + ledger;
negative results first-class; every claim from a measured run.
