# W-Engine Hardening Implementation Plan (W1, W2, W4, W5 — W3 deferred)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the Horizon pathway's work items W1, W2, W4, W5 (spec = `docs/dev/2026-07-29-horizon-100m-50h-pathway.md` §5; Baskar 2026-07-29: "Start W1, W2, W4, W5 now"). W3 (multi-GPU NCCL) DEFERRED: nova GH200 nodes are single-GPU; pattern-dev option on gpubox 2×Ada later, full test on Horizon.

**Architecture:** W1/W2 harden `src/diffsim/assembly/device_assembly.py` (bounded intermediates; constrained-scatter chunking); W4 is a hold-session measurement (no code); W5 lands two solver accelerators (fused `apply_dev` hook; node-block Jacobi) with fp32+IR and CUDA-graphs as stretch. GPU validation uses hold job 11777138 while it lives (~12 h), else gpubox/sbatch.

**Tech Stack:** `device_assembly.py` (`ChunkTable`, `_dof_indices_kernel_chunked`, the constraint-expansion scatter), `krylov_dev.py` (`cg_dev`/`bicgstab_dev`), `saddle_precond.py`, the ladder harness + hold-leg conventions from `cluster/a3-probes/run-legs.sh`.

## Global Constraints

- Branch `w-engine-hardening` off master (27accf2); one implementer at a time; W4 (remote measurement) may run concurrently with code tasks.
- Defaults bit-for-bit; all new behavior opt-in or provably-identical (chunk-size bounds must produce IDENTICAL assembled matrices — that is a parity gate, not a knob); gates green per commit: tests/test_device_assembly.py, test_saddle_ladder_cpu.py, test_saddle_precond.py, test_p2r1a_thin_plate_flow.py, test_p2r1c_thin_plate_flow_3d.py.
- Measured validation on the hold (srun --overlap, jobid 11777138, timeout-capped legs, one per process) while it remains; never cancel it; honest rows.
- Commits explicit paths + trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task W1: bound the chunked dof-indices intermediate

**Files:** Modify `src/diffsim/assembly/device_assembly.py` (the dof-indices/slot build that allocated 137,367,584,768 B in one array at ~68M — traceback in campaign doc §10.10); Test `tests/test_device_assembly.py` (extend).
**Interfaces — Produces:** the intermediate built in bounded chunks (target ≤ ~2 GB per chunk, constant chosen with a comment justifying it); assembled CSR BIT-IDENTICAL to the unchunked path (parity test on a small mesh comparing full assembled matrices); no API change.
Steps: failing parity/bound test (monkeypatch or size-threshold hook to force the chunked path on a small mesh) → implement → gates → commit `fix(device_assembly): bound the chunked dof-indices intermediate` + trailer.
**GPU gate (hold):** retry leg-5 (3d-L8 ~68M, bdiag, device, NSTEPS=2, timeout 2400): must pass the former OOM site; record iters/s_per_step/SMI/RSS (the first ≥30M device-assembly row).

### Task W2: constrained-scatter ChunkTable extension

**Files:** Modify `src/diffsim/assembly/device_assembly.py` (lift the `BackendError("chunking supports the identity-constraint...")` at ~line 442: extend block-row chunking over the constraint-expansion scatter — chunk over element bins so per-bin slot arrays stay < 2³¹ AND ≤ the W1 bound); Test `tests/test_device_assembly.py` (extend: constrained small mesh, chunked-vs-unchunked bit-identical CSR).
**GPU gate (hold):** 3d-L7r9 (adaptive 9.08M) with SADDLE_ASSEMBLY=device, bdiag, NSTEPS=5, timeout 2400 — the row that brings the 4.33× device win to adaptive meshes (reference: 247.8 s/step host). Commit `feat(device_assembly): constraint-aware scatter chunking (adaptive device assembly at scale)` + trailer.

### Task W4: developed-march warm-start/restart validation (measurement only — runs first, concurrently)

No code. On the hold: `SADDLE_POINTS=3d-L7 SADDLE_SOLVERS=fgmres_bdiag SADDLE_ASSEMBLY=device SADDLE_RESTART=120 SADDLE_X0=extrap SADDLE_NSTEPS=200`, timeout 14400. Deliverable: settled iters/step curve (first 5 vs last 50 mean), settled s/step, and the honest comparison vs the 5-step probe (676.2/40.76). Record in campaign doc §11 addendum. If the hold dies first: gpubox L6 equivalent or a queued sbatch — record which.

### Task W5a: fused `apply_dev` hook (short recurrence + preconditioner)

**Files:** Modify `src/diffsim/solvers/krylov_dev.py` (`bicgstab_dev` gains optional `apply_dev=None` right-preconditioner — verify the fused path's structure first; preconditioned BiCGStab inserts M⁻¹ at the standard two points), `src/diffsim/solvers/linsolve.py` (opt-in backend `fused_bdiag` or `fused` + meta knob routing make_bdiag_apply), Test `tests/test_saddle_precond.py` (vs splu on the level-4 saddle + parity of unpreconditioned default).
**GPU gate (hold/box):** 3d-L7 uniform device leg with the new backend, 5 steps — compare vs 40.76 (fgmres r120+warm) and 61.5 (raw fused): the question is whether short recurrence + bdiag beats FGMRES once basis/ortho costs are gone. Commit + trailer.

### Task W5d: node-block Jacobi (4×4) upgrade of bdiag

**Files:** Modify `src/diffsim/solvers/saddle_precond.py` (`make_bdiag_apply(..., block="node")` opt-in: extract per-node ndof×ndof diagonal blocks, invert once host-side (or batched device), apply = per-node small matmul kernel), `linsolve.py` meta routing (`bdiag_block="node"`), Test `tests/test_saddle_precond.py` (vs splu; iteration-count assertion node-block ≤ scalar bdiag with numbers printed — a finding either way).
**GPU gate:** 3d-L7 uniform device leg, 5 steps, vs the scalar-bdiag rows at matched restart. Commit + trailer.

### Task W5-stretch (only if hold time + review bandwidth remain): fp32+IR operator probe OR CUDA-graph capture — controller decides which, one only, probe-scale.

### Task W6: docs + verdicts + ledger

Campaign doc addendum §12 (W-round): all measured rows, per-item verdicts, updated 100M projection; runbook pointer; full regression; whole-branch review; merge decision to Baskar.

## Self-Review

Spec (§5 of the pathway) coverage: W1→T-W1, W2→T-W2, W4→T-W4, W5a/b→W5a, W5d→W5d, fp32/graphs→stretch, W3→explicitly deferred with reason. Bit-identical parity gates named for both assembly tasks (the non-negotiable). Hold-mortality fallbacks stated. No placeholders.
