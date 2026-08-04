# Track B: Projection Rescue — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the projection split stable at the corrected Re=250/L9 config and fixed-point-faithful for thin-shell (3-D Cd_reaction gap ≤ 10% vs monolithic), then demonstrate a ≥10M-DOF projection solve with no direct sub-solves — the Track B engine gate.

**Architecture:** Per the approved spec (Track B): (B1) audit + port the 3-D p′-outflow fix-set into the 2-D projection path — the O6-identified suspect for the structural divergence; (B2, conditional) knob-envelope probes with early-exit divergence detection; (B3) coupled/Uzawa predictor sub-iteration for the lagged-p* fixed-point gap, measured by the reaction arbiter against same-mesh monolithic; (B4) the engine-gate demonstrations.

**Tech Stack:** `LeraySBMShellStepper`/`LerayProjectionStepper` (the consistent-projection fix-set flags), the 2-D/3-D projection drivers, the consistent-reaction arbiter (transfers verbatim), `gpu_cg` PPE + iterative momentum, the gpubox toolkit + nova for the big SPD leg.

## Global Constraints

- Branch `track-b-projection` off current master (rebase onto Track A's merge if it lands first — the tracks interleave; controller sequences). Repo cd prefix; `.venv/bin/python`; toolkit-only box runs.
- The p2r0 one-sided path and ALL existing projection gates stay green at every task (tests/test_p2r1a_thin_plate_flow_projection.py, test_p2r1c_thin_plate_flow_3d_projection.py, test_p2r0_projection_sbm.py).
- Every stability claim from a measured run; probes carry early-exit divergence detection (`max|Cd| > 1e3` checked every 10 steps — the O6 lesson; a diverged probe costs minutes, not hours).
- Physics knobs stay behind flags; defaults bit-for-bit (parity tests per the house pattern).
- Commits: explicit paths; trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task B1: p′-scheme completeness audit + port (the divergence suspect)

**Files:**
- Read FIRST (the audit): `src/diffsim/steppers/leray.py` — enumerate the consistent-projection fix-set as ACTUALLY IMPLEMENTED (collocated PPE div, PSPG-consistent PPE, rotational-incremental update, `rotational_pin_outflow`, `rotational_pin_wall`, strong-wall pin, backflow β, `pressure_outflow_nodes` whole-line pin) and WHICH of them the 2-D driver's `run_flow_past_projection` actually activates vs what the 3-D `run_flow_past_3d_projection`/`LeraySBMShellStepper` path activates (the 3-D got the p′-outflow fix campaign; O6's suspicion: the 2-D path lacks pieces). Produce the diff table FIRST (in the report) before changing anything.
- Modify: `tests/p2r1a_thin_plate_flow.py` (the 2-D projection runner's stepper construction — activate the missing pieces; if a fix exists only in stepper code paths the 2-D shell stepper doesn't reach, the port lands in `src/diffsim/steppers/leray_sbm.py` behind existing flags — smallest faithful change)
- Test: `tests/test_p2r1a_thin_plate_flow_projection.py` (extend)

**Interfaces:**
- Produces: the 2-D projection path running the FULL fix-set (documented in the report's audit table: before/after per fix); no new public knobs unless a fix genuinely needs exposure (then default preserving today's behavior).

- [ ] **Step 1 (audit):** the table — fix-by-fix, 2-D vs 3-D activation, with file:line. If the audit finds NO missing pieces (2-D already complete), STOP: report NEEDS_CONTEXT with the table — the divergence cause lies elsewhere and B2 becomes the lead (controller decision point).
- [ ] **Step 2 (failing gate):** `test_projection_re250_bounded` — the EXACT O6 P0 config (corrected units: nu=2.5e-4, x_c=5/16, level=7/r9/wake9, dt=5e-4, cudss predictor, gpu_cg→splu on CPU... CPU can't march 1000 steps at r9 in test time; the GATE RUNS ON GPU in Step 4; the CPU test asserts a 50-step tiny-config (level=5/r7, nu=2.5e-4-scaled geometry) bounded march with the ported fixes — RED if it diverges pre-port on that config; VERIFY first whether the tiny config reproduces the divergence (run it before/after — if it doesn't diverge pre-port, the CPU gate is parity-only and the GPU probe is the real gate; say so honestly).
- [ ] **Step 3 (port):** smallest faithful activation of the missing pieces.
- [ ] **Step 4 (GPU gate):** box probe — O6 P0 config, 1000 steps, early-exit divergence check: BOUNDED = B1 gate passed. Record either way (a bounded-but-wrong-physics outcome is B3's business; divergence persisting = B2 leads).
- [ ] **Step 5:** full projection gates green; commit `fix(projection): port the p'-outflow fix-set to the 2-D thin-shell path` + trailer.

---

### Task B2 (conditional — only if B1's GPU gate fails): stability envelope

Knob triage with early-exit probes (200-step legs): `inner_relax` {0.3, 0.5}, `inner_max` {8, 25}, `dt` {5e-4, 2.5e-4}, `consistent_projection` {on, off — diagnostic}, on the O6 P0 config. Deliverable: the bounded/diverged table + the narrowest stabilizing change promoted into the driver defaults for this config (flag-gated). Commit + record. If NOTHING bounds it: Track B reports the stability blocker honestly (controller/Baskar decision — the spec's engine gate cannot be met without this).

---

### Task B3: fixed-point gap — coupled/Uzawa predictor sub-iteration

**Files:**
- Read FIRST: `src/diffsim/steppers/leray.py` `inner_iterate`/`inner_max`/`inner_relax`/`inner_accel` (the EXISTING within-step predictor↔PPE fixed-point machinery — the R2a-era code) — determine what it iterates (lagged-p* refresh? full re-predict?) and why it plateaued at the documented ~40% 3-D gap (p2r1c runbook: inner_max 8→25 + Anderson converged to the SAME 27.6 — the fixed point itself is offset, not under-iterated).
- The candidate (documented in the ladder verdicts): a COUPLED update inside the inner loop — re-solve the predictor with the UPDATED pressure (p* ← p̂ each inner pass, not lagged) so the inner fixed point approaches the monolithic's coupled solution; implement as `inner_coupled=True` (default False) in `LerayProjectionStepper`, reusing the existing inner-loop skeleton.
- Test: parity (default False bit-for-bit); 3-D smoke gate: `run_flow_past_3d_projection(..., inner_coupled=True)` on the L3/L4 smoke — Cd_reaction (port the reaction arbiter call into the 3-D projection driver OR compare Cd directly vs the monolithic smoke at matched config) within **10%** of same-mesh monolithic (the spec gate), vs the documented ~40% without.
- Honest failure mode: if coupled iteration does NOT close the gap ≤10% at smoke scale, record the measured gap-vs-inner-work curve and STOP for controller/Baskar review (the spec's declaration gate depends on this number).
- Commit `feat(leray): coupled predictor inner iteration (inner_coupled)` + trailer.

---

### Task B4: engine-gate demonstrations

1. **Re=250 2-D shedding on the rescued path:** the corrected-units config, 8000 steps, GPU — Cd_reaction + St vs the monolithic campaign values (band: within the monolithic's own r9→r11 spread); bounded + shedding sustained.
2. **≥10M-DOF projection solve, no direct sub-solves:** the WP0-verified base-L7/band-r9 3-D mesh (9.24M DOF): predictor iterative (fused/bdiag-FGMRES — whatever Track A's baseline provides by then; VERIFY convergence, record iterations) + PPE `gpu_cg`; 3-5 steps, finite fields, memory + s/step recorded. Platform: gpubox if it fits, else nova GH200 sbatch.
3. Recording: `docs/dev/2026-07-27-track-b-campaign.md` + runbook sections; the ENGINE GATE assessment verbatim against the spec's three conditions; ledger; full regression; final review; merge decision to Baskar.

## Self-Review

Spec WP1→B1 (audit-first with the no-missing-pieces escape), WP2→B2 (conditional), WP3→B3 (gate ≤10% via the arbiter, honest-failure stop), engine gate→B4 (all three conditions). Early-exit probes mandated (O6 lesson). p2r0 protection named at every task. No default-path drift; parity tests throughout.
