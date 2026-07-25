# P2-R1a — 2-D Flow Past a Thin Plate (Re=126/250), Cd/St, Both Solvers

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
> Spec: `docs/superpowers/specs/2026-07-25-p2r1-thinshell-cfd-spec.md`.
> Repo: `/Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim` (master `b5e132e`).

**Goal:** reproduce ThinShell.pdf §4.3 — 2-D flow past an infinitesimally thin
plate — and recover the mesh-converged **Cd≈3.29–3.45, St≈0.15** at Re=250
(Table 1; Najjar & Balachandar 3.36/0.14), with BOTH the monolithic VMS-saddle
solver and the projection/PPE solver.

**Architecture:** reuse the two-sided shell surrogate + `surrogate_traction`
(drag) from `tests/p2r1_thin_plate_blocked_channel.py`; the NEW pieces are a
FINITE plate geometry (`Segment` in 2-D) in a flow-past domain and a
TIME-ACCURATE transient march (BDF2) long enough to capture periodic vortex
shedding, recording Cd(t)/Cl(t); then Cd (time-average) + St (FFT of Cl) per the
`test_cylinder_strouhal` pattern.

## Global Constraints
- Method exactly as the paper/code: two-sided surrogate, Nitsche (α=β=20), VMS
  (C_M=36), **BDF2**, Re=1/ν. NS-only (no heat).
- BOTH solvers per case: monolithic (`api/ns_bricks.assemble_linear_ns` + a
  saddle solver) and projection (`steppers/leray.py`, `solver` in
  {"splu","cudss","gpu_cg"}). Report Cd/St from each + their agreement.
- Cd nondim by ½ρU²·L (L = plate length); St = f·L/U (f = shedding freq).
- In-CI smoke at tiny level (seconds); quantitative Cd/St from a full-resolution
  run on gpubox (document the level/dt).
- Validation tolerance: Cd within ~5% of the Table-1 band, St within ~±0.02.

## Case parameters (ThinShell.pdf §4.3)
- **Re=250 (headline):** domain [0,36]×[0,16], u∞=(1,0) on all boundaries except
  outlet (p=0); plate centered at (5,8), length L (the plate half-height/length
  per the paper — read the exact L from the figure/config; the reference Najjar
  uses domain [0,35]×[0,16], plate at (5,8)). Base L7, wake refined L9, plate
  L9–11 (mesh-convergence). Δt≈5e-5 near-plate; march to periodic shedding.
- **Re=126 (bridge/qualitative):** domain [0,8]×[0,4], plate at (3,2), local L10,
  Δt=5e-5; compare velocity/pressure jump + Cp + wake length to a carved-out ref
  (Figures 10–12). Secondary to the Re=250 Cd/St gate.

---

### Task 1: Finite thin-plate flow-past case + transient march + Cd/Cl history (monolithic)
**Files:** Create `tests/p2r1a_thin_plate_flow.py` (driver, mirrors
`p2r1_thin_plate_blocked_channel.py`'s structure); Create
`tests/test_p2r1a_thin_plate_flow.py` (in-CI smoke gate).
- Geometry: a finite vertical plate via `Segment` (2-D) at the plate center,
  length L, spanning [yc−L/2, yc+L/2]; two-sided shell surrogate
  (`classify_shell_intercepted` + `extract_two_sided_surrogate`).
- BCs: u∞=(1,0) Dirichlet on inflow + top/bottom (weak or strong per the existing
  harness), outlet p=0 (or do-nothing); no-slip on the plate via
  `sbm_vector_dirichlet_twosided`.
- Transient BDF2 march (time-accurate, NOT the pseudo-steady `_march`): record
  Cd(t) = drag from `surrogate_traction` (F+ + F−, nondim), Cl(t) = lift, per step.
- Smoke gate (tiny level, few steps): runs to completion, produces finite Cd/Cl
  arrays, force is nonzero, two-sided coupling load-bearing (drop one side →
  force collapses, as the blocked-channel gate checks). Fast (seconds).
- Run the smoke test on the Mac CPU venv → green. Commit.

### Task 2: Cd (time-avg) + St (FFT) extraction + Re=250 validation (monolithic)
**Files:** Create `src/diffsim/postproc/shedding.py` (or a test helper):
`time_avg_cd(t, cd, t_start)` and `strouhal(t, cl, U, L)` (FFT peak → f → St),
mirroring `tests/test_cylinder_strouhal.py`. Modify the driver to emit Cd/St.
- Validate at a resolution reaching the periodic regime: **Cd≈3.29–3.45,
  St≈0.15** (Table 1). This is a full-resolution run → gpubox (via the remote
  toolkit); document level/dt/wall. Add a documented expected-value assertion.
- Commit with the recorded Cd/St numbers in the report.

### Task 3: Projection-solver variant of the same case
**Files:** Modify the driver to run the SAME case through `steppers/leray.py`
(`solver="gpu_cg"` on GPU / `"splu"` on CPU), producing Cd(t)/Cl(t).
- Validate: projection Cd/St agree with the monolithic Cd/St (and Table 1) within
  tolerance — the both-solver comparison deliverable.
- If the projection path needs the two-sided shell wired into its predictor/PPE
  (the blocked-channel used monolithic assembly), do that minimal wiring here.
- Commit; report both-solver Cd/St table.

### Task 4: In-CI gate + gpubox full-resolution runbook
**Files:** Finalize `tests/test_p2r1a_thin_plate_flow.py` (tiny-level smoke, both
solvers, always-on); Create `docs/dev/p2r1a-thin-plate-runbook.md` (the
full-resolution gpubox commands + the recorded Cd/St vs Table 1, both solvers).
- Ledger entry. Commit.

---

## Self-review checklist
- Finite plate (not full-height); two-sided surrogate load-bearing.
- Time-accurate BDF2 march reaches periodic shedding before averaging.
- Cd/St match Table 1 within tol on BOTH solvers; both-solver agreement reported.
- Smoke gate fast + always-on; quantitative numbers from a documented gpubox run.
