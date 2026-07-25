# ThinShell GPU Device-Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the 2-D and 3-D thin-plate (ThinShell.pdf) drivers so BOTH solvers — monolithic VMS-saddle and PPE/projection — assemble and solve on the GPU, validate the 2-D Re=250 case on gpubox (Cd≈3.29–3.45, St≈0.15), and stand up the 3-D drivers on all three GPU targets (gpubox RTX 6000 Ada, nova A100, nova GH200).

**Architecture:** The device stack already exists and is parity-gated — this plan is WIRING, not new numerics. (1) Route the monolithic drivers' hardcoded `splu` through `solve_linear(solver=, device=)` (the `ladder_rung3d_cube.py` pattern) → cuDSS/fused on GPU for free. (2) Swap host `assemble_linear_ns` for `DeviceNSAssembler` per step, feeding the cached SBM face system through the existing `extra_matrix=`/`extra_rhs=` hooks and strong BCs through `strong_b_vals=`. (3) The projection stepper is ALREADY device-routed (`solve_linear(..., device=self.dm.device)` + `device_assembly` flag) — build the DeviceMesh on `cuda:0`, set predictor `solver="fused"` (device BiCGStab) and `ppe_solver="gpu_cg"` (device CG), and forward the `device_assembly` flag through `LeraySBMShellStepper`. (4) Launch the Re=250 both-solver validation on gpubox; scale-out 3-D smokes to gpubox, then nova A100 and GH200 via SLURM.

**Tech Stack:** Python, Warp (device kernels), torch (gpu_cg/fused device Krylov), nvmath cuDSS, scipy.sparse (host reference), the existing `DeviceNSAssembler`/`solve_linear`/`LerayProjectionStepper`/`LeraySBMShellStepper` framework, `scripts/remote/` toolkit (gpubox), SLURM sbatch (nova).

## Global Constraints

- Branch `thinshell-gpu` off `master` at `17d7842`; repo `/Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim`. Every bash command starts `cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && ` (REPO-IDENTITY GUARD — a stale clone exists at `~/DiffSim`).
- Mac = brain/master; Mac NEVER runs heavy solves. GPU work goes through `scripts/remote/gpubox-*.sh` (tmux + `.remote-run.lock`) or nova sbatch. Code flows Mac→worker; results/commits flow back (see `docs/dev/remote-workflow.md`).
- **gpubox is WSL2**: every GPU command MUST be prefixed `LD_LIBRARY_PATH=/usr/lib/wsl/lib` or Warp/torch sees no CUDA device (`nvidia-smi` works regardless — do not be fooled). Remote python is ALWAYS `.venv/bin/python`, never bare `python`.
- **Solver policy** (settled): cuDSS = direct GPU solve, valid to its measured **~812k-DOF wall on 48 GB** — use it where it fits and as the parity oracle. The scalable path is ITERATIVE: `gpu_cg` (SPD PPE), `fused` (device BiCGStab, nonsymmetric predictor/saddle), FGMRES/blockch beyond. AMGX never as the outer on the saddle. The full monolithic-saddle block preconditioner at L6+ 3-D is the R2b research track — this plan gates an honest `fused` attempt and documents the boundary, it does NOT promise L8 monolithic.
- Do NOT break existing CPU gates: `tests/test_p2r1a_thin_plate_flow.py`, `tests/test_p2r1a_thin_plate_flow_projection.py`, `tests/test_p2r1c_thin_plate_flow_3d*.py`, `tests/test_p2r0_projection_sbm.py`, `tests/test_device_assembly.py`. Default env-knob values must reproduce today's behavior bit-for-bit on CPU.
- GPU parity gates use DISTRIBUTION locks (measured tail × 5), never bit-equality (GPU atomicAdd is few-ULP nondeterministic; cuDSS factorization is not run-to-run reproducible).
- fp64 everywhere. Never edit a `.py` while a run is in flight (Warp lazy compile). Never pipe a producing run through head/tail (SIGPIPE); poll log FILES.
- Re=250 case facts: adaptive mesh `BASE_LEVEL=7 REFINE_LEVEL=9 WAKE_LEVEL=9` (~34k nodes ≈ 100k dofs — far under the cuDSS wall); `PERT_EPS=0.03 PERT_T_END=1.0` symmetry kick REQUIRED (symmetric flow never sheds); St uses PHYSICAL plate length L=1.0 (= `plate_L`·16), not the octree 0.0625; averaging from `T_START=20.0`.
- Commits end with: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## Scope

**In:** device solve + device assembly for both solvers, both dims; Re=250 2-D both-solver GPU validation (acceptance gate); 3-D both-solver GPU smokes on gpubox; nova A100 + GH200 bring-up smokes; runbook.
**Out (rides on this wiring later):** remaining ThinShell case geometries (two-plate tandem, cylinder+attachments θ-sweep — needs the `Segment.distance_vector` fix, R1b), the R2b monolithic-saddle block preconditioner at L6+ 3-D, multi-GPU NCCL runs, adaptive-dt for NS (variable-dt NS is a documented deferred research item — do NOT bolt it on here).

## Key existing interfaces (verified 2026-07-25, cite before changing)

```python
# src/diffsim/solvers/linsolve.py:1043
solve_linear(A, b, solver="splu", sym=False, tol=1e-10, maxiter=40000,
             device=None, cache=None, cache_key=None, return_result=False)
# solver="fused": CSROperator + (cg_dev if sym else bicgstab_dev), Jacobi diag,
#                 raises ConvergenceError on failure  (linsolve.py:1100)
# solver="gpu_cg": SPD-only device CG                  (linsolve.py:1118)
# solver="cudss":  device direct                       (linsolve.py:1416)

# src/diffsim/assembly/device_assembly.py:786
DeviceNSAssembler.assemble(aq_by_bin, div_aq_by_bin, fq_by_bin, nu, sigma,
    sig2tau=None, s_skew=0.5, strong_b_vals=None,
    extra_matrix=None,   # (slots_d, vals_d) — cached SBM face system, added
    extra_rhs=None)      # (dofs_d, vals_d)    after volume scatter, BEFORE strong rows
# .assemble_fill(...)   in-place fill (device buffers)      :894
# .assemble_device(...) device-resident torch CSR + rhs     :904
# .set_strong_rows(rows) once per epoch                     :758
# .csr_slots(rows, cols) host slot lookup                   :1110
# .add_matrix_values/.add_rhs_values (device atomic adds)   :1127/:1135

# src/diffsim/steppers/leray.py — ALREADY device-routed:
#   :834  predictor: solve_linear(..., solver=self.solver, sym=False, device=self.dm.device)
#   :1102/:1118/:1137 PPE+mass: solver=self._ppe_solver, sym=True, device=self.dm.device
#   :303  self.device_assembly = bool(device_assembly)
# tests/ladder_rung3d_cube.py:295-301 — the routed-monolithic pattern to copy.
# tests/test_device_assembly.py::test_device_assembly_consistency — the exact
#   aq/dq/fq → DeviceNSAssembler.assemble(...) adapter to copy (oracle for Task 4).
```

---

### Task 1: Route the 2-D monolithic solve through `solve_linear` (+ DEVICE knob)

**Files:**
- Read: `tests/p2r1a_thin_plate_flow.py` (march loop ~485-525; `__main__` env parsing ~1023-1070; `compare_solvers`; where `DeviceMesh.from_mesh(..., "cpu")` is called)
- Read: `tests/ladder_rung3d_cube.py:290-310` (the routed pattern)
- Modify: `tests/p2r1a_thin_plate_flow.py`
- Test: `tests/test_p2r1a_thin_plate_flow.py` (add one test; existing tests must stay green)

**Interfaces:**
- Consumes: `solve_linear(A, b, solver=, sym=False, device=)` from `diffsim.solvers.linsolve`.
- Produces: `run_flow_past(..., mono_solver="splu", device="cpu")` — new kwargs, defaults preserve today's behavior; env knobs `MONO_SOLVER`, `DEVICE` in `__main__` and forwarded by `compare_solvers`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_p2r1a_thin_plate_flow.py`:

```python
def test_mono_solver_routing_parity():
    """Routing the monolithic solve through solve_linear(solver="splu")
    must reproduce the legacy inline-splu march exactly (same host LU)."""
    from p2r1a_thin_plate_flow import run_flow_past
    kw = dict(level=4, nsteps=3, dt=0.01, nu=0.1, U_inf=1.0, verbose=False)
    res_legacy = run_flow_past(**kw)                       # default path
    res_routed = run_flow_past(mono_solver="splu", device="cpu", **kw)
    assert np.allclose(res_legacy["cd"], res_routed["cd"], rtol=0, atol=1e-12), (
        f"routed splu diverged from legacy: {res_legacy['cd']} vs {res_routed['cd']}")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && .venv/bin/python -m pytest tests/test_p2r1a_thin_plate_flow.py::test_mono_solver_routing_parity -q`
Expected: FAIL — `TypeError: run_flow_past() got an unexpected keyword argument 'mono_solver'`

- [ ] **Step 3: Add the kwargs and route the solve**

In `run_flow_past(...)` signature add (after existing kwargs): `mono_solver="splu", device="cpu"`. Where the driver builds the DeviceMesh, replace the hardcoded `"cpu"` with `device`. Replace the solve site (~line 516):

```python
        # Solve — routed through solve_linear so MONO_SOLVER/DEVICE select
        # the backend (splu host | cudss GPU-direct | fused GPU-BiCGStab).
        # Matrix changes every step (Picard convection + kick rows): no cache_key.
        Acsr = A.tocsr()
        if mono_solver == "splu":
            x_cur = splu(Acsr.tocsc()).solve(b)      # legacy path, bit-for-bit
        else:
            x_cur = solve_linear(Acsr, b, solver=mono_solver, sym=False,
                                 device=device)
```

Add the import near the top with the other diffsim imports: `from diffsim.solvers.linsolve import solve_linear`.

NOTE the legacy branch keeps literal `splu(...)` — `mono_solver="splu"` therefore also takes it. That makes the parity test trivially pass but STILL verifies the plumbing (kwargs flow through). To also exercise `solve_linear`'s splu path, run the routed branch when `device != "cpu"` OR `mono_solver != "splu"` — i.e. the exact code above. This is deliberate: zero risk to the CPU baseline.

In `__main__` add env parsing next to the existing knobs:

```python
    mono_solver = os.environ.get("MONO_SOLVER", "splu")
    device      = os.environ.get("DEVICE", "cpu")
```

and pass `mono_solver=mono_solver, device=device` into `run_flow_past(...)` and into `compare_solvers(...)`. In `compare_solvers`, accept and forward the same two kwargs to its internal `run_flow_past` call (monolithic leg) — read the function to find the call site.

- [ ] **Step 4: Run the new test + the full existing 2-D gates**

Run: `cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && .venv/bin/python -m pytest tests/test_p2r1a_thin_plate_flow.py tests/test_p2r1a_thin_plate_flow_projection.py -q`
Expected: all pass (previously 9 + 1 new).

- [ ] **Step 5: Commit**

```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && git add tests/p2r1a_thin_plate_flow.py tests/test_p2r1a_thin_plate_flow.py && git commit -m "$(cat <<'EOF'
feat(p2r1a): route 2-D monolithic solve through solve_linear (MONO_SOLVER/DEVICE)

run_flow_past gains mono_solver=/device= kwargs (env MONO_SOLVER/DEVICE);
default splu path bit-for-bit unchanged. Non-splu backends route through
solve_linear(solver=, sym=False, device=) per the ladder_rung3d_cube
pattern — cudss (GPU direct) and fused (GPU BiCGStab) become available
with zero further driver code. compare_solvers forwards both knobs.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Route the 3-D monolithic solve the same way

**Files:**
- Read: `tests/p2r1c_thin_plate_flow_3d.py` (imports ~40-60; assemble+solve at :418-:430; `__main__` env parsing; DeviceMesh build in `_build_shell_3d`)
- Modify: `tests/p2r1c_thin_plate_flow_3d.py`
- Test: `tests/test_p2r1c_thin_plate_flow_3d.py` (add one test)

**Interfaces:**
- Consumes: `solve_linear` (as Task 1).
- Produces: `run_flow_past_3d(..., mono_solver="splu", device="cpu")`; env `MONO_SOLVER`, `DEVICE`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_p2r1c_thin_plate_flow_3d.py`:

```python
def test_mono3d_solver_routing_parity():
    from p2r1c_thin_plate_flow_3d import run_flow_past_3d
    kw = dict(level=3, nsteps=2, dt=0.01, nu=0.1, U_inf=1.0, verbose=False)
    res_legacy = run_flow_past_3d(**kw)
    res_routed = run_flow_past_3d(mono_solver="splu", device="cpu", **kw)
    assert np.allclose(res_legacy["cd"], res_routed["cd"], rtol=0, atol=1e-12)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && .venv/bin/python -m pytest tests/test_p2r1c_thin_plate_flow_3d.py::test_mono3d_solver_routing_parity -q`
Expected: FAIL — unexpected keyword argument.

- [ ] **Step 3: Implement**

Mirror Task 1 exactly: add `mono_solver="splu", device="cpu"` kwargs to `run_flow_past_3d`; thread `device` into the `DeviceMesh.from_mesh(..., device)` call inside `_build_shell_3d` (add a `device="cpu"` parameter to it); replace the solve at :430 with the same routed block; add `from diffsim.solvers.linsolve import solve_linear`; add the two env knobs in `__main__`.

- [ ] **Step 4: Run the 3-D gates**

Run: `cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && .venv/bin/python -m pytest tests/test_p2r1c_thin_plate_flow_3d.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && git add tests/p2r1c_thin_plate_flow_3d.py tests/test_p2r1c_thin_plate_flow_3d.py && git commit -m "$(cat <<'EOF'
feat(p2r1c): route 3-D monolithic solve through solve_linear (MONO_SOLVER/DEVICE)

Same routing as the 2-D driver: mono_solver=/device= kwargs + env knobs,
splu default bit-for-bit, cudss/fused available via solve_linear.
_build_shell_3d gains device= for the DeviceMesh.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Encode the gpubox WSL CUDA env into the remote toolkit + GPU solve smoke (both solvers, both dims)

**Files:**
- Modify: `scripts/remote/config.sh` (add `GPUBOX_GPU_ENV`)
- Modify: `scripts/remote/gpubox-run.sh` (prepend it to the launched command)
- Create: `tests/gpu_smoke_thinshell.py` (small driver-level GPU smoke, runnable ONLY on a CUDA box — not collected by pytest on the Mac)

**Interfaces:**
- Consumes: Task 1+2 knobs (`MONO_SOLVER`, `DEVICE`, existing `PPE_SOLVER`).
- Produces: `GPUBOX_GPU_ENV` config var; `gpubox-run.sh` launches with it; `tests/gpu_smoke_thinshell.py` printing `SMOKE-OK 2d-mono 2d-proj 3d-mono 3d-proj` on success.

- [ ] **Step 1: Encode the WSL lib path (hazard encoded, not remembered)**

In `scripts/remote/config.sh`, after the `CLAUDE_BIN` block:

```bash
# gpubox is WSL2: the native libcuda stub shadows the WSL GPU-passthrough
# driver. EVERY GPU process must see /usr/lib/wsl/lib first or Warp/torch
# report "no CUDA-capable device" while nvidia-smi works fine.
export GPUBOX_GPU_ENV="LD_LIBRARY_PATH=/usr/lib/wsl/lib"
```

In `scripts/remote/gpubox-run.sh`, change the tmux launch line to prepend it:

```bash
  "trap 'rm -f \"$GPUBOX_LOCK\"' EXIT; cd '$GPUBOX_REPO_ABS' && { env $GPUBOX_GPU_ENV $cmd ; } 2>&1 | tee '$log'"
```

- [ ] **Step 2: Write the GPU smoke driver**

Create `tests/gpu_smoke_thinshell.py`:

```python
"""GPU smoke: both solvers x both dims solve ON DEVICE. Run on a CUDA box:

    LD_LIBRARY_PATH=/usr/lib/wsl/lib .venv/bin/python tests/gpu_smoke_thinshell.py

Tiny cases, ~1-2 min total. Asserts finite Cd everywhere and prints SMOKE-OK.
NOT a pytest file (no test_ prefix): Mac CI never collects it."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

import warp as wp
wp.init()
assert wp.is_cuda_available(), (
    "no CUDA device — on gpubox set LD_LIBRARY_PATH=/usr/lib/wsl/lib")
DEV = "cuda:0"

from p2r1a_thin_plate_flow import run_flow_past, run_flow_past_projection
from p2r1c_thin_plate_flow_3d import run_flow_past_3d

ok = []
# 2-D monolithic on cuDSS (direct GPU)
r = run_flow_past(level=4, nsteps=3, dt=0.01, nu=0.1,
                  mono_solver="cudss", device=DEV, verbose=False)
assert np.all(np.isfinite(r["cd"])), r["cd"]; ok.append("2d-mono")
# 2-D projection: predictor fused (GPU BiCGStab), PPE gpu_cg (GPU CG)
r = run_flow_past_projection(level=4, nsteps=3, dt=0.01, nu=0.1,
                             predictor_solver="fused", ppe_solver="gpu_cg",
                             device=DEV, verbose=False)
assert np.all(np.isfinite(r["cd"])), r["cd"]; ok.append("2d-proj")
# 3-D monolithic on cuDSS
r = run_flow_past_3d(level=3, nsteps=2, dt=0.01, nu=0.1,
                     mono_solver="cudss", device=DEV, verbose=False)
assert np.all(np.isfinite(r["cd"])), r["cd"]; ok.append("3d-mono")
# 3-D projection fused + gpu_cg
sys.path.insert(0, os.path.dirname(__file__))
from p2r1c_thin_plate_flow_3d_projection import run_flow_past_3d_projection
r = run_flow_past_3d_projection(level=3, nsteps=2, ppe_solver="gpu_cg",
                                predictor_solver="fused", device=DEV,
                                verbose=False)
assert np.all(np.isfinite(r["cd"])), r["cd"]; ok.append("3d-proj")
print("SMOKE-OK", *ok)
```

NOTE: `run_flow_past_projection` / `run_flow_past_3d_projection` must accept `device=` — read them; the projection drivers build the DeviceMesh (`_build_shell*`) with `"cpu"` hardcoded. Add a `device="cpu"` kwarg threaded to the DeviceMesh build in each (2-3 lines per driver, same as Tasks 1-2). The steppers pick it up automatically via `self.dm.device`.

- [ ] **Step 3: Sync to gpubox and run the smoke there**

gpubox's working tree currently equals old-master content but is UNCOMMITTED (box HEAD b101056). Reconcile first, on the box, then sync:

```bash
# 1. Snapshot the box working tree (safety, cheap):
ssh gpubox "cd /home/bglab/Baskar/DiffSim && git add -A && git commit -m 'box working-tree snapshot pre thinshell-gpu sync' && git branch box-snap-$(date +%Y%m%d)"
# 2. Push the plan branch from the Mac and check it out on the box:
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && git push gpubox thinshell-gpu:thinshell-gpu
ssh gpubox "cd /home/bglab/Baskar/DiffSim && git checkout thinshell-gpu"
# 3. Run the smoke under the run-lock toolkit:
bash scripts/remote/gpubox-run.sh ".venv/bin/python tests/gpu_smoke_thinshell.py" smoke-thinshell
# 4. Poll to completion:
bash scripts/remote/gpubox-poll.sh <log-path-printed-by-run> 60
```

Expected in the log: `SMOKE-OK 2d-mono 2d-proj 3d-mono 3d-proj`.
If `fused` raises `ConvergenceError` on the projection predictor (α=50 Nitsche conditioning), retry that leg with `predictor_solver="cudss"` and RECORD the failure in the ledger — the fused-on-SBM convergence envelope is a finding, not a silent skip.

- [ ] **Step 4: Commit**

```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && git add scripts/remote/config.sh scripts/remote/gpubox-run.sh tests/gpu_smoke_thinshell.py tests/p2r1a_thin_plate_flow.py tests/p2r1c_thin_plate_flow_3d_projection.py && git commit -m "$(cat <<'EOF'
feat(gpu): WSL CUDA env in remote toolkit + both-solver GPU solve smoke

config.sh: GPUBOX_GPU_ENV=LD_LIBRARY_PATH=/usr/lib/wsl/lib (WSL2 libcuda
shadowing — encoded, not remembered); gpubox-run.sh prepends it.
tests/gpu_smoke_thinshell.py: 2-D+3-D x monolithic(cudss)+projection
(fused predictor + gpu_cg PPE) solve-on-device smoke; projection drivers
gain device= threading to their DeviceMesh builds.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Device assembly for the 2-D monolithic march (`ASSEMBLY=device`)

**Files:**
- Read: `tests/test_device_assembly.py::test_device_assembly_consistency` (:43-:56 — the exact aq/dq/fq adapter and parity assert to copy), `src/diffsim/assembly/device_assembly.py` docstrings for `assemble` (:786) / `csr_slots` (:1110) / `set_strong_rows` (:758)
- Read: `tests/p2r1a_thin_plate_flow.py` march loop — confirm whether Af_c is geometry-cached only or per-step (backflow); note the exact bc_rows/bc_vals/p_pin construction
- Modify: `tests/p2r1a_thin_plate_flow.py`
- Test: `tests/test_p2r1a_thin_plate_flow.py`

**Interfaces:**
- Consumes: `DeviceNSAssembler(dm)`, `.assemble(aq_by_bin, div_aq_by_bin, fq_by_bin, nu, sigma, strong_b_vals=, extra_matrix=, extra_rhs=)`, `.csr_slots(rows, cols)`, `.set_strong_rows(rows)`.
- Produces: `run_flow_past(..., assembly="host")` kwarg (`"host"|"device"`), env `ASSEMBLY`.

- [ ] **Step 1: Write the failing parity test**

Append to `tests/test_p2r1a_thin_plate_flow.py`:

```python
def test_device_assembly_parity_cpu(device):
    """assembly="device" on the CPU Warp device must match assembly="host"
    to distribution tolerance (host-device assembly parity; on cpu the
    scatter is deterministic so the tolerance is tight)."""
    from p2r1a_thin_plate_flow import run_flow_past
    kw = dict(level=4, nsteps=3, dt=0.01, nu=0.1, U_inf=1.0, verbose=False)
    res_h = run_flow_past(assembly="host", **kw)
    res_d = run_flow_past(assembly="device", **kw)
    assert np.allclose(res_h["cd"], res_d["cd"], rtol=1e-9, atol=1e-11), (
        f"device-assembly Cd diverged: {res_h['cd']} vs {res_d['cd']}")
```

Run: `cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && .venv/bin/python -m pytest tests/test_p2r1a_thin_plate_flow.py::test_device_assembly_parity_cpu -q`
Expected: FAIL — unexpected keyword `assembly`.

- [ ] **Step 2: Implement the device-assembly march path**

In `run_flow_past`, add kwarg `assembly="host"`. Before the march loop (once per mesh epoch), when `assembly == "device"`:

```python
    if assembly == "device":
        from diffsim.assembly.device_assembly import DeviceNSAssembler
        import warp as wp
        asm = DeviceNSAssembler(dm)                    # symbolic once
        # --- SBM face system → fixed-pattern device slots (cached) ---
        Af_csr = Af_c.tocsr()
        _rows, _cols = Af_csr.nonzero()
        _slots = asm.csr_slots(_rows, _cols)
        assert (_slots >= 0).all(), (
            "SBM face entries missing from device pattern — constraint-aware "
            "slot map does not cover Af_c on this mesh; fall back to host")
        af_slots_d = wp.array(_slots.astype(asm._idx_np), dtype=asm._idx_dtype,
                              device=dm.device)
        af_vals_d = wp.array(np.ascontiguousarray(Af_csr.data, np.float64),
                             dtype=wp.float64, device=dm.device)
        _bf_nz = np.nonzero(bf_c)[0]
        bf_dofs_d = wp.array(_bf_nz.astype(np.int32), dtype=wp.int32,
                             device=dm.device)
        bf_vals_d = wp.array(np.ascontiguousarray(bf_c[_bf_nz], np.float64),
                             dtype=wp.float64, device=dm.device)
        # --- strong rows: BCs + inflow-kick rows + pressure pin, ONE plan ---
        strong_rows = np.concatenate([np.asarray(bc_rows, np.int64),
                                      np.asarray(inflow_vy_rows, np.int64),
                                      [int(p_pin)]])
        strong_rows, _uniq = np.unique(strong_rows, return_index=True)
        asm.set_strong_rows(strong_rows)
```

Inside the march loop, replace the host `assemble_linear_ns → +Af_c → LIL surgery → solve` block for the device path:

```python
        if assembly == "device":
            # strong values in set_strong_rows ORDER (np.unique-sorted):
            sb = np.zeros(len(strong_rows))
            _val_of = dict(zip(map(int, bc_rows), bc_vals))
            if _pert_active and t_new < pert_t_end:
                _val_of.update({int(r): _pert_vkick for r in inflow_vy_rows})
            else:
                _val_of.update({int(r): 0.0 for r in inflow_vy_rows})
            _val_of[int(p_pin)] = 0.0
            sb[:] = [_val_of.get(int(r), 0.0) for r in strong_rows]
            A_dev, b_dev = asm.assemble(
                {1: aq}, {1: dq}, {1: fq_raw}, nu, sigma,
                strong_b_vals=sb,
                extra_matrix=(af_slots_d, af_vals_d),
                extra_rhs=(bf_dofs_d, bf_vals_d))
            Acsr, b = A_dev, b_dev            # host CSR copies from assemble()
            if mono_solver == "splu":
                x_cur = splu(Acsr.tocsc()).solve(b)
            else:
                x_cur = solve_linear(Acsr, b, solver=mono_solver, sym=False,
                                     device=device)
        else:
            # ... existing host path, UNCHANGED ...
```

IMPORTANT verify-first notes for the implementer:
1. The `{1: aq}` by-bin dict form and the `assemble(...)` return convention MUST be copied from `test_device_assembly.py::test_device_assembly_consistency` — if that test passes flat arrays or a different bin key, follow the test, not this sketch.
2. If the march recomputes Af per step (backflow `a_face`), re-run only the VALUE upload each step (`af_vals_d` assign from the fresh `Af_csr.data`, pattern/slots unchanged — assert identical `nonzero()` first).
3. `strong_b_vals` ordering vs `set_strong_rows` ordering: read `set_strong_rows`' docstring; if it preserves input order rather than sorting, drop the `np.unique` sort and keep parallel arrays.
4. On the CPU Warp device this whole path runs on "cpu" — that is the point of the parity test.

Add `assembly=os.environ.get("ASSEMBLY", "host")` in `__main__` and forward through `compare_solvers`.

- [ ] **Step 3: Run parity + full 2-D gates**

Run: `cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && .venv/bin/python -m pytest tests/test_p2r1a_thin_plate_flow.py tests/test_p2r1a_thin_plate_flow_projection.py -q`
Expected: all pass. If the `csr_slots` coverage assert trips on the ADAPTIVE mesh (hanging-node constraint pattern), that is a REAL finding: gate device assembly to uniform meshes (`refine_to is None`) with a clear `ValueError`, record in the ledger, and keep `assembly="host"` for adaptive — do NOT silently skip.

- [ ] **Step 4: GPU check on gpubox (distribution tolerance)**

```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && git push gpubox thinshell-gpu:thinshell-gpu && ssh gpubox "cd /home/bglab/Baskar/DiffSim && git checkout thinshell-gpu && git pull . thinshell-gpu 2>/dev/null; env LD_LIBRARY_PATH=/usr/lib/wsl/lib ASSEMBLY=device MONO_SOLVER=cudss DEVICE=cuda:0 LEVEL=4 NSTEPS=3 .venv/bin/python tests/p2r1a_thin_plate_flow.py 2>&1 | tail -5"
```

Expected: finite Cd, no ConvergenceError. Cross-check the printed Cd against the Mac host-assembly value at the same config — agreement to ~1e-7 class (GPU atomics few-ULP; if larger, lock the tolerance from the measured tail × 5, not from one run).

- [ ] **Step 5: Commit**

```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && git add tests/p2r1a_thin_plate_flow.py tests/test_p2r1a_thin_plate_flow.py && git commit -m "$(cat <<'EOF'
feat(p2r1a): device assembly for the 2-D monolithic march (ASSEMBLY=device)

DeviceNSAssembler in the march loop: symbolic pattern once; per-step
volume fill on device; cached SBM face system injected via the
extra_matrix/extra_rhs slot hooks (csr_slots coverage asserted); strong
BCs + inflow kick + pressure pin via set_strong_rows/strong_b_vals.
Host path bit-for-bit unchanged (assembly="host" default). CPU parity
gate test_device_assembly_parity_cpu.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Device assembly for the 3-D monolithic march

**Files:**
- Read: `tests/p2r1c_thin_plate_flow_3d.py` march loop (:400-:440), Task 4's final 2-D implementation (the pattern to mirror)
- Modify: `tests/p2r1c_thin_plate_flow_3d.py`
- Test: `tests/test_p2r1c_thin_plate_flow_3d.py`

**Interfaces:**
- Consumes: identical to Task 4.
- Produces: `run_flow_past_3d(..., assembly="host")`, env `ASSEMBLY`.

- [ ] **Step 1: Failing test**

```python
def test_device_assembly_parity_cpu_3d(device):
    from p2r1c_thin_plate_flow_3d import run_flow_past_3d
    kw = dict(level=3, nsteps=2, dt=0.01, nu=0.1, verbose=False)
    res_h = run_flow_past_3d(assembly="host", **kw)
    res_d = run_flow_past_3d(assembly="device", **kw)
    assert np.allclose(res_h["cd"], res_d["cd"], rtol=1e-9, atol=1e-11)
```

Run: `cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && .venv/bin/python -m pytest tests/test_p2r1c_thin_plate_flow_3d.py::test_device_assembly_parity_cpu_3d -q` → FAIL (unexpected kwarg).

- [ ] **Step 2: Implement** — mirror Task 4's final code exactly (3-D driver has its own bc_rows/pin construction and TWO-SIDED face system `Af_c/bf_c` from `sbm_vector_dirichlet_twosided`; the slot/value treatment is identical since it is just a host CSR). 3-D has no inflow kick — strong rows are bc_rows + pin only.

- [ ] **Step 3: Run gates**: `cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && .venv/bin/python -m pytest tests/test_p2r1c_thin_plate_flow_3d.py -q` → all pass.

- [ ] **Step 4: Commit**

```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && git add tests/p2r1c_thin_plate_flow_3d.py tests/test_p2r1c_thin_plate_flow_3d.py && git commit -m "$(cat <<'EOF'
feat(p2r1c): device assembly for the 3-D monolithic march (ASSEMBLY=device)

Mirrors the 2-D wiring: DeviceNSAssembler volume fill + two-sided SBM
face system via extra_matrix/extra_rhs slots + strong rows on device.
Host default unchanged; CPU parity gate added.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Projection stepper — forward `device_assembly` through `LeraySBMShellStepper` and gate the full-device projection path

**Files:**
- Read: `src/diffsim/steppers/leray_sbm.py` (`LeraySBMShellStepper.__init__` — the `LerayProjectionStepper(...)` construction), `src/diffsim/steppers/leray.py:280-310` (how `device_assembly` is consumed)
- Modify: `src/diffsim/steppers/leray_sbm.py`
- Test: `tests/test_p2r1a_thin_plate_flow_projection.py`

**Interfaces:**
- Consumes: `LerayProjectionStepper(..., device_assembly=bool)` (exists, leray.py:303).
- Produces: `LeraySBMShellStepper(..., device_assembly=False)` — forwarded to the base stepper; `run_flow_past_projection(..., device_assembly=False)` kwarg + env `ASSEMBLY` mapping (`device` → True).

- [ ] **Step 1: Failing test**

Append to `tests/test_p2r1a_thin_plate_flow_projection.py`:

```python
def test_projection_device_assembly_parity(device):
    """Projection with device_assembly=True must match the host-assembly
    projection march (CPU Warp device: deterministic, tight tolerance)."""
    from p2r1a_thin_plate_flow import run_flow_past_projection
    kw = dict(level=4, nsteps=3, dt=0.01, nu=0.1, ppe_solver="splu",
              verbose=False)
    res_h = run_flow_past_projection(**kw)
    res_d = run_flow_past_projection(device_assembly=True, **kw)
    assert np.allclose(res_h["cd"], res_d["cd"], rtol=1e-9, atol=1e-11), (
        f"projection device-assembly diverged: {res_h['cd']} vs {res_d['cd']}")
```

Run: `cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && .venv/bin/python -m pytest tests/test_p2r1a_thin_plate_flow_projection.py::test_projection_device_assembly_parity -q` → FAIL (unexpected kwarg).

- [ ] **Step 2: Implement the forwarding**

In `LeraySBMShellStepper.__init__` add parameter `device_assembly=False` and pass `device_assembly=device_assembly` into the `LerayProjectionStepper(...)` construction. In `run_flow_past_projection` add kwarg `device_assembly=False` and pass through to the stepper. In `__main__`, map `ASSEMBLY=device` → `device_assembly=True` for the projection leg (and forward in `compare_solvers`).

VERIFY-FIRST: read how `LerayProjectionStepper` uses `device_assembly` (leray.py:303-310) — if its device path has preconditions (e.g. identity constraints, no `extra_block` support), the parity test on the SBM-coupled shell stepper is exactly the gate that finds it. If `extra_block` (the SBM system) is not consumed by the device-assembly branch, the finding is: projection device assembly applies to the BASE operators only — record it, keep `device_assembly=False` as the shell default, and file the extra_block-on-device wiring as a follow-on ledger item. Do not fake the gate.

- [ ] **Step 3: Run projection gates**

Run: `cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && .venv/bin/python -m pytest tests/test_p2r1a_thin_plate_flow_projection.py tests/test_p2r0_projection_sbm.py -q`
Expected: all pass (p2r0 one-sided path must remain untouched).

- [ ] **Step 4: Commit**

```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && git add src/diffsim/steppers/leray_sbm.py tests/p2r1a_thin_plate_flow.py tests/test_p2r1a_thin_plate_flow_projection.py && git commit -m "$(cat <<'EOF'
feat(leray-sbm): forward device_assembly through LeraySBMShellStepper

LeraySBMShellStepper gains device_assembly= (default False) forwarded to
the base LerayProjectionStepper; run_flow_past_projection exposes it and
ASSEMBLY=device maps to it. Parity gate added; p2r0 one-sided unchanged.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Re=250 2-D both-solver GPU validation on gpubox (THE ACCEPTANCE GATE)

**Files:**
- Read: `docs/dev/p2r1a-thin-plate-runbook.md` (the recipe + pass bands)
- Modify: `docs/dev/p2r1a-thin-plate-runbook.md` (record the results)

**Interfaces:**
- Consumes: everything above; `compare_solvers` (SOLVER=both) forwarding `mono_solver/device/assembly/ppe_solver/predictor_solver`.
- Produces: recorded Cd_mean/St for BOTH solvers on GPU vs ThinShell Table 1.

- [ ] **Step 1: Pre-flight sanity (short GPU run, ~100 steps)**

```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && git push gpubox thinshell-gpu:thinshell-gpu
bash scripts/remote/gpubox-run.sh "SOLVER=both MONO_SOLVER=cudss ASSEMBLY=device DEVICE=cuda:0 PPE_SOLVER=gpu_cg PRED_SOLVER=fused BASE_LEVEL=7 REFINE_LEVEL=9 WAKE_LEVEL=9 NSTEPS=100 DT=5e-5 NU=0.004 U_INF=1.0 PLATE_XC=0.1389 PLATE_YC=0.5 PLATE_L=0.0625 PERT_EPS=0.03 PERT_T_END=1.0 T_START=0.0 .venv/bin/python tests/p2r1a_thin_plate_flow.py" re250-preflight
bash scripts/remote/gpubox-poll.sh <log> 60
```

Gate: both legs run, finite Cd, and note the measured **s/step for each solver** — from it compute the projected wall-time for 1M steps. If projected > ~48 h for the pair, run the two solver legs as two sequential launches (one per GPU is NOT possible under the single run-lock — sequential is fine) and/or consult Baskar before burning the time. NOTE: `PRED_SOLVER` must be threaded to the projection leg's `predictor_solver` in `compare_solvers` — verify while reading; add if missing (2 lines).

- [ ] **Step 2: Launch the production run**

Same command with `NSTEPS=1000000 T_START=20.0`, tag `re250-both-gpu`. Poll periodically (`gpubox-poll.sh <log> 40`); the driver prints `Cd_mean=… St=…` per leg at completion. Do NOT sync or edit any `.py` while it runs (run-lock enforces).

- [ ] **Step 3: Validate + record**

Pass bands (ThinShell Table 1 / spec): **Cd_mean ∈ [3.29, 3.45]** (±5% vs Najjar & Balachandar 3.36), **St ∈ [0.13, 0.17]**, BOTH solvers; solver-vs-solver Cd agreement within ~5%. Append a dated results table to `docs/dev/p2r1a-thin-plate-runbook.md`: config, per-solver Cd_mean/St/s-per-step/backend (`cudss+device-assembly` vs `fused+gpu_cg`), log path, verdict vs Table 1. If OUT of band: this is a physics/resolution finding (mesh, dt, averaging window) — record honestly, do not tune tolerances; escalate to Baskar with the numbers.

- [ ] **Step 4: Commit runbook + fetch any box commits**

```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && bash scripts/remote/gpubox-fetch.sh && git add docs/dev/p2r1a-thin-plate-runbook.md && git commit -m "$(cat <<'EOF'
docs(p2r1a): Re=250 both-solver GPU validation results (gpubox)

Recorded Cd_mean/St for monolithic (cudss + device assembly) and
projection (fused predictor + gpu_cg PPE) on RTX 6000 Ada vs ThinShell
Table 1 (Najjar & Balachandar), with per-solver s/step and log paths.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: 3-D both-solver GPU ladder on gpubox (L4 → L5 → L6, honest boundaries)

**Files:**
- Modify: `docs/dev/p2r1c-3d-plate-runbook.md` (record the ladder)

**Interfaces:**
- Consumes: Tasks 2/3/5 knobs on the 3-D drivers.
- Produces: measured 3-D s/step + solver-backend matrix on gpubox; the documented monolithic cuDSS ceiling and the projection-iterative path beyond it.

- [ ] **Step 1: Run the ladder (sequential gpubox-run.sh launches, poll each out)**

| Rung | Command core | Expectation |
|---|---|---|
| L4 mono | `MONO_SOLVER=cudss ASSEMBLY=device DEVICE=cuda:0 LEVEL=4 NSTEPS=10 DT=0.005 NU=0.004 .venv/bin/python tests/p2r1c_thin_plate_flow_3d.py` | fast, finite Cd |
| L4 proj | `PPE_SOLVER=gpu_cg PRED_SOLVER=fused DEVICE=cuda:0 LEVEL=4 NSTEPS=10 DT=0.005 NU=0.004 .venv/bin/python tests/p2r1c_thin_plate_flow_3d_projection.py` | finite Cd |
| L5 both | same, `LEVEL=5 NSTEPS=10` | ~131k dof: cudss fits (under wall) |
| L6 proj | `LEVEL=6 NSTEPS=10` projection only | ~1M dof: gpu_cg PPE + fused predictor is THE path |
| L6 mono-attempt | `LEVEL=6 NSTEPS=3 MONO_SOLVER=fused` | HONEST GATE: may not converge (saddle + α=50). Record iterations/outcome either way |

- [ ] **Step 2: Record the ladder** in `docs/dev/p2r1c-3d-plate-runbook.md`: per-rung s/step, memory (`nvidia-smi --query-gpu=memory.used`), backend, outcome. Explicitly state: monolithic-direct ends at the cuDSS ~812k wall (L5/L6 boundary); monolithic-iterative at L6+ is the R2b block-preconditioner track; projection (fused + gpu_cg) is the working L6+ path today.

- [ ] **Step 3: Commit**

```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && git add docs/dev/p2r1c-3d-plate-runbook.md && git commit -m "$(cat <<'EOF'
docs(p2r1c): 3-D both-solver GPU ladder on gpubox (L4-L6)

Measured s/step + memory per rung; monolithic cudss to the ~812k wall,
projection fused+gpu_cg beyond; L6 monolithic-fused attempt recorded
honestly (R2b block-precond remains the L6+ monolithic track).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Nova bring-up — reconcile, sync, A100 + GH200 smokes

**Files:**
- Read: `scripts/remote/nova-sync-submit.sh`, `cluster/bootstrap.sh`, `cluster/a100-smoke.sh`, `cluster/gh200-smoke.sh`, `cluster/slurm/` (existing sbatch patterns), nova state (`/work/mech-ai/baskarg/DiffSim`, branch `chunked-csr` @ `fde8408d`, untracked `cluster/gh200_*.sh` + venvs)
- Create: `cluster/slurm/thinshell_a100_smoke.sbatch`, `cluster/slurm/thinshell_gh200_smoke.sbatch`
- Modify: (possibly) `cluster/` — reconciled nova-only scripts committed back to master

**Interfaces:**
- Consumes: the 3-D drivers + knobs from Tasks 2/5; nova venvs `.venv-nova` (x86) and `.venv-nova-arm` (GH200).
- Produces: both smokes green on nova; sbatch files committed.

- [ ] **Step 1: Reconcile nova's untracked work back to master (worker→Mac policy)**

```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim
scp nova:/work/mech-ai/baskarg/DiffSim/cluster/gh200_osub.sh nova:/work/mech-ai/baskarg/DiffSim/cluster/gh200_probe.sh nova:/work/mech-ai/baskarg/DiffSim/cluster/submit_probe.sh cluster/ 2>/dev/null
scp nova:/work/mech-ai/baskarg/DiffSim/cluster/slurm/gh200_hold12.sbatch cluster/slurm/
git status --short cluster/   # review, then commit what is genuinely new
```

Check whether `benchmarks/gh200_capacity_probe.py` (referenced by gh200_osub.sh) exists in master; if it only exists on nova, scp it back too. Commit as `chore(cluster): reconcile nova GH200 scripts back to master`.

- [ ] **Step 2: Ship the branch to nova and check it out**

Use the documented bundle path (nova has no GitHub auth): run `bash scripts/remote/nova-sync-submit.sh` if it supports branch shipping (read it first); otherwise manual:

```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim
git bundle create /tmp/thinshell-gpu.bundle master thinshell-gpu
scp /tmp/thinshell-gpu.bundle nova:/work/mech-ai/baskarg/bundles/
ssh nova "cd /work/mech-ai/baskarg/DiffSim && git fetch /work/mech-ai/baskarg/bundles/thinshell-gpu.bundle thinshell-gpu:thinshell-gpu && git checkout thinshell-gpu && .venv-nova/bin/pip install -q -e . --no-deps"
```

(Nova's untracked venvs/scripts survive checkout. NEVER run from HOME — everything under `/work/mech-ai/baskarg`.)

- [ ] **Step 3: A100 smoke sbatch**

Create `cluster/slurm/thinshell_a100_smoke.sbatch` (pattern: `gh200_hold12.sbatch` header + `a100-smoke.sh` body):

```bash
#!/bin/bash
#SBATCH --job-name=thinshell-a100-smoke
#SBATCH --account=mech-ai
#SBATCH --partition=nova
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100-pcie:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/work/mech-ai/baskarg/DiffSim/cluster/results/%x-%j.out
cd /work/mech-ai/baskarg/DiffSim
export WARP_CACHE_PATH=/work/mech-ai/baskarg/.warp-cache
. .venv-nova/bin/activate
nvidia-smi -L
python tests/gpu_smoke_thinshell.py
```

Submit + verify: `ssh nova "cd /work/mech-ai/baskarg/DiffSim && sbatch cluster/slurm/thinshell_a100_smoke.sbatch"`, then poll `squeue -u $USER` / read the `.out` for `SMOKE-OK 2d-mono 2d-proj 3d-mono 3d-proj`. (Respect the ≤4 concurrent Nova GPU jobs cap.)

- [ ] **Step 4: GH200 smoke sbatch**

Create `cluster/slurm/thinshell_gh200_smoke.sbatch` — same body but:

```bash
#SBATCH --partition=nova-arm
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=200G
RPM_DIR=/work/mech-ai/baskarg/python311-arm
export LD_LIBRARY_PATH="$RPM_DIR/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export WARP_CACHE_PATH=/work/mech-ai/baskarg/.warp-cache-arm
. .venv-nova-arm/bin/activate
```

Submit and verify `SMOKE-OK` in the `.out`. If the ARM venv lacks a dep (torch/nvmath drift), fix via `cluster/bootstrap_arm.sh` and record. (GH200 = coupled host-device regime — the same device-wired drivers run; managed-memory oversubscription levers from `gh200_osub.sh` are out of scope here.)

- [ ] **Step 5: Commit**

```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && git add cluster/slurm/thinshell_a100_smoke.sbatch cluster/slurm/thinshell_gh200_smoke.sbatch && git commit -m "$(cat <<'EOF'
feat(cluster): thinshell both-solver GPU smokes for nova A100 + GH200

sbatch wrappers running tests/gpu_smoke_thinshell.py (2-D+3-D x
monolithic-cudss + projection fused/gpu_cg) on gpu:a100-pcie:1 and
gpu:gh200:1 (nova-arm, ARM venv + RPM lib path).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: ThinShell GPU runbook + merge

**Files:**
- Create: `docs/dev/thinshell-gpu-runbook.md`
- Modify: `.superpowers/sdd/progress.md` (ledger)

- [ ] **Step 1: Write the runbook** — one page per target (gpubox / nova-A100 / nova-GH200): env prep (WSL lib path; venv per target), the knob matrix (`MONO_SOLVER`, `PRED_SOLVER`, `PPE_SOLVER`, `ASSEMBLY`, `DEVICE`, mesh/case env vars), the Re=250 production command, the 3-D ladder commands, the measured s/step tables from Tasks 7-8, and the solver decision tree (cudss ≤ ~812k dof; projection fused+gpu_cg beyond; R2b = monolithic L6+ track). Cross-link `p2r1a-thin-plate-runbook.md` and `p2r1c-3d-plate-runbook.md`.

- [ ] **Step 2: Full regression sweep on the Mac (CPU)**

Run: `cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && .venv/bin/python -m pytest tests/test_p2r1a_thin_plate_flow.py tests/test_p2r1a_thin_plate_flow_projection.py tests/test_p2r1c_thin_plate_flow_3d.py tests/test_p2r1c_thin_plate_flow_3d_projection.py tests/test_p2r0_projection_sbm.py tests/test_device_assembly.py -q`
Expected: all green. Optionally run the full suite on gpubox via `scripts/remote/gpubox-test.sh` (parallel xdist, covers CUDA-only behavior).

- [ ] **Step 3: Ledger + merge decision** — update `.superpowers/sdd/progress.md` with per-task outcomes (including the honest findings: fused-on-SBM convergence, adaptive-mesh slot coverage, L6 monolithic attempt). Then use superpowers:finishing-a-development-branch to merge `thinshell-gpu` → master. Baskar pushes to origin (Mac is the sole gatekeeper).

---

## Self-Review Against the Goal

1. ✅ Both solvers on GPU, 2-D: monolithic = routed solve (T1) + device assembly (T4) → cudss/fused; projection = device-routed already + fused/gpu_cg + device_assembly forwarding (T3/T6).
2. ✅ Both solvers on GPU, 3-D: T2/T5 mirror + T8 ladder.
3. ✅ Re=250 validation on GPU (T7, acceptance gate with Table-1 bands).
4. ✅ gpubox / A100 / GH200 targets: T3+T7+T8 (gpubox), T9 (nova both partitions), T10 runbook.
5. ✅ cuDSS-wall honesty: iterative path (fused/gpu_cg) wired and gated; R2b boundary documented, not promised.
6. ✅ Existing CPU gates protected at every task; GPU parity via distribution locks.
7. Placeholder scan: verify-first notes are explicit instructions with named oracles (`test_device_assembly_consistency`, leray.py line refs), not TBDs. Type check: `mono_solver/device/assembly` kwargs and env names consistent across T1-T9; `run_flow_past`/`run_flow_past_3d`/`run_flow_past_projection` signatures match their uses in `gpu_smoke_thinshell.py`.

**Known risks (tracked in-plan):** fused BiCGStab convergence on α=50 Nitsche systems (T3/T8 gates it, cudss fallback); adaptive-mesh slot coverage for the SBM face pattern (T4 asserts it, host fallback); `device_assembly` × `extra_block` interaction in the projection base stepper (T6 gates it); GH200 ARM venv drift (T9 bootstrap fallback).
