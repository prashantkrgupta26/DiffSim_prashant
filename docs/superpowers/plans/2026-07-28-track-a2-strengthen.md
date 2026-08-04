# Track A2: Strengthen the Monolithic Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Production-rate monolithic engine at ~10M DOF — PCD-class outer iteration counts at ≤~60 s/step (≈4× better than bdiag's measured 242 s at 9.08M), with the measured 243–245 GB host-assembly wall removed via device assembly.

**Architecture:** Per the approved spec (`docs/superpowers/specs/2026-07-28-track-a2-strengthen-design.md`): WP1 telemetry counters inside the PCD inner solves (the binding prerequisite); WP2 AMGX V-cycle on the EXTRACTED velocity sub-CSR as the PCD F-inner (`pcd_inner="amgx"` via the existing meta-through-cache plumbing — never AMGX on the raw saddle); WP3 `SADDLE_ASSEMBLY` pass-through so the ladder marches with the drivers' existing device assembly; WP4 the GPU validation matrix + binding success gate.

**Tech Stack:** `src/diffsim/solvers/saddle_precond.py` (A1/A3 module), `src/diffsim/solvers/amgx.py` (`amgx_solve`, `last_solve_stats`, `amgx_configs/PCG_CLASSICAL_V_JACOBI.json`), `linsolve.py` backend dispatch + `_LAST_ITERS` telemetry sentinel, `tests/gpu_saddle_ladder.py` harness, gpubox toolkit + nova GH200 sbatch kits (`cluster/slurm/saddle_ladder_gh200*.sbatch`).

## Global Constraints

- Branch `track-a2-strengthen` off current master; repo cd prefix; `.venv/bin/python`; box runs via the gpubox toolkit only; nova submissions are the CONTROLLER's (subagents record ready-to-submit configs, never submit).
- Default paths untouched: all new behavior opt-in; every existing gate stays green at every commit (tests/test_saddle_precond.py, test_saddle_ladder_cpu.py, test_p2r1a_thin_plate_flow.py, test_p2r1c_thin_plate_flow_3d.py, test_p2r0_projection_sbm.py, test_device_assembly.py).
- AMGX ONLY on the extracted velocity block — never the raw saddle (documented divergence). AMGX int32 nnz ceiling ~79.5M carried as a stated constraint in every projection to 100M.
- Every performance/stability claim from a measured run; negative results first-class; separate process per GPU ladder leg (cross-leg-accumulation rule); never touch tmux sessions/jobs you didn't start (a Track B confirm session may still be live on gpubox).
- SUCCESS GATE (spec, binding, time-boxed to THIS plan): at 9.08M, pcd-amgx outer growth from 3d-L6 ≤ ~2× AND s/step ≤ ~60 s; partial outcomes (e.g. AMGX-setup-dominates → lagged-setup probe) recorded honestly.
- Commits: explicit paths; trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task T1: PCD inner-solve telemetry

**Files:**
- Modify: `src/diffsim/solvers/saddle_precond.py` (the `_cg` helper + `make_pcd_apply`), `src/diffsim/solvers/linsolve.py` (the `fgmres_pcd` branch — attach the aggregate to the result path)
- Test: `tests/test_saddle_precond.py` (extend)

**Interfaces:**
- Consumes: `make_pcd_apply(A, meta, device)`, `_cg(...)` with `_INNER_TOL=1e-4, _INNER_MAX=500`, `_LAST_ITERS` sentinel in linsolve.py, `LinearSolveResult` (`.iterations` populated today).
- Produces: `make_pcd_apply(A, meta, device, stats=None)` — when `stats` is a dict, each apply accumulates per-block records: `stats = {"F": {"applies": int, "iters_total": int, "cap_hits": int, "max_exit_relres": float}, "Ap": {...}, "Mp": {...}}`. The `fgmres_pcd` branch creates the dict, passes it, and after the solve stores it in a NEW module sentinel `_LAST_INNER_STATS[0]` (list-of-one, same pattern as `_LAST_ITERS`); the `return_result=True` wrapper copies it onto `LinearSolveResult.meta["inner_stats"]` (verify `LinearSolveResult` has a meta/info dict — if not, add an optional `inner_stats=None` field to the dataclass, default None, no other backend touched). Default `stats=None` ⇒ behavior byte-identical.
- VERIFY-FIRST: how `_cg` learns iteration count and exit residual from `cg_dev` (read `cg_dev`'s return info dict; the A3 review confirmed it returns `(x_numpy, info)`); cap-hit = `iters >= _INNER_MAX`.

- [ ] **Step 1 (failing test):** extend `tests/test_saddle_precond.py`:

```python
def test_fgmres_pcd_inner_stats():
    Acsr, b, x_splu, cache = _get_system_with_meta()   # existing helper
    r = solve_linear(Acsr, b, solver="fgmres_pcd", sym=False, device="cpu",
                     tol=1e-10, cache=cache, cache_key="k", return_result=True)
    st = r.inner_stats            # or r.meta["inner_stats"] per the verify-first outcome
    assert st is not None
    for blk in ("F", "Ap", "Mp"):
        assert st[blk]["applies"] > 0
        assert st[blk]["iters_total"] > 0
        assert st[blk]["cap_hits"] >= 0
        assert 0.0 <= st[blk]["max_exit_relres"] < float("inf")
```

Run: `.venv/bin/pytest tests/test_saddle_precond.py::test_fgmres_pcd_inner_stats -q` → RED (no such attribute).
- [ ] **Step 2:** implement per Produces; keep the counters plain Python ints/floats (no device sync beyond what `cg_dev` already returns).
- [ ] **Step 3:** gates — full `tests/test_saddle_precond.py` + `tests/test_saddle_ladder_cpu.py` + `tests/test_p2r1a_thin_plate_flow.py -q` green (timeout 600000).
- [ ] **Step 4:** commit `feat(solvers): PCD inner-solve telemetry (per-block iters/cap-hits/exit-relres)` + trailer.

---

### Task T2: AMGX velocity inner (`pcd_inner="amgx"`)

**Files:**
- Read FIRST: `src/diffsim/solvers/amgx.py` (`amgx_solve` signature :115, `last_solve_stats` :42, `_sparsity_fingerprint`/setup-caching behavior — determine whether repeated calls with the same sparsity re-setup or reuse; the per-step Picard F changes VALUES but not sparsity), `amgx_configs/PCG_CLASSICAL_V_JACOBI.json`, the `blockamgx` branch in `linsolve.py:1445+` (meta-via-cache shape reference).
- Modify: `src/diffsim/solvers/saddle_precond.py`, `src/diffsim/solvers/linsolve.py` (extend the existing `fgmres_pcd` branch — no new backend name)
- Test: `tests/test_saddle_precond.py` (extend)

**Interfaces:**
- Consumes: T1's `stats` dict (the AMGX F-inner records `applies`, `iters_total` from `last_solve_stats()`, `cap_hits`=0 convention, `max_exit_relres`).
- Produces: `build_pcd_meta(dm, nu, sigma, p_pin, inner="jacobi")` gains the `inner` kwarg (values `"jacobi"` | `"amgx"`; default `"jacobi"` = today's behavior bit-for-bit). When `"amgx"`: `make_pcd_apply` extracts the velocity sub-CSR once per `make_pcd_apply` call (`u_ids` mask already exists; `F = A[u_ids][:,u_ids].tocsr()`) and the F-inner calls `amgx_solve(F, r_u_shifted, sym=False, tol=_INNER_TOL, maxiter=50)` instead of `_cg`. Ap/Mp inners unchanged. The extraction + any AMGX setup happens ONCE per outer solve (per step), NOT per apply — structure the closure so per-apply work is solve-only; whether AMGX internally re-setups on same-sparsity value changes is the VERIFY-FIRST question above, and the answer (measured, from `last_solve_stats` timings if available) goes in the report.
- CPU/CI story: pyamgx is GPU-only. The unit test monkeypatches `saddle_precond`'s imported `amgx_solve` with a scipy-splu stand-in to prove the ROUTING (F extracted correctly — assert the stand-in received a matrix of shape `(n_u, n_u)` with the expected nnz; result still matches x_splu to the same gates). A real-AMGX correctness check happens on GPU in T4 (`pytest.mark.skipif(no pyamgx)` for a small real-AMGX test that T4's box run executes).

- [ ] **Step 1 (failing test):**

```python
def test_fgmres_pcd_amgx_routing(monkeypatch):
    calls = {}
    def fake_amgx(F, rhs, sym=False, tol=0.0, maxiter=0, **kw):
        calls["shape"] = F.shape; calls["n"] = calls.get("n", 0) + 1
        import scipy.sparse.linalg as sla
        return sla.spsolve(F.tocsc(), rhs)
    monkeypatch.setattr("diffsim.solvers.saddle_precond.amgx_solve", fake_amgx)
    Acsr, b, x_splu, cache = _get_system_with_meta(inner="amgx")   # helper gains the kwarg
    r = solve_linear(Acsr, b, solver="fgmres_pcd", sym=False, device="cpu",
                     tol=1e-10, cache=cache, cache_key="k", return_result=True)
    n_u = (Acsr.shape[0] // 3) * 2          # 2-D: velocity dofs
    assert calls["shape"] == (n_u, n_u)
    assert calls["n"] >= 1
    _assert_pcd_accuracy(r.x, x_splu)        # same gates as the existing PCD test
```

RED first (import of `amgx_solve` into saddle_precond does not exist).
- [ ] **Step 2:** implement per Produces (import `amgx_solve` lazily inside the `"amgx"` path so CPU-only environments never import pyamgx).
- [ ] **Step 3:** parity — `inner="jacobi"` default still produces the T1-green results (run the whole extended test file); gates as T1 Step 3.
- [ ] **Step 4:** commit `feat(solvers): AMGX velocity-inner for PCD (pcd_inner=amgx, extracted block only)` + trailer.

---

### Task T3: device assembly through the ladder

**Files:**
- Read FIRST: how `run_flow_past` / `run_flow_past_3d` expose assembly (the `assembly=` kwarg / `ASSEMBLY` env per `docs/dev/thinshell-gpu-runbook.md` §2.1–2.2), and what `tests/test_device_assembly.py` already proves (kernel parity host-vs-device).
- Modify: `tests/gpu_saddle_ladder.py` (`run_ladder_point` gains `assembly=None` pass-through; `__main__` reads `SADDLE_ASSEMBLY` env, default None = today's host path)
- Test: `tests/test_saddle_ladder_cpu.py` (extend: one tiny point with `assembly="device"` on cpu device — the device stack runs on "cpu" — asserting finite result + iterations recorded + the default-None path byte-identical to a pre-change reference march)

**Interfaces:**
- Produces: `run_ladder_point(tag, solver, dim, nsteps=5, device=..., assembly=None, **cfg)`; env `SADDLE_ASSEMBLY` ∈ {unset, "device", "host"}.
- Steps: TDD as house pattern; gates: ladder CPU tests + both drivers' suites; commit `feat(solvers): SADDLE_ASSEMBLY pass-through — device assembly in the ladder path` + trailer.

---

### Task T4: GPU validation campaign + verdict

**Files:**
- Modify: `cluster/slurm/saddle_ladder_gh200.sbatch`, `cluster/slurm/saddle_ladder_gh200_pcd.sbatch` (add `SADDLE_ASSEMBLY=device` + solver name updates), create `docs/dev/2026-07-28-track-a2-campaign.md`, append dated section to `docs/dev/thinshell-gpu-runbook.md`.
- This is a measurement campaign task (the A5 pattern): gpubox legs directly; the GH200 legs are recorded as ready-to-submit configs for the controller.

**Campaign matrix (each leg its own process, `SADDLE_ASSEMBLY=device`):**
1. gpubox: `2d-r11 × pcd(inner=amgx)` — the graded-mesh retest (does the AMGX inner fix the divergence? 200-step early-exit probe first, then the 5-step ladder point if bounded). Requires pyamgx on gpubox — VERIFY it is installed in the box venv before planning legs; if absent, record and fall back to nova-only AMGX legs.
2. gpubox: `3d-L6 × {bdiag, pcd-amgx}` — the 1.1M reference rungs with device assembly (also measures the assembly-time saving vs the A5 host-path numbers).
3. GH200 (controller submits): `3d-L7r9 × {bdiag, pcd-amgx}` with device assembly — the gate legs. Record per leg: outer iters/step, T1 inner stats (iters, cap-hits), s/step SPLIT (assembly / AMGX setup / solve — from the telemetry + timestamps), RSS + SMI peaks (the sbatch samplers). Target evidence: setup RSS ≪ 100 GB (device assembly removes the 243 GB transient).
4. If AMGX setup dominates s/step (from the split): the flag-gated lagged-setup probe (refresh the AMGX hierarchy every k∈{5} steps, values-only updates between — smallest faithful implementation, parity-tested default-off) — ONE probe leg, recorded either way.

**Verdict (binding gate, verbatim):** at 9.08M, pcd-amgx outer growth from 3d-L6 ≤ ~2× AND s/step ≤ ~60 s ⇒ pcd-amgx is the production engine (record the 100M projection WITH the int32-nnz ceiling caveat). Otherwise bdiag remains the engine; record why + what it would take. Either way: campaign report + runbook section + ledger; commit docs.

## Self-Review

Spec coverage: WP1→T1, WP2→T2 (+lagged-setup fallback in T4.4), WP3→T3, WP4→T4 (gate verbatim). Verify-first items name their oracles (cg_dev info dict, AMGX setup-caching behavior, driver assembly knob, pyamgx-on-gpubox). Type consistency: `stats` dict schema defined in T1 and consumed by name in T2/T4; `run_ladder_point(assembly=None)` defined in T3, used in T4's env. No placeholders; the only deliberately deferred decision (`r.inner_stats` vs `r.meta["inner_stats"]`) is bounded by an explicit verify-first instruction with both landing spots named.
