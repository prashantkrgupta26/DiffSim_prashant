# Solver-Escalation Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Tasks 1–6 are code (Mac-testable, subagent-executable). Tasks 7–11 are OPERATOR tasks (cluster legs and human gates) — the session controller executes them directly; they are included so the campaign's gates are explicit.

**Goal:** Break the T5 solver wall — PCD-as-primary with AMGX inners, proven on offline snapshots, then Ahmed, then the truck Re=250 full-amplitude gate.

**Architecture:** Add a system-dump knob to the march driver so failing saddle systems become portable `.npz` snapshots; a standalone solver lab iterates preconditioner configs on them. Wire `fgmres_pcd` as a first-class primary solver (meta build + per-step σ/ν refresh). Add the Ahmed body as a second permanent case through the body-agnostic pipeline, and STL-in-domain renders for the human geometry gate.

**Tech Stack:** scipy/numpy (host saddle CSR), Warp + AMGX paths already in `linsolve.py`/`saddle_precond.py` (Track A2), pyvista (renders), pytest.

**Spec:** `docs/superpowers/specs/2026-08-01-solver-escalation-design.md` (governs; pass bars quoted below are copied from it).

## Global Constraints

- Every new knob defaults OFF ⇒ byte-identical default paths (project cardinal rule).
- Suite gate on this Mac: `.venv/bin/python -m pytest tests/test_truck_flow.py -q` → `17 passed, 1 skipped` before AND after each task (plus new tests passing).
- New CUDA-only test legs use the `@pytest.mark.skipif(not _HAS_CUDA, ...)` idiom from `tests/test_truck_flow.py`.
- Commit per task, prefix `feat(sesc):` (solver-escalation).
- Never modify `results/`, `cluster/slurm/`, or live-run configs.
- Cluster launches (Tasks 8/10/11): verify no live leg on the compute node first (`srun --overlap ... ps -ef | grep -E "t5_gate|ahmed_gate|solver_lab" | grep -v grep`), append-only logs, `setsid nohup ... </dev/null`.
- Rungs 2/3 (Tasks 10/11) are BLOCKED until the corresponding Rung-0 sign-off ledger line exists (Tasks 7/9).

## File Map

- `src/diffsim/cases/truck/truck_march.py` — MODIFY: `dump_system_steps`/`dump_system_dir` knob; PCD-primary meta build + per-step refresh (Tasks 1, 2).
- `cluster/solver_lab.py` — CREATE: snapshot solver lab CLI (Task 3).
- `tools/render_mesh.py` — MODIFY: STL-in-domain render mode (Task 4).
- `src/diffsim/cases/ahmed.py` — CREATE: Ahmed geometry generator + case defaults (Task 5).
- `tests/test_ahmed.py` — CREATE: watertightness, clearance, tiny march (Task 5).
- `cluster/t5_gate.py` — MODIFY: DUMP_SYSTEM knobs (Task 6).
- `cluster/ahmed_gate.py` — CREATE: Ahmed march gate (Task 6).
- `tests/test_truck_flow.py` — MODIFY: tests for Tasks 1–3.

---

### Task 1: `dump_system` knob in the march driver

**Files:**
- Modify: `src/diffsim/cases/truck/truck_march.py`
- Test: `tests/test_truck_flow.py`

**Interfaces:**
- Consumes: march-loop locals `Acsr` (scipy CSR on the host path), `b`, `sigma`, `nu_step`, `_tol`, `step`, `dt_step`, `dm`, `p_pin`, `_bd`; `build_pcd_meta` from `diffsim.solvers.saddle_precond`.
- Produces: `run_truck(..., dump_system_steps=(), dump_system_dir=None)`; snapshot file `sys_step{N:04d}.npz` with keys `A_indptr, A_indices, A_data, A_shape, b, ndof, nfree, tol, sigma, nu, dt, step, Mp_indptr, Mp_indices, Mp_data, Mp_shape, Ap_indptr, Ap_indices, Ap_data, Ap_shape, p_pin, bd_json`. Task 3's lab reads exactly these keys.

- [ ] **Step 1: Write the failing test** (append to `tests/test_truck_flow.py`)

```python
def test_dump_system_snapshot(tmp_path):
    """dump_system_steps writes a self-contained solvable saddle snapshot."""
    import json
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu as _splu
    from test_truck_viz import _tiny_tire_mesh, _CFG_PATH
    from diffsim.cases.truck_config import load_truck_config
    from diffsim.cases.truck import run_truck
    cfg = load_truck_config(_CFG_PATH)
    res = run_truck(cfg, nsteps=3, base_level=5, truck_band_to=6,
                    band_cells=2, merged=_tiny_tire_mesh(cfg),
                    region_refine=False, nu=1.0 / 50.0, dt=0.02,
                    verbose=False,
                    dump_system_steps=(1,), dump_system_dir=str(tmp_path))
    f = tmp_path / "sys_step0001.npz"
    assert f.exists()
    d = np.load(f)
    A = sp.csr_matrix((d["A_data"], d["A_indices"], d["A_indptr"]),
                      shape=tuple(d["A_shape"]))
    n = int(d["nfree"]) * int(d["ndof"])
    assert A.shape == (n, n)
    assert np.all(np.isfinite(d["b"])) and d["b"].shape == (n,)
    # snapshot must be solvable standalone
    x = _splu(A.tocsc()).solve(d["b"])
    assert np.all(np.isfinite(x))
    # PCD operators present and square in the pressure space (nfree x nfree)
    Ap = sp.csr_matrix((d["Ap_data"], d["Ap_indices"], d["Ap_indptr"]),
                       shape=tuple(d["Ap_shape"]))
    assert Ap.shape == (int(d["nfree"]), int(d["nfree"]))
    assert int(d["p_pin"]) >= 0
    json.loads(str(d["bd_json"]))          # knob record parses
    # knob OFF byte-identity: default run unaffected
    ref = run_truck(cfg, nsteps=3, base_level=5, truck_band_to=6,
                    band_cells=2, merged=_tiny_tire_mesh(cfg),
                    region_refine=False, nu=1.0 / 50.0, dt=0.02,
                    verbose=False)
    np.testing.assert_array_equal(res["cd"], ref["cd"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_truck_flow.py::test_dump_system_snapshot -x -q`
Expected: FAIL — `TypeError: run_truck() got an unexpected keyword argument 'dump_system_steps'`

- [ ] **Step 3: Implement.** In `truck_march.py`:

(a) Add to the `run_truck` signature (near the other diagnostics knobs):

```python
              dump_system_steps=(), dump_system_dir=None,
```

with docstring lines:

```python
        dump_system_steps / dump_system_dir : dump the assembled saddle
            system (A, b) plus the PCD pressure operators (Mp, Ap) at the
            listed step indices to ``sys_step{N:04d}.npz`` for the offline
            solver lab.  HOST-ASSEMBLY ONLY (device parity is trajectory-
            tight, so host-captured systems are the same matrices): a
            device-handoff Acsr raises ValueError.  Default () = OFF,
            byte-identical.
```

(b) Module-level helper (place next to the checkpoint helpers):

```python
def _dump_saddle_system(path, Acsr, b, *, ndof, nfree, tol, sigma, nu,
                        dt, step, pcd_meta, bd):
    """Write a self-contained solver-lab snapshot (see solver_lab.py)."""
    import json
    import scipy.sparse as sp
    if not sp.issparse(Acsr):
        raise ValueError(
            "dump_system requires host assembly (scipy CSR); re-run the "
            "capture leg with TRUCK_ASSEMBLY=host / assembly='host'")
    A = Acsr.tocsr()
    Mp = pcd_meta["Mp"].tocsr()
    Ap = pcd_meta["Ap"].tocsr()
    tmp = str(path) + ".tmp.npz"
    np.savez_compressed(
        tmp,
        A_indptr=A.indptr, A_indices=A.indices, A_data=A.data,
        A_shape=np.asarray(A.shape), b=np.asarray(b, np.float64),
        ndof=ndof, nfree=nfree, tol=float(tol), sigma=float(sigma),
        nu=float(nu), dt=float(dt), step=int(step),
        Mp_indptr=Mp.indptr, Mp_indices=Mp.indices, Mp_data=Mp.data,
        Mp_shape=np.asarray(Mp.shape),
        Ap_indptr=Ap.indptr, Ap_indices=Ap.indices, Ap_data=Ap.data,
        Ap_shape=np.asarray(Ap.shape),
        p_pin=int(pcd_meta["p_pin"]),
        bd_json=json.dumps({k: v for k, v in bd.items()
                            if isinstance(v, (int, float, str, bool))}))
    os.replace(tmp, str(path))
```

(If `os` is not already imported at module level in `truck_march.py`, import it; check first.)

(c) Setup (before the march loop, after `_bd` exists): normalize the knob and pre-build the PCD operators once if dumping is requested:

```python
    _dump_steps = set(int(s) for s in (dump_system_steps or ()))
    _dump_meta = None
    if _dump_steps:
        if dump_system_dir is None:
            raise ValueError("dump_system_steps set but dump_system_dir=None")
        import pathlib as _pl
        _pl.Path(dump_system_dir).mkdir(parents=True, exist_ok=True)
        from diffsim.solvers.saddle_precond import build_pcd_meta as _bpm
        _dump_meta = _bpm(dm, nu if nu is not None else 1.0, 1.0 / dt,
                          p_pin=int(p_pin))
```

(d) In the march loop, immediately BEFORE the solve on BOTH assembly branches (the `_iter_solve`/`splu` call sites at the host path and the device path — the device path will raise via the helper's issparse guard):

```python
                if step in _dump_steps:
                    _dump_saddle_system(
                        _pl_dump / f"sys_step{step:04d}.npz", Acsr, b,
                        ndof=ndof, nfree=nfree, tol=_tol, sigma=sigma,
                        nu=nu_step, dt=dt_step, step=step,
                        pcd_meta=_dump_meta, bd=_bd)
```

where `_pl_dump` is created in (c) as `_pl_dump = _pl.Path(dump_system_dir)` (hoist it; use a plain string join if pathlib shadowing is awkward at the loop site). Match the exact local names in the current loop (`_tol`, `sigma`, `nu_step`, `dt_step`) — read the call sites of `_iter_solve` first.

- [ ] **Step 4: Run the new test, then the full suite**

Run: `.venv/bin/python -m pytest tests/test_truck_flow.py::test_dump_system_snapshot -x -q` → PASS.
Run: `.venv/bin/python -m pytest tests/test_truck_flow.py -q` → `18 passed, 1 skipped`.

- [ ] **Step 5: Commit**

```bash
git add src/diffsim/cases/truck/truck_march.py tests/test_truck_flow.py
git commit -m "feat(sesc): dump_system knob — portable saddle snapshots for the solver lab"
```

---

### Task 2: PCD as a first-class primary solver

**Files:**
- Modify: `src/diffsim/cases/truck/truck_march.py`
- Test: `tests/test_truck_flow.py`

**Interfaces:**
- Consumes: `build_pcd_meta` (Task-1 import pattern), existing knobs `pcd_f_inner`, `pcd_ap_inner`, cache `_pcd_cache`, `_iter_solve`.
- Produces: `run_truck(..., mono_solver="fgmres_pcd")` works end-to-end: meta built at setup with `inner=pcd_f_inner, ap_inner=pcd_ap_inner`, σ/ν refreshed before every primary solve. Guard: `mono_solver="fgmres_pcd"` with `saddle_fallback="pcd"` raises `ValueError` (PCD cannot be its own fallback).

- [ ] **Step 1: Write the failing test** (append to `tests/test_truck_flow.py`)

```python
def test_pcd_primary_march():
    """mono_solver='fgmres_pcd' marches and tracks the splu trajectory."""
    from test_truck_viz import _tiny_tire_mesh, _CFG_PATH
    from diffsim.cases.truck_config import load_truck_config
    from diffsim.cases.truck import run_truck
    cfg = load_truck_config(_CFG_PATH)
    common = dict(nsteps=3, base_level=5, truck_band_to=6, band_cells=2,
                  region_refine=False, nu=1.0 / 50.0, dt=0.02, verbose=False)
    ref = run_truck(cfg, merged=_tiny_tire_mesh(cfg),
                    mono_solver="splu", **common)
    res = run_truck(cfg, merged=_tiny_tire_mesh(cfg),
                    mono_solver="fgmres_pcd", **common)
    for k in ("cd", "cd_surr"):
        assert np.all(np.isfinite(res[k]))
        np.testing.assert_allclose(res[k], ref[k], rtol=0, atol=1e-5)
    # PCD-as-primary cannot also be the fallback
    with pytest.raises(ValueError, match="fallback"):
        run_truck(cfg, merged=_tiny_tire_mesh(cfg),
                  mono_solver="fgmres_pcd", saddle_fallback="pcd", **common)
```

NOTE on the 1e-5 tolerance: it assumes the march's default iterative solve
tol is ≤1e-8 on the tiny system. Check what tol `_iter_solve` receives on a
default `run_truck` call (read the tol plumbing); if the default is looser,
pass the driver's tol knob explicitly in BOTH runs of this test (whatever
its real name is — e.g. `tol=1e-10`) so the comparison is meaningful. Do
not loosen the assert to make a loose default pass.

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_truck_flow.py::test_pcd_primary_march -x -q`
Expected: FAIL — `ConvergenceError`/`ValueError: fgmres_pcd requires ('pcd_meta', cache_key) in cache` from `linsolve.py` (nothing builds the meta today).

- [ ] **Step 3: Implement.** In `truck_march.py`:

(a) Next to the existing fallback-arming guard/blocks (search `saddle_fallback == "pcd"`), add BEFORE them:

```python
    if mono_solver == "fgmres_pcd" and saddle_fallback == "pcd":
        raise ValueError(
            "mono_solver='fgmres_pcd' cannot use saddle_fallback='pcd' "
            "(PCD is already the primary; no fallback for the fallback)")
    if mono_solver == "fgmres_pcd":
        from diffsim.solvers.saddle_precond import build_pcd_meta
        _t_pm = time.time()
        _pcd_cache[("pcd_meta", "truck")] = build_pcd_meta(
            dm, nu if nu is not None else 1.0, 1.0 / dt,
            p_pin=int(p_pin), inner=pcd_f_inner, ap_inner=pcd_ap_inner)
        print(f"[truck] PCD PRIMARY armed (F={pcd_f_inner}, "
              f"Ap={pcd_ap_inner}, meta {time.time()-_t_pm:.1f}s)",
              flush=True)
```

(b) In `_iter_solve`, before the primary `solve_linear` call, refresh the per-step coefficients (mirrors what the fallback path already does):

```python
        if mono_solver == "fgmres_pcd":
            _pm = _pcd_cache[("pcd_meta", "truck")]
            _pm["sigma"] = float(sigma)
            _pm["nu"] = float(nu_step)
```

No other change: `_slv_cache` already includes `"fgmres_pcd"`, and the fallback path is unreachable for this solver (the except-guard requires `_fb_meta is not None`, which stays `None`).

- [ ] **Step 4: Run the new test, then the full suite**

Run: `.venv/bin/python -m pytest tests/test_truck_flow.py::test_pcd_primary_march -x -q` → PASS.
Run: `.venv/bin/python -m pytest tests/test_truck_flow.py -q` → `19 passed, 1 skipped`.

- [ ] **Step 5: Commit**

```bash
git add src/diffsim/cases/truck/truck_march.py tests/test_truck_flow.py
git commit -m "feat(sesc): fgmres_pcd as first-class primary (meta build + per-step sigma/nu refresh)"
```

---

### Task 3: the solver lab

**Files:**
- Create: `cluster/solver_lab.py`
- Test: `tests/test_truck_flow.py`

**Interfaces:**
- Consumes: Task-1 snapshot keys (exact list in Task 1); `solve_linear` from `diffsim.solvers.linsolve`; `_LAST_ITERS`, `_LAST_INNER_STATS` sentinels in `linsolve` (read their exact semantics at `linsolve.py:~43` before using).
- Produces: `run_config(npz_path, config, *, device="cpu", restart=None, maxiter=None, tol=None) -> dict` with keys `config, converged, relres, outer, inner_stats, wall_s, n` — importable by tests and by the CLI `python cluster/solver_lab.py SNAP.npz --config pcd-amgx [--device cuda:0] [--restart N] [--maxiter M] [--tol T]`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_truck_flow.py`)

```python
def test_solver_lab_on_snapshot(tmp_path):
    """The lab reproduces a converged solve on a dumped tiny system for
    the bdiag control and the pcd-jacobi candidate."""
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..",
                                      "cluster"))
    from test_truck_viz import _tiny_tire_mesh, _CFG_PATH
    from diffsim.cases.truck_config import load_truck_config
    from diffsim.cases.truck import run_truck
    cfg = load_truck_config(_CFG_PATH)
    run_truck(cfg, nsteps=3, base_level=5, truck_band_to=6, band_cells=2,
              merged=_tiny_tire_mesh(cfg), region_refine=False,
              nu=1.0 / 50.0, dt=0.02, verbose=False,
              dump_system_steps=(1,), dump_system_dir=str(tmp_path))
    from solver_lab import run_config
    snap = str(tmp_path / "sys_step0001.npz")
    for config in ("bdiag", "pcd-jacobi"):
        row = run_config(snap, config, device="cpu", tol=1e-8)
        assert row["converged"], row
        assert row["relres"] < 1e-7
        assert row["n"] > 0 and row["wall_s"] > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_truck_flow.py::test_solver_lab_on_snapshot -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'solver_lab'`

- [ ] **Step 3: Create `cluster/solver_lab.py`**

```python
"""Offline solver lab — run candidate saddle solvers on dumped snapshots.

Rung 1 of the solver-escalation campaign (spec 2026-08-01).  A snapshot is a
self-contained system dumped by run_truck(dump_system_steps=...): the saddle
CSR A and rhs b at a march step, plus the PCD pressure operators (Mp, Ap)
and the knob record of the leg that produced it.

Configs:
  bdiag      — fgmres_bdiag at the leg's knobs (control A: must reproduce
               the leg's outcome; a captured miss-system must MISS)
  pcd-jacobi — fgmres_pcd, Jacobi inners (control B: the slow fallback)
  pcd-amgx   — fgmres_pcd, AMGX F-inner + AMG-on-Ap (the candidate;
               requires CUDA + pyamgx)

Usage:
  python cluster/solver_lab.py SNAP.npz --config pcd-amgx \
      [--device cuda:0] [--restart 60] [--maxiter 3000] [--tol 5e-4]
Prints one JSON row per run (append to a .jsonl for sweeps).
"""
import argparse
import json
import sys
import time

import numpy as np
import scipy.sparse as sp


def _load(npz_path):
    d = np.load(npz_path)
    A = sp.csr_matrix((d["A_data"], d["A_indices"], d["A_indptr"]),
                      shape=tuple(d["A_shape"]))
    Mp = sp.csr_matrix((d["Mp_data"], d["Mp_indices"], d["Mp_indptr"]),
                       shape=tuple(d["Mp_shape"]))
    Ap = sp.csr_matrix((d["Ap_data"], d["Ap_indices"], d["Ap_indptr"]),
                       shape=tuple(d["Ap_shape"]))
    meta = dict(ndof=int(d["ndof"]), nfree=int(d["nfree"]),
                tol=float(d["tol"]), sigma=float(d["sigma"]),
                nu=float(d["nu"]), dt=float(d["dt"]), step=int(d["step"]),
                p_pin=int(d["p_pin"]), bd=json.loads(str(d["bd_json"])))
    return A, np.asarray(d["b"], np.float64), Mp, Ap, meta


def _build_cache(config, Mp, Ap, meta):
    """Reconstruct the solver-cache entries solve_linear expects, without a
    mesh: bdiag uses the knob dict; pcd uses a hand-built pcd_meta (the
    same keys build_pcd_meta returns — see saddle_precond.py:~514)."""
    cache = {"ndof": meta["ndof"]}
    if config == "bdiag":
        bd = {"ndof": meta["ndof"]}
        for k in ("saddle_x0", "saddle_restart", "saddle_equilibrate",
                  "saddle_min_work"):
            if k in meta["bd"]:
                bd[k] = meta["bd"][k]
        cache[("blocktri_meta", "lab")] = bd
    else:
        inner = "amgx" if config == "pcd-amgx" else "jacobi"
        ap_inner = "amgx" if config == "pcd-amgx" else "jacobi"
        # Mirror the exact key set build_pcd_meta returns (verify against
        # saddle_precond.py before finalizing; p_pin_local == p_pin here
        # because the dump stores the local pinned index).
        cache[("pcd_meta", "lab")] = {
            "ndof": meta["ndof"], "dim": 3, "Mp": Mp, "Ap": Ap,
            "sigma": meta["sigma"], "nu": meta["nu"],
            "p_pin": meta["p_pin"], "p_pin_local": meta["p_pin"],
            "inner": inner, "ap_inner": ap_inner}
    return cache


def run_config(npz_path, config, *, device="cpu", restart=None,
               maxiter=None, tol=None):
    from diffsim.solvers import linsolve
    A, b, Mp, Ap, meta = _load(npz_path)
    cache = _build_cache(config, Mp, Ap, meta)
    if restart is not None and config == "bdiag":
        cache[("blocktri_meta", "lab")]["saddle_restart"] = int(restart)
    solver = "fgmres_bdiag" if config == "bdiag" else "fgmres_pcd"
    kw = dict(solver=solver, sym=False, tol=float(tol or meta["tol"]),
              device=device, cache=cache, cache_key="lab")
    if maxiter is not None:
        kw["maxiter"] = int(maxiter)
    t0 = time.time()
    converged, relres = True, float("nan")
    try:
        x = linsolve.solve_linear(A, b, **kw)
        relres = float(np.linalg.norm(b - A @ x) / np.linalg.norm(b))
    except Exception as e:
        if type(e).__name__ not in ("ConvergenceError",):
            raise
        converged = False
    wall = time.time() - t0
    inner = linsolve._LAST_INNER_STATS[0] if solver == "fgmres_pcd" else None
    return dict(config=config, snapshot=str(npz_path), step=meta["step"],
                n=A.shape[0], tol=kw["tol"], converged=converged,
                relres=relres, outer=int(linsolve._LAST_ITERS[0]),
                inner_stats=inner, wall_s=round(wall, 3), device=device)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--config", required=True,
                    choices=("bdiag", "pcd-jacobi", "pcd-amgx"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--restart", type=int, default=None)
    ap.add_argument("--maxiter", type=int, default=None)
    ap.add_argument("--tol", type=float, default=None)
    a = ap.parse_args(argv)
    row = run_config(a.snapshot, a.config, device=a.device,
                     restart=a.restart, maxiter=a.maxiter, tol=a.tol)
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
```

IMPLEMENTER NOTE (verify, don't trust): (1) the exact key set of
`build_pcd_meta`'s return at `saddle_precond.py:~514` — if it contains keys
beyond those mirrored above (e.g. factorizations or device buffers built
lazily), replicate what the `fgmres_pcd` branch actually reads; (2) whether
`ConvergenceError` on the fgmres_pcd path leaves `_LAST_ITERS` set; (3)
whether `p_pin` stored by Task 1 is the local pinned index the pcd branch
expects (`p_pin_local`) — Task 1 stores `pcd_meta["p_pin"]` from
`build_pcd_meta`'s own return, so mirror whichever of the two names the
branch consumes. Fix the lab (and Task 1's dump if needed) to match reality,
and record what you found in the report.

- [ ] **Step 4: Run the new test, then the full suite**

Run: `.venv/bin/python -m pytest tests/test_truck_flow.py::test_solver_lab_on_snapshot -x -q` → PASS.
Run: `.venv/bin/python -m pytest tests/test_truck_flow.py -q` → `20 passed, 1 skipped`.

- [ ] **Step 5: Commit**

```bash
git add cluster/solver_lab.py tests/test_truck_flow.py
git commit -m "feat(sesc): offline solver lab (bdiag control / pcd-jacobi / pcd-amgx on snapshots)"
```

---

### Task 4: STL-in-domain render mode

**Files:**
- Modify: `tools/render_mesh.py`
- Test: `tests/test_render_mesh.py` (create)

**Interfaces:**
- Consumes: pyvista (already a dependency of the tool); `MergedTriMesh`-style `(verts, tris)` arrays.
- Produces: `render_stl_in_domain(stl_path, out_prefix, *, position=(0.0, 0.0, 0.0), scale=1.0, domain=((0, 0, 0), (1.0, 0.125, 0.125)))` writing `{out_prefix}_iso.png`, `{out_prefix}_side.png`, `{out_prefix}_front.png`; CLI `python tools/render_mesh.py --stl PATH --out-prefix P [--position x,y,z] [--scale s]`.

- [ ] **Step 1: Write the failing test** (create `tests/test_render_mesh.py`)

```python
"""STL-in-domain render smoke test (offscreen pyvista)."""
import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))


def _write_box_stl(path):
    """Minimal binary STL: unit cube scaled to 0.05, 12 triangles."""
    v = np.array([[x, y, z] for x in (0, 1) for y in (0, 1)
                  for z in (0, 1)], np.float32) * 0.05
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5),
             (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
             (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(faces)))
        for a, b, c in faces:
            f.write(struct.pack("<3f", 0, 0, 0))
            for i in (a, b, c):
                f.write(struct.pack("<3f", *v[i]))
            f.write(struct.pack("<H", 0))


def test_render_stl_in_domain(tmp_path):
    from render_mesh import render_stl_in_domain
    stl = tmp_path / "box.stl"
    _write_box_stl(stl)
    prefix = str(tmp_path / "box_dom")
    render_stl_in_domain(str(stl), prefix, position=(0.3, 0.0, 0.04))
    for tag in ("iso", "side", "front"):
        p = tmp_path / f"box_dom_{tag}.png"
        assert p.exists() and p.stat().st_size > 1000
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_render_mesh.py -x -q`
Expected: FAIL — `ImportError: cannot import name 'render_stl_in_domain'`

- [ ] **Step 3: Implement.** Append to `tools/render_mesh.py` (before the `__main__` block):

```python
def render_stl_in_domain(stl_path, out_prefix, *, position=(0.0, 0.0, 0.0),
                         scale=1.0,
                         domain=((0.0, 0.0, 0.0), (1.0, 0.125, 0.125))):
    """Render the body STL placed inside the channel domain, 3 views.

    position/scale mirror the config placement convention: rendered
    verts = stl_verts * scale + position (apply the SAME transform the
    mesh pipeline applies to this body before carving)."""
    if not pathlib.Path(stl_path).exists():
        raise FileNotFoundError(f"[render_mesh] STL not found: {stl_path}")
    body = pv.read(str(stl_path))
    body.points = body.points * float(scale) + np.asarray(position,
                                                          np.float64)
    lo, hi = np.asarray(domain[0], float), np.asarray(domain[1], float)
    box = pv.Box(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
    ground = pv.Plane(center=((lo[0] + hi[0]) / 2, lo[1],
                              (lo[2] + hi[2]) / 2),
                      direction=(0, 1, 0),
                      i_size=hi[0] - lo[0], j_size=hi[2] - lo[2])
    for tag, cam in (("iso", "iso"), ("side", "xy"), ("front", "yz")):
        p = pv.Plotter(off_screen=True, window_size=(1920, 1080))
        p.add_mesh(box, style="wireframe", color="black", line_width=2)
        p.add_mesh(ground, color="tan", opacity=0.4)
        p.add_mesh(body, color="steelblue", show_edges=False)
        p.camera_position = cam
        p.add_text(f"{pathlib.Path(stl_path).name} in domain — {tag}",
                   font_size=12)
        png = f"{out_prefix}_{tag}.png"
        p.screenshot(png)
        p.close()
        print("  wrote", png)
```

And replace the `__main__` block's plain loop with flag handling (keep the npz path behavior byte-identical when no flags given):

```python
if __name__ == "__main__":
    args = sys.argv[1:]
    if "--stl" in args:
        def _get(flag, default=None):
            return (args[args.index(flag) + 1]
                    if flag in args else default)
        pos = tuple(float(x) for x in
                    _get("--position", "0,0,0").split(","))
        render_stl_in_domain(_get("--stl"),
                             _get("--out-prefix", "results/mesh_renders/body"),
                             position=pos,
                             scale=float(_get("--scale", "1.0")))
    else:
        for f in args:
            if not pathlib.Path(f).exists():
                sys.exit(f"[render_mesh] ERROR: input path not found: {f}")
            render(f)
```

(The missing-path check already exists from the cleanup campaign — merge with it rather than duplicating; keep whichever form is present.)

- [ ] **Step 4: Run the new test**

Run: `.venv/bin/python -m pytest tests/test_render_mesh.py -x -q` → PASS.
Run: `.venv/bin/python -m pytest tests/test_truck_flow.py -q` → unchanged (`20 passed, 1 skipped`).

- [ ] **Step 5: Commit**

```bash
git add tools/render_mesh.py tests/test_render_mesh.py
git commit -m "feat(sesc): STL-in-domain render mode for the geometry QA gate"
```

---

### Task 5: Ahmed body — generator, case module, tests

**Files:**
- Create: `src/diffsim/cases/ahmed.py`
- Test: `tests/test_ahmed.py`

**Interfaces:**
- Consumes: `MergedTriMesh` from `diffsim.geometry.merged_trimesh` (constructor `MergedTriMesh(verts, tris)` — verify signature); `run_truck` (body-agnostic `merged=` argument).
- Produces:
  - `ahmed_profile(slant_deg=25.0, front_radius=0.0958, n_arc=16) -> np.ndarray` — closed convex CCW (x,y) polygon, unit body length, x=0 front.
  - `ahmed_verts_tris(length=0.06, slant_deg=25.0, n_arc=16) -> (verts, tris)` — watertight triangulation, width/height at standard Ahmed ratios (W=0.3726·L, H=0.2759·L), body-local coords (min corner at origin).
  - `ahmed_merged(length=0.06, clearance=0.003, x_front=0.32, band_level=11, slant_deg=25.0) -> MergedTriMesh` — placed in the unit channel: y shifted by `clearance` (assert `clearance >= 4 * 2.0**-band_level`), z centered in [0, 0.125], x front at `x_front`.
  - `write_ahmed_stl(path, **kwargs)` — binary STL of `ahmed_verts_tris` for the render gate.

- [ ] **Step 1: Write the failing tests** (create `tests/test_ahmed.py`)

```python
"""Ahmed body: watertightness, placement asserts, tiny march gate."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from diffsim.cases.ahmed import (ahmed_profile, ahmed_verts_tris,
                                 ahmed_merged)


def _signed_volume(verts, tris):
    a, b, c = (verts[tris[:, i]] for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def test_ahmed_watertight_and_oriented():
    verts, tris = ahmed_verts_tris(length=1.0)
    # every edge shared by exactly two triangles, opposite orientation
    edges = {}
    for t in tris:
        for i in range(3):
            e = (int(t[i]), int(t[(i + 1) % 3]))
            edges[e] = edges.get(e, 0) + 1
    for (u, v), cnt in edges.items():
        assert cnt == 1, f"duplicate directed edge {(u, v)}"
        assert edges.get((v, u), 0) == 1, f"unmatched edge {(u, v)}"
    # outward orientation => positive enclosed volume, plausible magnitude
    vol = _signed_volume(verts, tris)
    # H=0.2759, W=0.3726, minus front rounding + slant cut: ~0.09-0.103
    assert 0.080 < vol < 0.103


def test_ahmed_profile_convex_ccw():
    p = ahmed_profile()
    x, y = p[:, 0], p[:, 1]
    # closed CCW: shoelace area positive
    area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert area > 0
    # convex: all cross products of consecutive edge vectors >= 0 (tol)
    d = np.diff(np.vstack([p, p[:1]]), axis=0)
    cross = d[:-1, 0] * d[1:, 1] - d[:-1, 1] * d[1:, 0]
    assert np.all(cross > -1e-12)


def test_ahmed_clearance_assert():
    with pytest.raises(AssertionError):
        ahmed_merged(length=0.06, clearance=1e-4, band_level=11)


def test_ahmed_tiny_march():
    """Enlarged Ahmed through the body-agnostic pipeline: 3 steps finite."""
    from test_truck_viz import _CFG_PATH
    from diffsim.cases.truck_config import load_truck_config
    from diffsim.cases.truck import run_truck
    cfg = load_truck_config(_CFG_PATH)
    merged = ahmed_merged(length=0.2, clearance=0.02, x_front=0.28,
                          band_level=6)
    res = run_truck(cfg, nsteps=3, base_level=5, truck_band_to=6,
                    band_cells=2, merged=merged, region_refine=False,
                    nu=1.0 / 50.0, dt=0.02, verbose=False)
    assert res["n_excluded"] > 0 and res["sf_faces"] > 0
    for k in ("cd", "cd_surr"):
        assert res[k].shape == (3,)
        assert np.all(np.isfinite(res[k]))
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ahmed.py -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'diffsim.cases.ahmed'`

- [ ] **Step 3: Create `src/diffsim/cases/ahmed.py`**

```python
"""Ahmed reference body (Ahmed et al. 1984) for the solver-escalation
campaign — clean-geometry rung between snapshot lab and the truck gate.

Standard proportions at unit body length L=1: W=389/1044, H=288/1044,
front-edge radius R=100/1044, slant length 222/1044 at ``slant_deg``.
DELIBERATE deviations, recorded per the spec: (1) NO support stilts
(sub-resolution features are the T5 lesson); (2) front rounding is applied
in the x-y profile only (2.5-D extrusion, flat sides) rather than 3-D
fillets — adequate for the solver-survival rung; noted for the production
accuracy phase.
"""
import numpy as np

_W = 389.0 / 1044.0
_H = 288.0 / 1044.0
_R = 100.0 / 1044.0
_SLANT_LEN = 222.0 / 1044.0


def ahmed_profile(slant_deg=25.0, front_radius=_R, n_arc=16):
    """Closed convex CCW (x, y) polygon of the side profile, unit length.

    x: 0 (front) -> 1 (rear); y: 0 (bottom) -> H (roof)."""
    r = float(front_radius)
    th = np.deg2rad(float(slant_deg))
    dx, dy = _SLANT_LEN * np.cos(th), _SLANT_LEN * np.sin(th)
    pts = []
    # front-bottom arc: (0, r) -> (r, 0), center (r, r)
    for a in np.linspace(np.pi, 1.5 * np.pi, n_arc + 1):
        pts.append((r + r * np.cos(a), r + r * np.sin(a)))
    pts.append((1.0, 0.0))            # bottom rear
    pts.append((1.0, _H - dy))        # rear face top (below slant)
    pts.append((1.0 - dx, _H))        # slant leading edge on the roof
    # front-top arc: (r, H) -> (0, H - r), center (r, H - r)
    for a in np.linspace(0.5 * np.pi, np.pi, n_arc + 1):
        pts.append((r + r * np.cos(a), _H - r + r * np.sin(a)))
    return np.asarray(pts, np.float64)


def ahmed_verts_tris(length=0.06, slant_deg=25.0, n_arc=16):
    """Watertight triangulated Ahmed body, body-local coords, min corner
    at the origin; returns (verts[N,3] float64, tris[M,3] int64)."""
    prof = ahmed_profile(slant_deg=slant_deg, n_arc=n_arc) * float(length)
    m = len(prof)
    w = _W * float(length)
    verts = np.vstack([np.column_stack([prof, np.zeros(m)]),
                       np.column_stack([prof, np.full(m, w)])])
    tris = []
    # side walls: quad (i, i+1) x (z=0, z=w) -> 2 tris, outward normals
    for i in range(m):
        j = (i + 1) % m
        a, b_, c, d = i, j, m + j, m + i
        tris.append((a, c, b_))
        tris.append((a, d, c))
    # caps: fan from vertex 0 (profile is convex)
    for i in range(1, m - 1):
        tris.append((0, i + 1, i))              # z=0 cap, normal -z
        tris.append((m, m + i, m + i + 1))      # z=w cap, normal +z
    verts = np.ascontiguousarray(verts, np.float64)
    tris = np.asarray(tris, np.int64)
    # orientation guard: enclosed volume must be positive (outward normals)
    a, b_, c = (verts[tris[:, i]] for i in range(3))
    vol = float(np.einsum("ij,ij->i", a, np.cross(b_, c)).sum() / 6.0)
    if vol < 0:
        tris = tris[:, [0, 2, 1]]
    return verts, tris


def ahmed_merged(length=0.06, clearance=0.003, x_front=0.32,
                 band_level=11, slant_deg=25.0):
    """Ahmed placed in the unit channel [0,1] x [0,1/8] x [0,1/8].

    clearance is the ground gap; it must be resolvable (>= 4 cells at the
    band level) — the T5 sub-resolution lesson, enforced here."""
    h_band = 2.0 ** (-int(band_level))
    assert clearance >= 4.0 * h_band, (
        f"Ahmed clearance {clearance} < 4 cells at band level {band_level} "
        f"(h={h_band}) — unresolvable gap (T5 lesson)")
    from diffsim.geometry.merged_trimesh import MergedTriMesh
    verts, tris = ahmed_verts_tris(length=length, slant_deg=slant_deg)
    w = verts[:, 2].max()
    verts = verts + np.asarray(
        [float(x_front), float(clearance), 0.0625 - w / 2.0], np.float64)
    return MergedTriMesh(verts, tris)


def write_ahmed_stl(path, **kwargs):
    """Binary STL of the Ahmed body (for the render gate)."""
    import struct
    verts, tris = ahmed_verts_tris(**kwargs)
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(tris)))
        for t in tris:
            a, b_, c = verts[t[0]], verts[t[1]], verts[t[2]]
            n = np.cross(b_ - a, c - a)
            nn = np.linalg.norm(n)
            n = n / nn if nn > 0 else n
            f.write(struct.pack("<3f", *n))
            for p in (a, b_, c):
                f.write(struct.pack("<3f", *p))
            f.write(struct.pack("<H", 0))
    return path
```

IMPLEMENTER NOTE: verify `MergedTriMesh(verts, tris)` is the real
constructor signature (`git grep -n "class MergedTriMesh" src/`) — the
tests in `tests/test_truck_flow.py` build one from `(v, t)` after
`_read_stl`, so match that call form exactly.

- [ ] **Step 4: Run the new tests, then the full suite**

Run: `.venv/bin/python -m pytest tests/test_ahmed.py -x -q` → 4 passed.
Run: `.venv/bin/python -m pytest tests/test_truck_flow.py tests/test_ahmed.py -q` → `24 passed, 1 skipped`.

- [ ] **Step 5: Commit**

```bash
git add src/diffsim/cases/ahmed.py tests/test_ahmed.py
git commit -m "feat(sesc): Ahmed body — analytic watertight generator, placement asserts, tiny-gate test"
```

---

### Task 6: gate wiring — t5 dump knobs + Ahmed gate

**Files:**
- Modify: `cluster/t5_gate.py`
- Create: `cluster/ahmed_gate.py`

**Interfaces:**
- Consumes: Task-1 knob (`dump_system_steps`, `dump_system_dir`), Task-5 `ahmed_merged`, Task-2 PCD-primary; `run_truck` and the knob-echo/telemetry idioms of `t5_gate.py` (read it first).
- Produces: env knobs `DUMP_SYSTEM_STEPS` (comma-separated ints), `DUMP_SYSTEM_DIR` in the t5 gate; `cluster/ahmed_gate.py` runnable with `AHMED_LENGTH, AHMED_CLEARANCE, AHMED_XFRONT, BAND_TO, BASE_LEVEL, NSTEPS, RE_TARGET, SOFT_START, MONO_SOLVER, PCD_F_INNER, PCD_AP_INNER, MARCH_CKPT, RESUME, U_CAP` env knobs.

- [ ] **Step 1: t5_gate dump knobs.** In `cluster/t5_gate.py`, with the other scalar env reads at module top:

```python
DUMP_SYS_STEPS = tuple(int(s) for s in
                       os.environ.get("DUMP_SYSTEM_STEPS", "").split(",")
                       if s.strip())
DUMP_SYS_DIR = os.environ.get("DUMP_SYSTEM_DIR", "") or None
```

and thread `dump_system_steps=DUMP_SYS_STEPS, dump_system_dir=DUMP_SYS_DIR`
into the `run_truck(...)` call (next to the checkpoint knobs). Follow the
existing eager-validation style: if steps are set but dir is empty, raise at
module top, not at step time.

- [ ] **Step 2: Create `cluster/ahmed_gate.py`** — the Rung-2 leg runner, mirroring the t5 gate's structure but minimal (no staged BC, no seal knobs):

```python
"""AHMED GATE LEG — Rung 2 of the solver-escalation campaign.

Clean-geometry full-amplitude march: Ahmed body (no stilts, resolvable
clearance) in the unit channel, PCD-primary by default.  Pass bar (spec
§6): >=500 consecutive steps at amplitude 1.0 with zero misses
post-transient, umax bounded (< U_CAP/10), finite cd_react.
"""
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, "/work/mech-ai/baskarg/DiffSim/src")

CONF = ("/work/mech-ai/baskarg/DiffSim/local_code_old/"
        "truck_4case_fresh_inputs/NewRun-no-shell-slope0p25/config.txt")

BASE_LEVEL = int(os.environ.get("BASE_LEVEL", "7"))
BAND_TO = int(os.environ.get("BAND_TO", "11"))
NSTEPS = int(os.environ.get("NSTEPS", "900"))
DT = float(os.environ.get("DT", "6.25e-4"))
RE_TARGET = float(os.environ.get("RE_TARGET", "250"))
SOFT_START = float(os.environ.get("SOFT_START", "300"))
U_CAP = float(os.environ.get("U_CAP", "5000"))
MONO = os.environ.get("MONO_SOLVER", "fgmres_pcd")
F_INNER = os.environ.get("PCD_F_INNER", "amgx")
AP_INNER = os.environ.get("PCD_AP_INNER", "amgx")
TOL = float(os.environ.get("TOL", "5e-4"))
A_LEN = float(os.environ.get("AHMED_LENGTH", "0.06"))
A_CLR = float(os.environ.get("AHMED_CLEARANCE", "0.003"))
A_XF = float(os.environ.get("AHMED_XFRONT", "0.32"))
ASSEMBLY = os.environ.get("TRUCK_ASSEMBLY", "device")
CKPT_DIR = os.environ.get("MARCH_CKPT", "") or None
RESUME = os.environ.get("RESUME", "0") == "1"
OUT = pathlib.Path(os.environ.get(
    "OUT_DIR", "/work/mech-ai/baskarg/DiffSim/results/ahmed-r2"))
OUT.mkdir(parents=True, exist_ok=True)

from diffsim.cases.truck_config import load_truck_config
from diffsim.cases.truck import run_truck
from diffsim.cases.ahmed import ahmed_merged
from diffsim.solvers import linsolve

cfg = load_truck_config(CONF)
merged = ahmed_merged(length=A_LEN, clearance=A_CLR, x_front=A_XF,
                      band_level=BAND_TO)
nu_unit = cfg.domain_scale / RE_TARGET

print("=" * 72, flush=True)
print(f"[AHMED] base={BASE_LEVEL} band={BAND_TO} nsteps={NSTEPS} "
      f"Re={RE_TARGET} nu_unit={nu_unit:.3e} mono={MONO} "
      f"F={F_INNER} Ap={AP_INNER} soft_start={SOFT_START} "
      f"L={A_LEN} clr={A_CLR}", flush=True)

_rows = []
_miss_post = [0]


def on_step(step, info):
    miss = bool(info.get("accepted_miss", False))
    if step >= SOFT_START and miss:
        _miss_post[0] += 1
    _rows.append(dict(step=step, cd=info.get("cd"),
                      umax=info.get("umax"), miss=miss,
                      iters=linsolve._LAST_ITERS[0]))
    if step % 10 == 0:
        r = _rows[-1]
        print(f"[AHMED] step {step:4d} cd={r['cd']:+.5f} "
              f"umax={r['umax']:.2f} iters={r['iters']} miss={miss}",
              flush=True)
        (OUT / "rows.jsonl").open("a").write(
            "\n".join(json.dumps(q) for q in _rows[-10:]) + "\n")


t0 = time.time()
res = run_truck(cfg, nsteps=NSTEPS, base_level=BASE_LEVEL,
                truck_band_to=BAND_TO, band_cells=2, merged=merged,
                region_refine=False, nu=nu_unit, dt=DT,
                mono_solver=MONO, pcd_f_inner=F_INNER,
                pcd_ap_inner=AP_INNER,
                soft_start=SOFT_START, u_cap=U_CAP,
                assembly=ASSEMBLY, on_step=on_step,
                checkpoint_interval=200 if CKPT_DIR else None,
                checkpoint_dir=CKPT_DIR, resume=RESUME,
                verbose=False)
post = [r for r in _rows if r["step"] >= SOFT_START]
print(f"[AHMED] DONE {time.time()-t0:.0f}s steps={len(_rows)} "
      f"post-transient={len(post)} misses_post={_miss_post[0]} "
      f"umax_final={post[-1]['umax'] if post else float('nan')}",
      flush=True)
print(f"[AHMED] PASS-BAR: steps_at_amp1={len(post)}>=500? "
      f"misses=0? {_miss_post[0] == 0}", flush=True)
```

IMPLEMENTER NOTE: verify against the real `run_truck` signature (Task 2's
final state) — in particular the exact parameter names for `soft_start`,
`assembly`, `tol` handling (the t5 gate passes a tol schedule; if
`run_truck` takes `tol=`/`tol_schedule=`, pass `TOL` the same way the t5
gate does for a constant tol), and whether `checkpoint_interval=None` is
the correct OFF value. Mirror `cluster/t5_gate.py`'s actual call and trim.

- [ ] **Step 3: Verify**

Run: `.venv/bin/python -m py_compile cluster/t5_gate.py cluster/ahmed_gate.py` → exit 0.
Run: `.venv/bin/python -m pytest tests/test_truck_flow.py -q` → unchanged.

- [ ] **Step 4: Commit**

```bash
git add cluster/t5_gate.py cluster/ahmed_gate.py
git commit -m "feat(sesc): DUMP_SYSTEM gate knobs + Ahmed Rung-2 gate leg"
```

---

### Task 7 (OPERATOR): Rung 0a — truck render package + sign-off

Not a subagent task. Controller: regenerate the sealed truck mesh dump
(`DUMP_BOX`/`DUMP_DIR` path or existing `results/mesh_*.npz`), then:

```bash
.venv/bin/python tools/render_mesh.py results/mesh_<sealed>.npz
.venv/bin/python tools/render_mesh.py --stl local_code_old/truck/Truck.stl \
    --out-prefix results/mesh_renders/truck_domain \
    --position <config placement in unit coords> --scale <config scale>
```

(The exact position/scale: the config places bodies at (0.0, -0.002, -7.0)
pre-scale; use the SAME transform `build_truck_mesh` applies — read it and
echo the numbers into the ledger with the render.)

Deliver `results/mesh_renders/` package; verify the automated checks in the
render log (single component, zero pockets, surrogate sanity). **GATE:
Baskar reviews; approval ledgered as
`RUNG-0a SIGN-OFF: truck mesh <file> approved (Baskar, <date>)`.**

### Task 8 (OPERATOR): Rung 1 — snapshot capture + lab verdict

1. On Nova (after the on-node no-live-leg check): short capture legs with
   `TRUCK_ASSEMBLY=host`, `DUMP_SYSTEM_STEPS` at the crest steps, from
   existing checkpoints (`RESUME=1`) or fresh (`NSTEPS` to just past the
   crest). Capture ≥3 miss-systems + 2 converged controls (leg tols in the
   snapshot record).
2. Validate every snapshot: `--config bdiag` must reproduce the leg outcome
   (miss-systems miss, controls converge).
3. Sweeps: `pcd-jacobi` and `pcd-amgx` × inner budgets × `--restart` ×
   `--maxiter`; rows to `results/solver_lab/rung1.jsonl`.
4. **Pass bar (spec §5):** candidate converges every miss-system at its
   leg tol at wall ≤2× the leg's bdiag per-step time. Escalation trigger
   for convective Fp: inners converge, outer stagnates. Verdict ledgered.

### Task 9 (OPERATOR): Rung 0b — Ahmed render package + sign-off

```bash
.venv/bin/python - <<'EOF'
from diffsim.cases.ahmed import write_ahmed_stl
write_ahmed_stl("results/mesh_renders/ahmed.stl", length=0.06)
EOF
.venv/bin/python tools/render_mesh.py --stl results/mesh_renders/ahmed.stl \
    --out-prefix results/mesh_renders/ahmed_domain --position 0.32,0.003,0.0507
```

plus a tiny-mesh dump + slice renders of the actual Ahmed carve at
production `BAND_TO` (same `render_mesh.py` npz path). **GATE: ledgered
`RUNG-0b SIGN-OFF: ahmed mesh <file> approved (Baskar, <date>)`.**

### Task 10 (OPERATOR): Rung 2 — Ahmed march

`cluster/ahmed_gate.py` on a GH200 hold with the Rung-1-winning config.
Pass bar quoted in the gate's final print (≥500 steps at amp 1.0, zero
post-transient misses, umax < U_CAP/10, finite cd). Verdict ledgered.

### Task 11 (OPERATOR): Rung 3 — truck gate

`cluster/t5_gate.py`, fresh start, sealed mesh, Re=250,
`MONO_SOLVER=fgmres_pcd` + winning inners, `SOFT_START<=300`,
`NSTEPS>=SOFT_START+500`. Same pass bar. Pass ⇒ campaign closed; fail ⇒
ASM(LU) build triggers (spec §8, new spec + plan).

---

## Deferred (recorded, not planned)

- Convective Fp (built only on the Rung-1 stagnation signature — if
  triggered, it gets its own task brief against `saddle_precond.py`).
- ASM(LU) reserve (spec §8; own spec if triggered).
- p=2 band, band-13, Re ladder past 250 (spec §10).
