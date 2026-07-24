# Tutorial/Course Visualization — Implementation Plan (Phase 1)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
> Spec: `docs/superpowers/specs/2026-07-24-tutorial-viz-design.md`.

**Goal:** Wire `diffsim.viz` into ~13 high-value tutorials/course-modules, additive
and optional/guarded — base installs still run and print results; figures appear
when the `[viz]` extra is installed.

**Architecture:** One shared guarded helper `tutorials/_viz.py` wraps `diffsim.viz`
in try/except; tutorials call the helper's emitters (never import viz directly).
matplotlib for convergence/scaling/phase-field; PyVista+VTU (GL-guarded) for
flow/SBM/3-D.

**Tech:** diffsim.viz (`export`/`plots`/`renders`/`share`), the `.venv` CPU python
(`/Users/baskarg/DiffSim/.venv/bin/python`), pytest.

## Global Constraints
- **No hard dependency**: no tutorial imports `diffsim.viz` directly; all viz goes
  through `tutorials/_viz.py`, which degrades to a printed hint when `[viz]` absent.
- Every touched tutorial must **run to completion + exit 0 on the base venv**
  (viz uninstalled), printing its numeric results unchanged.
- **Renders are GL-guarded** via `renders.offscreen_gl_ok()`; never raise without GL.
- **Tests**: export (VTU/VTP) tested on CPU (deterministic); renders GL-guarded
  smoke only (no pixel asserts); one no-`[viz]` smoke test per touched family.
- **Do NOT modify `src/diffsim/viz/`** — consume as-is.
- Figures → `<tutorial dir>/figures/`; for ns-sbm-course, where the LaTeX expects them.

---

### Task 1: Shared guarded helper + foundation tests
**Files:** Create `tutorials/_viz.py`; Create `tests/test_tutorial_viz.py`.
**Interfaces (Produces):** `HAS_VIZ`, `figures_dir(file)->Path`, and guarded
emitters `convergence`, `history`, `surface_profile`, `field`, `vtu`, `vtu_sbm`,
`body_vtp`, `paraview_state` — each no-ops with a one-line install hint when
`HAS_VIZ` is False (and, for renders, when `offscreen_gl_ok()` is False), else
dispatches to the corresponding `diffsim.viz` fn and saves under `figures_dir`.

- Read the real signatures in `src/diffsim/viz/{plots,renders,export,share}.py`
  and bind the emitters to them exactly.
- Test: (a) with viz importable, `vtu(...)` on a tiny mesh writes a parseable file
  (CPU, deterministic); (b) monkeypatch `HAS_VIZ=False` → every emitter no-ops,
  returns None, prints a hint, raises nothing; (c) `figures_dir` creates the dir.
- Run: `.venv/bin/python -m pytest tests/test_tutorial_viz.py -q` → PASS. Commit.

### Task 2: Wire the .py tutorial track
**Files:** Modify `tutorials/A_foundations/A1_mms_convergence.py`,
`tutorials/D_flow/{D1_ns_mms,D2_lid_driven_cavity,D3_cylinder}.py`,
`tutorials/F_phasefield/{F1_allen_cahn,F2_cahn_hilliard,F3_adaptivity}.py`,
`tutorials/E_differentiable/E1_shape_optimization.py`,
`tutorials/P_performance/{P1_cost_model_and_scaling,P2_solver_showdown}.py`;
Create `tests/test_tutorial_viz_track.py`.
**Consumes:** `tutorials/_viz.py` (Task 1).

- For each tutorial: keep all existing prints/behavior; near the end (after the
  results exist) add guarded `_viz.*` calls for that family's figure(s) per the
  spec table (convergence→`convergence`; flow→`vtu`+`field`+`history`;
  phase-field→`vtu`+`field`; shape-opt→`history`+geometry `vtu`; performance→
  `convergence`/`history`-style scaling curves). Import `tutorials._viz as _viz`
  (add `tutorials/__init__.py` if needed for import).
- Test (`tests/test_tutorial_viz_track.py`): each tutorial's `main()` (or a small
  driver) runs to completion with viz FORCED-unavailable (monkeypatch
  `_viz.HAS_VIZ=False`) at a tiny problem size — exit 0, no raise. Keep sizes small.
- Run the new test on the base venv → PASS. Commit.

### Task 3: Wire the ns-sbm-course modules + LaTeX figures
**Files:** Modify the runnable scripts under `tutorials/ns-sbm-course/{01_monolithic_vms,
02_pressure_projection,04_shifted_boundary,05_three_dimensions}/`; add figure
outputs where the course LaTeX includes them (inspect the module + latex dir);
Create/extend a course viz smoke test.
**Consumes:** `tutorials/_viz.py` (Task 1).

- Add guarded field renders + SBM surrogate/true-boundary + VTU + a `paraview_state`
  per module, saving into the module's figure dir referenced by the course LaTeX.
- If a LaTeX chapter should `\includegraphics` the new figure, wire the include
  (guarded so the PDF still builds if the figure is absent — e.g. commit a
  placeholder or use `\IfFileExists`).
- Test: each touched module script runs to completion with viz forced-unavailable
  (exit 0). Run on base venv → PASS. Commit.

---

## Self-review checklist (per task)
- No direct `import diffsim.viz` in any tutorial (only via `tutorials/_viz`).
- Base-venv run of every touched file exits 0 and prints results.
- Export tested on CPU; renders GL-guarded; no pixel asserts; no viz-module edits.
