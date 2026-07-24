# Tutorial/Course Visualization Rollout — Design Spec (Phase 1)

**Date:** 2026-07-24
**Status:** APPROVED. Phased, high-value subset, optional/guarded.
**Depends on:** `diffsim.viz` (module on master @ `26a57c8`: `export`, `plots`,
`renders`, `share`; optional `[viz]` extra = matplotlib + PyVista).

## Goal
Wire `diffsim.viz` into the highest-value teaching materials so tutorials emit
figures (fields, convergence, morphology, flow, SBM boundaries), **without adding
a hard dependency** — every tutorial still runs and prints its numeric results on
a base install; figures are additive when `[viz]` is present.

## Principles
- **Optional/guarded**: a single shared helper wraps `diffsim.viz` in a
  `try/except`. No tutorial imports viz directly. Base install → tutorial runs,
  prints a one-line install hint instead of a figure. Never crashes.
- **Right viz per family**: matplotlib (CPU-friendly) for convergence / scaling /
  phase-field snapshots; PyVista + VTU (GL-guarded) for flow / SBM / 3-D fields.
- **Tests lean on the deterministic path**: VTU/VTP export is numpy/meshio →
  CPU-deterministic, test it. Renders need GL → guarded smoke only (no pixel
  asserts). Plus a no-`[viz]` smoke test that each touched tutorial still runs.

## The shared helper — `tutorials/_viz.py`
A thin, guarded facade over `diffsim.viz`. Interface (implementer finalizes
signatures against the real viz API — `plots.convergence/surface_profile/history`,
`renders.lic/mesh_slice/contour/offscreen_gl_ok/Style`,
`export.export_vtu/export_vtu_sbm/export_body_vtp`, `share.paraview_state`):

- `HAS_VIZ: bool` — True iff `diffsim.viz` imported.
- `figures_dir(tutorial_file) -> Path` — `<tutorial dir>/figures/`, created on demand.
- Guarded emitters that no-op-with-hint when `HAS_VIZ` is False and otherwise
  dispatch to the viz fn + save into `figures_dir`, printing `"[viz] wrote <path>"`:
  `convergence(...)`, `history(...)`, `surface_profile(...)`, `field(...)`
  (contour/mesh_slice render, GL-guarded via `renders.offscreen_gl_ok()`),
  `vtu(...)` / `vtu_sbm(...)` / `body_vtp(...)`, `paraview_state(...)`.
- GL emitters must additionally guard on `renders.offscreen_gl_ok()` and
  fall back to the hint (never raise) when GL is unavailable.

## Phase-1 targets (~13)
| Family | Tutorials | Figures |
|---|---|---|
| A_foundations | A1_mms_convergence | log-log error-vs-h convergence plot |
| D_flow | D1_ns_mms, D2_lid_driven_cavity, D3_cylinder | velocity/vorticity field (VTU + render), centerline profile, drag/lift history |
| F_phasefield | F1_allen_cahn, F2_cahn_hilliard, F3_adaptivity | morphology field snapshots (VTU + contour) |
| E_differentiable | E1_shape_optimization | objective-vs-iteration history + geometry snapshots |
| P_performance | P1_cost_model_and_scaling, P2_solver_showdown | scaling / solver-showdown curves |
| ns-sbm-course | 01_monolithic_vms, 02_pressure_projection, 04_shifted_boundary, 05_three_dimensions | field renders + SBM surrogate/true boundary + VTU + ParaView state; figures land where the course LaTeX `\includegraphics` expects them |

## Non-goals (Phase 1)
- The remaining tutorials (B_nonlinear, C_time, A2–A6, E0*, orgelmorph-course) —
  deferred to a later phase.
- No pixel-level render assertions. No new required dependency.
- No changes to `diffsim.viz` itself (consume it as-is; if a genuine gap is found,
  note it, don't expand scope).

## Success criteria
- Each touched tutorial runs to completion on the **base (no-`[viz]`) venv**,
  prints its numeric results + a `[viz]` install hint, exit 0.
- With `[viz]` installed, each emits its figure(s) into `figures/` (or the course
  LaTeX figure dir).
- Export (VTU/VTP) tests pass on CPU; render tests are GL-guarded and skip cleanly
  without GL; the no-`[viz]` smoke test passes with viz uninstalled/monkeypatched
  unavailable.
