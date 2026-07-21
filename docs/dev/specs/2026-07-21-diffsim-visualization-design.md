# DiffSim visualization — design spec

**Design spec.** Brainstormed 2026-07-21. A streamlined path from DiffSim's
octree field outputs to shareable/interactive views and publication figures,
calibrated to the group's paper-figure style. Approach 1 (PyVista engine +
VTU exporter + ParaView state templates), chosen for delivering all three
share mechanisms from one data path.

**Priority note:** the *implementation* queues behind P2-R2 per Baskar
(2026-07-21); this spec + its plan are written now, execution deferred.

## 1. Goal & audience

Make DiffSim results easy to (a) **share interactively** with collaborators /
students without a DiffSim install, and (b) turn into **reproducible
publication figures**. Not optimized for the dev inner-loop or for streaming
the 100M-DOF hero (those are non-goals here). Primary share mechanisms:
ParaView Glance web viewer (#2) and desktop-ParaView state files (#3), with a
self-contained HTML file (#1) as a convenience secondary.

## 2. What DiffSim writes today (the starting point)

- Film: `<name>_final.npz` (fields + h + t + 2-D grids), height-milestone
  `<name>_h{...}.npz` snapshots (`src/diffsim/film/run.py`).
- XDD: fields via `xdd/run.py`; J–V curves + traces (`xdd/observables.py`).
- Mesh: adaptive **octree** (not a regular grid) — the key format constraint.
- No VTK/VTU/XDMF/ParaView export and no in-tree 3-D visualization exist.

## 3. Architecture — one exporter, thin layers

```
run / .npz  ──► export_vtu() ──►  .vtu (+ .pvd time series)
                                    │
        ┌───────────────────────────┼───────────────────────────┐
   field renders (PyVista)     share mechanisms          quantitative plots
   LIC / mesh / contours    HTML | Glance | .pvsm         (matplotlib)
```

The **exporter is the foundation**; everything else consumes its VTU. The
module lives at `src/diffsim/viz/`.

## 4. The exporter (`viz/export.py`)

`export_vtu(source, path, *, fields=None, time_series=False) -> Path`
converts an octree field source (a run object or an `.npz`) to a VTK
unstructured grid (`.vtu`; a `.pvd` collection for transient runs). It writes,
as point/cell data, exactly what the group's figures are built on:

- **Physics fields** — velocity (vector), pressure, temperature, composition
  phases, XDD's 5 fields — as named arrays.
- **Element size / refinement level** (cell data) — for the octree-mesh
  figures (matches the group's Fig 19: slices colored by "Element Size").
- **Element-type classification** — interior / true-intercepted / surrogate
  (from `sbm/surrogate.py::classify_lambda`) as a cell label — the SBM
  carve-out story (their Fig 3).
- **Surrogate distance `d`** — the shifted-boundary distance field (their
  Fig 17b uses `d` directly).
- **Embedded-body surface** — the immersed geometry as a companion `.vtp`
  (the white/gray obstacle in the renders).

Format decision: **VTU (unstructured) is primary** — it opens in ParaView
Glance (web, #2), desktop ParaView (#3), and PyVista, so all three share
mechanisms work from one file. `vtkHyperTreeGrid` (octree-native, compact) is
a **Phase-2 optimization** for large octrees on desktop ParaView; vtk.js /
Glance HTG support is less mature, so VTU keeps the web path working today.

## 5. Field renders (`viz/renders.py`, PyVista) — matches Figs 14/19

Scriptable, reproducible helpers that read the VTU and render off-screen
(publication) or export interactive (share). Calibrated to the group's figures:

- `lic(vtu, color_by="velocity_magnitude", ...)` — **surface Line Integral
  Convolution** colored by a scalar (their signature flow figure, Fig 14).
  PyVista/VTK surface-LIC.
- `mesh_slice(vtu, color_by="element_size", normal=..., ...)` — an **octree
  slice colored by element size / refinement level** (their Fig 19), showing
  the refinement shells around the body.
- `contour(vtu, field, ...)` — scalar contours (T, p, composition) with the
  group's colormaps.
- `body(vtp)` — the embedded obstacle surface (white/gray).
- `small_multiples(items, render_fn, layout=...)` — a **grid of cases / time
  instants** (their angle-of-attack and time-instant grids).

Renders take a shared **style** object (colormap, background, camera,
scalar-bar) so figures are consistent and regenerate from a run.

## 6. Quantitative plots (`viz/plots.py`, matplotlib) — matches Figs 15–18

Consistent-style helpers drawing from a run's scalar data (not the VTU):

- `convergence(levels, values, reference=..., ...)` — **log-log Cd / L2-error
  vs h** with a reference line (literally the P2-R0 Task-9 output; their
  spatial-convergence figures).
- `surface_profile(x, series, reference=..., inset=...)` — **Cp / Cf / Nu vs
  arc-length** with literature-comparison overlays and **inset zooms** (their
  Figs 16–18).
- `history(t, series, ...)` — **Cd / Cl time histories** (drag histories,
  shedding).
- `parameter_sweep(param, values, reference=...)` — e.g. Cl/Cd vs AoA with
  literature markers (their Fig 15).

Shared style: the blue-line / red-marker + grid + legend convention seen in
the papers.

## 7. Share mechanisms (`viz/share.py`) — thin layers on the exporter

- **ParaView state templates (#3, primary):** per-physics `.pvsm` generators
  (film / XDD / NS) — a collaborator opens `state.pvsm` + the VTU in desktop
  ParaView and lands on the exact scene (cameras, colormaps, LIC/slice
  filters). Templates parameterized by the run's fields.
- **Glance web (#2, primary):** the exported VTU/VTP drops directly into
  ParaView Glance (browser, no install) — documented as a one-page "how to
  view" note; no code beyond the exporter.
- **Self-contained HTML (#1, secondary):** `export_html(vtu, path)` via
  PyVista/trame → one `.html` a collaborator double-clicks (browser, rotate/
  slice, no install). Best for 2-D + moderate 3-D (embeds the data).

## 8. Testing

- Exporter: round-trip a small octree `.npz` → VTU, assert the fields +
  element-size + element-type + distance arrays are present and correct
  against an independent hand-computed reference on a tiny mesh (the house
  gate-hygiene: independent reference, not self-comparison).
- Renders/plots: smoke-render off-screen to a PNG on a tiny fixture and assert
  non-empty / correct dimensions (they're figure code — pixel-exact gating is
  brittle; assert they run + produce output).
- Share: assert the `.pvsm` template parses and references the written arrays;
  assert `export_html` produces a self-contained file.
- Host-side (no GPU); CI-cheap on tiny fixtures.

## 9. File structure

- `src/diffsim/viz/__init__.py` — public API (`export_vtu`, `lic`,
  `mesh_slice`, `contour`, `convergence`, `surface_profile`, `history`,
  `paraview_state`, `export_html`).
- `src/diffsim/viz/export.py` — the octree→VTU exporter.
- `src/diffsim/viz/renders.py` — PyVista field renders + the `style` object.
- `src/diffsim/viz/plots.py` — matplotlib quantitative plots.
- `src/diffsim/viz/share.py` — `.pvsm` templates + `export_html`.
- `tests/test_viz_export.py`, `tests/test_viz_renders_plots.py`,
  `tests/test_viz_share.py`.
- Dependencies: add `pyvista` (+ `meshio`) to an optional `[viz]` extra;
  matplotlib already present; ParaView stays external (consumed, not a dep).

## 10. Phasing (YAGNI)

- **Phase 1** (the useful core): the VTU exporter + `lic`/`mesh_slice`/
  `contour` renders + `convergence`/`surface_profile`/`history` plots + the
  ParaView `.pvsm` state templates + the Glance "how to view" note. Delivers
  your primary #2/#3 and the figure vocabulary.
- **Phase 2** (polish/scale): `export_html` (self-contained #1),
  `small_multiples` polish, `parameter_sweep`, and `vtkHyperTreeGrid` export
  for large octrees on desktop ParaView.

## 11. Non-goals

- Not a dev inner-loop live viewer (a later, separate concern if wanted).
- Not streaming/LOD/server-rendering the 100M-DOF hero (that's a ParaView-
  server deployment question, out of scope here).
- Not a bespoke web app — we lean on ParaView Glance / PyVista-HTML rather
  than building a viewer.
