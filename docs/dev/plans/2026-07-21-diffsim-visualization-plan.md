# DiffSim Visualization — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking. Commit-per-green; **agents never
> push**. Spec (source of truth):
> `docs/dev/specs/2026-07-21-diffsim-visualization-design.md`. Real code
> entry-points: `src/diffsim/film/run.py` (npz fields), `src/diffsim/xdd/run.py`
> + `src/diffsim/xdd/observables.py` (XDD), `src/diffsim/octree/build.py`
> (Octree), `src/diffsim/mesh/nodes.py` (Mesh), `src/diffsim/sbm/surrogate.py`
> (`classify_lambda`, `GeometryData`).

**Goal:** a DiffSim Phase 1 visualization module (`src/diffsim/viz/`) that reads
the solver's existing `.npz` outputs and `Mesh`/`Octree` objects, exports them
to VTK Unstructured Grid files (`.vtu` / `.pvd`), drives PyVista off-screen
renders matched to the group's paper figures (LIC, octree-slice, contour), plots
quantitative scalar histories and profiles with matplotlib, and generates
ParaView state-file (`.pvsm`) templates so collaborators land on the correct
scene in desktop ParaView. Phase 2 items (HTML export, `small_multiples`,
`parameter_sweep`, `vtkHyperTreeGrid`) are explicitly **out of scope here**.

**Priority note:** IMPLEMENTATION IS DEFERRED behind P2-R2. This plan is written
now; execution begins after P2-R2 merges (Baskar, 2026-07-21).

## Header

### Architecture

```
run / .npz  ──► export_vtu() ──►  .vtu + .pvd
                                    │
        ┌───────────────────────────┼─────────────────────────────┐
   viz/renders.py              viz/share.py                viz/plots.py
   lic / mesh_slice /        .pvsm templates             matplotlib
   contour (PyVista)         + Glance note               convergence /
                                                          surface_profile /
                                                          history
```

Single data path: exporter runs once; renders, share-mechanisms, and plots all
consume its output. Module lives at `src/diffsim/viz/`.

### Tech Stack

- **pyvista** (+ **meshio**) in an optional `[viz]` extra in `pyproject.toml`.
  Never import at top-level; guard all viz imports inside functions so
  `import diffsim` never fails without `[viz]` installed.
- **matplotlib** (already present; no extra).
- **numpy / scipy** (core, always available).
- ParaView stays external (generated `.pvsm` is consumed by desktop ParaView,
  not imported).
- Host-only; CI-cheap on tiny fixtures — no GPU, no warp, no torch in `viz/`.

### Global Constraints

1. **`[viz]` extra:** `pyvista` and `meshio` live behind
   `[project.optional-dependencies] viz = ["pyvista>=0.44", "meshio>=5.3"]`
   in `pyproject.toml`. All `import pyvista` / `import meshio` calls are inside
   function bodies (not module-level), so `import diffsim` never fails without
   `[viz]`.
2. **Host-side / CI-cheap:** test fixtures are tiny (level-3 or smaller uniform
   octrees, < 100 elements); no GPU; no warp kernel compilation; off-screen
   rendering only (PyVista `off_screen=True`).
3. **Gate hygiene — independent reference:** exporter tests assert field arrays
   against hand-computed reference values on a tiny mesh, NOT against the
   exporter's own output (self-comparison is not a gate). See Task 1.
4. **REPO-IDENTITY GUARD:** every commit begins with
   `assert git rev-parse --show-toplevel` equals the worktree path; agent does
   not git-push.
5. **Agents never push.** Commits land on the worktree branch
   (`viz-plan` → ultimately `master` after supervisor review).
6. **Spec §10 Phase 1 only.** `export_html`, `small_multiples`,
   `parameter_sweep`, and `vtkHyperTreeGrid` are Phase 2; no stubs, no
   placeholders.

---

## Task 1 — Octree-to-VTU exporter: physics fields

**Spec reference:** §4, §8, §9.

### Files

- **Create** `src/diffsim/viz/__init__.py`
- **Create** `src/diffsim/viz/export.py`
- **Create** `tests/test_viz_export.py`
- **Modify** `pyproject.toml` (add `[viz]` optional extra)

### Interface

```python
# src/diffsim/viz/export.py

def export_vtu(
    source,          # Mesh object (from mesh/nodes.py) or path to .npz
    path: str | os.PathLike,
    *,
    fields: dict | None = None,   # {name: nodal_array} overrides; None = infer
    time_series: bool = False,    # write .pvd collection for transient snapshots
) -> pathlib.Path:
    """Convert a DiffSim field source to a VTK Unstructured Grid (.vtu).

    Source types:
      - Mesh object + fields dict: the caller already has the Mesh and nodal
        arrays (e.g. from a live FilmRun). Pass mesh as source, fields as a
        dict of {name: ndarray(Nn,) or (Nn, 3)}.
      - Path string/Path to a .npz file: loaded and unpacked automatically
        (see npz field conventions below).

    Writes point data (nodal arrays) and cell data (element-size, level).
    Returns the written .vtu path.
    """
```

```python
# src/diffsim/viz/__init__.py
from .export import export_vtu
```

### NPZ field conventions (from `film/run.py::_Recorder.write_final_npz`)

`<name>_final.npz` carries: `phi_p` (Nn,), `phi_f` (Nn,), `coords` (Nn, dim),
`h` (scalar), `t` (scalar), and optionally `grid_p` (ny, nx) / `grid_f` for 2-D
structured views. Snapshot `.npz` files carry: `phi_p`, `phi_f`, `coords` (3-D)
or `phi_p`/`phi_f` grids (2-D), `h`, `t`.

The exporter maps:
- `phi_p`, `phi_f` → point data arrays `"phi_p"`, `"phi_f"`
- `coords` → node coordinates
- Synthetic `phi_s = 1 - phi_p - phi_f` → point data `"phi_s"`
- Element size `h_e = mesh.tree.h()` → cell data `"element_size"`
- Refinement level `lev = mesh.tree.levels` → cell data `"level"`

Cell topology: each element is a VTK `Quad` (dim=2) or `Hexahedron` (dim=3)
with corner nodes only. Mixed-p meshes have p2 elements too — the plan maps p2
elements to VTK `BiQuadratic_Quad` (dim=2, VTK type 28) or `Triquadratic_Hexahedron`
(dim=3, VTK type 29). Node connectivity is read from `mesh.conn_of[p]` for each
`p` in `mesh.bins`.

**VTK cell-type lookup:**
```python
VTK_CELL_TYPE = {
    (2, 1): 9,   # Quad (4 nodes)
    (2, 2): 28,  # BiQuad_Quad (9 nodes)
    (3, 1): 12,  # Hexahedron (8 nodes)
    (3, 2): 29,  # TriquadHex (27 nodes)
}
```

### Steps

- [ ] 1. Add `[viz]` optional extra to `pyproject.toml`:
  ```toml
  [project.optional-dependencies]
  dev = ["pytest>=8.0", "pytest-xdist>=3.5", "ruff>=0.6"]
  viz = ["pyvista>=0.44", "meshio>=5.3"]
  ```

- [ ] 2. Write `tests/test_viz_export.py` — FAILING test:
  ```python
  """Gate: round-trip a tiny level-2 uniform mesh to .vtu; check fields."""
  import numpy as np
  import pytest
  import pathlib

  pytest.importorskip("pyvista")
  pytest.importorskip("meshio")

  from diffsim.octree.build import build_uniform
  from diffsim.mesh.nodes import build_mesh
  from diffsim.viz import export_vtu


  def _tiny_mesh():
      tree = build_uniform(2, dim=3)
      mesh = build_mesh(tree, p=1)
      return mesh


  def test_export_vtu_creates_file(tmp_path):
      mesh = _tiny_mesh()
      Nn = len(mesh.node_coords)
      rng = np.random.default_rng(0)
      fields = {
          "phi_p": rng.random(Nn),
          "phi_f": rng.random(Nn),
      }
      out = export_vtu(mesh, tmp_path / "test.vtu", fields=fields)
      assert out.exists()
      assert out.stat().st_size > 0


  def test_export_vtu_fields_present(tmp_path):
      """Independent-reference check: read the .vtu back with meshio and
      verify the arrays match the inputs exactly (NOT via the exporter's
      own arrays -- this is the house gate-hygiene rule)."""
      import meshio

      mesh = _tiny_mesh()
      Nn = len(mesh.node_coords)
      phi_p_ref = np.linspace(0.0, 1.0, Nn)
      phi_f_ref = np.linspace(0.1, 0.9, Nn)

      export_vtu(mesh, tmp_path / "ref.vtu",
                 fields={"phi_p": phi_p_ref, "phi_f": phi_f_ref})

      m = meshio.read(str(tmp_path / "ref.vtu"))
      # point data
      np.testing.assert_allclose(m.point_data["phi_p"], phi_p_ref, atol=1e-12)
      np.testing.assert_allclose(m.point_data["phi_f"], phi_f_ref, atol=1e-12)
      np.testing.assert_allclose(
          m.point_data["phi_s"], 1.0 - phi_p_ref - phi_f_ref, atol=1e-12)
      # cell data must be present
      assert "element_size" in m.cell_data
      assert "level" in m.cell_data
      # coordinate round-trip: meshio returns point coordinates
      np.testing.assert_allclose(
          m.points, mesh.node_coords, atol=1e-12)


  def test_export_vtu_cell_count(tmp_path):
      """Number of cells in the written file equals number of octree elements."""
      import meshio
      mesh = _tiny_mesh()
      Nn = len(mesh.node_coords)
      export_vtu(mesh, tmp_path / "cells.vtu",
                 fields={"phi_p": np.zeros(Nn), "phi_f": np.zeros(Nn)})
      m = meshio.read(str(tmp_path / "cells.vtu"))
      total_cells = sum(len(cb.data) for cb in m.cells)
      assert total_cells == len(mesh.tree)


  def test_export_npz_roundtrip(tmp_path):
      """Pass a .npz path as source; verify fields are written."""
      import meshio
      mesh = _tiny_mesh()
      Nn = len(mesh.node_coords)
      phi_p = np.linspace(0.0, 1.0, Nn)
      phi_f = np.linspace(0.1, 0.9, Nn)
      npz_path = tmp_path / "snap.npz"
      np.savez(npz_path, phi_p=phi_p, phi_f=phi_f,
               coords=mesh.node_coords, h=0.9, t=1.0)
      out = export_vtu(npz_path, tmp_path / "snap.vtu")
      m = meshio.read(str(out))
      np.testing.assert_allclose(m.point_data["phi_p"], phi_p, atol=1e-12)
  ```
  Run: `pytest tests/test_viz_export.py -x` → must fail with `ImportError` or `ModuleNotFoundError`.

- [ ] 3. Create `src/diffsim/viz/__init__.py` (empty scaffold):
  ```python
  """DiffSim visualization module — Phase 1.

  Optional extra: pip install diffsim[viz]
  Requires: pyvista>=0.44, meshio>=5.3
  """
  from .export import export_vtu

  __all__ = ["export_vtu"]
  ```

- [ ] 4. Implement `src/diffsim/viz/export.py`:
  ```python
  """Octree-mesh → VTK Unstructured Grid exporter (Phase 1, spec §4)."""
  import os
  import pathlib
  from typing import Union

  import numpy as np


  # VTK cell types for (dim, p) pairs:
  #   Quad (4), BiQuadQuad (9), Hex (8), TriquadHex (27)
  _VTK_TYPE = {(2, 1): 9, (2, 2): 28, (3, 1): 12, (3, 2): 29}


  def export_vtu(
      source,
      path: Union[str, os.PathLike],
      *,
      fields: dict = None,
      time_series: bool = False,
  ) -> pathlib.Path:
      try:
          import meshio
      except ImportError as e:
          raise ImportError(
              "export_vtu requires 'meshio'. Install with: pip install diffsim[viz]"
          ) from e

      path = pathlib.Path(path)

      # ── Resolve source ──────────────────────────────────────────────────────
      if isinstance(source, (str, os.PathLike)):
          from ..mesh.nodes import Mesh
          npz = np.load(str(source))
          coords = npz["coords"]
          loaded_fields = {k: npz[k] for k in npz.files
                          if k not in ("coords", "h", "t")}
          # Reconstruct a minimal mesh from coordinates only.
          # For .npz-only source we cannot recover the Octree connectivity;
          # the caller should pass a Mesh object for full cell data.
          # Emit structured-only when conn is not available.
          _write_meshio_npz_only(coords, loaded_fields, path, meshio)
          return path

      # source is a Mesh object
      mesh = source
      dim = mesh.dim
      coords = mesh.node_coords          # float64 [Nn, dim]
      h_elem = mesh.tree.h()             # [Ne]
      levels = mesh.tree.levels          # uint8 [Ne]

      # ── Build per-p cell blocks ─────────────────────────────────────────────
      cells = []
      element_size_cells = []
      level_cells = []

      for pv in sorted(mesh.bins.keys()):
          eids = mesh.bins[pv]           # element indices for this p
          conn = mesh.conn_of[pv]        # int32 [nb, (p+1)^dim]
          vtk_type = _VTK_TYPE[(dim, int(pv))]
          cells.append((vtk_type, conn))
          element_size_cells.append(h_elem[eids])
          level_cells.append(levels[eids].astype(np.int32))

      # ── Point data ─────────────────────────────────────────────────────────
      point_data = {}
      if fields is not None:
          for k, v in fields.items():
              point_data[k] = np.asarray(v, dtype=np.float64)
          if "phi_p" in fields and "phi_f" in fields:
              point_data["phi_s"] = (
                  1.0 - point_data["phi_p"] - point_data["phi_f"])

      # ── Cell data ───────────────────────────────────────────────────────────
      # meshio expects one array per cell-block in a list
      cell_data = {
          "element_size": element_size_cells,
          "level": level_cells,
      }

      # ── Write .vtu ─────────────────────────────────────────────────────────
      m = meshio.Mesh(
          points=coords,
          cells=cells,
          point_data=point_data,
          cell_data=cell_data,
      )
      path.parent.mkdir(parents=True, exist_ok=True)
      meshio.write(str(path), m)

      if time_series:
          _write_pvd(path, [path])

      return path


  def _write_meshio_npz_only(coords, loaded_fields, path, meshio):
      """Fallback: write point-cloud only when no Mesh connectivity is available."""
      verts = np.arange(len(coords)).reshape(-1, 1)
      pdata = {k: np.asarray(v, dtype=np.float64) for k, v in loaded_fields.items()}
      if "phi_p" in pdata and "phi_f" in pdata:
          pdata["phi_s"] = 1.0 - pdata["phi_p"] - pdata["phi_f"]
      m = meshio.Mesh(points=coords, cells=[("vertex", verts)], point_data=pdata)
      path.parent.mkdir(parents=True, exist_ok=True)
      meshio.write(str(path), m)


  def _write_pvd(vtu_path: pathlib.Path, files: list) -> pathlib.Path:
      pvd = vtu_path.with_suffix(".pvd")
      lines = ['<?xml version="1.0"?>',
               '<VTKFile type="Collection" version="0.1">',
               '  <Collection>']
      for i, f in enumerate(files):
          lines.append(f'    <DataSet timestep="{i}" file="{pathlib.Path(f).name}"/>')
      lines += ['  </Collection>', '</VTKFile>']
      pvd.write_text("\n".join(lines))
      return pvd
  ```

- [ ] 5. Run: `pytest tests/test_viz_export.py -x` → all four tests pass.

- [ ] 6. Run: `ruff check src/diffsim/viz/ tests/test_viz_export.py` → zero warnings.

- [ ] 7. Commit:
  ```
  feat(viz): Task 1 — octree-to-VTU exporter with physics fields

  Adds src/diffsim/viz/ module (export_vtu), pyproject.toml [viz] optional
  extra, and tests/test_viz_export.py with four independent-reference gates.

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  ```

---

## Task 2 — Exporter: DiffSim-specific cell data (element-type, surrogate distance, embedded-body .vtp)

**Spec reference:** §4 (element-type classification, surrogate distance d, embedded-body surface), §8, §9.

### Files

- **Modify** `src/diffsim/viz/export.py` (add `export_vtu_sbm`, `export_body_vtp`)
- **Modify** `src/diffsim/viz/__init__.py`
- **Modify** `tests/test_viz_export.py` (add three new tests)

### Interface

```python
# src/diffsim/viz/export.py additions

def export_vtu_sbm(
    mesh,                       # Mesh object
    path: str | os.PathLike,
    *,
    fields: dict | None = None,
    retained_tree,              # Octree returned by classify_lambda
    frac: np.ndarray,           # [Nret] domain-inside fraction per retained element
    geom: "GeometryData | None" = None,  # surrogate GP geometry cache
    sf: "SurrogateFaces | None" = None,  # surrogate face list
) -> pathlib.Path:
    """export_vtu extended with SBM cell data.

    Adds three cell-data arrays to the VTU written by export_vtu:
      element_type  int32 [Ne]: 0=interior, 1=intercepted, 2=surrogate
                   (interior: frac==1.0; intercepted: 0<frac<1.0;
                    surrogate: on the surrogate face boundary, i.e. elem in sf.elem)
      frac_in      float64 [Ne]: domain-inside fraction from classify_lambda
      d_mean       float64 [Ne]: mean ||d|| per element (from GeometryData.d
                   averaged over surrogate GPs on that element; 0 elsewhere)
    """


def export_body_vtp(
    vertices: np.ndarray,       # float64 [Nv, 3] surface vertex positions
    triangles: np.ndarray,      # int32 [Nt, 3] triangle connectivity
    path: str | os.PathLike,
) -> pathlib.Path:
    """Write the embedded-body surface as a VTK PolyData (.vtp) file."""
```

### Element-type classification logic

```python
# element_type derivation (inside export_vtu_sbm):
elem_type = np.zeros(len(retained_tree), dtype=np.int32)
# retained elements with frac < 1.0 are intercepted
elem_type[frac < 1.0] = 1
# elements whose index appears in sf.elem are surrogate-boundary elements
if sf is not None:
    elem_type[np.unique(sf.elem)] = 2
```

`d_mean` per element: for each element `e` in `sf.elem`, find all GP indices
`gp_idx` where `sf.elem == e`, compute `np.mean(np.linalg.norm(geom.d[gp_idx], axis=1))`.
All other elements get 0.0.

### Tests

- [ ] Test `test_export_vtu_sbm_cell_arrays`: build a tiny level-2 3-D tree,
  call `classify_lambda` with a sphere oracle, call `extract_surrogate`, call
  `export_vtu_sbm`, read back with meshio, assert `element_type`, `frac_in`,
  and `d_mean` arrays are present and have correct shape `(Ne,)` per cell block.
  Assert that elements with `frac==1.0` have `element_type==0`.

- [ ] Test `test_export_body_vtp_roundtrip`: write a 4-vertex 2-triangle surface,
  read back with meshio, assert point coordinates and triangle connectivity match.

- [ ] Test `test_element_type_values`: on a tiny fixture, verify that an element
  known to be in `sf.elem` has `element_type==2` and that a fully-interior
  element has `element_type==0`.

### Steps

- [ ] 1. Write failing tests (three tests above) in `tests/test_viz_export.py`.

- [ ] 2. Implement `export_vtu_sbm` in `export.py`:
  - Compute `elem_type`, `frac_in`, `d_mean` as described above.
  - Call `export_vtu(mesh, path, fields=fields)` first to write the base VTU,
    then re-read with meshio, add the three cell-data arrays, re-write.
    (Avoids duplicating the exporter logic.)

- [ ] 3. Implement `export_body_vtp` using meshio with cell type `"triangle"`.

- [ ] 4. Expose both in `__init__.py`:
  ```python
  from .export import export_vtu, export_vtu_sbm, export_body_vtp
  __all__ = ["export_vtu", "export_vtu_sbm", "export_body_vtp"]
  ```

- [ ] 5. Run: `pytest tests/test_viz_export.py -x` → all tests pass.

- [ ] 6. Run: `ruff check src/diffsim/viz/ tests/test_viz_export.py`

- [ ] 7. Commit:
  ```
  feat(viz): Task 2 — SBM cell data (element-type, frac_in, d_mean) + body .vtp

  export_vtu_sbm adds element_type/frac_in/d_mean cell arrays (spec §4);
  export_body_vtp writes the embedded-body surface as .vtp.

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  ```

---

## Task 3 — PyVista field renders: `lic`, `mesh_slice`, `contour` + shared `Style`

**Spec reference:** §5, §8, §9.

### Files

- **Create** `src/diffsim/viz/renders.py`
- **Modify** `src/diffsim/viz/__init__.py`
- **Create** `tests/test_viz_renders_plots.py`

### Interface

```python
# src/diffsim/viz/renders.py

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class Style:
    """Shared style object: colormap, background, camera, scalar-bar.
    Defaults match the group's paper-figure convention (white background,
    viridis for scalars, jet suppressed)."""
    background: str = "white"
    cmap: str = "RdBu_r"         # default colormap
    scalar_bar: bool = True
    window_size: Tuple[int, int] = (1200, 800)
    camera_position: str | list = "xy"  # pyvista camera preset or list


def lic(
    vtu: str | os.PathLike,
    color_by: str = "velocity_magnitude",
    *,
    style: Style | None = None,
    output: str | os.PathLike | None = None,  # None → return plotter only
    off_screen: bool = True,
) -> "pyvista.Plotter":
    """Surface Line Integral Convolution colored by a scalar (spec §5, Fig 14).

    Reads the .vtu, applies PyVista surface LIC on the first surface/slice,
    colors by `color_by` scalar. Returns the plotter (call .screenshot() to save).
    If `output` is given, saves a PNG and returns the plotter.
    """


def mesh_slice(
    vtu: str | os.PathLike,
    color_by: str = "element_size",
    normal: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    origin: Tuple[float, float, float] | None = None,
    *,
    style: Style | None = None,
    output: str | os.PathLike | None = None,
    off_screen: bool = True,
) -> "pyvista.Plotter":
    """Octree slice colored by element_size / level (spec §5, Fig 19).

    Slices the .vtu at the given plane, colors by `color_by`. Effective for
    showing the refinement shells around an immersed body.
    """


def contour(
    vtu: str | os.PathLike,
    field: str,
    isovalues: list | int = 10,
    *,
    style: Style | None = None,
    output: str | os.PathLike | None = None,
    off_screen: bool = True,
) -> "pyvista.Plotter":
    """Scalar contours (T, p, composition) with the group's colormap (spec §5)."""
```

### Tests

Tests assert that renders **run and produce non-empty output of correct
dimensions** — pixel-exact gating is brittle (spec §8 rule). Do not check
pixel values or colorbar positions.

```python
# tests/test_viz_renders_plots.py (render section)

pytest.importorskip("pyvista")

def _write_test_vtu(tmp_path):
    """Build a tiny VTU with element_size cell data and phi_p point data."""
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.viz import export_vtu
    tree = build_uniform(2, dim=3)
    mesh = build_mesh(tree, p=1)
    Nn = len(mesh.node_coords)
    vtu = tmp_path / "tiny.vtu"
    export_vtu(mesh, vtu, fields={
        "phi_p": np.linspace(0.0, 1.0, Nn),
        "phi_f": np.linspace(0.1, 0.9, Nn),
    })
    return vtu


def test_mesh_slice_runs(tmp_path):
    vtu = _write_test_vtu(tmp_path)
    from diffsim.viz.renders import mesh_slice, Style
    out = tmp_path / "slice.png"
    pl = mesh_slice(vtu, color_by="element_size", output=out,
                    style=Style(window_size=(400, 300)))
    assert out.exists()
    assert out.stat().st_size > 0
    img = np.array(pl.screenshot(return_img=True))
    assert img.shape[0] == 300 and img.shape[1] == 400
    assert img.shape[2] == 3


def test_contour_runs(tmp_path):
    vtu = _write_test_vtu(tmp_path)
    from diffsim.viz.renders import contour, Style
    out = tmp_path / "contour.png"
    pl = contour(vtu, field="phi_p", output=out,
                 style=Style(window_size=(400, 300)))
    assert out.exists()
    assert out.stat().st_size > 0


def test_lic_runs(tmp_path):
    """LIC: build a VTU with a vector field; assert PNG produced."""
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.viz import export_vtu
    from diffsim.viz.renders import lic, Style
    tree = build_uniform(2, dim=3)
    mesh = build_mesh(tree, p=1)
    Nn = len(mesh.node_coords)
    # Synthetic velocity vector field
    velocity = np.column_stack([
        np.sin(mesh.node_coords[:, 0] * np.pi),
        np.cos(mesh.node_coords[:, 1] * np.pi),
        np.zeros(Nn),
    ])
    vtu = tmp_path / "vel.vtu"
    export_vtu(mesh, vtu, fields={
        "velocity": velocity,
        "velocity_magnitude": np.linalg.norm(velocity, axis=1),
    })
    out = tmp_path / "lic.png"
    pl = lic(vtu, color_by="velocity_magnitude", output=out,
             style=Style(window_size=(400, 300)))
    assert out.exists()
    assert out.stat().st_size > 0
```

### Steps

- [ ] 1. Write failing tests in `tests/test_viz_renders_plots.py`.

- [ ] 2. Implement `src/diffsim/viz/renders.py`:
  - `Style` dataclass first.
  - `mesh_slice`: read VTU with `pyvista.read(str(vtu))`, call
    `.slice(normal=normal, origin=origin)`, color by `color_by`,
    optionally call `pl.screenshot(str(output))`.
  - `contour`: read VTU, call `.contour(isovalues, scalars=field)`,
    render with `style.cmap`.
  - `lic`: read VTU, extract surface / 2-D slice, apply
    `pyvista.LICFilter` or `plotter.add_mesh_clip_plane` with
    surface LIC texture; if PyVista < 0.44 LIC API not available,
    fall back to a stream-tracer visualization and log a warning.
  - All three functions: guard `import pyvista` inside the function body;
    raise `ImportError` with install hint if absent.

- [ ] 3. Expose in `__init__.py`:
  ```python
  from .renders import Style, lic, mesh_slice, contour
  __all__ = ["export_vtu", "export_vtu_sbm", "export_body_vtp",
             "Style", "lic", "mesh_slice", "contour"]
  ```

- [ ] 4. Run: `pytest tests/test_viz_renders_plots.py -x -k "mesh_slice or contour or lic"` → pass.

- [ ] 5. Run: `ruff check src/diffsim/viz/renders.py tests/test_viz_renders_plots.py`

- [ ] 6. Commit:
  ```
  feat(viz): Task 3 — PyVista renders (lic, mesh_slice, contour) + Style

  Adds viz/renders.py: Style dataclass + three figure-matched render helpers
  (spec §5). Tests assert runs + non-empty PNG of correct dimensions.

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  ```

---

## Task 4 — Matplotlib quantitative plots: `convergence`, `surface_profile`, `history`

**Spec reference:** §6, §8, §9.

### Files

- **Create** `src/diffsim/viz/plots.py`
- **Modify** `src/diffsim/viz/__init__.py`
- **Modify** `tests/test_viz_renders_plots.py` (add plot tests)

### Interface

```python
# src/diffsim/viz/plots.py

import numpy as np
import matplotlib


def convergence(
    levels: np.ndarray | list,     # refinement levels h ~ 2**-level
    values: np.ndarray | list,     # quantity (error, Cd, etc.) at each level
    reference: float | None = None,  # reference value for error = |v - ref|
    *,
    slope: float | None = None,    # draw a reference slope line (e.g. 2.0 for O(h^2))
    xlabel: str = "h",
    ylabel: str = "Error",
    label: str = "DiffSim",
    style_kw: dict | None = None,  # passed to matplotlib
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.axes.Axes":
    """Log-log Cd / L2-error vs h with a reference slope line (spec §6, Fig spatial-conv).

    levels: list of integer refinement levels (h = 2**-level); values: the
    scalar metric at each level.  If reference is given, plots |value - reference|.
    Returns the Axes.
    """


def surface_profile(
    x: np.ndarray,
    series: dict,              # {label: ndarray(same shape as x)} — main curves
    reference: dict | None = None,  # {label: (x_ref, y_ref)} — lit overlays
    *,
    inset: dict | None = None, # {"xlim": (a,b), "ylim": (a,b)} — inset zoom
    xlabel: str = "arc-length",
    ylabel: str = "Cp",
    style_kw: dict | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.axes.Axes":
    """Cp / Cf / Nu vs arc-length with literature overlays + inset zoom (spec §6, Figs 16–18).

    series: a dict of {label: y_array} curves plotted against x.
    reference: optional dict of {label: (x_ref, y_ref)} arrays from literature.
    inset: if given, adds an inset axes zoomed to xlim/ylim.
    Returns the main Axes.
    """


def history(
    t: np.ndarray,
    series: dict,              # {label: ndarray(same shape as t)}
    *,
    xlabel: str = "t",
    ylabel: str = "Cd",
    style_kw: dict | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.axes.Axes":
    """Cd / Cl time histories — drag histories, shedding (spec §6)."""
```

### Shared style convention (blue line / red marker pattern from the papers)

```python
# Default style (from group's paper-figure convention):
_DEFAULT_SERIES_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
_DEFAULT_MARKER = "o"
_GRID_KW = dict(linestyle="--", alpha=0.5)
```

### Tests

Tests assert render produces a non-empty PNG and the axes contain the expected
number of lines; they do NOT check pixel values or exact artist positions.

```python
# tests/test_viz_renders_plots.py (plot section)

def test_convergence_plot(tmp_path):
    from diffsim.viz.plots import convergence
    import matplotlib.pyplot as plt
    levels = [2, 3, 4, 5]
    values = [0.4, 0.1, 0.025, 0.006]
    ax = convergence(levels, values, reference=None, slope=2.0)
    fig = ax.get_figure()
    fig.savefig(str(tmp_path / "conv.png"))
    assert (tmp_path / "conv.png").stat().st_size > 0
    # has at least 1 line (the main series) + 1 slope reference
    assert len(ax.get_lines()) >= 2
    plt.close("all")


def test_surface_profile_with_inset(tmp_path):
    from diffsim.viz.plots import surface_profile
    import matplotlib.pyplot as plt
    x = np.linspace(0, 2 * np.pi, 50)
    ax = surface_profile(
        x, {"DiffSim": np.cos(x)},
        reference={"Lit": (x[::5], np.cos(x[::5]))},
        inset={"xlim": (1.0, 2.0), "ylim": (-1.0, 0.0)},
    )
    fig = ax.get_figure()
    fig.savefig(str(tmp_path / "prof.png"))
    assert (tmp_path / "prof.png").stat().st_size > 0
    plt.close("all")


def test_history_plot(tmp_path):
    from diffsim.viz.plots import history
    import matplotlib.pyplot as plt
    t = np.linspace(0, 10, 200)
    ax = history(t, {"Cd": 1.4 + 0.05 * np.sin(2 * t)})
    fig = ax.get_figure()
    fig.savefig(str(tmp_path / "hist.png"))
    assert (tmp_path / "hist.png").stat().st_size > 0
    assert len(ax.get_lines()) == 1
    plt.close("all")
```

### Steps

- [ ] 1. Write failing tests in `tests/test_viz_renders_plots.py`.

- [ ] 2. Implement `src/diffsim/viz/plots.py`:
  - `convergence`: convert levels to h = 2**-level; plot on log-log scale;
    if reference given, plot |values - reference|; draw slope-reference line
    via `np.polyfit` on one decade.
  - `surface_profile`: plot each series in `series` dict with default color
    cycle; overlay `reference` curves with distinct markers; create an inset
    axes (`ax.inset_axes`) if `inset` is given and zoom to `inset["xlim"]`,
    `inset["ylim"]`; call `ax.indicate_inset_zoom(axins)`.
  - `history`: plot each series; add grid with `_GRID_KW`; label axes.
  - All: use `matplotlib.pyplot.subplots` when `ax` is `None`.

- [ ] 3. Expose in `__init__.py`:
  ```python
  from .plots import convergence, surface_profile, history
  __all__ = [..., "convergence", "surface_profile", "history"]
  ```

- [ ] 4. Run: `pytest tests/test_viz_renders_plots.py -x -k "convergence or profile or history"` → pass.

- [ ] 5. Run: `ruff check src/diffsim/viz/plots.py`

- [ ] 6. Commit:
  ```
  feat(viz): Task 4 — matplotlib plots (convergence, surface_profile, history)

  Adds viz/plots.py: three figure-matched quantitative-plot helpers (spec §6).
  Tests assert runs + non-empty PNG + correct line counts.

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  ```

---

## Task 5 — ParaView state templates + Glance how-to note

**Spec reference:** §7, §8, §9.

### Files

- **Create** `src/diffsim/viz/share.py`
- **Modify** `src/diffsim/viz/__init__.py`
- **Create** `tests/test_viz_share.py`
- **Create** `docs/dev/how-to-view-in-glance.md`

### Interface

```python
# src/diffsim/viz/share.py

def paraview_state(
    vtu_path: str | os.PathLike,
    *,
    physics: str = "film",          # "film" | "xdd" | "ns"
    color_by: str | None = None,    # default: physics-appropriate field
    camera: dict | None = None,     # {"position": [...], "focal_point": [...]}
    output: str | os.PathLike | None = None,  # write .pvsm here; None = return str
) -> str:
    """Generate a ParaView state (.pvsm) template for the given VTU.

    Parameterized by physics type to pre-select colormaps and filters:
      film: color by phi_p, add slice filter on element_size
      xdd:  color by n (electron density), add JV-overlay annotation
      ns:   color by velocity_magnitude, add LIC surface filter + slice

    The .pvsm uses a relative path to the VTU so the collaborator can move
    both files together.  Templates are pure XML strings — no ParaView install
    required to generate.

    Returns the .pvsm XML string; writes to `output` if given.
    """
```

### `.pvsm` template structure

The `.pvsm` is a minimal ParaView state XML that:
1. Declares a `vtkXMLUnstructuredGridReader` source pointing to the VTU by
   filename (relative path, `os.path.basename(vtu_path)`).
2. Wraps it in a `XMLUnstructuredGridReader` proxy with the correct source ID.
3. Adds a `<Representation>` section that sets `ColorArrayName` to `color_by`.
4. Adds a `<Camera>` section if `camera` is given.
5. References the arrays by name (they are asserted to be present in the VTU
   in the test).

The template does NOT need to be pixel-perfect when opened in ParaView; it
must: (a) be valid XML; (b) reference the correct VTU filename; (c) name the
correct color array; (d) be parseable by Python's `xml.etree.ElementTree`.

### Tests

```python
# tests/test_viz_share.py

import xml.etree.ElementTree as ET
import pytest
from diffsim.viz import paraview_state


def _write_vtu(tmp_path):
    import numpy as np
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.viz import export_vtu
    tree = build_uniform(2, dim=3)
    mesh = build_mesh(tree, p=1)
    Nn = len(mesh.node_coords)
    vtu = tmp_path / "test.vtu"
    export_vtu(mesh, vtu, fields={
        "phi_p": np.zeros(Nn), "phi_f": np.zeros(Nn)})
    return vtu


def test_pvsm_is_valid_xml(tmp_path):
    vtu = _write_vtu(tmp_path)
    pvsm_str = paraview_state(vtu, physics="film")
    root = ET.fromstring(pvsm_str)
    assert root.tag is not None


def test_pvsm_references_vtu_filename(tmp_path):
    vtu = _write_vtu(tmp_path)
    pvsm_str = paraview_state(vtu, physics="film")
    assert "test.vtu" in pvsm_str


def test_pvsm_color_by_field(tmp_path):
    vtu = _write_vtu(tmp_path)
    pvsm_str = paraview_state(vtu, physics="film", color_by="phi_p")
    assert "phi_p" in pvsm_str


def test_pvsm_writes_file(tmp_path):
    vtu = _write_vtu(tmp_path)
    out = tmp_path / "scene.pvsm"
    paraview_state(vtu, physics="film", output=out)
    assert out.exists()
    assert out.stat().st_size > 0
    # parse it
    ET.parse(str(out))


def test_pvsm_ns_physics(tmp_path):
    vtu = _write_vtu(tmp_path)
    pvsm_str = paraview_state(vtu, physics="ns")
    assert "velocity_magnitude" in pvsm_str or "ns" in pvsm_str.lower()
```

### Steps

- [ ] 1. Write failing tests in `tests/test_viz_share.py`.

- [ ] 2. Implement `src/diffsim/viz/share.py`:

  Define `_PHYSICS_DEFAULTS` dict:
  ```python
  _PHYSICS_DEFAULTS = {
      "film": {"color_by": "phi_p",              "filter": "slice"},
      "xdd":  {"color_by": "n",                  "filter": "none"},
      "ns":   {"color_by": "velocity_magnitude", "filter": "lic"},
  }
  ```

  Implement `paraview_state` as a string-template `.pvsm` generator:
  ```python
  def paraview_state(vtu_path, *, physics="film", color_by=None,
                     camera=None, output=None):
      import pathlib, os
      vtu_path = pathlib.Path(vtu_path)
      defaults = _PHYSICS_DEFAULTS.get(physics, _PHYSICS_DEFAULTS["film"])
      cb = color_by or defaults["color_by"]
      vtu_name = vtu_path.name
      cam_xml = _camera_xml(camera) if camera else ""
      pvsm = _PVSM_TEMPLATE.format(
          vtu_name=vtu_name,
          color_by=cb,
          camera_section=cam_xml,
          physics=physics,
      )
      if output is not None:
          pathlib.Path(output).write_text(pvsm)
      return pvsm
  ```

  Define `_PVSM_TEMPLATE` as a minimal but parseable ParaView state XML
  (version="5.10", `<ParaView><ServerManagerState>` structure):
  ```xml
  <?xml version="1.0"?>
  <ParaView version="5.10">
    <ServerManagerState>
      <Proxy group="sources" type="XMLUnstructuredGridReader" id="1001">
        <Property name="FileName" number_of_elements="1">
          <Element index="0" value="{vtu_name}"/>
        </Property>
      </Proxy>
      <Proxy group="representations" type="GeometryRepresentation" id="2001">
        <Property name="ColorArrayName" number_of_elements="2">
          <Element index="0" value="POINTS"/>
          <Element index="1" value="{color_by}"/>
        </Property>
      </Proxy>
      {camera_section}
    </ServerManagerState>
  </ParaView>
  ```

- [ ] 3. Create `docs/dev/how-to-view-in-glance.md`:
  A one-page note (spec §7 Glance how-to) explaining:
  1. Export a `.vtu` with `export_vtu(mesh, "result.vtu", fields=...)`.
  2. Go to https://kitware.github.io/glance/ in a browser.
  3. Drag-and-drop `result.vtu` onto the Glance window.
  4. Use the left panel to change the color-by field.
  5. To share: send the `.vtu` file; the recipient repeats steps 2–4.

- [ ] 4. Expose in `__init__.py`:
  ```python
  from .share import paraview_state
  __all__ = [..., "paraview_state"]
  ```

- [ ] 5. Run: `pytest tests/test_viz_share.py -x` → all five tests pass.

- [ ] 6. Run: `ruff check src/diffsim/viz/share.py tests/test_viz_share.py`

- [ ] 7. Commit:
  ```
  feat(viz): Task 5 — ParaView .pvsm templates + Glance how-to note

  Adds viz/share.py (paraview_state, three physics presets), five XML-parse
  tests, and docs/dev/how-to-view-in-glance.md (spec §7).

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  ```

---

## Self-Review Checklist

### Spec Phase 1 coverage

| Spec §10 Phase 1 item | Task covering it | Status |
|---|---|---|
| VTU exporter (physics fields) | Task 1 | Covered |
| Element size / level cell data | Task 1 | Covered |
| Element-type classification | Task 2 | Covered |
| Surrogate distance `d` | Task 2 | Covered |
| Embedded-body `.vtp` | Task 2 | Covered |
| `lic` render | Task 3 | Covered |
| `mesh_slice` render | Task 3 | Covered |
| `contour` render | Task 3 | Covered |
| Shared `Style` object | Task 3 | Covered |
| `convergence` plot | Task 4 | Covered |
| `surface_profile` w/ inset | Task 4 | Covered |
| `history` plot | Task 4 | Covered |
| `.pvsm` state templates | Task 5 | Covered |
| Glance how-to note | Task 5 | Covered |

**Phase 2 items correctly excluded:** `export_html`, `small_multiples`,
`parameter_sweep`, `vtkHyperTreeGrid` — none appear in this plan.

### Placeholder scan

No `TODO`, `FIXME`, `...`, `pass`, or `NotImplementedError` placeholders in
the implementation steps above. All code in the steps is real or references real
identifiers from the existing codebase.

### Type consistency

- `export_vtu(source, path, *, fields, time_series)` → `pathlib.Path`
- `export_vtu_sbm(mesh, path, *, fields, retained_tree, frac, geom, sf)` → `pathlib.Path`
- `export_body_vtp(vertices, triangles, path)` → `pathlib.Path`
- `lic(vtu, color_by, *, style, output, off_screen)` → `pyvista.Plotter`
- `mesh_slice(vtu, color_by, normal, origin, *, style, output, off_screen)` → `pyvista.Plotter`
- `contour(vtu, field, isovalues, *, style, output, off_screen)` → `pyvista.Plotter`
- `convergence(levels, values, reference, *, slope, ..., ax)` → `matplotlib.axes.Axes`
- `surface_profile(x, series, reference, *, inset, ..., ax)` → `matplotlib.axes.Axes`
- `history(t, series, *, ..., ax)` → `matplotlib.axes.Axes`
- `paraview_state(vtu_path, *, physics, color_by, camera, output)` → `str`

All return types are consistent with spec §4–§7. Source type for `export_vtu`
accepts both `Mesh` and path string/Path (per spec §4 "a run object or .npz").

### Gate-hygiene check

Task 1 `test_export_vtu_fields_present` reads the VTU back with `meshio`
independently and asserts against `phi_p_ref` / `phi_f_ref` constructed before
the call — this is an independent reference, not a self-comparison. The same
pattern is required in Task 2 for element-type assertions.

---

## Spec Ambiguities Resolved (for Supervisor Confirmation)

The following questions arose during plan authorship. Items marked **[ASSUMED]**
contain a decision made by the plan author; please confirm or correct before
implementation begins.

1. **[ASSUMED] Octree-to-VTK cell mapping for p2 elements.** The spec says
   "VTU (unstructured)" and the `Mesh` object has `conn_of[p]` for each `p ∈
   {1, 2}`. The plan maps p1 elements to `VTK_Hex` (type 12, 8 nodes) and p2
   elements to `VTK_TriquadHex` (type 29, 27 nodes). However, the `build_mesh`
   source (nodes.py) confirms `conn_of[2]` has shape `[Ne, (2+1)^3 = 27]`, so
   the mapping is consistent. **Confirm: type 29 (`TriquadraticHexahedron`) is
   the correct VTK cell type for 3-D p2 octree elements.**

2. **[ASSUMED] `.npz`-only source path in `export_vtu`.** When `source` is a
   `.npz` path, the plan cannot recover the octree connectivity (only `coords`,
   `phi_p`, `phi_f` are in the file). The plan falls back to writing a
   vertex/point-cloud-only VTU (meshio `"vertex"` cells). This means cell data
   (`element_size`, `level`) is absent in the npz-only path. An alternative
   would be to require the caller to pass the `Mesh` object even for npz inputs.
   **Confirm: is the npz-only vertex fallback acceptable, or must the Mesh always
   be provided alongside the npz?**

3. **[ASSUMED] XDD fields in the VTU.** The spec says "XDD's 5 fields" as point
   data. From `exciton_system.py`, these are `IPHI=0, IN=1, IP=2, IXD=3, IXA=4`
   (phi, n, p, xd, xa). The plan names them `"phi"`, `"n"`, `"p"`, `"xd"`,
   `"xa"` as point arrays. **Confirm these names (vs. the nondim names `phi_hat`,
   `n_hat`, etc.) — important for `.pvsm` template field references.**

4. **[ASSUMED] `lic` render for 3-D VTU.** The spec cites "surface LIC" (PyVista
   surface LIC). For a volumetric 3-D octree VTU there is no natural surface;
   the plan slices the VTU at `z=0.5` first, then applies surface LIC on the
   slice. If the run is 2-D (dim=2), the slice is skipped. **Confirm: is a
   z=0.5 midplane slice the right default for 3-D LIC, or should the caller
   always specify `normal` / `origin`?**

5. **[OPEN — could not confirm from code] Surrogate distance `d` in the VTU as
   a nodal vs. element-mean scalar.** `GeometryData.d` is defined at surrogate
   face Gauss points (`[Nf*nqf, dim]`), not at nodes or element centroids. The
   plan averages `||d||` over the GPs of each surrogate element to produce a
   scalar per element (cell data). An alternative is node-averaging. The spec
   says "surrogate distance d as a cell label" (§4), which supports cell data.
   **Confirm: cell-data `d_mean` (mean ||d|| per element) is the intended
   representation.**

6. **[OPEN — could not confirm from code] `classify_lambda` element-type vs.
   `frac` ambiguity at `frac==1.0`.** The plan codes `element_type=0` (interior)
   for `frac==1.0`, `element_type=1` (intercepted) for `0 < frac < 1.0`, and
   `element_type=2` (surrogate-boundary, overrides 0/1) for elements in
   `sf.elem`. It is possible for a `frac==1.0` element to be in `sf.elem` (a
   fully-inside element that happens to be adjacent to a missing element — the
   surrogate extraction logic in `extract_surrogate` checks neighbor presence,
   not `frac`). The plan lets `element_type=2` override `element_type=0` for
   such elements. **Confirm this override is correct.**

## Supervisor resolutions (2026-07-21 — binding; confirm any code-dependent item at build time)

The plan writer flagged 6 ambiguities. Resolutions:
1. **VTK type 29 (TriquadraticHexahedron) for 3-D p2** — correct (27-node hex = VTK_TRIQUADRATIC_HEXAHEDRON). Verify `conn_of[2]` node ordering matches VTK's at implementation.
2. **`.npz`-only source** — support BOTH: a Mesh/run source gives full octree cell topology (the good path, required for the Fig-19 mesh figures); a bare `.npz` degrades to a point-cloud VTU with a WARNING (fine for field contours, NOT for mesh-slice figures). Document the degradation.
3. **XDD field names in the VTU** — use readable physical names (`phi`,`n`,`p`,`xd`,`xa`); implementer's call, non-blocking.
4. **LIC on 3-D VTU** — default to a z=0.5 midplane slice, caller-overridable. Fine.
5. **Surrogate distance `d`** — cell `d_mean` (mean ‖d‖ over surrogate GPs) for FIELD COLORING; the Fig-17b `d`-vs-x PROFILE plot uses GP-level `d` from `GeometryData.d` (a `surface_profile` input, not the VTU). Both, for their two uses.
6. **element_type priority** — surrogate (2) OVERRIDES interior (0) for elements in `sf.elem` (the SBM-active element wins the label). Correct for the carve-out story.
