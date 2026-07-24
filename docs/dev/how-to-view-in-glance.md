# How to view a DiffSim result in ParaView Glance (no install)

ParaView Glance is a browser-based viewer — collaborators and students can open
a DiffSim `.vtu`/`.vtp` with no DiffSim, ParaView, or Python install. This is
the primary "share interactively" path (spec §7, mechanism #2).

## 1. Export a VTU from a run

```python
from diffsim.viz import export_vtu

# `mesh` is a DiffSim Mesh (from build_mesh); `fields` maps names to nodal arrays.
export_vtu(mesh, "result.vtu", fields={"phi_p": phi_p, "phi_f": phi_f})
```

For the SBM carve-out cell data (element type / surrogate distance) use
`export_vtu_sbm(...)`; for the embedded body surface use `export_body_vtp(...)`.

> A bare `.npz` path also works (`export_vtu("run_final.npz", "result.vtu")`)
> but degrades to a point cloud with a warning — it has no octree cell topology,
> so the mesh-slice / element-size figures need the `Mesh` object instead.

## 2. Open Glance in a browser

Go to <https://kitware.github.io/glance/app/> (nothing to install).

## 3. Drag and drop

Drag `result.vtu` (or `body.vtp`) from your file manager onto the Glance
window. It loads immediately.

## 4. Explore

- Use the left **pipeline / coloring** panel to change the *Color By* field
  (e.g. `phi_p`, `phi_s`, `element_size`, `velocity_magnitude`).
- Add a **Slice** or **Clip** representation to look inside a 3-D octree.
- Rotate / pan / zoom with the mouse.

## 5. Share

Send the `.vtu` (and `.vtp`) file to a collaborator. They repeat steps 2–4 —
no install, no account. The file is self-contained (fields + mesh embedded).

## Related: desktop ParaView state files

For a reproducible desktop-ParaView scene (pre-set camera, colormap, filters),
generate a `.pvsm` alongside the VTU:

```python
from diffsim.viz import paraview_state
paraview_state("result.vtu", physics="ns", output="scene.pvsm")
```

Open `scene.pvsm` in desktop ParaView (File → Load State) and point it at the
VTU when prompted; it lands on the intended scene.
