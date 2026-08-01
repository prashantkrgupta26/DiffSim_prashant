"""Render octree-mesh structure views from mesh_*.npz dumps (anchors/levels/h).

Views per variant:
  1. centerline slice z=0.0625 (level-colored cell outlines) — full + truck zoom
  2. underbody close-up crop, 3-D
  3. ground-height y-slice (y=0.001) level-colored
Also writes cropped .vtu files for ParaView.

Usage: .venv/bin/python tools/render_mesh.py results/mesh_current.npz [more.npz]
"""
import sys
import pathlib

import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True


def load(npz_path):
    d = np.load(npz_path)
    return d["anchors"].astype(np.float64), d["levels"], d["h"].astype(np.float64)


def hex_grid(anchors, levels, h, mask):
    a = anchors[mask]
    hh = h[mask]
    n = len(a)
    # 8 corners per hex, VTK hexahedron ordering
    offs = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], np.float64)
    pts = (a[:, None, :] + offs[None, :, :] * hh[:, None, None]).reshape(-1, 3)
    cells = np.empty((n, 9), np.int64)
    cells[:, 0] = 8
    cells[:, 1:] = np.arange(8 * n).reshape(n, 8)
    grid = pv.UnstructuredGrid(cells.ravel(),
                               np.full(n, pv.CellType.HEXAHEDRON), pts)
    grid.cell_data["level"] = levels[mask]
    return grid


def slab_mask(anchors, h, axis, coord, lo=None, hi=None):
    m = (anchors[:, axis] <= coord) & (anchors[:, axis] + h > coord)
    for ax, (l, u) in (lo or {}).items():
        m &= (anchors[:, ax] + h > l) & (anchors[:, ax] < u)
    return m


def render(npz_path):
    name = pathlib.Path(npz_path).stem
    out = pathlib.Path("results/mesh_renders")
    out.mkdir(parents=True, exist_ok=True)
    anchors, levels, h = load(npz_path)
    print(f"{name}: {len(levels)} cells, levels {levels.min()}-{levels.max()}")

    views = [
        # (tag, axis, coord, window {axis: (lo,hi)}, camera normal)
        ("zslice_full", 2, 0.0625, {0: (0.0, 1.0), 1: (0.0, 0.125)}, "z"),
        ("zslice_truck", 2, 0.0625, {0: (0.25, 0.45), 1: (0.0, 0.05)}, "z"),
        ("yslice_gap", 1, 0.001, {0: (0.25, 0.45), 2: (0.03, 0.095)}, "y"),
        ("inlet_ground", 2, 0.0625, {0: (0.0, 0.12), 1: (0.0, 0.03)}, "z"),
    ]
    for tag, axis, coord, win, normal in views:
        m = slab_mask(anchors, h, axis, coord, lo=win)
        if not m.any():
            continue
        g = hex_grid(anchors, levels, h, m)
        p = pv.Plotter(off_screen=True, window_size=(1920, 1080))
        p.add_mesh(g, scalars="level", show_edges=True, cmap="viridis",
                   line_width=1, clim=[7, 12])
        p.camera_position = {"z": "xy", "y": "xz"}[normal]
        p.add_text(f"{name} — {tag} (level-colored)", font_size=12)
        png = out / f"{name}_{tag}.png"
        p.screenshot(str(png))
        p.close()
        print("  wrote", png)

    # underbody 3-D crop + .vtu for ParaView
    m3 = ((anchors[:, 0] + h > 0.28) & (anchors[:, 0] < 0.42) &
          (anchors[:, 1] < 0.025) & (anchors[:, 2] + h > 0.045) &
          (anchors[:, 2] < 0.085))
    g3 = hex_grid(anchors, levels, h, m3)
    vtu = out / f"{name}_underbody.vtu"
    g3.save(str(vtu))
    p = pv.Plotter(off_screen=True, window_size=(1920, 1080))
    p.add_mesh(g3, scalars="level", show_edges=True, cmap="viridis",
               opacity=0.9, clim=[7, 12])
    p.camera_position = "iso"
    p.add_text(f"{name} — underbody crop", font_size=12)
    p.screenshot(str(out / f"{name}_underbody3d.png"))
    p.close()
    print("  wrote", vtu, "and 3-D view")


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


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage: render_mesh.py results/mesh_*.npz [more.npz ...]",
              file=sys.stderr)
        sys.exit(1)
    if "--stl" in args:
        def _get(flag, default=None):
            if flag not in args:
                return default
            i = args.index(flag) + 1
            if i >= len(args) or args[i].startswith("--"):
                sys.exit(f"[render_mesh] ERROR: {flag} requires a value "
                         f"(usage: --stl PATH [--out-prefix P] "
                         f"[--position x,y,z] [--scale s])")
            return args[i]
        _pos_raw = _get("--position", "0,0,0")
        try:
            pos = tuple(float(x) for x in _pos_raw.split(","))
        except ValueError:
            sys.exit(f"[render_mesh] ERROR: --position must be three "
                     f"comma-separated numbers, got {_pos_raw!r}")
        if len(pos) != 3:
            sys.exit(f"[render_mesh] ERROR: --position needs exactly 3 "
                     f"components (x,y,z), got {len(pos)}: {_pos_raw!r}")
        render_stl_in_domain(_get("--stl"),
                             _get("--out-prefix", "results/mesh_renders/body"),
                             position=pos,
                             scale=float(_get("--scale", "1.0")))
    else:
        missing = [f for f in args if not pathlib.Path(f).exists()]
        if missing:
            for m in missing:
                print(f"[render_mesh] ERROR: input path not found: {m}", file=sys.stderr)
            sys.exit(1)
        for f in args:
            render(f)
