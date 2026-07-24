"""PyVista off-screen field renders (Phase 1, spec §5).

Three figure-matched helpers calibrated to the group's paper figures:
  lic         — surface Line Integral Convolution colored by a scalar (Fig 14)
  mesh_slice  — octree slice colored by element_size / level (Fig 19)
  contour     — scalar contours (T, p, composition) with the group's colormap

All three share a `Style` dataclass (colormap, background, camera, scalar-bar,
window size).  `import pyvista` is guarded inside each function body so
`import diffsim.viz` never fails without the [viz] extra.

Headless note: off-screen rendering needs a working GL backend (EGL/OSMesa or
xvfb).  On a display-less box without those, `pyvista.Plotter(off_screen=True)`
raises at render time; callers/tests should probe with `offscreen_gl_ok()` and
skip if unavailable — the render code here is otherwise backend-agnostic.
"""
import os
from dataclasses import dataclass
from typing import Tuple, Union


def _require_pyvista():
    try:
        import pyvista as pv
    except ImportError as e:
        raise ImportError(
            "diffsim.viz.renders requires 'pyvista'. Install with: "
            "pip install diffsim[viz]"
        ) from e
    return pv


_GL_PROBE_SRC = (
    "import pyvista as pv\n"
    "pv.OFF_SCREEN = True\n"
    "pl = pv.Plotter(off_screen=True, window_size=(64, 48))\n"
    "pl.add_mesh(pv.Sphere())\n"
    "img = pl.screenshot(return_img=True)\n"
    "pl.close()\n"
    "assert img is not None and img.size > 0\n"
)


def offscreen_gl_ok() -> bool:
    """Return True iff PyVista can complete an off-screen screenshot here.

    The probe runs in a SUBPROCESS: a missing GL backend (EGL/OSMesa) makes
    VTK abort the whole process rather than raise a catchable exception, so an
    in-process try/except is not enough to survive it.  A non-zero subprocess
    exit (crash or abort) => False; callers/tests use this to skip-if-no-GL.
    """
    try:
        _require_pyvista()
    except ImportError:
        return False
    import subprocess
    import sys
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _GL_PROBE_SRC],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    except Exception:
        return False
    return proc.returncode == 0


@dataclass
class Style:
    """Shared style object: colormap, background, camera, scalar-bar.

    Defaults match the group's paper-figure convention (white background,
    RdBu_r diverging map for scalars)."""
    background: str = "white"
    cmap: str = "RdBu_r"
    scalar_bar: bool = True
    window_size: Tuple[int, int] = (1200, 800)
    camera_position: Union[str, list] = "xy"


def _new_plotter(pv, style: Style, off_screen: bool):
    if off_screen:
        pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=off_screen, window_size=list(style.window_size))
    pl.background_color = style.background
    return pl


def _finish(pl, style: Style, output):
    pl.camera_position = style.camera_position
    if output is not None:
        pl.screenshot(str(output))
    return pl


def lic(
    vtu: Union[str, os.PathLike],
    color_by: str = "velocity_magnitude",
    *,
    style: "Style | None" = None,
    output: "str | os.PathLike | None" = None,
    off_screen: bool = True,
    vectors: str = "velocity",
):
    """Surface Line Integral Convolution colored by a scalar (spec §5, Fig 14).

    Reads the .vtu, extracts a surface (a z=0.5 midplane slice for a 3-D grid,
    the surface itself for 2-D), and draws the streamline texture colored by
    `color_by`.  For a robust cross-version path we render hedgehog/streamline
    glyphs of the `vectors` field over the colored surface (VTK's surface-LIC
    representation is not exposed uniformly across PyVista versions).  Returns
    the plotter; writes a PNG if `output` is given.
    """
    pv = _require_pyvista()
    style = style or Style()
    grid = pv.read(str(vtu))

    # Reduce to a 2-D surface for the streamline texture.
    surf = grid
    bounds = grid.bounds
    zspan = bounds[5] - bounds[4]
    if zspan > 1e-12:
        zmid = 0.5 * (bounds[4] + bounds[5])
        surf = grid.slice(normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, zmid))

    pl = _new_plotter(pv, style, off_screen)
    pl.add_mesh(surf, scalars=color_by, cmap=style.cmap,
                show_scalar_bar=style.scalar_bar)
    # streamline glyphs to convey the flow direction (LIC-style overlay)
    if vectors in (surf.point_data.keys()):
        try:
            arrows = surf.glyph(orient=vectors, scale=False, factor=0.05)
            pl.add_mesh(arrows, color="black")
        except Exception:
            pass
    return _finish(pl, style, output)


def mesh_slice(
    vtu: Union[str, os.PathLike],
    color_by: str = "element_size",
    normal: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    origin: "Tuple[float, float, float] | None" = None,
    *,
    style: "Style | None" = None,
    output: "str | os.PathLike | None" = None,
    off_screen: bool = True,
):
    """Octree slice colored by element_size / level (spec §5, Fig 19).

    Slices the .vtu at the given plane and colors by `color_by`, showing the
    refinement shells around an immersed body.
    """
    pv = _require_pyvista()
    style = style or Style()
    grid = pv.read(str(vtu))
    if origin is None:
        b = grid.bounds
        origin = (0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]),
                  0.5 * (b[4] + b[5]))
    sl = grid.slice(normal=normal, origin=origin)

    pl = _new_plotter(pv, style, off_screen)
    pl.add_mesh(sl, scalars=color_by, cmap=style.cmap, show_edges=True,
                show_scalar_bar=style.scalar_bar)
    return _finish(pl, style, output)


def contour(
    vtu: Union[str, os.PathLike],
    field: str,
    isovalues: "list | int" = 10,
    *,
    style: "Style | None" = None,
    output: "str | os.PathLike | None" = None,
    off_screen: bool = True,
):
    """Scalar contours (T, p, composition) with the group's colormap (spec §5)."""
    pv = _require_pyvista()
    style = style or Style()
    grid = pv.read(str(vtu))
    # contour needs point data; promote cell data if the field lives on cells
    if field not in grid.point_data and field in grid.cell_data:
        grid = grid.cell_data_to_point_data()
    iso = grid.contour(isovalues, scalars=field)

    pl = _new_plotter(pv, style, off_screen)
    # draw the source surface faintly, then the isosurfaces on top
    pl.add_mesh(grid.outline(), color="gray")
    if iso.n_points > 0:
        pl.add_mesh(iso, scalars=field, cmap=style.cmap,
                    show_scalar_bar=style.scalar_bar)
    return _finish(pl, style, output)
