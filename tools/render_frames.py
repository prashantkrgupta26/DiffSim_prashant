#!/usr/bin/env python
"""render_frames.py — Offscreen PNG renderer for truck viz animation extracts.

Usage
-----
python tools/render_frames.py VIZ_DIR [options]

Reads per-frame .vtp files written by the TruckVizHook and renders numbered
PNG sequences per shot using PyVista offscreen rendering.

Shots
-----
1. q_iso     : Q-isosurface colored by velocity magnitude + truck surface
2. centerline: Centerline slice colored by |u|
3. surface_cp: Truck surface colored by Cp

Each shot writes:
  VIZ_DIR/renders/shot_<name>_NNNNNN.png

Camera positions and scalar ranges are FIXED across all frames within a shot
(no per-frame rescaling) — this is load-bearing for smooth video output.

Scalar ranges
-------------
Determined by scanning ALL frames first (two-pass: scan -> render).
Override via --range_umag, --range_Q, --range_Cp.

Requirements
------------
pyvista >= 0.44 (offscreen rendering on macOS via pv.Plotter(off_screen=True);
no Xvfb needed). If pyvista is not available this script prints a message and
exits 0 (graceful degradation for CI without the viz extra).

Camera presets (fixed)
----------------------
Shot 1 (q_iso)     : isometric from front-right-top
Shot 2 (centerline): looking along z (plan view of the slice)
Shot 3 (surface_cp): isometric from front-left-top for body surface
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import pathlib


def _parse_args():
    p = argparse.ArgumentParser(
        description="Offscreen render truck viz frames to PNG sequences.")
    p.add_argument("viz_dir", help="Root viz directory (contains frames/)")
    p.add_argument("--shot", choices=["q_iso", "centerline", "surface_cp", "all"],
                   default="all", help="Which shot(s) to render (default: all)")
    p.add_argument("--range_umag", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="Fixed velocity-magnitude scalar range (e.g. 0 1.2)")
    p.add_argument("--range_Q", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="Fixed Q-criterion scalar range (e.g. -0.5 2.0)")
    p.add_argument("--range_Cp", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="Fixed Cp scalar range (e.g. -2.0 1.0)")
    p.add_argument("--width", type=int, default=1920, help="PNG width (px)")
    p.add_argument("--height", type=int, default=1080, help="PNG height (px)")
    p.add_argument("--window_size", type=int, nargs=2, default=None,
                   metavar=("W", "H"), help="Override width and height together")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Offscreen probe — called once at import time so we can bail early.
# ---------------------------------------------------------------------------

def _check_pyvista():
    try:
        import pyvista as pv
        return pv
    except ImportError:
        print("[render_frames] pyvista not available. "
              "Install with: pip install 'diffsim[viz]'", file=sys.stderr)
        sys.exit(0)


# ---------------------------------------------------------------------------
# Camera presets (unit-cube domain)
# ---------------------------------------------------------------------------

_CAMERAS = {
    "q_iso": {
        # Front-right-top isometric: see wake vortices behind truck
        "position": (1.5, 0.5, 0.5),
        "focal_point": (0.4, 0.08, 0.08),
        "up": (0.0, 1.0, 0.0),
    },
    "centerline": {
        # Looking along +z axis (plan/top-down view of the z-mid slice)
        "position": (0.5, 0.08, 2.0),
        "focal_point": (0.5, 0.08, 0.08),
        "up": (0.0, 1.0, 0.0),
    },
    "surface_cp": {
        # Front-left-top for body surface Cp
        "position": (-0.5, 0.3, 0.3),
        "focal_point": (0.4, 0.08, 0.08),
        "up": (0.0, 1.0, 0.0),
    },
}


# ---------------------------------------------------------------------------
# Range scan (two-pass: scan all frames first for consistent color bar)
# ---------------------------------------------------------------------------

def _scan_range(files, scalar_name, pv):
    """Scan all .vtp files for the min/max of a named scalar. Returns (lo, hi).

    Skipped files (unreadable or missing scalar) are logged to stderr.
    NaN-containing fields are handled via np.nanmin/np.nanmax; a one-line
    warning is printed to stderr naming the field and NaN count.
    """
    import numpy as np
    lo, hi = float("inf"), float("-inf")
    for f in files:
        try:
            mesh = pv.read(f)
            if scalar_name in mesh.point_data:
                arr = mesh.point_data[scalar_name]
            elif scalar_name in mesh.cell_data:
                arr = mesh.cell_data[scalar_name]
            else:
                print(f"[render_frames] skip {f}: scalar '{scalar_name}' not found",
                      file=sys.stderr)
                continue
            if len(arr) == 0:
                print(f"[render_frames] skip {f}: scalar '{scalar_name}' is empty",
                      file=sys.stderr)
                continue
            arr = np.asarray(arr, dtype=np.float64)
            nan_count = int(np.isnan(arr).sum())
            if nan_count > 0:
                print(f"[render_frames] warning: {f}: field '{scalar_name}' "
                      f"contains {nan_count} NaN(s); using nanmin/nanmax",
                      file=sys.stderr)
            lo = min(lo, float(np.nanmin(arr)))
            hi = max(hi, float(np.nanmax(arr)))
        except Exception as exc:
            print(f"[render_frames] skip {f}: {exc}", file=sys.stderr)
            continue
    if lo > hi or not (np.isfinite(lo) and np.isfinite(hi)):
        return (0.0, 1.0)  # empty / all-NaN fallback
    if lo == hi:
        return (lo - 0.5, hi + 0.5)
    return (lo, hi)


# ---------------------------------------------------------------------------
# Render one shot
# ---------------------------------------------------------------------------

def render_shot(
    shot_name: str,
    files: list[str],
    out_dir: pathlib.Path,
    pv,
    *,
    scalar: str,
    scalar_range: tuple,
    camera: dict,
    width: int = 1920,
    height: int = 1080,
    cmap: str = "viridis",
) -> list[pathlib.Path]:
    """Render each file in `files` to a numbered PNG using fixed camera + range.

    Parameters
    ----------
    shot_name : str, e.g. "q_iso"
    files : sorted list of .vtp paths
    out_dir : directory for PNGs
    pv : pyvista module
    scalar : name of the scalar field to color by
    scalar_range : (lo, hi) — fixed across all frames
    camera : dict with keys position, focal_point, up
    width, height : PNG resolution
    cmap : colormap name

    Returns list of written PNG paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for idx, fpath in enumerate(sorted(files)):
        out_png = out_dir / f"shot_{shot_name}_{idx:06d}.png"
        try:
            mesh = pv.read(fpath)
            pl = pv.Plotter(off_screen=True, window_size=[width, height])

            # Prefer cell/point data depending on what's available
            if scalar in mesh.point_data or scalar in mesh.cell_data:
                pl.add_mesh(mesh, scalars=scalar, clim=list(scalar_range),
                            cmap=cmap, show_scalar_bar=True,
                            scalar_bar_args={"title": scalar})
            else:
                pl.add_mesh(mesh, color="lightgrey")

            pl.camera_position = [
                camera["position"],
                camera["focal_point"],
                camera["up"],
            ]
            pl.screenshot(str(out_png))
            pl.close()
            written.append(out_png)
        except Exception as exc:
            print(f"[render_frames] warning: could not render {fpath}: {exc}",
                  file=sys.stderr)

    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    pv = _check_pyvista()

    viz_dir = pathlib.Path(args.viz_dir)
    frames_dir = viz_dir / "frames"
    renders_dir = viz_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    w, h = args.width, args.height
    if args.window_size:
        w, h = args.window_size

    shots_todo = (
        ["q_iso", "centerline", "surface_cp"]
        if args.shot == "all" else [args.shot]
    )

    # Shot definitions: subdir -> scalar -> scalar_range override key
    shot_cfg = {
        "q_iso": {
            "subdir": "q_iso",
            "scalar": "velocity_magnitude",
            "range_override": args.range_umag,
            "cmap": "plasma",
        },
        "centerline": {
            "subdir": "centerline",
            "scalar": "velocity_magnitude",
            "range_override": args.range_umag,
            "cmap": "viridis",
        },
        "surface_cp": {
            "subdir": "surface_cp",
            "scalar": "Cp",
            "range_override": args.range_Cp,
            "cmap": "RdBu_r",
        },
    }

    for shot_name in shots_todo:
        cfg = shot_cfg[shot_name]
        shot_frames_dir = frames_dir / cfg["subdir"]
        files = sorted(glob.glob(str(shot_frames_dir / "*.vtp")))
        if not files:
            print(f"[render_frames] no .vtp files found in {shot_frames_dir}; "
                  f"skipping shot '{shot_name}'")
            continue

        print(f"[render_frames] shot '{shot_name}': {len(files)} frames, "
              f"scalar='{cfg['scalar']}'")

        # Determine scalar range
        if cfg["range_override"] is not None:
            srange = tuple(cfg["range_override"])
        else:
            print(f"[render_frames]   scanning {len(files)} files for "
                  f"'{cfg['scalar']}' range...")
            srange = _scan_range(files, cfg["scalar"], pv)
        print(f"[render_frames]   scalar range: {srange[0]:.4g} .. {srange[1]:.4g}")

        # Render
        out_dir = renders_dir / shot_name
        written = render_shot(
            shot_name, files, out_dir, pv,
            scalar=cfg["scalar"],
            scalar_range=srange,
            camera=_CAMERAS[shot_name],
            width=w, height=h,
            cmap=cfg["cmap"],
        )
        print(f"[render_frames]   wrote {len(written)} PNGs to {out_dir}")

    print("[render_frames] done.")


if __name__ == "__main__":
    main()
