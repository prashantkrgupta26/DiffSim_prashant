"""results.py — Write visualization artifacts to the Dropbox-synced results/ folder.

Public API
----------
results_dir(base=None) -> pathlib.Path
    Returns <repo-root>/results, creating results/figures/ and results/vtu/.
    Pass `base` to override the repo root (useful in tests).

save_flow_run(name, *, mesh=None, node_fields=None, histories=None, body=None,
              base=None) -> dict[str, pathlib.Path]
    High-level emitter that writes one or more artifacts under results/:

    histories : dict {label: (t_array, values_array)} or {label: values_array}
        Written as results/figures/<name>_forces.png via plots.history.
        If only values are provided (no t), step indices [1..n] are used.

    mesh + node_fields : (Mesh, dict {name: ndarray})
        Written as results/vtu/<name>.vtu (meshio) and results/vtu/<name>.pvsm
        (ParaView state, physics="ns", color_by="velocity_magnitude").

    body : (vertices_array, triangles_array)
        Written as results/vtu/<name>_body.vtp via export_body_vtp.
        In 2-D, the "body" is typically a degenerate surface (a line segment
        represented as two vertices + one degenerate triangle), which pyvista
        cannot write reliably as a .vtp.  We fall back gracefully: if pyvista
        is missing or fails, we emit a .vtu triangle mesh via meshio (rename
        path to <name>_body.vtu).  Skip entirely if the geometry cannot be
        expressed as a triangle mesh (e.g. a line segment with < 3 vertices).

    base : str | pathlib.Path | None
        Override the repo-root for results/ (used by tests via monkeypatch or
        explicit arg so the test never litters the real results/ folder).

Each section is wrapped in try/except; a missing optional dep prints a hint
(mirroring tutorials/_viz.py) and never crashes the run.

Returns a dict {artifact_kind: path} for paths actually written.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Install hint (mirror tutorials/_viz.py style)
# ---------------------------------------------------------------------------

_HINT = (
    "[viz] Install hint: pip install 'diffsim[viz]'  "
    "(needs matplotlib + pyvista + meshio)"
)


def _hint() -> None:
    print(_HINT, file=sys.stdout, flush=True)


# ---------------------------------------------------------------------------
# results_dir
# ---------------------------------------------------------------------------

def _repo_root() -> pathlib.Path:
    """Find the repository root robustly.

    Walks from this file's location upward looking for pyproject.toml or
    setup.cfg (project markers).  Falls back to the grandparent of the src/
    package directory (src/diffsim/viz/results.py -> src/diffsim/viz ->
    src/diffsim -> src -> <repo-root>).
    """
    here = pathlib.Path(__file__).resolve()
    # Walk upward looking for pyproject.toml or setup.cfg
    candidate = here
    for _ in range(10):
        candidate = candidate.parent
        if (candidate / "pyproject.toml").exists() or (
                candidate / "setup.cfg").exists():
            return candidate
    # Fallback: src/diffsim/viz/results.py -> ../../../../
    return here.parent.parent.parent.parent


def results_dir(base: str | pathlib.Path | None = None) -> pathlib.Path:
    """Return <base>/results, creating figures/ and vtu/ subdirs on demand.

    Parameters
    ----------
    base : str | pathlib.Path | None
        Override the directory that contains results/.  Defaults to the
        repository root (auto-detected via pyproject.toml search).
    """
    root = pathlib.Path(base) if base is not None else _repo_root()
    rd = root / "results"
    (rd / "figures").mkdir(parents=True, exist_ok=True)
    (rd / "vtu").mkdir(parents=True, exist_ok=True)
    return rd


# ---------------------------------------------------------------------------
# Internal save helper (mirrors tutorials/_viz.py _save_fig)
# ---------------------------------------------------------------------------

def _save_fig(fig_or_ax, out_path: pathlib.Path) -> None:
    """Save a matplotlib Figure or Axes to *out_path*."""
    try:
        import matplotlib.pyplot as plt
        fig = getattr(fig_or_ax, "figure", None) or fig_or_ax
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"[viz] warning: could not save figure {out_path}: {exc}",
              flush=True)


# ---------------------------------------------------------------------------
# save_flow_run
# ---------------------------------------------------------------------------

def save_flow_run(
    name: str,
    *,
    mesh=None,
    node_fields: dict | None = None,
    histories: dict | None = None,
    body=None,
    base: str | pathlib.Path | None = None,
) -> dict:
    """Write visualization artifacts for a completed flow run to results/.

    Parameters
    ----------
    name : str
        Base name for all artifacts, e.g. "p2r1a_re10_L5".

    mesh : Mesh object or None
        DiffSim Mesh (from build_mesh).  Required together with node_fields
        for VTU export.

    node_fields : dict {field_name: ndarray(Nn,) or (Nn, d)} or None
        Per-node field arrays (velocity_magnitude, pressure, …).  Written
        as point_data into the VTU.  At least "velocity_magnitude" is
        recommended so the ParaView state colors correctly.

    histories : dict or None
        Force/coefficient histories.  Each entry may be:
          - (t_array, values_array) — explicit time axis
          - values_array            — step indices 1..n used as t
        All series are plotted on a single axes and saved as
        results/figures/<name>_forces.png.

    body : (vertices, triangles) or None
        Embedded-body surface.  Typically None or skipped for 2-D runs
        where the body is a line segment (cannot be represented as a valid
        triangle mesh with area > 0).  When provided:
          - Tries .vtp via pyvista (best for ParaView).
          - Falls back to .vtu triangle mesh via meshio.
          - Skips with a warning if triangles is empty or has < 3 vertices.

    base : str | pathlib.Path | None
        Override the results root directory (for tests).

    Returns
    -------
    dict {artifact_kind: pathlib.Path}
        Keys: "forces_png", "vtu", "pvsm", "body_vtp" (or "body_vtu"),
        only for artifacts actually written.
    """
    rd = results_dir(base)
    written = {}

    # ------------------------------------------------------------------ #
    # 1. Force/coefficient histories -> results/figures/<name>_forces.png #
    # ------------------------------------------------------------------ #
    if histories is not None:
        try:
            from diffsim.viz.plots import history as _history_plot

            # Normalize to {label: values_array}; build a single t axis.
            series_norm: dict[str, np.ndarray] = {}
            t_axis: np.ndarray | None = None
            for label, val in histories.items():
                if isinstance(val, (list, tuple)) and len(val) == 2 and \
                        hasattr(val[0], "__len__") and hasattr(val[1], "__len__"):
                    t_cand, y = np.asarray(val[0], np.float64), np.asarray(val[1], np.float64)
                    if t_axis is None:
                        t_axis = t_cand
                    series_norm[label] = y
                else:
                    y = np.asarray(val, np.float64)
                    if t_axis is None:
                        t_axis = np.arange(1, len(y) + 1, dtype=np.float64)
                    series_norm[label] = y

            if t_axis is None or not series_norm:
                print("[viz] save_flow_run: histories dict is empty — skipping",
                      flush=True)
            else:
                out_png = rd / "figures" / f"{name}_forces.png"
                ax = _history_plot(
                    t_axis,
                    series_norm,
                    xlabel="t",
                    ylabel="Force coefficient",
                )
                ax.set_title(f"{name} — Cd / Cl history")
                _save_fig(ax, out_png)
                print(f"[viz] wrote {out_png}", flush=True)
                written["forces_png"] = out_png

        except ImportError:
            _hint()
        except Exception as exc:
            print(f"[viz] save_flow_run: histories export failed: {exc}",
                  flush=True)

    # ------------------------------------------------------------------ #
    # 2. Mesh + fields -> results/vtu/<name>.vtu + .pvsm                  #
    # ------------------------------------------------------------------ #
    if mesh is not None and node_fields is not None:
        # --- VTU ---
        try:
            from diffsim.viz.export import export_vtu as _export_vtu

            out_vtu = rd / "vtu" / f"{name}.vtu"
            _export_vtu(mesh, out_vtu, fields=node_fields)
            print(f"[viz] wrote {out_vtu}", flush=True)
            written["vtu"] = out_vtu

        except ImportError:
            _hint()
        except Exception as exc:
            print(f"[viz] save_flow_run: VTU export failed: {exc}", flush=True)

        # --- ParaView state (.pvsm) --- only if VTU was written -------
        if "vtu" in written:
            try:
                from diffsim.viz.share import paraview_state as _pvsm

                out_pvsm = rd / "vtu" / f"{name}.pvsm"
                color_by = (
                    "velocity_magnitude"
                    if "velocity_magnitude" in node_fields
                    else next(iter(node_fields))
                )
                _pvsm(
                    written["vtu"],
                    physics="ns",
                    color_by=color_by,
                    output=out_pvsm,
                )
                print(f"[viz] wrote {out_pvsm}", flush=True)
                written["pvsm"] = out_pvsm

            except Exception as exc:
                print(f"[viz] save_flow_run: ParaView state failed: {exc}",
                      flush=True)

    # ------------------------------------------------------------------ #
    # 3. Body surface -> results/vtu/<name>_body.vtp (or .vtu fallback)   #
    # ------------------------------------------------------------------ #
    if body is not None:
        try:
            verts, tris = body
            verts = np.asarray(verts, dtype=np.float64)
            tris = np.asarray(tris, dtype=np.int64)

            if len(tris) == 0 or len(verts) < 3:
                print(
                    f"[viz] save_flow_run: body has {len(verts)} vertices and "
                    f"{len(tris)} triangles — too degenerate for VTP/VTU export "
                    "(2-D line body skipped)",
                    flush=True,
                )
            else:
                # Try .vtp (pyvista)
                try:
                    from diffsim.viz.export import export_body_vtp as _body_vtp

                    out_body = rd / "vtu" / f"{name}_body.vtp"
                    _body_vtp(verts, tris, out_body)
                    print(f"[viz] wrote {out_body}", flush=True)
                    written["body_vtp"] = out_body

                except ImportError:
                    # pyvista missing: fall back to meshio .vtu triangle mesh
                    try:
                        import meshio

                        v3 = verts if verts.shape[1] == 3 else \
                            np.column_stack([verts, np.zeros(len(verts))])
                        m = meshio.Mesh(
                            points=v3,
                            cells=[("triangle", tris.astype(np.int32))],
                        )
                        out_body = rd / "vtu" / f"{name}_body.vtu"
                        meshio.write(str(out_body), m)
                        print(f"[viz] wrote {out_body} (meshio fallback, pyvista absent)",
                              flush=True)
                        written["body_vtu"] = out_body

                    except ImportError:
                        _hint()
                    except Exception as exc2:
                        print(f"[viz] save_flow_run: body VTU fallback failed: {exc2}",
                              flush=True)

        except Exception as exc:
            print(f"[viz] save_flow_run: body export failed: {exc}", flush=True)

    return written
