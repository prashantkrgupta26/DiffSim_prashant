"""Shared guarded visualization helper for DiffSim tutorials.

ALL tutorial viz goes through this module — no tutorial imports diffsim.viz
directly.  When the [viz] extra is absent (matplotlib / pyvista / meshio not
installed), every emitter no-ops, prints a one-line install hint, and returns
None.  No exception is ever raised for a missing dependency.

Public interface
----------------
HAS_VIZ : bool
    True iff ``import diffsim.viz`` succeeded at module load time.

figures_dir(tutorial_file) -> pathlib.Path
    Returns ``<tutorial_dir>/figures/``, creating it on demand.

Emitters (all return None on failure):
  convergence(tutorial_file, levels, values, stem, *, reference=None, slope=None,
              xlabel="h", ylabel="Error", label="DiffSim")
      Log-log error-vs-h convergence plot via plots.convergence.

  history(tutorial_file, t, series, stem, *, xlabel="t", ylabel="value")
      Time-history plot (Cd/Cl/energy/…) via plots.history.

  bar_chart(tutorial_file, labels, values, stem, *, xlabel="", ylabel="value",
            title=None)
      Categorical bar chart (per-solver time, method comparison) — matplotlib
      direct, for categorical x-axes where history/convergence do not apply.

  surface_profile(tutorial_file, x, series, stem, *, reference=None,
                  xlabel="arc-length", ylabel="Cp")
      Surface profile (centerline u, Cp, …) via plots.surface_profile.

  field(tutorial_file, vtu_path, stem, *, field_name="velocity_magnitude",
        physics="ns")
      GL-guarded field render (LIC or contour) via renders.lic/contour.
      No-ops with a hint if GL is unavailable.

  vtu(tutorial_file, source, stem, *, fields=None)
      Export a DiffSim Mesh (or .npz path) to a .vtu file via export.export_vtu.

  vtu_sbm(tutorial_file, mesh, stem, *, fields=None, retained_tree,
          frac, geom=None, sf=None)
      SBM-extended VTU export via export.export_vtu_sbm.

  body_vtp(tutorial_file, vertices, triangles, stem)
      Embedded-body surface export via export.export_body_vtp.

  paraview_state(tutorial_file, vtu_path, stem, *, physics="film",
                 color_by=None, camera=None)
      Write a ParaView .pvsm state file via share.paraview_state.
"""
from __future__ import annotations

import pathlib
import sys

# ──────────────────────────────────────────────────────────────────────────────
# One-time guarded import of diffsim.viz
# ──────────────────────────────────────────────────────────────────────────────
try:
    import diffsim.viz as _dv
    from diffsim.viz import renders as _renders
    HAS_VIZ = True
except Exception:
    _dv = None
    _renders = None
    HAS_VIZ = False

_HINT = (
    "[viz] Install hint: pip install 'diffsim[viz]'  "
    "(needs matplotlib + pyvista + meshio)"
)


def _hint() -> None:
    """Print the one-line install hint to stdout."""
    print(_HINT)


# ──────────────────────────────────────────────────────────────────────────────
# figures_dir
# ──────────────────────────────────────────────────────────────────────────────

def figures_dir(tutorial_file: str | pathlib.Path) -> pathlib.Path:
    """Return ``<tutorial_dir>/figures/``, creating it on demand.

    Pass ``__file__`` of the calling tutorial.
    """
    d = pathlib.Path(tutorial_file).resolve().parent / "figures"
    d.mkdir(exist_ok=True)
    return d


# ──────────────────────────────────────────────────────────────────────────────
# Internal save helper
# ──────────────────────────────────────────────────────────────────────────────

def _save_fig(fig_or_ax, out_path: pathlib.Path) -> None:
    """Save a matplotlib Figure or Axes to *out_path*."""
    try:
        import matplotlib.pyplot as plt  # local guard
        fig = getattr(fig_or_ax, "figure", None) or fig_or_ax
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        print(f"[viz] warning: could not save {out_path}: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Emitters
# ──────────────────────────────────────────────────────────────────────────────

def convergence(
    tutorial_file: str | pathlib.Path,
    levels,
    values,
    stem: str,
    *,
    reference=None,
    slope: float = None,
    xlabel: str = "h",
    ylabel: str = "Error",
    label: str = "DiffSim",
) -> pathlib.Path | None:
    """Log-log convergence plot (error vs h).  Wraps plots.convergence."""
    if not HAS_VIZ:
        _hint()
        return None
    try:
        out = figures_dir(tutorial_file) / f"{stem}.png"
        ax = _dv.convergence(
            levels, values, reference,
            slope=slope, xlabel=xlabel, ylabel=ylabel, label=label,
        )
        _save_fig(ax, out)
        print(f"[viz] wrote {out}")
        return out
    except Exception as exc:
        print(f"[viz] convergence failed: {exc}")
        return None


def history(
    tutorial_file: str | pathlib.Path,
    t,
    series: dict,
    stem: str,
    *,
    xlabel: str = "t",
    ylabel: str = "value",
) -> pathlib.Path | None:
    """Time-history plot (Cd, Cl, energy, …).  Wraps plots.history."""
    if not HAS_VIZ:
        _hint()
        return None
    try:
        out = figures_dir(tutorial_file) / f"{stem}.png"
        ax = _dv.history(t, series, xlabel=xlabel, ylabel=ylabel)
        _save_fig(ax, out)
        print(f"[viz] wrote {out}")
        return out
    except Exception as exc:
        print(f"[viz] history failed: {exc}")
        return None


def bar_chart(
    tutorial_file: str | pathlib.Path,
    labels,
    values,
    stem: str,
    *,
    xlabel: str = "",
    ylabel: str = "value",
    title: str | None = None,
) -> pathlib.Path | None:
    """Categorical bar chart (e.g. per-solver wall time).

    Uses matplotlib directly — diffsim.viz has no bar helper, and HAS_VIZ
    implies matplotlib is importable.  For categorical x-axes (solver names,
    method labels) where the time-history/convergence plots do not apply.
    """
    if not HAS_VIZ:
        _hint()
        return None
    try:
        import matplotlib.pyplot as plt

        out = figures_dir(tutorial_file) / f"{stem}.png"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([str(x) for x in labels], list(values))
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        _save_fig(ax, out)
        print(f"[viz] wrote {out}")
        return out
    except Exception as exc:
        print(f"[viz] bar_chart failed: {exc}")
        return None


def surface_profile(
    tutorial_file: str | pathlib.Path,
    x,
    series: dict,
    stem: str,
    *,
    reference: dict = None,
    xlabel: str = "arc-length",
    ylabel: str = "Cp",
) -> pathlib.Path | None:
    """Surface profile plot (centerline u, Cp, …).  Wraps plots.surface_profile."""
    if not HAS_VIZ:
        _hint()
        return None
    try:
        out = figures_dir(tutorial_file) / f"{stem}.png"
        ax = _dv.surface_profile(
            x, series, reference, xlabel=xlabel, ylabel=ylabel,
        )
        _save_fig(ax, out)
        print(f"[viz] wrote {out}")
        return out
    except Exception as exc:
        print(f"[viz] surface_profile failed: {exc}")
        return None


def field(
    tutorial_file: str | pathlib.Path,
    vtu_path: str | pathlib.Path,
    stem: str,
    *,
    field_name: str = "velocity_magnitude",
    physics: str = "ns",
) -> pathlib.Path | None:
    """GL-guarded field render (LIC for flow, contour for scalars).

    Uses renders.lic for ns physics, renders.contour otherwise.
    No-ops with a hint if pyvista is missing or GL is unavailable.
    """
    if not HAS_VIZ:
        _hint()
        return None
    try:
        if not _renders.offscreen_gl_ok():
            print("[viz] field: no offscreen GL — skipping render "
                  "(install EGL/OSMesa or use a display)")
            return None
        out = figures_dir(tutorial_file) / f"{stem}.png"
        if physics == "ns":
            _dv.lic(vtu_path, color_by=field_name, output=out)
        else:
            _dv.contour(vtu_path, field=field_name, output=out)
        print(f"[viz] wrote {out}")
        return out
    except Exception as exc:
        print(f"[viz] field render failed: {exc}")
        return None


def vtu(
    tutorial_file: str | pathlib.Path,
    source,
    stem: str,
    *,
    fields: dict = None,
) -> pathlib.Path | None:
    """Export a DiffSim Mesh (or .npz path) to a .vtu file.

    Wraps export.export_vtu.  Requires meshio (part of [viz] extra).
    """
    if not HAS_VIZ:
        _hint()
        return None
    try:
        out = figures_dir(tutorial_file) / f"{stem}.vtu"
        _dv.export_vtu(source, out, fields=fields)
        print(f"[viz] wrote {out}")
        return out
    except Exception as exc:
        print(f"[viz] vtu export failed: {exc}")
        return None


def vtu_sbm(
    tutorial_file: str | pathlib.Path,
    mesh,
    stem: str,
    *,
    fields: dict = None,
    retained_tree,
    frac,
    geom=None,
    sf=None,
) -> pathlib.Path | None:
    """SBM-extended VTU export.  Wraps export.export_vtu_sbm."""
    if not HAS_VIZ:
        _hint()
        return None
    try:
        out = figures_dir(tutorial_file) / f"{stem}.vtu"
        _dv.export_vtu_sbm(
            mesh, out,
            fields=fields,
            retained_tree=retained_tree,
            frac=frac,
            geom=geom,
            sf=sf,
        )
        print(f"[viz] wrote {out}")
        return out
    except Exception as exc:
        print(f"[viz] vtu_sbm export failed: {exc}")
        return None


def body_vtp(
    tutorial_file: str | pathlib.Path,
    vertices,
    triangles,
    stem: str,
) -> pathlib.Path | None:
    """Write an embedded-body surface as .vtp.  Wraps export.export_body_vtp."""
    if not HAS_VIZ:
        _hint()
        return None
    try:
        out = figures_dir(tutorial_file) / f"{stem}.vtp"
        _dv.export_body_vtp(vertices, triangles, out)
        print(f"[viz] wrote {out}")
        return out
    except Exception as exc:
        print(f"[viz] body_vtp export failed: {exc}")
        return None


def paraview_state(
    tutorial_file: str | pathlib.Path,
    vtu_path: str | pathlib.Path,
    stem: str,
    *,
    physics: str = "film",
    color_by: str = None,
    camera: dict = None,
) -> pathlib.Path | None:
    """Write a ParaView .pvsm state file.  Wraps share.paraview_state."""
    if not HAS_VIZ:
        _hint()
        return None
    try:
        out = figures_dir(tutorial_file) / f"{stem}.pvsm"
        _dv.paraview_state(
            vtu_path,
            physics=physics,
            color_by=color_by,
            camera=camera,
            output=out,
        )
        print(f"[viz] wrote {out}")
        return out
    except Exception as exc:
        print(f"[viz] paraview_state failed: {exc}")
        return None
