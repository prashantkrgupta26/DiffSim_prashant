"""Matplotlib quantitative plots (Phase 1, spec §6).

Three figure-matched helpers drawing from a run's scalar data (not the VTU):
  convergence      — log-log Cd / L2-error vs h with a reference-slope line
  surface_profile  — Cp / Cf / Nu vs arc-length w/ literature overlays + inset
  history          — Cd / Cl time histories

matplotlib is a core dep (no [viz] extra needed).  The `Agg` backend is forced
so the helpers work head-less (no display) — they are figure code, not a viewer.
"""
import matplotlib
matplotlib.use("Agg", force=False)  # head-less; respect an already-set backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


# Group paper-figure convention: blue line / red marker + dashed grid.
_DEFAULT_SERIES_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
_DEFAULT_MARKER = "o"
_GRID_KW = dict(linestyle="--", alpha=0.5)


def _get_ax(ax):
    if ax is None:
        _, ax = plt.subplots()
    return ax


def convergence(
    levels,
    values,
    reference=None,
    *,
    slope=None,
    xlabel: str = "h",
    ylabel: str = "Error",
    label: str = "DiffSim",
    style_kw: dict = None,
    ax=None,
):
    """Log-log Cd / L2-error vs h with a reference-slope line (spec §6).

    `levels`: integer refinement levels (h = 2**-level). `values`: the scalar
    metric per level. If `reference` is given, plots |value - reference|.
    Returns the Axes.
    """
    ax = _get_ax(ax)
    levels = np.asarray(levels, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    h = 2.0 ** (-levels)
    y = np.abs(values - reference) if reference is not None else values

    kw = dict(color=_DEFAULT_SERIES_COLORS[0], marker=_DEFAULT_MARKER)
    if style_kw:
        kw.update(style_kw)
    ax.loglog(h, y, label=label, **kw)

    if slope is not None:
        # anchor the reference triangle at the finest (h, y) point
        h_ref = np.array([h.min(), h.max()])
        y_anchor = y[np.argmin(h)]
        y_ref = y_anchor * (h_ref / h.min()) ** slope
        ax.loglog(h_ref, y_ref, "k--",
                  label=fr"$\mathcal{{O}}(h^{{{slope:g}}})$")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", **_GRID_KW)
    ax.legend()
    return ax


def surface_profile(
    x,
    series: dict,
    reference: dict = None,
    *,
    inset: dict = None,
    xlabel: str = "arc-length",
    ylabel: str = "Cp",
    style_kw: dict = None,
    ax=None,
):
    """Cp / Cf / Nu vs arc-length with literature overlays + inset zoom (spec §6).

    `series`: {label: y_array} main curves against `x`.
    `reference`: optional {label: (x_ref, y_ref)} literature overlays (markers).
    `inset`: if given, adds an inset axes zoomed to inset["xlim"], inset["ylim"].
    Returns the main Axes.
    """
    ax = _get_ax(ax)
    x = np.asarray(x, dtype=np.float64)

    def _draw(target):
        for i, (lbl, y) in enumerate(series.items()):
            kw = dict(color=_DEFAULT_SERIES_COLORS[i % len(_DEFAULT_SERIES_COLORS)])
            if style_kw:
                kw.update(style_kw)
            target.plot(x, np.asarray(y, dtype=np.float64), label=lbl, **kw)
        if reference:
            for j, (lbl, (xr, yr)) in enumerate(reference.items()):
                c = _DEFAULT_SERIES_COLORS[
                    (len(series) + j) % len(_DEFAULT_SERIES_COLORS)]
                target.plot(np.asarray(xr), np.asarray(yr), linestyle="none",
                            marker=_DEFAULT_MARKER, color=c, label=lbl)

    _draw(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, **_GRID_KW)
    ax.legend()

    if inset:
        axins = ax.inset_axes([0.55, 0.55, 0.4, 0.4])
        _draw(axins)
        axins.set_xlim(*inset["xlim"])
        axins.set_ylim(*inset["ylim"])
        axins.grid(True, **_GRID_KW)
        ax.indicate_inset_zoom(axins, edgecolor="black")
    return ax


def history(
    t,
    series: dict,
    *,
    xlabel: str = "t",
    ylabel: str = "Cd",
    style_kw: dict = None,
    ax=None,
):
    """Cd / Cl time histories — drag histories, shedding (spec §6)."""
    ax = _get_ax(ax)
    t = np.asarray(t, dtype=np.float64)
    for i, (lbl, y) in enumerate(series.items()):
        kw = dict(color=_DEFAULT_SERIES_COLORS[i % len(_DEFAULT_SERIES_COLORS)])
        if style_kw:
            kw.update(style_kw)
        ax.plot(t, np.asarray(y, dtype=np.float64), label=lbl, **kw)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, **_GRID_KW)
    ax.legend()
    return ax
