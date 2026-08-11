"""benchmarks/chns/movie.py — Interface movie writer.

Produces an animated GIF of the phi=0 contour evolution using matplotlib
with the Agg backend (no display required). Frames are assembled with
matplotlib.animation.PillowWriter.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def write_interface_movie(
    snapshots: list[np.ndarray],
    coords: np.ndarray,
    path: str,
    fps: int = 10,
) -> None:
    """Write a GIF animation of the phi=0 interface contour for each snapshot.

    Each frame plots the phi=0 level set on a 2-D domain inferred from coords.
    The phi field is reconstructed onto a structured 2-D grid from the scattered
    (coords, phi) data, then matplotlib.pyplot.contour is used to draw the
    zero level set.

    Args:
        snapshots: list of [n] phi arrays, one per frame.
        coords   : [n, 2] node coordinates (uniform structured mesh assumed).
        path     : output file path; should end in '.gif'.
        fps      : frames per second for the animation (default 10).

    Raises:
        ValueError: if snapshots is empty.
        RuntimeError: if the output file is not created (write failed).
    """
    if not snapshots:
        raise ValueError("snapshots list is empty; nothing to write.")

    coords = np.asarray(coords, dtype=float)

    # Reconstruct grid dimensions from coords
    xs = np.unique(np.round(coords[:, 0], decimals=12))
    ys = np.unique(np.round(coords[:, 1], decimals=12))
    nx, ny = len(xs), len(ys)

    ix = np.searchsorted(xs, np.round(coords[:, 0], decimals=12))
    iy = np.searchsorted(ys, np.round(coords[:, 1], decimals=12))

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)

    # Build phi grid helper
    def _phi_grid(phi: np.ndarray) -> np.ndarray:
        grid = np.zeros((nx, ny))
        grid[ix, iy] = np.asarray(phi, dtype=float)
        return grid

    # Use PillowWriter directly with grab_frame per frame
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("phi = 0 interface")

    writer = animation.PillowWriter(fps=fps)
    with writer.saving(fig, path, dpi=80):
        for phi in snapshots:
            ax.clear()
            ax.set_xlim(xs[0], xs[-1])
            ax.set_ylim(ys[0], ys[-1])
            ax.set_aspect("equal")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("phi = 0 interface")

            grid = _phi_grid(phi)
            phi_min, phi_max = grid.min(), grid.max()
            if phi_min < 0.0 < phi_max:
                ax.contour(xs, ys, grid.T, levels=[0.0], colors=["blue"])

            writer.grab_frame()

    plt.close(fig)

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"Movie write failed: '{path}' is missing or empty.")
