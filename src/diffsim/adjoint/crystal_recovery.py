"""Structure-factor descriptor S(k) and its field-space gradient — Task 2.

Provides three pure-numpy functions for the crystal free-energy recovery harness:

- ``nodal_to_grid``     — reshape a (shuffled) nodal field to a 2-D grid
- ``structure_factor``  — radially-averaged power spectrum S(k) of a 2-D field
- ``structure_factor_grad`` — ∂(0.5‖S(field)−target‖²)/∂field, analytic adjoint

These are format-agnostic (no engine imports) and dtype-clean (real fields in,
real gradient out).  The gradient is verified against central finite differences
in tests/test_crystal_recovery.py (rtol 1e-4).

Normalization notes
-------------------
For a real input ``f``, ``F_k = fft2(f)[k] = sum_j f_j exp(−2πi·k·j/N)``.
The chain rule through ``P_k = |F_k|²`` back to ``f_j`` gives:

    d|F_k|²/df_j = 2·Re(conj(F_k) · exp(−2πi·k·j/N))

So the full spatial gradient is:

    dL/df_j = 2·Re( sum_k dL_dP_k · conj(F_k) · exp(−2πi·k·j/N) )
            = 2·Re( fft2(dL_dP · conj(F))[j] )

This uses ``fft2`` (forward DFT, exponent −2πi), NOT ``ifft2``.
The mean-subtraction ``field → field − mean(field)`` has adjoint ``y → y − mean(y)``.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# nodal_to_grid
# ---------------------------------------------------------------------------

def nodal_to_grid(field_1d: np.ndarray, mesh) -> np.ndarray:
    """Reshape a (potentially shuffled) nodal field to a regular 2-D grid.

    Parameters
    ----------
    field_1d : (Nn,) array
        Per-node scalar values in node-index order.
    mesh : mesh-like object
        Must expose ``node_coords`` (shape ``(Nn, ≥2)``), coordinates of each
        node.  The first two columns are used (x = col 0, y = col 1).

    Returns
    -------
    grid : (ny, nx) ndarray
        Values on the 2-D grid; ``grid[iy, ix]`` corresponds to the node at
        ``(xs[ix], ys[iy])``.
    """
    coords = np.asarray(mesh.node_coords, dtype=float)[:, :2]  # (Nn, 2): x, y
    x_raw = np.round(coords[:, 0], 12)
    y_raw = np.round(coords[:, 1], 12)
    xs = np.unique(x_raw)
    ys = np.unique(y_raw)
    # lexsort: primary key = y (slowest-varying → rows), secondary = x (fastest)
    # np.lexsort sorts by LAST key first, so we pass (x, y) to sort (y, x).
    order = np.lexsort((x_raw, y_raw))
    return field_1d[order].reshape(len(ys), len(xs))


# ---------------------------------------------------------------------------
# structure_factor
# ---------------------------------------------------------------------------

def structure_factor(field_2d: np.ndarray, nbins: int | None = None) -> np.ndarray:
    """Radially-averaged power spectrum S(k) of a real 2-D field.

    Computes ``P = |FFT2(field − mean(field))|²``, bins by integer wavenumber
    magnitude, and returns the per-bin MEAN.

    Parameters
    ----------
    field_2d : (ny, nx) real array
    nbins : int, optional
        Number of radial bins (0 … nbins-1).  Default ``min(ny, nx) // 2``.

    Returns
    -------
    S : (nbins,) float64 array
        Per-bin mean power.  Bins with no wavenumber contribute 0.
    """
    field_2d = np.asarray(field_2d, dtype=float)
    ny, nx = field_2d.shape
    if nbins is None:
        nbins = min(ny, nx) // 2

    F = np.fft.fft2(field_2d - field_2d.mean())
    P = np.abs(F) ** 2  # (ny, nx), real

    # integer wavenumber magnitudes (pixel-frequency units, not physical)
    kx = np.fft.fftfreq(nx) * nx   # shape (nx,)
    ky = np.fft.fftfreq(ny) * ny   # shape (ny,)
    kr = np.round(np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)).astype(int)  # (ny, nx)

    # radial sum and count, sliced to [0, nbins)
    kr_flat = kr.ravel()
    P_flat = P.ravel()
    S_sum = np.bincount(kr_flat, weights=P_flat, minlength=nbins)[:nbins]
    S_cnt = np.bincount(kr_flat, minlength=nbins)[:nbins].astype(float)
    # guard divide-by-zero on empty bins
    S = np.where(S_cnt > 0, S_sum / S_cnt, 0.0)
    return S


# ---------------------------------------------------------------------------
# structure_factor_grad
# ---------------------------------------------------------------------------

def structure_factor_grad(
    field_2d: np.ndarray,
    target_S: np.ndarray,
    nbins: int | None = None,
) -> np.ndarray:
    """Field-space gradient of the structure-factor mismatch loss.

    Computes ``∂L/∂field`` where ``L = 0.5 ‖S(field) − target_S‖²``
    analytically via the chain rule through:
      1. mean-subtraction
      2. fft2 (DFT)
      3. element-wise |·|²
      4. radial-averaging (linear operator A)

    Parameters
    ----------
    field_2d : (ny, nx) real array
        Current field.
    target_S : (nbins,) array
        Target structure factor (e.g. from experiment or reference snapshot).
    nbins : int, optional
        Must match the nbins used to compute target_S.

    Returns
    -------
    grad : (ny, nx) float64 array
        ∂L/∂field — real, same shape as ``field_2d``.

    Normalization
    -------------
    The chain rule through ``|fft2(f)|²`` for a real input ``f`` gives:

        d|F_k|²/df_j = 2 · Re(conj(F_k) · exp(−2πi·k·j/N))

    Summing over k:

        dL/df_j = 2 · Re( sum_k dL/dP_k · conj(F_k) · exp(−2πi·k·j/N) )
                = 2 · Re( fft2(dL/dP · conj(F))[j] )

    (This uses fft2, NOT ifft2 — the exponent is −2πi, same as the forward DFT.)
    """
    field_2d = np.asarray(field_2d, dtype=float)
    ny, nx = field_2d.shape
    N = ny * nx  # total number of pixels (used in dL_dP flat array sizing)
    if nbins is None:
        nbins = min(ny, nx) // 2

    # ---------- forward pass (re-use structure_factor logic inline) ----------
    field_c = field_2d - field_2d.mean()   # mean-subtracted
    F = np.fft.fft2(field_c)               # complex (ny, nx)
    P = np.abs(F) ** 2                     # real   (ny, nx)

    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    kr = np.round(np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)).astype(int)
    kr_flat = kr.ravel()

    P_flat = P.ravel()
    S_sum = np.bincount(kr_flat, weights=P_flat, minlength=nbins)[:nbins]
    S_cnt = np.bincount(kr_flat, minlength=nbins)[:nbins].astype(float)
    S = np.where(S_cnt > 0, S_sum / S_cnt, 0.0)

    # ---------- backward pass ------------------------------------------------
    # dL/dS  (per-bin cotangent)
    dL_dS = S - np.asarray(target_S, dtype=float)  # (nbins,)

    # dL/dP  — adjoint of radial averaging:
    #   each pixel in bin b gets  dL_dS[b] / count[b]
    dL_dP_flat = np.zeros(N, dtype=float)
    # only pixels whose bin index < nbins contribute
    mask = kr_flat < nbins
    dL_dP_flat[mask] = (dL_dS[kr_flat[mask]]
                        / np.where(S_cnt[kr_flat[mask]] > 0,
                                   S_cnt[kr_flat[mask]], 1.0))
    dL_dP = dL_dP_flat.reshape(ny, nx)

    # dL/d(field_centered) — chain rule through |F|² and fft2 together:
    #   d|F_k|²/df_j = 2·Re(conj(F_k)·exp(−2πi·k·j/N))
    #   dL/df_j = 2·Re( sum_k dL_dP_k · conj(F_k) · exp(−2πi·k·j/N) )
    #           = 2·Re( fft2(dL_dP · conj(F))[j] )
    # The forward fft2 exponent (−2πi) is what we want here — NOT ifft2.
    dL_dfc = 2.0 * np.real(np.fft.fft2(dL_dP * np.conj(F)))  # real (ny, nx)

    # dL/dfield — adjoint of mean-subtraction:
    #   field → field − mean(field) has adjoint y → y − mean(y)
    grad = dL_dfc - dL_dfc.mean()
    return grad
