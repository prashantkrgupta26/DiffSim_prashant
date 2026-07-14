"""Morphology diagnostics — structure factor, characteristic length scales,
two-point correlation, interfacial area, anisotropy, and phase fractions.

These operate on a **gridded** field snapshot (a regular ``(N, N)`` or
``(N, N, N)`` array), which is what the tutorials save as morphology output.
They quantify the length scale and topology of a phase-separated pattern so
coarsening ``L(t)`` and the linear-stability prediction (fastest ``k`` ->
wavelength) can be compared against a measurement (spec P1).

Units
-----
Lengths are returned in the same physical units as the supplied grid spacing
``dx`` (default ``dx=1`` -> "grid cells"). Wavevectors are in ``2*pi/length``.
The structure factor is dimensionless (power per mode of the demeaned field).
"""
from __future__ import annotations

import numpy as np


def _demean(field):
    f = np.asarray(field, dtype=float)
    return f - f.mean()


def structure_factor(field, dx=1.0):
    """Radially averaged structure factor ``S(q)`` of a gridded field.

    ``S(q) = <|FFT(phi - <phi>)|^2>`` averaged over shells of constant
    ``|q|``. The peak of ``S(q)`` locates the dominant modulation wavevector
    of the morphology.

    Parameters
    ----------
    field : ndarray
        2-D or 3-D real field snapshot.
    dx : float
        Grid spacing (physical length per cell).

    Returns
    -------
    q : ndarray
        Shell wavenumbers ``|q|`` (units ``2*pi/length``), monotone increasing.
    S : ndarray
        Radially averaged power at each ``q`` (dimensionless).
    """
    f = _demean(field)
    dim = f.ndim
    fk = np.fft.fftn(f)
    power = (fk * np.conj(fk)).real / f.size
    freqs = [np.fft.fftfreq(n, d=dx) * 2.0 * np.pi for n in f.shape]
    grids = np.meshgrid(*freqs, indexing="ij")
    qmag = np.sqrt(sum(g ** 2 for g in grids))
    # bin by |q| using the fundamental spacing as bin width
    dq = 2.0 * np.pi / (max(f.shape) * dx)
    nbins = int(np.ceil(qmag.max() / dq)) + 1
    idx = np.minimum((qmag / dq).astype(int), nbins - 1).ravel()
    p = power.ravel()
    S = np.bincount(idx, weights=p, minlength=nbins)
    counts = np.bincount(idx, minlength=nbins)
    counts = np.where(counts == 0, 1, counts)
    S = S / counts
    q = (np.arange(nbins) + 0.5) * dq
    # drop the DC bin (q~0) which is ~0 after demeaning
    return q[1:], S[1:]


def peak_wavelength(field, dx=1.0):
    """Dominant morphology wavelength from the peak of ``S(q)``.

    ``lambda_peak = 2*pi / q_peak`` where ``q_peak = argmax S(q)``.
    Units: same as ``dx``.
    """
    q, S = structure_factor(field, dx=dx)
    if q.size == 0:
        return float("nan")
    qp = q[int(np.argmax(S))]
    return float(2.0 * np.pi / qp) if qp > 0 else float("nan")


def first_moment_wavelength(field, dx=1.0):
    """Characteristic wavelength from the first moment of ``S(q)``.

    ``q1 = sum(q S) / sum(S)``, ``lambda_1 = 2*pi / q1``. More stable than the
    raw peak for noisy spectra; the two together bracket the length scale
    (spec P1: ">=2 length definitions"). Units: same as ``dx``.
    """
    q, S = structure_factor(field, dx=dx)
    denom = float(np.sum(S))
    if denom <= 0:
        return float("nan")
    q1 = float(np.sum(q * S) / denom)
    return float(2.0 * np.pi / q1) if q1 > 0 else float("nan")


def two_point_correlation(field, max_lag=None, dx=1.0):
    """Radially averaged two-point autocorrelation ``C(r)`` of the field.

    Computed via the Wiener-Khinchin theorem (inverse FFT of the power
    spectrum) on the demeaned field. ``C(0)`` equals the field variance; the
    first zero crossing is a real-space correlation length.

    Parameters
    ----------
    field : ndarray
        2-D or 3-D field.
    max_lag : int, optional
        Largest lag (in cells) to return. Default ``min(shape)//2``.
    dx : float
        Grid spacing.

    Returns
    -------
    r : ndarray
        Lag distances (units of ``dx``).
    C : ndarray
        Autocovariance at each lag; ``C[0] == var(field)``.
    """
    f = _demean(field)
    fk = np.fft.fftn(f)
    corr = np.fft.ifftn((fk * np.conj(fk))).real / f.size
    corr = np.fft.fftshift(corr)
    center = np.array(corr.shape) // 2
    grids = np.meshgrid(*[np.arange(n) - c for n, c in zip(corr.shape, center)],
                        indexing="ij")
    rmag = np.sqrt(sum(g ** 2 for g in grids))
    if max_lag is None:
        max_lag = min(corr.shape) // 2
    idx = np.minimum(np.round(rmag).astype(int), max_lag).ravel()
    C = np.bincount(idx, weights=corr.ravel(), minlength=max_lag + 1)
    counts = np.bincount(idx, minlength=max_lag + 1)
    counts = np.where(counts == 0, 1, counts)
    C = C[:max_lag + 1] / counts[:max_lag + 1]
    r = np.arange(max_lag + 1) * dx
    return r, C


def correlation_length(field, dx=1.0):
    """Real-space correlation length: first zero crossing of ``C(r)``.

    Units: same as ``dx``. Returns ``nan`` if ``C(r)`` has no zero crossing in
    the sampled range.
    """
    r, C = two_point_correlation(field, dx=dx)
    sign = np.sign(C)
    crossings = np.where(np.diff(sign) != 0)[0]
    if crossings.size == 0:
        return float("nan")
    i = crossings[0]
    # linear interpolation of the zero crossing between r[i], r[i+1]
    c0, c1 = C[i], C[i + 1]
    if c1 == c0:
        return float(r[i])
    frac = c0 / (c0 - c1)
    return float(r[i] + frac * (r[i + 1] - r[i]))


def interfacial_area(field, dx=1.0, threshold=None):
    """Interfacial length (2-D) / area (3-D) of the level set at ``threshold``.

    Estimated as ``INT |grad H(phi - t)| ~ sum over cells crossed`` via the
    magnitude of the discrete gradient of the thresholded indicator. This is
    the diffuse-interface analogue used to track total interface as domains
    coarsen (it should DECREASE during coarsening).

    Parameters
    ----------
    field : ndarray
        2-D or 3-D field.
    dx : float
        Grid spacing.
    threshold : float, optional
        Level value. Default is the field mean (the natural 50% level).

    Returns
    -------
    float
        Interface measure. Units: ``length**(dim-1)``.
    """
    f = np.asarray(field, dtype=float)
    if threshold is None:
        threshold = float(f.mean())
    ind = (f >= threshold).astype(float)
    grads = np.gradient(ind, dx)
    if f.ndim == 1:
        grads = [grads]
    gmag = np.sqrt(sum(g ** 2 for g in grads))
    # integrate |grad indicator| over the domain
    cell_vol = dx ** f.ndim
    return float(np.sum(gmag) * cell_vol)


def phase_fractions(field, threshold=None):
    """Area/volume fractions of the two phases split at ``threshold``.

    Returns ``(frac_low, frac_high)`` summing to 1 (dimensionless). Default
    threshold is the field mean.
    """
    f = np.asarray(field, dtype=float)
    if threshold is None:
        threshold = float(f.mean())
    high = float(np.mean(f >= threshold))
    return (1.0 - high, high)


def anisotropy(field, dx=1.0):
    """Directional anisotropy of the pattern from the power spectrum.

    Returns a dimensionless ratio in ``[0, 1)``: 0 for an isotropic
    (direction-independent) morphology, larger for a pattern with a preferred
    orientation. Defined as the normalised magnitude of the second angular
    Fourier moment of the 2-D power spectrum. 2-D only.
    """
    f = _demean(field)
    if f.ndim != 2:
        raise ValueError("anisotropy is defined for 2-D fields")
    fk = np.fft.fftshift(np.fft.fftn(f))
    power = (fk * np.conj(fk)).real
    ny, nx = f.shape
    ky = (np.arange(ny) - ny // 2)[:, None]
    kx = (np.arange(nx) - nx // 2)[None, :]
    theta = np.arctan2(ky, kx)
    mask = (kx ** 2 + ky ** 2) > 0
    w = power[mask]
    t = theta[mask]
    denom = float(np.sum(w))
    if denom <= 0:
        return 0.0
    # second angular moment (period pi for an unoriented axis)
    c2 = float(np.sum(w * np.cos(2 * t)) / denom)
    s2 = float(np.sum(w * np.sin(2 * t)) / denom)
    return float(np.hypot(c2, s2))


__all__ = [
    "structure_factor", "peak_wavelength", "first_moment_wavelength",
    "two_point_correlation", "correlation_length", "interfacial_area",
    "phase_fractions", "anisotropy",
]
