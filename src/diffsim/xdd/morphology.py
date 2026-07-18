"""XDD morphology module (SP-1 Task A2).

Provides:
- Morphology dataclass — structured Cartesian grid morphology
- read_cpu_cloud()    — parse the CPU legacy .txt cloud format
- write_cpu_cloud()   — inverse writer for round-trip support
- signed_distance()   — scipy EDT-based signed distance utility with mid-plane correction
- from_film_npz()     — reader for DiffSim film-output npz files
- tanh_mask()         — relaxed phase indicator (→1 in acceptor)
- region_weights()    — (w_donor, w_acceptor, w_interface) weights
- interface_mask()    — smooth / sharp interface indicator
- descriptors()       — GraSPI-compatible morphology statistics

Block A: PURE numpy/scipy/stdlib — do NOT import warp or torch.

Sign convention (CPU ground truth):
  dist > 0 : acceptor side
  dist < 0 : donor side
  dist = 0 : at the interface

signed_distance() formula and mid-plane correction:
  EDT is voxel-center-to-voxel-center, so the raw field
      raw = edt(morph >= 0.5, sampling) - edt(morph < 0.5, sampling)
  places the zero-crossing half a voxel into the acceptor region rather than
  at the physical interface mid-plane.  signed_distance() applies the correction
      dist = raw - 0.5 * max(spacing) * sign(raw)
  shifting every non-zero value half a voxel toward zero.  After correction,
  deviations from the CPU ground-truth file (morph_bilayer_2D.txt) are
  symmetric ±0.5 voxel on both donor and acceptor sides.

CPU file convention and parity:
  The CPU files (morph_bilayer_2D.txt) anchor dist=0 at the last acceptor
  node rather than the geometric mid-plane.  Against that convention, our
  utility deviates by up to ±0.5 * max(spacing) symmetrically.
  The CPU-parity path (read_cpu_cloud) uses the file-provided dist field
  directly and is unaffected by this utility.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt


# ──────────────────────────────────────────────────────────────────────────────
# Morphology dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Morphology:
    """Structured Cartesian morphology grid.

    Parameters
    ----------
    morph : np.ndarray
        Float array of shape (nx, ny) or (nx, ny, nz).  Values in {0., 1.}
        where 0 = donor, 1 = acceptor.
    dist : np.ndarray
        Signed distance field [m], same shape as morph.
        Negative = donor side, positive = acceptor side, 0 at interface.
        (CPU sign convention from morph_bilayer_2D.txt.)
    spacing : tuple
        (dx, dy) or (dx, dy, dz) in metres.
    """
    morph:   np.ndarray
    dist:    np.ndarray
    spacing: tuple

    @property
    def shape(self) -> tuple:
        return self.morph.shape

    @property
    def ndim(self) -> int:
        return self.morph.ndim

    @property
    def donor_fraction(self) -> float:
        """Fraction of nodes labelled donor (morph < 0.5)."""
        return float(np.mean(self.morph < 0.5))

    @property
    def acceptor_fraction(self) -> float:
        """Fraction of nodes labelled acceptor (morph >= 0.5)."""
        return float(np.mean(self.morph >= 0.5))


# ──────────────────────────────────────────────────────────────────────────────
# CPU cloud reader / writer
# ──────────────────────────────────────────────────────────────────────────────

def read_cpu_cloud(
    path: str | Path,
    physical_size: Sequence[float],
) -> Morphology:
    """Parse a CPU legacy morphology cloud file and return a Morphology.

    File format:
      Line 1: ``n_total  n_nodes_x  n_nodes_y [n_nodes_z]``  (optional comment)
      Then n_total lines: ``ix  iy  morph  dist``  (2-D)
                      or  ``ix  iy  iz  morph  dist``  (3-D)
      ix, iy[, iz] are 0-based integer grid indices.
      morph: float label (0=donor, 1=acceptor).
      dist: signed distance to D/A interface [m] (negative=donor, positive=acceptor).

    Parameters
    ----------
    path : path-like
        Path to the .txt cloud file.
    physical_size : sequence of float
        (Lx, Ly) or (Lx, Ly, Lz) in metres.
        spacing[d] = L[d] / (n[d] - 1).

    Returns
    -------
    Morphology
        Arrays ordered by (ix, iy[, iz]) regardless of file row order.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]

    # ── Header ────────────────────────────────────────────────────────────────
    # Strip inline comments from header line
    header_raw = lines[0].split("#")[0].strip()
    header_parts = header_raw.split()
    n_total = int(header_parts[0])
    nx      = int(header_parts[1])
    ny      = int(header_parts[2])
    if len(header_parts) >= 4:
        nz = int(header_parts[3])
        ndim = 3
    else:
        nz   = None
        ndim = 2

    # ── Validate node count ───────────────────────────────────────────────────
    expected = nx * ny * (nz if nz is not None else 1)
    if n_total != expected:
        raise ValueError(
            f"read_cpu_cloud: header says n_total={n_total} but "
            f"nx*ny{'*nz' if nz else ''}={expected}"
        )

    data_lines = lines[1:]
    if len(data_lines) < n_total:
        raise ValueError(
            f"read_cpu_cloud: expected {n_total} data lines, "
            f"found {len(data_lines)}"
        )

    # ── Parse data lines ──────────────────────────────────────────────────────
    if ndim == 2:
        morph_arr = np.empty((nx, ny), dtype=np.float64)
        dist_arr  = np.empty((nx, ny), dtype=np.float64)
        for i in range(n_total):
            parts = data_lines[i].split()
            ix, iy = int(parts[0]), int(parts[1])
            morph_arr[ix, iy] = float(parts[2])
            dist_arr[ix, iy]  = float(parts[3])
        spacing = (
            float(physical_size[0]) / (nx - 1),
            float(physical_size[1]) / (ny - 1),
        )
    else:
        morph_arr = np.empty((nx, ny, nz), dtype=np.float64)
        dist_arr  = np.empty((nx, ny, nz), dtype=np.float64)
        for i in range(n_total):
            parts = data_lines[i].split()
            ix, iy, iz_ = int(parts[0]), int(parts[1]), int(parts[2])
            morph_arr[ix, iy, iz_] = float(parts[3])
            dist_arr[ix, iy, iz_]  = float(parts[4])
        spacing = (
            float(physical_size[0]) / (nx - 1),
            float(physical_size[1]) / (ny - 1),
            float(physical_size[2]) / (nz - 1),
        )

    return Morphology(morph=morph_arr, dist=dist_arr, spacing=spacing)


def write_cpu_cloud(m: Morphology, path: str | Path) -> None:
    """Write a Morphology to a CPU legacy cloud text file.

    Inverse of read_cpu_cloud — round-trip support.  Node order: fastest index
    is the last spatial axis (C order), matching the file format.

    Parameters
    ----------
    m : Morphology
    path : path-like
    """
    path = Path(path)
    ndim = m.ndim
    shape = m.shape

    nx = shape[0]
    ny = shape[1]
    nz = shape[2] if ndim == 3 else None
    n_total = int(np.prod(shape))

    lines: list[str] = []

    if ndim == 2:
        header = f"{n_total}  {nx}  {ny}"
        lines.append(header)
        for ix in range(nx):
            for iy in range(ny):
                lines.append(
                    f"{ix}\t {iy} {m.morph[ix, iy]:.1f} {m.dist[ix, iy]:.6e}"
                )
    else:
        header = f"{n_total}  {nx}  {ny}  {nz}"
        lines.append(header)
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    lines.append(
                        f"{ix}\t {iy} {iz} "
                        f"{m.morph[ix, iy, iz]:.1f} {m.dist[ix, iy, iz]:.6e}"
                    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Signed distance utility
# ──────────────────────────────────────────────────────────────────────────────

def signed_distance(
    morph: np.ndarray,
    spacing: Sequence[float],
) -> np.ndarray:
    """Compute a signed distance field from a binary morphology label array.

    Uses scipy.ndimage.distance_transform_edt with per-axis sampling=spacing,
    then applies a half-voxel mid-plane correction so the zero-crossing sits
    at the physical interface mid-plane between donor and acceptor voxel centres.

    Formula (raw EDT, then mid-plane correction):
        raw  = edt(morph >= 0.5, sampling) - edt(morph < 0.5, sampling)
        dist = raw - 0.5 * max(spacing) * sign(raw)

    EDT is voxel-centre-to-voxel-centre; the correction shifts each non-zero
    value half the maximum voxel spacing toward zero.  np.sign(0)=0, so exact
    zeros are untouched.  The raw field has no values in (-s, s) minus {0} by EDT
    construction, so the correction never flips signs.

    Sign convention (matches CPU ground truth):
        dist > 0 : acceptor side (morph=1 region)
        dist < 0 : donor side (morph=0 region)
        dist ~ 0 : near the D/A interface

    Parity with CPU files:
        The CPU files (e.g. morph_bilayer_2D.txt) anchor dist=0 at the last
        acceptor node rather than the geometric mid-plane.  After the mid-plane
        correction, deviations vs that convention are symmetric ±0.5*max(spacing)
        on both donor and acceptor sides (Gate 2 tolerance: 0.51*max(spacing)).
        The CPU-parity path (read_cpu_cloud) uses file-provided dist directly
        and is unaffected.

    Parameters
    ----------
    morph : np.ndarray
        Float or bool array of any shape (nx[, ny[, nz]]).
        Values >= 0.5 are treated as acceptor; < 0.5 as donor.
    spacing : sequence of float
        Per-axis voxel spacing [m].  Length must equal morph.ndim.

    Returns
    -------
    dist : np.ndarray  [m]
        Signed distance field, same shape as morph.
    """
    sampling = tuple(float(s) for s in spacing)
    acceptor_mask = morph >= 0.5
    donor_mask    = ~acceptor_mask

    d_from_donor    = distance_transform_edt(acceptor_mask, sampling=sampling)
    d_from_acceptor = distance_transform_edt(donor_mask,    sampling=sampling)

    # Positive in acceptor, negative in donor (raw: voxel-center-to-voxel-center)
    raw = d_from_donor - d_from_acceptor

    # Half-voxel mid-plane correction:
    # EDT measures from voxel *centre* to nearest opposite-phase voxel centre, so
    # every non-zero value is shifted by +0.5*spacing relative to the physical
    # interface mid-plane.  Subtract half the max-spacing in the direction of the
    # current sign so that the zero-crossing sits at the true mid-plane.
    # np.sign(0)=0, so exact zeros are untouched.
    # The raw field has no values in (-s, 0) or (0, s) by EDT construction,
    # so this shift cannot flip any sign.
    s = float(max(spacing))
    dist = raw - 0.5 * s * np.sign(raw)
    return dist


# ──────────────────────────────────────────────────────────────────────────────
# Film npz reader
# ──────────────────────────────────────────────────────────────────────────────

def from_film_npz(
    path: str | Path,
    threshold: float = 0.5,
    species: str = "phi",
    physical_size: Sequence[float] | None = None,
) -> Morphology:
    """Read a DiffSim film-output npz file and return a Morphology.

    Loads the composition field array, thresholds it, and computes the signed
    distance field via signed_distance().

    Species selection:
        1. Try key ``species`` (default "phi").
        2. Fall back to the first 2-D or 3-D float array found in the file.

    Spacing resolution (in priority order):
        1. npz key ``dx`` (isotropic, scalar).
        2. npz key ``spacing`` (per-axis tuple or scalar).
        3. ``physical_size`` argument (Lx, Ly[, Lz]) → spacing[d] = L[d] / (n[d]-1).

    Parameters
    ----------
    path : path-like
    threshold : float
        Voxels with field > threshold are labelled acceptor (morph=1).
    species : str
        Primary npz key for the composition field.
    physical_size : sequence of float or None
        Required only if no spacing metadata found in the npz.

    Returns
    -------
    Morphology
    """
    path = Path(path)
    data = np.load(str(path), allow_pickle=False)

    # ── Load composition field ────────────────────────────────────────────────
    field: np.ndarray | None = None
    if species in data:
        field = data[species]
    else:
        # Fall back to first 2-D or 3-D float array
        for key in data.files:
            arr = data[key]
            if arr.ndim in (2, 3) and np.issubdtype(arr.dtype, np.floating):
                field = arr
                break
    if field is None:
        raise KeyError(
            f"from_film_npz: could not find species key '{species}' or any "
            f"2-D/3-D float array in {path}"
        )

    morph = (field > threshold).astype(np.float64)

    # ── Determine spacing ─────────────────────────────────────────────────────
    spacing: tuple | None = None

    if "dx" in data:
        dx_val = float(data["dx"])
        spacing = tuple(dx_val for _ in range(morph.ndim))
    elif "spacing" in data:
        sp = data["spacing"]
        if sp.ndim == 0:
            s = float(sp)
            spacing = tuple(s for _ in range(morph.ndim))
        else:
            spacing = tuple(float(v) for v in sp)
    elif physical_size is not None:
        spacing = tuple(
            float(physical_size[d]) / (morph.shape[d] - 1)
            for d in range(morph.ndim)
        )
    else:
        raise ValueError(
            "from_film_npz: no spacing metadata (dx/spacing) found in npz "
            "and physical_size argument not provided."
        )

    dist = signed_distance(morph, spacing)
    return Morphology(morph=morph, dist=dist, spacing=spacing)


# ──────────────────────────────────────────────────────────────────────────────
# Relaxed masks
# ──────────────────────────────────────────────────────────────────────────────

def tanh_mask(dist: np.ndarray, width: float) -> np.ndarray:
    """Relaxed acceptor phase indicator via hyperbolic tangent.

    ``mask = 0.5 * (1 + tanh(dist / width))``

    Properties:
        - mask → 1 deep in the acceptor region (dist >> 0)
        - mask → 0 deep in the donor region (dist << 0)
        - mask(dist=0) = 0.5  (at the interface)
        - Monotonically non-decreasing in dist

    Parameters
    ----------
    dist : np.ndarray
        Signed distance field [m].
    width : float
        Interface half-width parameter [m].  Smaller → sharper transition.

    Returns
    -------
    mask : np.ndarray  ∈ [0, 1], same shape as dist.
    """
    return 0.5 * (1.0 + np.tanh(dist / width))


def region_weights(
    dist: np.ndarray,
    width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute relaxed region weight functions.

    Returns
    -------
    w_donor : np.ndarray
        Weight for the donor region: ``1 - tanh_mask(dist, width)``.
    w_acceptor : np.ndarray
        Weight for the acceptor region: ``tanh_mask(dist, width)``.
    w_interface : np.ndarray
        Un-normalised sech² interface bump:
        ``1 / cosh(dist / width)^2`` (= sech²(dist/width)).
        Peaks at dist=0; does NOT integrate to 1.

    Notes
    -----
    w_donor + w_acceptor = 1 everywhere (partition of unity).
    w_interface is used as an un-normalised smooth localiser for interface
    properties; normalise by volume-integrating if a probability density is
    required.
    """
    w_acceptor  = tanh_mask(dist, width)
    w_donor     = 1.0 - w_acceptor
    w_interface = 1.0 / np.cosh(dist / width) ** 2
    return w_donor, w_acceptor, w_interface


def interface_mask(
    dist: np.ndarray,
    half_thickness: float,
    sharp: bool = False,
) -> np.ndarray:
    """Smooth (or sharp) indicator for the D/A interface region.

    Smooth variant (default):
        ``0.5 * (tanh((half_thickness - |dist|) / (0.25 * half_thickness)) + 1)``
        → 1 at dist=0, decays smoothly to 0 outside ±half_thickness.

    Sharp variant (``sharp=True``):
        ``(|dist| < half_thickness).astype(float)``
        Binary 0/1 indicator.

    Parameters
    ----------
    dist : np.ndarray
        Signed distance field [m].
    half_thickness : float
        Half-width of the interface region [m].
    sharp : bool
        If True, return binary indicator; else smooth tanh indicator.

    Returns
    -------
    mask : np.ndarray  ∈ [0, 1], same shape as dist.
    """
    if sharp:
        return (np.abs(dist) < half_thickness).astype(float)
    # Smooth: tanh-based indicator centred on the interface
    return 0.5 * (
        np.tanh((half_thickness - np.abs(dist)) / (0.25 * half_thickness)) + 1.0
    )


# ──────────────────────────────────────────────────────────────────────────────
# Descriptors
# ──────────────────────────────────────────────────────────────────────────────

def descriptors(m: Morphology) -> dict:
    """Compute GraSPI-compatible morphology descriptors.

    Returns
    -------
    dict with keys:
        phase_frac_donor : float
            Fraction of nodes labelled donor (morph < 0.5).
        phase_frac_acceptor : float
            Fraction of nodes labelled acceptor (morph >= 0.5).
        interface_edges : int
            Count of axis-aligned neighbour pairs with opposite phase labels.
            Analogue of GraSPI STAT_e.
        interface_area_per_volume : float
            interface_edges × (face area per edge) / domain_volume  [m^-1].
            Face area for each edge is the product of spacings of all axes
            EXCEPT the axis along which the pair is measured.
            Domain volume is the physical extent: product of (n[d]-1)*spacing[d]
            for all d (consistent with spacing = L/(n-1) in read_cpu_cloud).
    """
    morph   = m.morph
    spacing = m.spacing
    ndim    = m.ndim
    shape   = m.shape

    # Phase fractions
    acceptor_mask = morph >= 0.5
    phase_frac_acceptor = float(np.mean(acceptor_mask))
    phase_frac_donor    = float(np.mean(~acceptor_mask))

    # Interface edge count: pairs of axis-adjacent nodes with opposite labels
    label = acceptor_mask.astype(np.int8)
    n_edges = 0
    for ax in range(ndim):
        # Slice off first and last along axis; compare
        sl_left  = [slice(None)] * ndim
        sl_right = [slice(None)] * ndim
        sl_left[ax]  = slice(None, -1)
        sl_right[ax] = slice(1, None)
        diff = label[tuple(sl_left)] != label[tuple(sl_right)]
        n_edges += int(np.sum(diff))

    # Interface area per volume
    # Face area for edges along axis `ax` = product of spacings for all OTHER axes
    # Domain volume = product of (n[d] * spacing[d]) for all d
    domain_volume = float(np.prod([(shape[d] - 1) * spacing[d] for d in range(ndim)]))

    total_face_area = 0.0
    label = acceptor_mask.astype(np.int8)
    for ax in range(ndim):
        sl_left  = [slice(None)] * ndim
        sl_right = [slice(None)] * ndim
        sl_left[ax]  = slice(None, -1)
        sl_right[ax] = slice(1, None)
        diff = label[tuple(sl_left)] != label[tuple(sl_right)]
        n_edges_ax = int(np.sum(diff))
        # Face area perpendicular to axis `ax`
        face_area = float(np.prod([spacing[d] for d in range(ndim) if d != ax]))
        total_face_area += n_edges_ax * face_area

    interface_area_per_volume = total_face_area / domain_volume

    return {
        "phase_frac_donor":         phase_frac_donor,
        "phase_frac_acceptor":      phase_frac_acceptor,
        "interface_edges":          n_edges,
        "interface_area_per_volume": interface_area_per_volume,
    }
