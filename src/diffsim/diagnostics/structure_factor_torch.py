r"""Differentiable structure factor S(q) (Torch) — an instrument-space observable
with a gradient, for beyond-Flory-Huggins learning (roadmap "Learning rung 2").

diagnostics/morphology.structure_factor is numpy/FFT post-processing: it has no
gradient, so it cannot drive an inverse problem.  This is a faithful Torch twin
of the same estimator

    S(q) = < |FFT(phi - <phi>)|^2 > / N   averaged over |q| shells (width dq),

built entirely from differentiable ops (torch.fft.fftn + index_add radial
binning), so a scattering-like misfit ||S_sim(q,t) - S_data(q,t)||^2 backprops to
the free-energy / material parameters through the autograd twin.  Value parity
with the numpy estimator and a finite-difference gradient check are gated in
tests/test_structure_factor_torch.py.

A nodal field on a uniform (p1) octree mesh is mapped to the (ny, nx) grid the
FFT needs via ``grid_index_from_coords`` (a fixed permutation → differentiable
gather)."""
import numpy as np
import torch


class TorchStructureFactor:
    """Precompute the |q| shell binning for a fixed grid ``shape``; call on a
    Torch field of that shape to get differentiable ``(q, S)``.

    Parameters
    ----------
    shape : tuple[int]
        Grid shape (2-D ``(ny, nx)`` or 3-D).  Fixed at construction.
    dx : float
        Grid spacing (physical length per cell); ``q`` is in ``2*pi/length``.
    device, dtype : torch specifics for the cached index/weight tensors.
    """

    def __init__(self, shape, dx=1.0, device="cpu", dtype=torch.float64):
        self.shape = tuple(int(n) for n in shape)
        self.dx = float(dx)
        self.dev = device
        freqs = [np.fft.fftfreq(n, d=dx) * 2.0 * np.pi for n in self.shape]
        grids = np.meshgrid(*freqs, indexing="ij")
        qmag = np.sqrt(sum(g ** 2 for g in grids))
        dq = 2.0 * np.pi / (max(self.shape) * dx)
        nbins = int(np.ceil(qmag.max() / dq)) + 1
        idx = np.minimum((qmag / dq).astype(int), nbins - 1).ravel()
        counts = np.bincount(idx, minlength=nbins)
        self.nbins = nbins
        self.dq = dq
        self.idx = torch.tensor(idx, dtype=torch.int64, device=device)
        self.counts = torch.tensor(np.where(counts == 0, 1, counts),
                                   dtype=dtype, device=device)
        self.q_all = torch.tensor((np.arange(nbins) + 0.5) * dq, dtype=dtype,
                                  device=device)

    def __call__(self, field):
        """``field``: real Torch tensor of the construction ``shape``.
        Returns ``(q, S)`` with the DC (q~0) bin dropped, both differentiable
        in ``field``."""
        f = field - field.mean()
        fk = torch.fft.fftn(f)
        power = (fk.real ** 2 + fk.imag ** 2) / f.numel()
        S = torch.zeros(self.nbins, dtype=power.dtype, device=self.dev)
        S = S.index_add(0, self.idx, power.reshape(-1)) / self.counts
        return self.q_all[1:], S[1:]


def grid_index_from_coords(coords):
    """Map a uniform (p1) mesh's nodes to a regular ``(ny, nx)`` grid.

    Returns ``(shape, gidx)`` where ``field_nodal[gidx].reshape(shape)`` is the
    field on the grid (a fixed gather — differentiable).  Assumes the node
    coordinates form a complete axis-aligned lattice (the adjoint-gate uniform
    mesh); raises if they do not.
    """
    coords = np.asarray(coords, dtype=float)
    dim = coords.shape[1]
    axes = [np.unique(np.round(coords[:, d], 9)) for d in range(dim)]
    shape = tuple(len(a) for a in axes)
    if int(np.prod(shape)) != coords.shape[0]:
        raise ValueError(
            f"coords ({coords.shape[0]} nodes) are not a complete "
            f"{shape} lattice — grid_index_from_coords needs a uniform mesh")
    # multi-index of each node, then the flat grid position (row-major, 'ij')
    midx = [np.searchsorted(axes[d], np.round(coords[:, d], 9))
            for d in range(dim)]
    flat = np.ravel_multi_index(midx, shape)
    gidx = np.empty(coords.shape[0], dtype=np.int64)
    gidx[flat] = np.arange(coords.shape[0])      # grid cell -> node id
    return shape, torch.tensor(gidx, dtype=torch.int64)
