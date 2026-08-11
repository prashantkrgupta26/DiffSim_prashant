"""Tests for crystal_recovery.py — structure-factor descriptor and its field-space gradient.

TDD: test_structure_factor_grad_vs_fd is the ground truth for the gradient
implementation. The normalization must match central-FD to rtol 1e-4, atol 1e-6.
"""
import numpy as np
import pytest
from diffsim.adjoint.crystal_recovery import (
    structure_factor, structure_factor_grad, nodal_to_grid)


# ---------------------------------------------------------------------------
# Shared mesh helper for multi-snapshot tests (copied from test_crystallization_multi)
# ---------------------------------------------------------------------------

def _dm(level, dim=2):
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh
    tree = build_uniform(level, dim=dim)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim), "cpu")
    return dm, mesh


def test_structure_factor_output_shape():
    """S(k) should return a 1-D array of length nbins (default min(shape)//2)."""
    f = np.ones((8, 12))
    S = structure_factor(f)
    assert S.ndim == 1
    assert len(S) == min(8, 12) // 2  # default nbins


def test_structure_factor_uniform_field_is_zero():
    """A uniform field has zero mean-subtracted content → S should be all zeros."""
    f = 3.0 * np.ones((16, 16))
    S = structure_factor(f)
    assert np.allclose(S, 0.0)


def test_structure_factor_nbins_kwarg():
    """nbins keyword should control the output length."""
    f = np.random.default_rng(1).standard_normal((16, 16))
    for nb in [4, 8, 12]:
        S = structure_factor(f, nbins=nb)
        assert len(S) == nb


def test_structure_factor_grad_shape():
    """Gradient should have the same shape as the input field."""
    rng = np.random.default_rng(2)
    f = rng.standard_normal((8, 8))
    tgt = structure_factor(rng.standard_normal((8, 8)))
    g = structure_factor_grad(f, tgt)
    assert g.shape == f.shape


def test_structure_factor_grad_dtype():
    """Gradient should be real-valued float64."""
    rng = np.random.default_rng(3)
    f = rng.standard_normal((8, 8))
    tgt = structure_factor(rng.standard_normal((8, 8)))
    g = structure_factor_grad(f, tgt)
    assert np.isrealobj(g)
    assert g.dtype == np.float64


def test_structure_factor_grad_vs_fd():
    """Analytic gradient must match central finite differences to rtol 1e-4."""
    rng = np.random.default_rng(0)
    f = 0.3 + 0.1 * rng.standard_normal((8, 8))
    tgt = structure_factor(0.3 + 0.1 * rng.standard_normal((8, 8)))
    g = structure_factor_grad(f, tgt)

    def loss(x):
        return 0.5 * float(((structure_factor(x) - tgt) ** 2).sum())

    h = 1e-6
    for (a, b) in [(0, 0), (3, 5), (7, 2), (4, 4)]:
        fp = f.copy(); fp[a, b] += h
        fm = f.copy(); fm[a, b] -= h
        fd = (loss(fp) - loss(fm)) / (2 * h)
        assert np.isclose(g[a, b], fd, rtol=1e-4, atol=1e-6), (
            f"pixel ({a},{b}): analytic={g[a,b]:.8g}, fd={fd:.8g}"
        )


class _FakeMesh:
    """Minimal mesh stub for nodal_to_grid tests."""
    def __init__(self, nx, ny):
        xs = np.linspace(0.0, 1.0, nx)
        ys = np.linspace(0.0, 1.0, ny)
        xg, yg = np.meshgrid(xs, ys)      # shape (ny, nx)
        # store in random node order
        rng = np.random.default_rng(42)
        perm = rng.permutation(nx * ny)
        coords_ordered = np.stack([xg.ravel(), yg.ravel()], axis=1)  # (ny*nx, 2), row-major
        self.node_coords = coords_ordered[perm]
        self._perm = perm
        self._nx = nx
        self._ny = ny
        # canonical field: value = row*nx + col  (row-major grid)
        self._field_canonical = np.arange(ny * nx, dtype=float)
        self.field_nodal = self._field_canonical[perm]  # shuffled


def test_nodal_to_grid_shape():
    """nodal_to_grid should return a (ny, nx) 2-D array."""
    m = _FakeMesh(6, 4)
    grid = nodal_to_grid(m.field_nodal, m)
    assert grid.shape == (m._ny, m._nx)


def test_nodal_to_grid_values():
    """nodal_to_grid must correctly unshuffle the nodal field to the 2-D grid."""
    m = _FakeMesh(6, 4)
    grid = nodal_to_grid(m.field_nodal, m)
    expected = m._field_canonical.reshape(m._ny, m._nx)
    assert np.allclose(grid, expected), (
        f"Max error: {np.abs(grid - expected).max()}"
    )


# ===========================================================================
# Task 3: Multi-snapshot recovery — identifiability lift
# ===========================================================================

def test_multisnapshot_recovers_higher_mode():
    """Multi-snapshot recovery lifts identifiability of cpl_0_2 (higher ψ-mode).

    Plants {"cpl_0_1": 0.15, "cpl_0_2": -0.12}, recovers with snapshots at
    steps (2, 4, 6).  Asserts:
      - loss drop ≥ 20×  (convergence)
      - cpl_0_2 within 30% of planted (higher mode identifiable from multi-snapshot)
    """
    from diffsim.adjoint.crystal_recovery import recover_multisnapshot
    dm, mesh = _dm(2)
    planted = {"cpl_0_1": 0.15, "cpl_0_2": -0.12}
    names = ["cpl_0_1", "cpl_0_2"]
    # multi-snapshot: observe several times -> higher mode cpl_0_2 identifiable
    lh, th, tt = recover_multisnapshot(
        dm, mesh, planted, names, n_steps=6, snapshots=(2, 4, 6),
        order=1, n_iter=60, lr=0.5)
    assert lh[-1] <= lh[0] / 20.0, (
        f"Loss drop {lh[0]:.4e} → {lh[-1]:.4e} is < 20× (ratio {lh[0]/lh[-1]:.1f}×)")
    # the higher mode cpl_0_2 recovers where single-snapshot (② Task 5) fails
    assert abs(th["cpl_0_2"] - tt["cpl_0_2"]) <= 0.30 * abs(tt["cpl_0_2"]), (
        f"cpl_0_2: planted={tt['cpl_0_2']:.4f}, recovered={th['cpl_0_2']:.4f}, "
        f"relerr={abs(th['cpl_0_2']-tt['cpl_0_2'])/abs(tt['cpl_0_2']):.3f}")


def test_multisnapshot_descriptor_smoke():
    """descriptor=True path runs and still reduces the loss (no crash, no diverge)."""
    from diffsim.adjoint.crystal_recovery import recover_multisnapshot
    dm, mesh = _dm(2)
    planted = {"cpl_0_1": 0.15, "cpl_0_2": -0.12}
    names = ["cpl_0_1", "cpl_0_2"]
    lh, th, tt = recover_multisnapshot(
        dm, mesh, planted, names, n_steps=6, snapshots=(2, 4, 6),
        order=1, n_iter=20, lr=0.5, descriptor=True, lam_desc=1e-3)
    # descriptor term must not cause divergence: final loss < initial loss
    assert lh[-1] < lh[0], (
        f"Descriptor-on smoke: loss did not decrease ({lh[0]:.4e} → {lh[-1]:.4e})")
