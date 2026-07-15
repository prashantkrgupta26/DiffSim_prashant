r"""Differentiable structure factor S(q) gates.

  * PARITY — the Torch estimator reproduces diagnostics/morphology.structure_factor
    (the numpy reference) bin-for-bin.
  * DIFFERENTIABLE — d/dfield of a scalar functional of S(q) matches central FD.
  * NODAL->GRID — the uniform-mesh gather reconstructs the grid field exactly.
"""
import numpy as np
import pytest
import torch

from diffsim.diagnostics.morphology import structure_factor as sf_np
from diffsim.diagnostics.structure_factor_torch import (TorchStructureFactor,
                                                        grid_index_from_coords)

pytestmark = pytest.mark.ad


def test_parity_with_numpy():
    rng = np.random.default_rng(0)
    for shape in [(16, 16), (24, 20), (8, 8, 8)]:
        f = rng.standard_normal(shape)
        q_np, S_np = sf_np(f, dx=1.0)
        tsf = TorchStructureFactor(shape, dx=1.0)
        q_t, S_t = tsf(torch.tensor(f))
        assert q_t.shape == q_np.shape
        rel_q = float(np.abs(q_t.numpy() - q_np).max())
        rel_S = float(np.abs(S_t.numpy() - S_np).max()
                      / max(np.abs(S_np).max(), 1e-30))
        print(f"shape {shape}  |dq|={rel_q:.2e}  rel|dS|={rel_S:.2e}")
        assert rel_q < 1e-12
        assert rel_S < 1e-12


def test_differentiable_vs_fd():
    """Autograd d/dfield of a random-weighted S(q) functional == central FD."""
    rng = np.random.default_rng(1)
    shape = (16, 16)
    f0 = rng.standard_normal(shape)
    tsf = TorchStructureFactor(shape, dx=1.0)
    w = torch.tensor(rng.standard_normal(tsf.nbins - 1))     # random dual

    def J(field_t):
        _, S = tsf(field_t)
        return (w * S).sum()

    ft = torch.tensor(f0, requires_grad=True)
    J(ft).backward()
    g = ft.grad.detach().clone()

    d = torch.tensor(rng.standard_normal(shape))
    d = d / d.norm()
    eps = 1e-6
    with torch.no_grad():
        Jp = float(J(torch.tensor(f0) + eps * d))
        Jm = float(J(torch.tensor(f0) - eps * d))
    dd_fd = (Jp - Jm) / (2 * eps)
    dd_ad = float((g * d).sum())
    rel = abs(dd_ad - dd_fd) / max(abs(dd_fd), 1e-12)
    print(f"S(q) grad  adj={dd_ad:+.6e} fd={dd_fd:+.6e} rel={rel:.2e}")
    assert rel < 1e-6, (dd_ad, dd_fd)


def test_sq_loss_gradient_through_twin():
    """End-to-end: march the autograd twin, take S(q) of the final morphology,
    and check dJ/d(beyond-FH coeffs) for an instrument-space misfit
    ||S_sim(q) - S_data(q)||^2 against central FD.  This is the differentiable
    S(q,t) observable wired for learning."""
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh
    from diffsim.adjoint.torch_twin import CHTwin
    from diffsim.adjoint.neural_energy import BasisCorrEnergy

    tree = build_uniform(3, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), "cpu")
    coords = mesh.node_coords
    shape, gidx = grid_index_from_coords(coords)
    dx = float(np.unique(np.round(coords[:, 0], 9))[1]
               - np.unique(np.round(coords[:, 0], 9))[0])
    tsf = TorchStructureFactor(shape, dx=dx)

    c0 = torch.tensor(0.5 + 0.12 * np.cos(np.pi * coords[:, 0])
                      * np.cos(np.pi * coords[:, 1]))
    M, kap = torch.tensor(1.0), torch.tensor(0.01)
    n_steps = 4

    def sq_of_final(energy):
        twin = CHTwin(dm, energy=energy, dt=0.01, order=1, device="cpu")
        out = twin.march(c0, None, M, kap, {}, n_steps)
        _, S = tsf(out[-1][0][gidx].reshape(shape))
        return S

    truth = BasisCorrEnergy(A=1.0, B=2.5, degrees=(2, 3), coeffs=(0.3, 0.15))
    for p in truth.parameters():
        p.requires_grad_(False)
    S_data = sq_of_final(truth).detach()

    model = BasisCorrEnergy(A=1.0, B=2.5, degrees=(2, 3), coeffs=(0.1, 0.05))
    model.A.requires_grad_(False)
    model.B.requires_grad_(False)

    def loss():
        return ((sq_of_final(model) - S_data) ** 2).sum()

    loss().backward()
    g = model.gamma.grad.detach().clone()
    d = torch.tensor([0.6, -0.8], dtype=torch.float64)
    d = d / d.norm()
    eps = 1e-6
    with torch.no_grad():
        model.gamma.add_(eps * d)
        Jp = float(loss())
        model.gamma.add_(-2 * eps * d)
        Jm = float(loss())
        model.gamma.add_(eps * d)
    dd_fd = (Jp - Jm) / (2 * eps)
    dd_ad = float((g * d).sum())
    rel = abs(dd_ad - dd_fd) / max(abs(dd_fd), 1e-12)
    print(f"S(q) loss grad through twin  adj={dd_ad:+.6e} fd={dd_fd:+.6e} "
          f"rel={rel:.2e}")
    assert rel < 1e-5, (dd_ad, dd_fd)


def test_nodal_to_grid_roundtrip():
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    tree = build_uniform(3, dim=2)
    mesh = build_mesh(tree, p=1)
    coords = mesh.node_coords
    shape, gidx = grid_index_from_coords(coords)
    # a known analytic field: reconstruct on the grid and compare to meshgrid
    xs = np.unique(np.round(coords[:, 0], 9))
    ys = np.unique(np.round(coords[:, 1], 9))
    fn = np.cos(2 * np.pi * coords[:, 0]) * np.sin(np.pi * coords[:, 1])
    grid = torch.tensor(fn)[gidx].reshape(shape).numpy()
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    ref = np.cos(2 * np.pi * X) * np.sin(np.pi * Y)
    err = float(np.abs(grid - ref).max())
    print(f"nodal->grid roundtrip max err = {err:.2e}  shape={shape}")
    assert err < 1e-12, err
    # and S(q) through the gather matches the numpy estimator on that grid
    tsf = TorchStructureFactor(shape, dx=float(xs[1] - xs[0]))
    _, S_t = tsf(torch.tensor(fn)[gidx].reshape(shape))
    _, S_np = sf_np(ref, dx=float(xs[1] - xs[0]))
    assert float(np.abs(S_t.numpy() - S_np).max()) < 1e-12
