r"""Beyond-Flory-Huggins neural free-energy head gates.

The M4 differentiable-thermo gate is three-way (hand IFT adjoint == autograd
twin == finite difference).  A many-weight MLP free-energy correction cannot use
the per-scalar-name numpy hand adjoint, so its correctness bar is:

  * GAUGE ANCHORING  — the learned correction f'_corr(c) is L2-orthogonal to the
    {1, c} modes the M4 findings proved unidentifiable (T0/T1), by construction.
  * CURVATURE CONSISTENCY — the closed-form f''(c) equals d f'/dc (autograd),
    i.e. the analytic MLP input-derivative is right (no nested autograd needed
    inside the Newton solve).
  * ADJOINT vs FD — the autograd-through-convergence gradient of an end-to-end
    trajectory loss w.r.t. every energy parameter {A, B, theta} matches central
    finite differences along a random direction (the whole learn-a-functional
    chain, verified).

Gates: uniform level-3 mesh, interior FH data (logs away from the walls),
BDF1 + BDF2."""
import numpy as np
import pytest
import torch

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.adjoint.torch_twin import CHTwin
from diffsim.adjoint.neural_energy import NeuralCHEnergy, BasisCorrEnergy

pytestmark = pytest.mark.ad


def _dm(level, device):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    return dm, mesh


def _ic(coords):
    return 0.5 + 0.1 * np.cos(np.pi * coords[:, 0]) \
        * np.cos(np.pi * coords[:, 1])


def test_gauge_anchoring():
    """The gauge-anchored correction is L2-orthogonal to {1, c}: a pure
    constant (T0) and a pure linear (T1) mode carry zero learnable content."""
    en = NeuralCHEnergy(A=1.0, B=2.5, hidden=(16, 16), corr_scale=0.3, seed=3)
    r0, r1 = en.gauge_residual()
    print(f"gauge residual  <corr,1>={r0:+.3e}  <corr,c>={r1:+.3e}")
    assert abs(r0) < 1e-11, r0
    assert abs(r1) < 1e-11, r1


def test_curvature_consistency():
    """f''(c) (closed form) == d f'/dc (autograd) across the composition
    domain — validates the analytic MLP input-derivative and the gauge term."""
    en = NeuralCHEnergy(A=1.0, B=2.5, hidden=(16, 16), corr_scale=0.3, seed=1)
    c = torch.linspace(0.06, 0.94, 41, requires_grad=True)
    fp = en.fp(c)
    (dfp,) = torch.autograd.grad(fp.sum(), c, create_graph=False)
    fpp = en.fpp(c)
    rel = float(((fpp - dfp).abs().max() / dfp.abs().max()).detach())
    print(f"curvature rel |fpp - d fp/dc| = {rel:.2e}")
    assert rel < 1e-10, rel


def test_basis_gauge_orthogonality():
    """The Legendre correction (degrees 2,3,4) is L2-orthogonal to {1, c} over
    the composition domain by construction — the P0/P1 gauge modes carry no
    content in the basis (shifted-Legendre orthogonality)."""
    c_lo, c_hi = 0.05, 0.95
    en = BasisCorrEnergy(A=1.0, B=2.5, degrees=(2, 3, 4),
                         coeffs=(0.4, 0.25, 0.15), c_lo=c_lo, c_hi=c_hi)
    # exact Gauss-Legendre quadrature (integrand is a degree<=5 polynomial)
    x, w = np.polynomial.legendre.leggauss(16)
    mid, half = 0.5 * (c_hi + c_lo), 0.5 * (c_hi - c_lo)
    cq = torch.tensor(mid + half * x)
    wq = torch.tensor(half * w)
    with torch.no_grad():
        r = en.corr_fp(cq)
    s0 = float((wq * r).sum())
    s1 = float((wq * cq * r).sum())
    print(f"basis gauge  <corr,1>={s0:+.2e}  <corr,c>={s1:+.2e}")
    assert abs(s0) < 1e-12, s0
    assert abs(s1) < 1e-12, s1


def test_basis_curvature_consistency():
    """BasisCorrEnergy f''(c) (closed form) == d f'/dc (autograd)."""
    en = BasisCorrEnergy(A=1.0, B=2.5, degrees=(2, 3, 4),
                         coeffs=(0.4, 0.25, 0.15))
    c = torch.linspace(0.06, 0.94, 41, requires_grad=True)
    fp = en.fp(c)
    (dfp,) = torch.autograd.grad(fp.sum(), c)
    fpp = en.fpp(c)
    rel = float(((fpp - dfp).abs().max() / dfp.abs().max()).detach())
    print(f"basis curvature rel = {rel:.2e}")
    assert rel < 1e-12, rel


@pytest.mark.parametrize("order,n_steps", [(1, 3), (2, 4)])
def test_neural_head_gradient_vs_fd(order, n_steps):
    """Autograd-through-convergence dJ/d{A,B,theta} matches central FD along a
    random direction — the beyond-FH learn-a-functional chain, verified."""
    dm, mesh = _dm(3, "cpu")
    coords = mesh.node_coords
    nn = dm.n_nodes
    target = torch.full((nn,), 0.5, dtype=torch.float64)
    c0 = torch.tensor(_ic(coords))
    dt = 0.01

    en = NeuralCHEnergy(A=1.0, B=2.5, hidden=(12, 12), corr_scale=0.2, seed=7)
    twin = CHTwin(dm, energy=en, dt=dt, order=order, device="cpu")
    M = torch.tensor(1.0)
    kap = torch.tensor(0.01)

    def loss():
        out = twin.march(c0, None, M, kap, {}, n_steps)
        return 0.5 * ((out[-1][0] - target) ** 2).sum()

    # analytic gradient
    J = loss()
    J.backward()
    params = [p for p in en.parameters()]
    grads = [p.grad.detach().clone() for p in params]

    # random unit direction in parameter space (seeded — no Math.random)
    g = torch.Generator().manual_seed(11)
    dirs = [torch.randn(p.shape, generator=g) for p in params]
    nrm = np.sqrt(sum(float((d * d).sum()) for d in dirs))
    dirs = [d / nrm for d in dirs]
    dderiv_adj = sum(float((gr * d).sum()) for gr, d in zip(grads, dirs))

    # central FD along the same direction
    eps = 1e-6
    with torch.no_grad():
        for p, d in zip(params, dirs):
            p.add_(eps * d)
        Jp = float(loss())
        for p, d in zip(params, dirs):
            p.add_(-2.0 * eps * d)
        Jm = float(loss())
        for p, d in zip(params, dirs):        # restore
            p.add_(eps * d)
    dderiv_fd = (Jp - Jm) / (2.0 * eps)

    rel = abs(dderiv_adj - dderiv_fd) / max(abs(dderiv_fd), 1e-12)
    print(f"order={order} n={n_steps}  adj={dderiv_adj:+.6e} "
          f"fd={dderiv_fd:+.6e}  rel={rel:.2e}")
    assert rel < 1e-5, (dderiv_adj, dderiv_fd, rel)
