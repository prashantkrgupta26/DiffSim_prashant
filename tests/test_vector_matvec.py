"""M1b Task 1b gates: vector-DOF matrix-free constrained matvec — CSR parity
at 1e-13 (node-major kron structure) and single-sync CG solve compatibility."""
import numpy as np
import pytest
import scipy.sparse as sp
from diffsim.octree.build import build_uniform, refine_elements
from diffsim.octree.balance import balance2to1
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh, assemble_csr
from diffsim.assembly.matvec import ConstrainedVectorOperator

pytestmark = pytest.mark.tier3


def _dm(dim, p, device, adapt=True):
    t = build_uniform(3 if dim == 2 else 2, dim=dim)
    if adapt:
        mask = np.zeros(len(t), bool)
        mask[0] = True
        t = balance2to1(refine_elements(t, mask))     # hanging constraints on
    m = build_mesh(t, p=p)
    c = build_constraints(m)
    return DeviceMesh.from_mesh(m, c, basis_tables(p, dim=dim), device)


@pytest.mark.parametrize("dim,p,ndof", [(2, 1, 3), (2, 2, 3), (3, 1, 4)])
def test_vector_matvec_matches_kron_csr(dim, p, ndof, device):
    dm = _dm(dim, p, device)
    op = ConstrainedVectorOperator(dm, ndof)
    A = assemble_csr(dm)                              # scalar constrained CSR
    A_vec = sp.kron(A, sp.identity(ndof, format="csr"), format="csr")
    rng = np.random.default_rng(4)
    x = rng.standard_normal(op.n_free)
    y_mf = op.matvec_numpy(x)
    y_csr = A_vec @ x
    scale = np.abs(y_csr).max()
    assert np.abs(y_mf - y_csr).max() < 1e-13 * max(scale, 1.0), (
        np.abs(y_mf - y_csr).max(), scale)


def test_vector_operator_with_cg_dev(device):
    # the M1b solver stack end-to-end: matrix-free vector operator + fused CG
    from diffsim.solvers.krylov_dev import cg_dev
    dm = _dm(2, 1, device)
    ndof = 2
    op = ConstrainedVectorOperator(dm, ndof)
    # SPD after adding a mass-like shift on the diagonal via CSR for the gate:
    # solve (A_vec + I) x = b matrix-free by wrapping
    A = assemble_csr(dm)
    A_vec = sp.kron(A, sp.identity(ndof, format="csr"), format="csr")

    class Shifted:
        device = op.device
        n_free = op.n_free

        def matvec(self, x, y):
            import warp as wp
            op.matvec(x, y)
            # y += x  (identity shift, on device)
            from diffsim.solvers import blas
            blas.axpy(1.0, x, y, op.device)

    rng = np.random.default_rng(6)
    b = rng.standard_normal(op.n_free)
    x_d, info = cg_dev(Shifted(), b, tol=1e-11, maxiter=4000,
                       diag=np.asarray(A_vec.diagonal()) + 1.0)
    assert info["converged"]
    from scipy.sparse.linalg import spsolve
    x_ref = spsolve((A_vec + sp.identity(op.n_free)).tocsr(), b)
    assert np.abs(x_d - x_ref).max() < 1e-8 * np.abs(x_ref).max()
