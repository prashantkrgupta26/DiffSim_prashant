"""Device-resident scalar Poisson (K_p) assembler gates: the device CSR
must equal the host assemble_csr(dm) (T^T K T) to fp tolerance, for both
the identity-T (uniform mesh -> node-graph pattern, the 100M PPE case) and
the hanging-node (weighted constraint-expansion) paths.  This is the K_p
assembler that kills the host COO->CSR wall on the projection engine's
pressure-Poisson operator.
"""
import numpy as np
import pytest

from diffsim.octree.build import build_uniform, refine_elements
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import (
    DeviceMesh, assemble_csr, assemble_csr_device,
    DeviceScalarPoissonAssembler)

pytestmark = pytest.mark.tier2


def _dm(dim, level, graded, device):
    tree = build_uniform(level, dim=dim)
    if graded:
        c = tree.centers()
        mask = np.linalg.norm(c, axis=1) < 0.5 * np.sqrt(dim)
        tree = refine_elements(tree, mask)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim), device)
    return dm, cons


@pytest.mark.parametrize("dim,level,graded", [
    (2, 4, False), (3, 3, False),          # identity-T (node-graph pattern)
    (2, 3, True), (3, 3, True),            # hanging-node (weighted T^T K T)
])
def test_device_scalar_matches_host(dim, level, graded, device):
    dm, cons = _dm(dim, level, graded, device)
    Kh = assemble_csr(dm).tocsr()
    Kh.sort_indices()
    Kh.eliminate_zeros()
    Kd = assemble_csr_device(dm).tocsr()
    Kd.sort_indices()
    Kd.eliminate_zeros()
    scale = max(np.abs(Kh.data).max(), 1e-30)
    D = Kh - Kd
    err = (np.abs(D.data).max() / scale) if D.nnz else 0.0
    assert err < 1e-11, (dim, level, graded, err)


def test_reuse_fill(device):
    """A single assembler filled twice yields identical CSR values — the
    per-step reuse contract (symbolic once, device scatter per fill)."""
    dm, _ = _dm(3, 3, False, device)
    asm = DeviceScalarPoissonAssembler(dm)
    K1 = asm.fill().to_csr()
    K2 = asm.fill().to_csr()
    assert np.array_equal(K1.indptr, K2.indptr)
    assert np.abs(K1.data - K2.data).max() == 0.0


def test_stepper_device_assembly_bit_for_bit(device):
    """The Leray stepper's K_p with device_assembly=True equals the
    default host-assembled K_p to fp tolerance (default OFF is unchanged)."""
    from diffsim.steppers.leray import LerayProjectionStepper
    dm, _ = _dm(3, 3, False, device)

    def f_fn(x, t):
        return np.zeros((len(x), 3))

    kw = dict(f_fn=f_fn, g_fn=f_fn, order=1, picard_iters=1,
              timestab=False, solver="splu")
    st_h = LerayProjectionStepper(dm, 0.01, 0.05, **kw)
    st_d = LerayProjectionStepper(dm, 0.01, 0.05, device_assembly=True, **kw)
    Kh = st_h.K_p.tocsr(); Kh.sort_indices()
    Kd = st_d.K_p.tocsr(); Kd.sort_indices()
    scale = max(np.abs(Kh.data).max(), 1e-30)
    D = Kh - Kd
    err = (np.abs(D.data).max() / scale) if D.nnz else 0.0
    assert err < 1e-11, err
