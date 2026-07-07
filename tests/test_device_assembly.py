"""M1d gates for DeviceNSAssembler (spec D2): consistency vs the host
assembler at 1e-12 + speed measurement, both scatter variants."""
import time

import numpy as np
import pytest

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.assembly.device_assembly import DeviceNSAssembler
from diffsim.api.ns_bricks import assemble_linear_ns
from diffsim.physics.poisson import gauss_points

pytestmark = pytest.mark.tier2


def _setup(dim, level, device):
    tree = build_uniform(level, dim=dim)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim), device)
    xq = gauss_points(mesh, dm.tables_by_p)
    rng = np.random.default_rng(3)
    pv = list(dm.bins)[0]
    ngp = len(xq[pv])
    aq = {pv: rng.standard_normal((ngp, dim)) * 0.5}
    dq = {pv: rng.standard_normal(ngp) * 0.1}
    fq = {pv: rng.standard_normal((ngp, dim))}
    return dm, aq, dq, fq


@pytest.mark.parametrize("dim,level", [(2, 5), (3, 3)])
@pytest.mark.parametrize("coloring", [False, True])
def test_device_assembly_consistency(dim, level, coloring, device):
    dm, aq, dq, fq = _setup(dim, level, device)
    nu, sigma = 0.05, 20.0
    A_h, b_h = assemble_linear_ns(dm, aq, dq, fq, nu, sigma=sigma)
    asm = DeviceNSAssembler(dm, coloring=coloring)
    A_d, b_d = asm.assemble(aq, dq, fq, nu, sigma)
    diff = (A_h - A_d)
    scale = np.abs(A_h.data).max()
    a_err = np.abs(diff.data).max() / scale if diff.nnz else 0.0
    b_err = np.abs(b_h - b_d).max() / max(np.abs(b_h).max(), 1e-30)
    assert a_err < 1e-12, (coloring, a_err)
    assert b_err < 1e-12, (coloring, b_err)


def test_device_assembly_speed(device):
    dim, level = 2, 7
    dm, aq, dq, fq = _setup(dim, level, device)
    nu, sigma = 0.05, 20.0
    t0 = time.perf_counter()
    for _ in range(3):
        assemble_linear_ns(dm, aq, dq, fq, nu, sigma=sigma)
    t_host = (time.perf_counter() - t0) / 3
    asm = DeviceNSAssembler(dm)                      # symbolic once
    asm.assemble(aq, dq, fq, nu, sigma)              # warm
    t0 = time.perf_counter()
    for _ in range(3):
        asm.assemble(aq, dq, fq, nu, sigma)
    t_dev = (time.perf_counter() - t0) / 3
    print(f"assembly 2D L7: host {t_host*1e3:.0f} ms vs "
          f"device-scatter {t_dev*1e3:.0f} ms ({t_host/t_dev:.1f}x)")
    assert t_dev < t_host, (t_dev, t_host)


def test_strong_rows_fold_in(device):
    """D1 item 2 gate: device strong rows == host LIL surgery, 1e-12."""
    dm, aq, dq, fq = _setup(2, 5, device)
    nu, sigma = 0.05, 20.0
    ndof = 3
    A_h, b_h = assemble_linear_ns(dm, aq, dq, fq, nu, sigma=sigma)
    rng = np.random.default_rng(4)
    rows = np.unique(rng.integers(0, A_h.shape[0], 200))
    bvals = rng.standard_normal(len(rows))
    Al = A_h.tolil()
    for k, r in enumerate(rows):
        Al.rows[r] = [int(r)]
        Al.data[r] = [1.0]
        b_h[r] = bvals[k]
    A_ref = Al.tocsr()
    asm = DeviceNSAssembler(dm)
    asm.set_strong_rows(rows)
    A_d, b_d = asm.assemble(aq, dq, fq, nu, sigma, strong_b_vals=bvals)
    diff = (A_ref - A_d)
    scale = np.abs(A_ref.data).max()
    assert (np.abs(diff.data).max() / scale if diff.nnz else 0) < 1e-12
    assert np.abs(b_h - b_d).max() / max(np.abs(b_h).max(), 1e-30) < 1e-12


def test_constraint_aware_scatter(device):
    """D1 item 3 gate: adapted mesh (hanging constraints, non-identity T)
    — device constrained assembly == host T^T K T at 1e-12."""
    from diffsim.octree.build import refine_elements
    from diffsim.octree.balance import balance2to1
    from diffsim.octree.build import build_uniform as bu
    from diffsim.physics.poisson import gauss_points
    tree = bu(4, dim=2)
    mask = np.zeros(len(tree), bool)
    mask[0] = True
    mask[len(tree) // 2] = True
    tree = balance2to1(refine_elements(tree, mask))
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    xq = gauss_points(mesh, dm.tables_by_p)
    rng = np.random.default_rng(3)
    aq, dq, fq = {}, {}, {}
    for pv in dm.bins:
        ngp = len(xq[pv])
        aq[pv] = rng.standard_normal((ngp, 2)) * 0.5
        dq[pv] = rng.standard_normal(ngp) * 0.1
        fq[pv] = rng.standard_normal((ngp, 2))
    nu, sigma = 0.05, 20.0
    A_h, b_h = assemble_linear_ns(dm, aq, dq, fq, nu, sigma=sigma)
    asm = DeviceNSAssembler(dm)
    A_d, b_d = asm.assemble(aq, dq, fq, nu, sigma)
    assert A_d.shape == A_h.shape
    diff = (A_h - A_d)
    scale = np.abs(A_h.data).max()
    assert (np.abs(diff.data).max() / scale if diff.nnz else 0) < 1e-12
    assert np.abs(b_h - b_d).max() / max(np.abs(b_h).max(), 1e-30) < 1e-12


def test_stepper_device_assembly_parity(device):
    """M1d stepper wiring gate: use_device_assembly=True produces the
    same trajectory as the host path (uniform cavity, 3 steps)."""
    from diffsim.steppers.linearized import LinearizedMonolithicStepper

    def lid(x, t):
        g = np.zeros((len(x), 2))
        g[np.abs(x[:, 1] - 1.0) < 1e-12, 0] = 1.0
        return g

    def run(dev_asm):
        tree = build_uniform(5, dim=2)
        mesh = build_mesh(tree, p=1)
        cons = build_constraints(mesh)
        dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2),
                                  device)
        st = LinearizedMonolithicStepper(
            dm, 0.01, 0.05, f_fn=lambda x, t: np.zeros((len(x), 2)),
            g_fn=lid, order=2, use_device_assembly=dev_asm)
        st.set_initial(lambda x: np.zeros((len(x), 2)))
        return [st.step().copy() for _ in range(3)]

    xs_h = run(False)
    xs_d = run(True)
    for a, b_ in zip(xs_h, xs_d):
        scale = max(np.abs(a).max(), 1e-30)
        assert np.abs(a - b_).max() / scale < 1e-11
