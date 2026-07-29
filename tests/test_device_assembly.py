"""M1d gates for DeviceNSAssembler (spec D2): consistency vs the host
assembler at 1e-12 + speed measurement, both scatter variants."""
import importlib.util
import time

import numpy as np
import pytest

_has_nvmath = importlib.util.find_spec("nvmath") is not None
_skip_nvmath = pytest.mark.skipif(
    not _has_nvmath, reason="nvmath (cuDSS) not available on this machine"
)

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


def test_assemble_extra_matrix_rhs(device):
    """M1d closure: cached extra contributions (SBM face system on the
    fixed pattern) added before the strong rows == host A + Af, b + bf
    at 1e-12 (the hero steady-driver composition)."""
    import warp as wp
    dm, aq, dq, fq = _setup(2, 4, device)
    nu, sigma = 0.05, 20.0
    asm = DeviceNSAssembler(dm)
    A0, b0 = asm.assemble(aq, dq, fq, nu, sigma)
    # On CPU warp, vals_d.numpy() returns a memory view (not a copy), so
    # A0.data shares the assembler's internal buffer — the second assemble()
    # call zeros it.  Force an independent copy immediately.
    A0 = A0.copy()
    b0 = b0.copy()
    rng = np.random.default_rng(9)
    # extra entries on EXISTING pattern positions + rhs adds
    take = rng.choice(asm.nnz, 500, replace=False)
    rows = np.searchsorted(asm.indptr, take, side="right") - 1
    cols = asm.indices[take]
    slots = asm.csr_slots(rows, cols)
    assert (slots == take).all()
    mv = rng.standard_normal(len(slots))
    dofs = rng.integers(0, asm.Nfull, 300)
    rv = rng.standard_normal(len(dofs))
    A1, b1 = asm.assemble(
        aq, dq, fq, nu, sigma,
        extra_matrix=(wp.array(slots.astype(np.int32), dtype=wp.int32,
                               device=device),
                      wp.array(mv, dtype=wp.float64, device=device)),
        extra_rhs=(wp.array(dofs.astype(np.int32), dtype=wp.int32,
                            device=device),
                   wp.array(rv, dtype=wp.float64, device=device)))
    Aref = A0.copy()
    Aref.data[slots] += mv
    bref = b0.copy()
    np.add.at(bref, dofs, rv)
    scale = np.abs(Aref.data).max()
    diff = (Aref - A1)
    assert (np.abs(diff.data).max() / scale if diff.nnz else 0) < 1e-12
    assert np.abs(bref - b1).max() / max(np.abs(bref).max(),
                                         1e-30) < 1e-12


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


def _host_gp_fields(dm, node_vals):
    """The stepper-style host einsums (the oracle for gp_field.py)."""
    full = np.asarray(dm.constraints.T @ node_vals)
    aq, dq, gu = {}, {}, {}
    for pv, b in dm.bins.items():
        tb = dm.tables_by_p[pv]
        conn = dm.mesh.conn_of[pv]
        vals = full[conn]
        aq[pv] = np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dm.dim)
        h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
        dq[pv] = (np.einsum("qad,ead->eq", tb.dN, vals)
                  * (2.0 / h)[:, None]).reshape(-1)
        gu[pv] = (np.einsum("qad,eac->eqdc", tb.dN, vals)
                  * (2.0 / h)[:, None, None, None]).reshape(
            -1, dm.dim * dm.dim)
    return aq, dq, gu


@pytest.mark.parametrize("dim,level,p,adapt", [
    (2, 4, 1, False), (3, 2, 1, False), (2, 3, 2, False),
    (2, 4, 1, True)])
def test_gp_field_device_parity(dim, level, p, adapt, device):
    """M1d D1-item-4 gate: device GP-field kernels (interp, consistent
    div, velocity/scalar gradients) == the host einsums at 1e-13, incl.
    hanging-constraint T application and p2 (nbf 9)."""
    from diffsim.assembly.gp_field import DeviceGPField
    tree = build_uniform(level, dim=dim)
    if adapt:
        from diffsim.octree.build import refine_elements
        from diffsim.octree.balance import balance2to1
        mask = np.zeros(len(tree), bool)
        mask[0] = True
        mask[len(tree) // 2] = True
        tree = balance2to1(refine_elements(tree, mask))
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=dim), device)
    rng = np.random.default_rng(7)
    u = rng.standard_normal((dm.n_free, dim))
    s = rng.standard_normal(dm.n_free)
    aq_h, dq_h, gu_h = _host_gp_fields(dm, u)
    full_s = np.asarray(cons.T @ s)
    gpf = DeviceGPField(dm)
    full_d = gpf.to_full(u, "u")
    aq_d, dq_d = gpf.interp_div(full_d, "u")
    gu_d = gpf.grad_vec(full_d, "u")
    gs_d = gpf.grad_scalar(gpf.to_full_scalar(s, "s"), "s")
    for pv, b in dm.bins.items():
        tb = dm.tables_by_p[pv]
        h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
        gs_h = (np.einsum("qad,ea->eqd", tb.dN,
                          full_s[dm.mesh.conn_of[pv]])
                * (2.0 / h)[:, None, None]).reshape(-1, dim)
        for got, ref in [(aq_d[pv].numpy(), aq_h[pv]),
                         (dq_d[pv].numpy(), dq_h[pv]),
                         (gu_d[pv].numpy(), gu_h[pv]),
                         (gs_d[pv].numpy(), gs_h)]:
            scale = max(np.abs(ref).max(), 1e-30)
            assert np.abs(got - ref).max() / scale < 1e-13


def _cavity_traj(device, dim, level, dev_asm, steps, solver="splu",
                 f_none=False):
    from diffsim.steppers.linearized import LinearizedMonolithicStepper

    def lid(x, t):
        g = np.zeros((len(x), dim))
        g[np.abs(x[:, 1] - 1.0) < 1e-12, 0] = 1.0
        return g

    tree = build_uniform(level, dim=dim)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim),
                              device)
    f_fn = None if f_none else (lambda x, t: np.zeros((len(x), dim)))
    st = LinearizedMonolithicStepper(
        dm, 0.01, 0.05, f_fn=f_fn,
        g_fn=lid, order=2, use_device_assembly=dev_asm, solver=solver)
    st.set_initial(lambda x: np.zeros((len(x), dim)))
    return [st.step().copy() for _ in range(steps)]


def test_stepper_device_assembly_parity(device):
    """M1d stepper gate: use_device_assembly=True (device GP fields ->
    device assembly, no host round-trip) produces the same trajectory as
    the host path (uniform cavity, 4 steps: covers BDF1 bootstrap, the
    fine-scale corrected field and the 2 c_n - c_m extrapolation)."""
    xs_h = _cavity_traj(device, 2, 5, False, 4)
    xs_d = _cavity_traj(device, 2, 5, True, 4)
    for a, b_ in zip(xs_h, xs_d):
        scale = max(np.abs(a).max(), 1e-30)
        assert np.abs(a - b_).max() / scale < 1e-11


def test_stepper_device_assembly_parity_3d(device):
    """Same gate in 3-D (nbf 8, rolled-loop kernels)."""
    xs_h = _cavity_traj(device, 3, 3, False, 4)
    xs_d = _cavity_traj(device, 3, 3, True, 4)
    for a, b_ in zip(xs_h, xs_d):
        scale = max(np.abs(a).max(), 1e-30)
        assert np.abs(a - b_).max() / scale < 1e-11


def test_stepper_device_resident_cudss_parity(device):
    """M1d D3 gate: the fully device-resident step (device GP fields ->
    device assembly -> zero-copy cuDSS refactorize+solve, plan reused
    across steps) matches the host splu trajectory. f_fn=None (zero
    body force, no per-step callback) on BOTH paths."""
    if device == "cpu":
        pytest.skip("cuDSS path needs a GPU")
    xs_h = _cavity_traj(device, 2, 5, False, 4, f_none=True)
    xs_d = _cavity_traj(device, 2, 5, True, 4, solver="cudss",
                        f_none=True)
    for a, b_ in zip(xs_h, xs_d):
        scale = max(np.abs(a).max(), 1e-30)
        assert np.abs(a - b_).max() / scale < 1e-11


def test_stepper_device_resident_fused_parity(device):
    """The fused-Krylov device step (zero-copy CSR operator, vals never
    leave the GPU — the factorization-exceeds-HBM fallback) matches the
    host splu trajectory at the iterative tol: velocity 1e-6, pressure
    1e-5 (BiCGStab rtol 1e-10; typical 2.3e-8 velocity / 3.3e-7
    near-uniform pressure wiggle — the pin fixes one node; the rest
    rides the residual). TOLERANCE NOTE (measured 2026-07-09): run-to-run
    GPU nondeterminism in the iterative stopping point reaches 2.0e-7
    velocity — the original 1e-7 gate flaked 4/10 standalone. Tolerances
    sit 5x above the observed tail; a real parity break is O(1e-2)+."""
    xs_h = _cavity_traj(device, 2, 5, False, 4)
    xs_d = _cavity_traj(device, 2, 5, True, 4, solver="fused")
    for a, b_ in zip(xs_h, xs_d):
        scale = max(np.abs(a).max(), 1e-30)
        assert np.abs(a[:, :2] - b_[:, :2]).max() / scale < 1e-6
        assert np.abs(a[:, 2] - b_[:, 2]).max() / scale < 1e-5


@_skip_nvmath
def test_assemble_device_resident(device):
    """D3 gate: device-resident CSR solves identically to the host path."""
    import torch
    from nvmath.sparse.advanced import DirectSolver
    dm, aq, dq, fq = _setup(2, 5, device)
    asm = DeviceNSAssembler(dm)
    A_h, b_h = asm.assemble(aq, dq, fq, 0.05, 20.0)
    A_t, b_t = asm.assemble_device(aq, dq, fq, 0.05, 20.0)
    slv = DirectSolver(A_t, b_t.clone())
    slv.plan(); slv.factorize()
    x_t = slv.solve().cpu().numpy()
    res = np.linalg.norm(A_h @ x_t - b_h) / max(
        np.linalg.norm(b_h), 1e-30)
    assert res < 1e-11, res


def test_device_assembly_p2(device):
    """Pure-p2 framework: DeviceNSAssembler parity at p=2 (nbf-generic
    slot maps)."""
    tree = build_uniform(4, dim=2)
    mesh = build_mesh(tree, p=2)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(2, dim=2), device)
    from diffsim.physics.poisson import gauss_points
    xq = gauss_points(mesh, dm.tables_by_p)
    rng = np.random.default_rng(4)
    pv = 2
    ngp = len(xq[pv])
    aq = {pv: rng.standard_normal((ngp, 2)) * 0.4}
    dq = {pv: rng.standard_normal(ngp) * 0.1}
    fq = {pv: rng.standard_normal((ngp, 2))}
    A_h, b_h = assemble_linear_ns(dm, aq, dq, fq, 0.05, sigma=20.0)
    asm = DeviceNSAssembler(dm)
    A_d, b_d = asm.assemble(aq, dq, fq, 0.05, 20.0)
    diff = (A_h - A_d)
    scale = np.abs(A_h.data).max()
    assert (np.abs(diff.data).max() / scale if diff.nnz else 0) < 1e-12
    assert np.abs(b_h - b_d).max() / max(np.abs(b_h).max(), 1e-30) < 1e-12


def test_stepper_p2_sanity(device):
    """Pure-p2 framework: LinearizedMonolithicStepper at p=2 — 10 cavity
    steps finite + device-assembly parity."""
    from diffsim.steppers.linearized import LinearizedMonolithicStepper

    def lid(x, t):
        g = np.zeros((len(x), 2))
        g[np.abs(x[:, 1] - 1.0) < 1e-12, 0] = 1.0
        return g

    def run(dev):
        tree = build_uniform(4, dim=2)
        mesh = build_mesh(tree, p=2)
        cons = build_constraints(mesh)
        dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(2, dim=2),
                                  device)
        st = LinearizedMonolithicStepper(
            dm, 0.01, 0.05, f_fn=lambda x, t: np.zeros((len(x), 2)),
            g_fn=lid, order=2, use_device_assembly=dev)
        st.set_initial(lambda x: np.zeros((len(x), 2)))
        for _ in range(10):
            x = st.step()
        return x.copy()

    x_h = run(False)
    assert np.isfinite(x_h).all() and np.abs(x_h).max() < 10
    x_d = run(True)
    assert np.abs(x_h - x_d).max() / max(np.abs(x_h).max(), 1e-30) < 1e-11


# ---------------------------------------------------------------------
# G5 rung c: node-graph pattern build (dof CSR = kron(G, ones(4,4))
# structurally; closed-form in-kernel slot maps) vs the original dof-COO
# build — the EXACTNESS gate for the full-res film assembler.
# ---------------------------------------------------------------------
def _strip3d(device, nx, ny, nz, level):
    from diffsim.octree.build import Octree
    tree0 = build_uniform(level, dim=3)
    hc = 2.0 ** -level
    cen = tree0.centers()
    keep = ((cen[:, 0] < nx * hc) & (cen[:, 1] < ny * hc)
            & (cen[:, 2] < nz * hc))
    tree = Octree(tree0.keys[keep], tree0.levels[keep], dim=3,
                  periodic=tree0.periodic)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    return DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=3),
                                device)


def test_node_pattern_exactness(device):
    """32x32x16 3-D strip (74k dofs, 4-dof node-major): the node-graph
    pattern build must reproduce the dof-COO build EXACTLY — identical
    CSR indptr/indices, and identical scattered values (same random
    element blocks through both slot-map paths; tolerance covers only
    atomic-order rounding on shared slots)."""
    import warp as wp
    dm = _strip3d(device, 32, 32, 16, 5)
    a_old = DeviceNSAssembler(dm, ndof=4, node_pattern=False)
    a_new = DeviceNSAssembler(dm, ndof=4, node_pattern=True)
    assert a_new.node_mode and not a_old.node_mode
    assert a_old.nnz == a_new.nnz
    assert np.array_equal(np.asarray(a_old.indptr, np.int64),
                          np.asarray(a_new.indptr, np.int64))
    assert np.array_equal(np.asarray(a_old.indices, np.int64),
                          np.asarray(a_new.indices, np.int64))
    pv, b, ne, nbf, _ = a_old._bins[0]
    nl = 4 * nbf
    rng = np.random.default_rng(11)
    Ae = rng.standard_normal((ne, nl, nl))
    be = rng.standard_normal((ne, nl))
    Ae_d = wp.array(Ae, dtype=wp.float64, device=device)
    be_d = wp.array(be, dtype=wp.float64, device=device)
    for asm in (a_old, a_new):
        asm.zero_fill()
        asm.scatter_bin(0, Ae_d, be_d)
    v_o, v_n = a_old.vals_d.numpy(), a_new.vals_d.numpy()
    f_o, f_n = a_old.F_d.numpy(), a_new.F_d.numpy()
    va = np.abs(v_o - v_n).max()
    fa = np.abs(f_o - f_n).max()
    print(f"node-pattern exactness: nnz={a_new.nnz} "
          f"max|dA|={va:.2e} max|dF|={fa:.2e}")
    assert va < 1e-14 * max(1.0, np.abs(v_o).max()), va
    assert fa < 1e-14 * max(1.0, np.abs(f_o).max()), fa
    # batched scatter == whole-bin scatter (the rung-c launch unit)
    a_new.zero_fill()
    step = ne // 3 + 1
    for e0 in range(0, ne, step):
        nb = min(step, ne - e0)
        Ab = wp.array(np.ascontiguousarray(Ae[e0:e0 + nb]),
                      dtype=wp.float64, device=device)
        bb = wp.array(np.ascontiguousarray(be[e0:e0 + nb]),
                      dtype=wp.float64, device=device)
        a_new.scatter_batch(0, e0, Ab, bb, nb)
    v_b = a_new.vals_d.numpy()
    assert np.abs(v_b - v_n).max() < 1e-14 * max(1.0, np.abs(v_n).max())


def test_film_node_pattern_march_parity(device):
    """The ternary film system through BOTH pattern builds: a short 3-D
    film march (blockch, device assembly + device blockch setup) with
    the node-graph pattern forced on/off must agree — the whole-pipeline
    value gate (element scatter + top-flux csr_slots adds + F_d)."""
    from diffsim.physics.wodo_film import WodoFilmStepper

    def run(node_pattern, nsteps=3):
        dm = _strip3d(device, 16, 16, 8, 4)
        st = WodoFilmStepper(dm, chi=(1.0, 0.3, 0.3), N=(5.0, 5.0, 1.0),
                             M=(0.225, 0.0, 0.225), kappa=(2e-4, 2e-4),
                             k_e=1.0, dt=1e-3, linsolver="blockch",
                             use_device_assembly=True)
        st._node_pattern = node_pattern
        rng = np.random.default_rng(3)
        st.set_initial(
            lambda x: 0.2 + 0.01 * rng.standard_normal(len(x)),
            lambda x: 0.2 + 0.01 * rng.standard_normal(len(x)))
        rec = []
        st.march(h_min=0.8, phis_stop=0.05, max_steps=nsteps,
                 callback=lambda s, K, dt, it:
                 rec.append((dt, s.x.copy())))
        return rec, st.n_reject

    rec_o, rej_o = run(False)
    rec_n, rej_n = run(True)
    assert rej_o == rej_n and len(rec_o) == len(rec_n)
    assert all(abs(a[0] - b[0]) < 1e-9 * a[0]
               for a, b in zip(rec_o, rec_n)), "dt ladder diverged"
    errs = [np.abs(a[1] - b[1]).max() / max(np.abs(a[1]).max(), 1e-30)
            for a, b in zip(rec_o, rec_n)]
    print(f"film node-pattern march parity: {len(rec_o)} steps, "
          f"max rel {max(errs):.2e}")
    assert max(errs) < 1e-9, errs


def test_constraint_aware_csr_int64_indptr(device):
    """Regression: constrained-path COO uses sp.coo_array (int64 indptr).

    sp.coo_matrix.tocsr() produces int32 indptr/indices (overflows at ~2^31
    NNZ for large adaptive meshes, e.g. 3d-L7r9 with hanging nodes).
    sp.coo_array.tocsr() produces int64 indptr.  This test asserts that
    DeviceNSAssembler's constraint-aware scatter path (identity_T=False,
    non-uniform AMR mesh) stores int64 indptr so the overflow cannot occur
    at large scale.
    """
    from diffsim.octree.build import refine_elements
    from diffsim.octree.balance import balance2to1
    from diffsim.octree.build import build_uniform as bu
    from diffsim.physics.poisson import gauss_points

    # AMR mesh with hanging nodes (non-identity T) — triggers the
    # constraint-expansion path in DeviceNSAssembler.__init__.
    tree = bu(3, dim=2)
    mask = np.zeros(len(tree), bool)
    mask[0] = True
    tree = balance2to1(refine_elements(tree, mask))
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    asm = DeviceNSAssembler(dm)
    # The constraint-expansion path sets self.indptr from K.indptr.
    # With coo_array (the fix), K.indptr is int64; with the old coo_matrix
    # it would be int32 (would silently overflow at ~2^31 NNZ).
    assert asm.indptr.dtype == np.int64, (
        f"DeviceNSAssembler.indptr must be int64 on constrained mesh "
        f"(got {asm.indptr.dtype}); coo_matrix overflow regression"
    )


# ---------------------------------------------------------------------
# W1: bound the per-step element-block (Ae/be) intermediate.  The
# whole-bin `Ae = wp.zeros((ne, nl, nl))` in assemble() sized to the
# ENTIRE element set — measured OOM: a single 137.4 GB allocation at
# ~68M DOF (3d-L8, ne=16.77M, nl=32 -> ne*1024*8 B), NOT a capacity
# wall (persistent state was 62.7 GiB).  The fix computes Ae/be in
# element BATCHES bounded by AE_BATCH_BYTES and scatters each batch;
# the assembled CSR must be BIT-IDENTICAL to the whole-bin path.  The
# `_ae_batch` hook forces the multi-batch path at toy sizes here.
# ---------------------------------------------------------------------
def _assemble_csr(asm, aq, dq, fq, nu, sigma):
    A, b = asm.assemble(aq, dq, fq, nu, sigma)
    A = A.tocsr()
    A.sort_indices()
    return A, b


@pytest.mark.parametrize("coloring", [False, True])
def test_ae_batch_parity_identity(coloring, device):
    """Identity-T uniform mesh: multi-batch Ae assembly (forced via a
    tiny _ae_batch) produces a BIT-IDENTICAL CSR (indptr/indices/data)
    and rhs vs the whole-bin path.  Both colored and uncolored scatter."""
    dm, aq, dq, fq = _setup(3, 3, device)
    nu, sigma = 0.05, 20.0
    ref = DeviceNSAssembler(dm, coloring=coloring)
    A_ref, b_ref = _assemble_csr(ref, aq, dq, fq, nu, sigma)
    ne = ref._bins[0][2]
    asm = DeviceNSAssembler(dm, coloring=coloring)
    asm._ae_batch = max(1, ne // 4)          # force >=4 batches
    A_b, b_b = _assemble_csr(asm, aq, dq, fq, nu, sigma)
    assert np.array_equal(A_ref.indptr, A_b.indptr)
    assert np.array_equal(A_ref.indices, A_b.indices)
    assert np.array_equal(A_ref.data, A_b.data), (
        "Ae-batched CSR data must be BIT-IDENTICAL to whole-bin",
        np.abs(A_ref.data - A_b.data).max())
    assert np.array_equal(b_ref, b_b)


def test_ae_batch_parity_node_pattern(device):
    """Node-graph pattern (forced): multi-batch Ae assembly through the
    in-kernel-slot scatter is BIT-IDENTICAL to the whole-bin path."""
    dm = _strip3d(device, 8, 8, 8, 4)
    rng = np.random.default_rng(7)
    pv = list(dm.bins)[0]
    from diffsim.physics.poisson import gauss_points
    xq = gauss_points(dm.mesh, dm.tables_by_p)
    ngp = len(xq[pv])
    aq = {pv: rng.standard_normal((ngp, 3)) * 0.5}
    dq = {pv: rng.standard_normal(ngp) * 0.1}
    fq = {pv: rng.standard_normal((ngp, 3))}
    nu, sigma = 0.05, 20.0
    ref = DeviceNSAssembler(dm, ndof=4, node_pattern=True)
    A_ref, b_ref = _assemble_csr(ref, aq, dq, fq, nu, sigma)
    ne = ref._bins[0][2]
    asm = DeviceNSAssembler(dm, ndof=4, node_pattern=True)
    asm._ae_batch = max(1, ne // 3)
    A_b, b_b = _assemble_csr(asm, aq, dq, fq, nu, sigma)
    assert np.array_equal(A_ref.indptr, A_b.indptr)
    assert np.array_equal(A_ref.indices, A_b.indices)
    assert np.array_equal(A_ref.data, A_b.data), (
        np.abs(A_ref.data - A_b.data).max())
    assert np.array_equal(b_ref, b_b)


def test_ae_batch_default_bound(device):
    """The derived default Ae batch keeps the per-batch element-block
    intermediate <= AE_BATCH_BYTES (the ~2 GB bound) and, at these toy
    sizes where ne fits in one batch, stays single-batch (bit-for-bit
    the pre-W1 path)."""
    from diffsim.assembly.device_assembly import AE_BATCH_BYTES
    dm, aq, dq, fq = _setup(3, 3, device)
    asm = DeviceNSAssembler(dm)
    pv, b, ne, nbf, gdof = asm._bins[0]
    npair = (nbf * asm.ndof) ** 2
    nb = asm._ae_batch_for(npair)
    assert nb * npair * 8 <= AE_BATCH_BYTES
    assert nb >= ne          # toy mesh: single batch, unchanged path
