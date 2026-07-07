"""M2-A1 gates: scalar advection-diffusion brick MMS (2D/3D, p1/p2),
Poisson limit, and advection-dominated sanity."""
import numpy as np
import pytest
from scipy.sparse.linalg import splu

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.scalar_transport import assemble_scalar_ad
from diffsim.physics.poisson import gauss_points, l2_error

pytestmark = pytest.mark.tier2


def _solve(dim, level, p, a_fn, kappa, u_star, f_fn, sigma=0.0,
           device="cuda:0"):
    tree = build_uniform(level, dim=dim)
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=dim),
                              device)
    xq = gauss_points(mesh, dm.tables_by_p)
    aq = {pv: a_fn(xq[pv]) for pv in xq}
    fq = {pv: f_fn(xq[pv]) for pv in xq}
    A, b = assemble_scalar_ad(dm, aq, fq, kappa, sigma=sigma)
    coords = mesh.node_coords[cons.free_nodes]
    bdry = np.zeros(len(coords), bool)
    for c in range(dim):
        bdry |= (np.abs(coords[:, c]) < 1e-12) | \
                (np.abs(coords[:, c] - 1.0) < 1e-12)
    A = A.tolil()
    for i in np.where(bdry)[0]:
        A.rows[i] = [int(i)]
        A.data[i] = [1.0]
        b[i] = u_star(coords[i:i + 1])[0]
    x = splu(A.tocsr().tocsc()).solve(b)
    u_all = np.asarray(cons.T @ x)
    return l2_error(dm, u_all, u_star)


def _case(dim):
    if dim == 2:
        u = lambda x: np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
        gu = lambda x: np.stack(
            [np.pi * np.cos(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1]),
             -np.pi * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])],
            axis=1)
        lap = lambda x: -2 * np.pi ** 2 * u(x)
    else:
        u = lambda x: (np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
                       * np.sin(np.pi * x[:, 2]))
        gu = lambda x: np.stack(
            [np.pi * np.cos(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
             * np.sin(np.pi * x[:, 2]),
             -np.pi * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
             * np.sin(np.pi * x[:, 2]),
             np.pi * np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
             * np.cos(np.pi * x[:, 2])], axis=1)
        lap = lambda x: -3 * np.pi ** 2 * u(x)
    return u, gu, lap


def _a_rot(x):
    a = np.zeros_like(x)
    a[:, 0] = 1.0 + 0.5 * x[:, 1]
    a[:, 1] = -0.5 + 0.3 * x[:, 0]
    return a


@pytest.mark.parametrize("dim,levels,p,order_lo", [
    (2, (4, 5, 6), 1, 1.85),
    (2, (3, 4, 5), 2, 2.7),
    (3, (3, 4), 1, 1.7),
])
def test_scalar_ad_mms_orders(dim, levels, p, order_lo, device):
    """A1 gate: advection-diffusion MMS orders (kappa=0.7, sigma=0.4,
    rotating advecting field)."""
    u, gu, lap = _case(dim)
    kappa, sigma = 0.7, 0.4

    def f_fn(x):
        a = _a_rot(x[:, :dim])
        adv = (a[:, :dim] * gu(x)[:, :dim]).sum(1) if dim == 2 else \
            (np.pad(a, ((0, 0), (0, 1)))[:, :dim] * gu(x)).sum(1)
        return sigma * u(x) + adv - kappa * lap(x)

    def a_fn(x):
        a = np.zeros((len(x), dim))
        a[:, :2] = _a_rot(x)[:, :2]
        return a

    errs = [
        _solve(dim, lv, p, a_fn, kappa, u, f_fn, sigma, device)
        for lv in levels]
    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    print(f"{dim}D p{p}: errs {[f'{e:.2e}' for e in errs]} "
          f"orders {[f'{o:.2f}' for o in orders]}")
    assert orders[-1] > order_lo, (errs, orders)


def test_poisson_limit(device):
    """a=0, sigma=0 must reproduce the Poisson brick's accuracy class."""
    u, gu, lap = _case(2)
    err = _solve(2, 5, 1, lambda x: np.zeros((len(x), 2)), 1.0, u,
                 lambda x: -lap(x), 0.0, device)
    assert err < 2e-3, err


def test_advection_dominated_stability(device):
    """Pe_h >> 1 sanity: SUPG keeps the solution bounded (no blowup) on
    an advection-dominated case (kappa=1e-4, |a|~1)."""
    u, gu, lap = _case(2)
    kappa = 1e-4

    def f_fn(x):
        a = _a_rot(x)
        return (a * gu(x)).sum(1) - kappa * lap(x)

    err = _solve(2, 5, 1, _a_rot, kappa, u, f_fn, 0.0, device)
    assert np.isfinite(err) and err < 0.05, err


def test_species_brick_genericity(device):
    """A2: two scalars (T, C) from the SAME factory with different
    coefficient sets, independently correct (one assembly module, two
    physics)."""
    u, gu, lap = _case(2)
    for name, kap, sig in (("temperature", 0.7, 0.4),
                           ("species", 0.013, 0.0)):
        def f_fn(x, kap=kap, sig=sig):
            a = _a_rot(x)
            return sig * u(x) + (a * gu(x)).sum(1) - kap * lap(x)
        err = _solve(2, 5, 1, _a_rot, kap, u, f_fn, sig, device)
        assert err < 2e-3, (name, err)


def test_coupled_transient_mms(device):
    """A3 gate: transient forced convection over a frozen flow —
    BDF2 space-time MMS. T* = sin(pi x)cos(pi y) cos(t); the advecting
    field is the rotating a(x). Verifies the coupler's history/BDF
    wiring + brick composition."""
    from diffsim.steppers.coupled import ScalarTransportStepper
    from diffsim.mesh.basis import basis_tables as bt

    dim, level, kappa = 2, 5, 0.7
    u_sp, gu_sp, lap_sp = _case(2)

    def T_star(x, t):
        return u_sp(x) * np.cos(t)

    tree = build_uniform(level, dim=dim)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, bt(1, dim=dim), device)
    xq = gauss_points(mesh, dm.tables_by_p)
    aq = {pv: _a_rot(xq[pv]) for pv in xq}

    def f_fn(x, t):
        a = _a_rot(x)
        return (-u_sp(x) * np.sin(t)
                + np.cos(t) * (a * gu_sp(x)).sum(1)
                - kappa * np.cos(t) * lap_sp(x))

    coords = mesh.node_coords[cons.free_nodes]
    bdry = np.zeros(len(coords), bool)
    for c in range(dim):
        bdry |= (np.abs(coords[:, c]) < 1e-12) | \
                (np.abs(coords[:, c] - 1.0) < 1e-12)
    dir_nodes = np.where(bdry)[0]

    T_END = 0.5

    def run(dt):
        st = ScalarTransportStepper(
            dm, kappa, dt,
            g_fn=lambda x, t: T_star(x, t),
            dirichlet_nodes=dir_nodes, order=2, f_fn=f_fn)
        st.set_initial(lambda x: T_star(x, 0.0))
        for _ in range(int(round(T_END / dt))):
            Tn = st.step(aq)
        return Tn

    # accuracy vs exact: on the spatial floor (~6e-4 at L5)
    Tn = run(0.05)
    u_all = np.asarray(cons.T @ Tn)
    e_abs = l2_error(dm, u_all, lambda x: T_star(x, T_END))
    assert e_abs < 1.5e-3, e_abs
    # TEMPORAL order isolated against a fine-dt reference run (the
    # exact-solution comparison measures SPACE at these dt — measured
    # 6e-4 floor; first gate version failed on that)
    ref = run(0.0125)
    errs = [np.linalg.norm(run(dt) - ref) for dt in (0.1, 0.05, 0.025)]
    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
    print(f"coupled BDF2 temporal: errs {[f'{e:.2e}' for e in errs]} "
          f"orders {[f'{o:.2f}' for o in orders]}")
    assert orders[-1] > 1.7, (errs, orders)


def test_advection_dominated_p2(device):
    """p2 + VMS-complete SUPG at Pe_h >> 1: stable AND third-order-class
    accurate (the discriminating case for the completed residual)."""
    u, gu, lap = _case(2)
    kappa = 1e-4

    def f_fn(x):
        a = _a_rot(x)
        return (a * gu(x)).sum(1) - kappa * lap(x)

    err = _solve(2, 4, 2, _a_rot, kappa, u, f_fn, 0.0, device)
    assert np.isfinite(err) and err < 5e-3, err


def test_scalar_cotangents_fd(device):
    """A5 gate: taped scalar cotangents vs FD — scalar-kappa bump (sum of
    per-GP dkappa) and per-GP aq bumps."""
    from diffsim.sbm.scalar_adjoint import scalar_volume_cotangents
    from diffsim.mesh.basis import basis_tables as bt
    dim, level, kappa, sigma = 2, 4, 0.7, 0.4
    tree = build_uniform(level, dim=dim)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, bt(1, dim=dim), device)
    xq = gauss_points(mesh, dm.tables_by_p)
    rng = np.random.default_rng(5)
    pv = 1
    ngp = len(xq[pv])
    aq = {pv: rng.standard_normal((ngp, 2)) * 0.4}
    kq = {pv: np.full(ngp, kappa)}
    fq = {pv: np.zeros(ngp)}
    nfull = dm.n_nodes
    T_full = rng.standard_normal(nfull)
    lam_full = rng.standard_normal(nfull)

    def J_of(kq_np, aq_np):
        A, _ = assemble_scalar_ad(dm, {pv: aq_np},
                                  fq, lambda x_, k=kq_np: np.interp(
                                      np.zeros(1), [0], [0]) * 0 + k[
                                      :len(x_)] if False else None,
                                  sigma=sigma)
        return 0.0  # placeholder (assembly path uses callable kappa)

    # residual-based J: lam . R(T; kq, aq) via the taped kernel forward
    from diffsim.sbm.scalar_adjoint import make_scalar_residual
    import warp as wp

    def resid(kq_np, aq_np):
        b = dm.bins[pv]
        k = make_scalar_residual(b["nbf"], b["nqp"])
        r = wp.zeros(nfull, dtype=wp.float64, device=device)
        wp.launch(k, dim=len(b["eids"]),
                  inputs=[b["conn"], b["h"], b["N"], b["dN"], b["lapN"],
                          b["w"],
                          wp.array(np.ascontiguousarray(aq_np),
                                   dtype=wp.float64, device=device),
                          wp.array(np.ascontiguousarray(kq_np),
                                   dtype=wp.float64, device=device),
                          wp.float64(sigma),
                          wp.float64((2 * sigma) ** 2), wp.float64(1.0),
                          wp.array(T_full, dtype=wp.float64,
                                   device=device), r], device=device)
        return float(lam_full @ r.numpy())

    cot = scalar_volume_cotangents(dm, aq, kq, sigma, T_full, lam_full)
    dk_gp, da_gp = cot[pv]
    # scalar-kappa: dJ/dkappa = -sum(dk_gp)  (cotangent = -lam dR/dk)
    eps = 1e-6
    fd_k = (resid(kq[pv] + eps, aq[pv]) - resid(kq[pv] - eps,
                                                aq[pv])) / (2 * eps)
    rel_k = abs(-dk_gp.sum() - fd_k) / max(abs(fd_k), 1e-30)
    assert rel_k < 1e-6, (rel_k, fd_k, dk_gp.sum())
    # per-GP aq bumps
    for gpi in rng.integers(0, ngp, 3):
        for c in range(2):
            ap = aq[pv].copy(); ap[gpi, c] += eps
            am = aq[pv].copy(); am[gpi, c] -= eps
            fd = (resid(kq[pv], ap) - resid(kq[pv], am)) / (2 * eps)
            rel = abs(-da_gp[gpi, c] - fd) / max(abs(fd), 1e-12)
            assert rel < 1e-5, (gpi, c, rel)
    # consistency: taped forward == assembled A@T (volume, same config)
    A_v, _ = assemble_scalar_ad(dm, aq, fq, kappa, sigma=sigma)
    r_tape = resid(kq[pv], aq[pv])
    lam_free = np.asarray(cons.T.T @ lam_full)
    T_free_c = np.asarray(cons.T.T @ T_full)  # NOTE: only valid T=I mesh
    r_asm = float(lam_free @ (A_v @ np.asarray(
        np.linalg.lstsq(cons.T.toarray(), T_full, rcond=None)[0])))
    assert abs(r_tape - r_asm) / max(abs(r_asm), 1e-30) < 1e-10, (
        r_tape, r_asm)


def test_field_kappa_gradient_fd(device):
    """B1 gate: per-GP field-kappa gradient vs FD (3 random GPs) on the
    box config with wall-flux QoI."""
    from diffsim.sbm.scalar_adjoint import field_kappa_gradient
    from scipy.sparse.linalg import splu as _splu
    from diffsim.mesh.basis import basis_tables as bt
    tree = build_uniform(4, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, bt(1, dim=2), device)
    xq = gauss_points(mesh, dm.tables_by_p)
    pv, ngp = 1, len(xq[1])
    aq = {pv: np.tile([1.0, 0.3], (ngp, 1))}
    fq = {pv: np.zeros(ngp)}
    coords = mesh.node_coords[cons.free_nodes]
    bdry = np.zeros(len(coords), bool)
    for c in range(2):
        bdry |= (np.abs(coords[:, c]) < 1e-12) | \
                (np.abs(coords[:, c] - 1) < 1e-12)
    dirn = np.where(bdry)[0]
    hot = np.where(np.abs(coords[:, 0]) < 1e-12)[0]
    chi = np.zeros(len(coords)); chi[hot] = 1.0

    def solve_Q(kq_np):
        A, b = assemble_scalar_ad(dm, aq, fq,
                                  lambda x_: kq_np[:len(x_)], sigma=0.0)
        Av, bv = A.copy(), b.copy()
        A = A.tolil()
        hs = set(hot)
        for i in dirn:
            A.rows[i] = [int(i)]; A.data[i] = [1.0]
            b[i] = 1.0 if i in hs else 0.0
        T = _splu(A.tocsr().tocsc()).solve(b)
        return float(chi @ (Av @ T - bv)), A.tocsr(), Av, T

    kq0 = np.full(ngp, 0.05)
    Q0, A, Av, T = solve_Q(kq0)
    rhs = Av.T @ chi
    rhs[dirn] = 0.0
    lam = _splu(A.tocsc().T).solve(rhs)
    lam[dirn] = 0.0
    g = field_kappa_gradient(dm, aq, {pv: kq0}, 0.0,
                             np.asarray(cons.T @ T),
                             np.asarray(cons.T @ lam),
                             np.asarray(cons.T @ chi))[pv]
    rng = np.random.default_rng(9)
    eps = 1e-6
    for gpi in rng.integers(0, ngp, 3):
        kp = kq0.copy(); kp[gpi] += eps
        km = kq0.copy(); km[gpi] -= eps
        fd = (solve_Q(kp)[0] - solve_Q(km)[0]) / (2 * eps)
        err = abs(g[gpi] - fd)
        rel = err / max(abs(fd), 1e-12)
        # low-sensitivity GPs: FD noise dominates rel — absolute backstop
        assert rel < 1e-5 or err < 1e-10, (int(gpi), rel, g[gpi], fd)
