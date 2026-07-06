"""M1c gates for the linearized s=1/2 NS adjoint (tier order per findings
4c: tape-vs-kernel-FD FIRST, physics gradients after)."""
import os
import sys

import numpy as np
import pytest
import warp as wp

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.poisson import gauss_points
from diffsim.sbm.ns_adjoint import make_lin_ns_residual, ns_volume_cotangents
from diffsim.api.ns_bricks import assemble_linear_ns

sys.path.insert(0, os.path.dirname(__file__))

pytestmark = pytest.mark.ad


def _setup(level, device):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    xq = gauss_points(mesh, dm.tables_by_p)
    rng = np.random.default_rng(3)
    pv = list(dm.bins)[0]
    ngp = len(xq[pv])
    aq = {pv: rng.standard_normal((ngp, 2)) * 0.5}
    dq = {pv: rng.standard_normal(ngp) * 0.1}
    x_full = rng.standard_normal(dm.n_nodes * 3)
    lam_full = rng.standard_normal(dm.n_nodes * 3)
    return dm, xq, aq, dq, x_full, lam_full


def _r_dot_lam(dm, aq, dq, nu, sigma, s_skew, x_full, lam_full):
    """<lam, r(aq, dq, nu)> via a plain (untaped) launch."""
    d = dm.device
    dim = dm.dim
    pv, b = next(iter(dm.bins.items()))
    k = make_lin_ns_residual(b["nbf"], b["nqp"], dim)
    r = wp.zeros(dm.n_nodes * 3, dtype=wp.float64, device=d)
    wp.launch(k, dim=len(b["eids"]),
              inputs=[b["conn"], b["h"], b["N"], b["dN"], b["w"],
                      wp.array(np.ascontiguousarray(aq[pv]),
                               dtype=wp.float64, device=d),
                      wp.array(np.ascontiguousarray(dq[pv]),
                               dtype=wp.float64, device=d),
                      wp.array(np.ascontiguousarray(aq[pv]),
                               dtype=wp.float64, device=d),
                      wp.array(np.array([nu]), dtype=wp.float64, device=d),
                      wp.float64(sigma), wp.float64((2 * sigma) ** 2),
                      wp.float64(s_skew),
                      wp.array(x_full, dtype=wp.float64, device=d), r],
              device=d)
    return float(lam_full @ r.numpy())


def test_tape_vs_kernel_fd(device):
    """THE findings-4c contract for every new taped kernel."""
    dm, xq, aq, dq, x_full, lam_full = _setup(3, device)
    nu, sigma, s = 0.05, 20.0, 0.5
    aq_bar, dnu = ns_volume_cotangents(dm, aq, dq, nu, sigma, s,
                                       x_full, lam_full)
    pv = list(dm.bins)[0]
    g_aq, g_dq = aq_bar[pv]
    eps = 1e-6
    # aq entries (three probes)
    for gp, c in ((0, 0), (7, 1), (3, 0)):
        ap = {pv: aq[pv].copy()}; ap[pv][gp, c] += eps
        am = {pv: aq[pv].copy()}; am[pv][gp, c] -= eps
        # NOTE: tau uses the FROZEN copy inside cotangents, but this FD
        # perturbs the aq used for tau too — so probe with tau ALSO from
        # the perturbed field is wrong; replicate frozen-tau semantics by
        # passing the perturbed aq as BOTH aq and dq unchanged... the
        # kernel's aq_f input in _r_dot_lam is set to the same perturbed
        # array — to match the tape's frozen-tau, hold aq_f at BASE:
        d = dm.device
        pvb = dm.bins[pv]
        k = make_lin_ns_residual(pvb["nbf"], pvb["nqp"], dm.dim)

        def rdot(afield):
            r = wp.zeros(dm.n_nodes * 3, dtype=wp.float64, device=d)
            wp.launch(k, dim=len(pvb["eids"]),
                      inputs=[pvb["conn"], pvb["h"], pvb["N"], pvb["dN"],
                              pvb["w"],
                              wp.array(np.ascontiguousarray(afield),
                                       dtype=wp.float64, device=d),
                              wp.array(np.ascontiguousarray(dq[pv]),
                                       dtype=wp.float64, device=d),
                              wp.array(np.ascontiguousarray(aq[pv]),
                                       dtype=wp.float64, device=d),  # BASE
                              wp.array(np.array([0.05]), dtype=wp.float64,
                                       device=d),
                              wp.float64(20.0), wp.float64(1600.0),
                              wp.float64(0.5),
                              wp.array(x_full, dtype=wp.float64, device=d),
                              r], device=d)
            return float(lam_full @ r.numpy())
        fd = (rdot(ap[pv]) - rdot(am[pv])) / (2 * eps)
        tape_g = -g_aq[gp, c]                     # cotangents carry -lam^T
        assert abs(fd - tape_g) < 1e-6 * max(abs(fd), 1e-10), (
            (gp, c), fd, tape_g)
    # nu (frozen-tau semantics: tau uses nu too — the tape DOES carry nu
    # through tau since nu_arr feeds tau_m_metric; FD must match exactly)
    f0p = _r_dot_lam(dm, aq, dq, nu + eps, sigma, s, x_full, lam_full)
    f0m = _r_dot_lam(dm, aq, dq, nu - eps, sigma, s, x_full, lam_full)
    fd_nu = (f0p - f0m) / (2 * eps)
    assert abs(fd_nu - (-dnu)) < 1e-6 * max(abs(fd_nu), 1e-10), (fd_nu, -dnu)


def test_residual_matches_assembled(device):
    """r(x) from the taped kernel == A x from the brick assembly (volume
    consistency; strong rows excluded by comparing FULL unconstrained)."""
    dm, xq, aq, dq, x_full, _ = _setup(3, device)
    nu, sigma = 0.05, 20.0
    pv = list(dm.bins)[0]
    d = dm.device
    b = dm.bins[pv]
    k = make_lin_ns_residual(b["nbf"], b["nqp"], dm.dim)
    r = wp.zeros(dm.n_nodes * 3, dtype=wp.float64, device=d)
    wp.launch(k, dim=len(b["eids"]),
              inputs=[b["conn"], b["h"], b["N"], b["dN"], b["w"],
                      wp.array(np.ascontiguousarray(aq[pv]),
                               dtype=wp.float64, device=d),
                      wp.array(np.ascontiguousarray(dq[pv]),
                               dtype=wp.float64, device=d),
                      wp.array(np.ascontiguousarray(aq[pv]),
                               dtype=wp.float64, device=d),
                      wp.array(np.array([nu]), dtype=wp.float64, device=d),
                      wp.float64(sigma), wp.float64((2 * sigma) ** 2),
                      wp.float64(0.5),
                      wp.array(x_full, dtype=wp.float64, device=d), r],
              device=d)
    # assembled path on the same GP data: unconstrained volume matrix
    import scipy.sparse as sp
    fq = {pv: np.zeros((len(xq[pv]), 2))}
    A_c, _ = assemble_linear_ns(dm, aq, dq, fq, nu, sigma=sigma)
    T = dm.constraints.T.tocsr()
    T_vec = sp.kron(T, sp.identity(3, format="csr"), format="csr")
    # uniform mesh: T is identity, so constrained == full
    r_assembled = np.asarray(A_c @ (T_vec.T @ x_full))
    r_kernel = np.asarray(T_vec.T @ r.numpy())
    scale = np.abs(r_assembled).max()
    assert np.abs(r_kernel - r_assembled).max() < 1e-12 * scale


def test_nu_gradient_adjoint_vs_fd(device):
    """First NS PHYSICS gradient: dJ/dnu on the linearized system at a
    frozen advecting field (the production epoch pattern: aq held, x
    solves A(aq, nu) x = b). J = 0.5 sum(u^2) over velocity dofs.
    Adjoint: A^T lam = dJ/dx; dJ/dnu = -lam^T dR/dnu via the taped kernel.
    FD: re-solve the LINEAR system at nu +- eps (aq frozen — matches the
    partial derivative the adjoint computes)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    dm, xq, aq, dq, x_full, _ = _setup(3, device)
    nu, sigma = 0.05, 20.0
    pv = list(dm.bins)[0]
    fq = {pv: np.zeros((len(xq[pv]), 2))}

    def solve(nu_val):
        A, bvec = assemble_linear_ns(dm, aq, dq, fq, nu_val, sigma=sigma)
        rng = np.random.default_rng(11)
        bvec = rng.standard_normal(A.shape[0])   # generic rhs, same seed
        return A, bvec, spla.spsolve(A.tocsc(), bvec)

    A, bvec, xc = solve(nu)
    T = dm.constraints.T.tocsr()
    T_vec = sp.kron(T, sp.identity(3, format="csr"), format="csr")
    x_node = np.asarray(T_vec @ xc)
    # J = 0.5 * sum over velocity dofs of x^2 (constrained dofs)
    vel_mask = np.tile([1.0, 1.0, 0.0], len(x_node) // 3)
    J = 0.5 * float((x_node * vel_mask) @ x_node)
    dJdx_node = x_node * vel_mask
    dJdx = np.asarray(T_vec.T @ dJdx_node)
    lam = spla.spsolve(A.tocsc().T, dJdx)
    lam_node = np.asarray(T_vec @ lam)
    _, dnu = ns_volume_cotangents(dm, aq, dq, nu, sigma, 0.5,
                                  x_node, lam_node)
    eps = 1e-6
    _, _, xp = solve(nu + eps)
    _, _, xm = solve(nu - eps)
    xpn = np.asarray(T_vec @ xp)
    xmn = np.asarray(T_vec @ xm)
    Jp = 0.5 * float((xpn * vel_mask) @ xpn)
    Jm = 0.5 * float((xmn * vel_mask) @ xmn)
    fd = (Jp - Jm) / (2 * eps)
    assert abs(fd - dnu) < 1e-5 * max(abs(fd), 1e-12), (fd, dnu, J)


def test_load_kernel_consistency_and_tape(device):
    """4c contract for the taped LOAD kernel: (i) matches the assembled b
    exactly; (ii) tape == kernel-FD for aq, fq, nu."""
    dm, xq, aq, dq, x_full, lam_full = _setup(3, device)
    nu, sigma = 0.05, 20.0
    pv = list(dm.bins)[0]
    rng = np.random.default_rng(5)
    fqv = {pv: rng.standard_normal((len(xq[pv]), 2))}
    from diffsim.sbm.ns_adjoint import make_lin_ns_load, ns_load_cotangents
    d = dm.device
    b = dm.bins[pv]
    k = make_lin_ns_load(b["nbf"], b["nqp"], dm.dim)

    def run_b(aqv, fv, nuv):
        r = wp.zeros(dm.n_nodes * 3, dtype=wp.float64, device=d)
        wp.launch(k, dim=len(b["eids"]),
                  inputs=[b["conn"], b["h"], b["N"], b["dN"], b["w"],
                          wp.array(np.ascontiguousarray(aqv),
                                   dtype=wp.float64, device=d),
                          wp.array(np.ascontiguousarray(fv),
                                   dtype=wp.float64, device=d),
                          wp.array(np.array([nuv]), dtype=wp.float64,
                                   device=d),
                          wp.float64((2 * sigma) ** 2), r], device=d)
        return r.numpy()

    # (i) consistency vs assemble_linear_ns's b
    import scipy.sparse as sp
    _, b_asm = assemble_linear_ns(dm, aq, dq, fqv, nu, sigma=sigma)
    T = dm.constraints.T.tocsr()
    T_vec = sp.kron(T, sp.identity(3, format="csr"), format="csr")
    b_kernel = np.asarray(T_vec.T @ run_b(aq[pv], fqv[pv], nu))
    scale = np.abs(b_asm).max()
    assert np.abs(b_kernel - b_asm).max() < 1e-12 * scale

    # (ii) tape vs kernel-FD
    aq_bar, fq_bar, dnu = ns_load_cotangents(dm, aq, fqv, nu, sigma,
                                             lam_full)
    eps = 1e-6
    for arrs, bar, probes in (
            ((aq[pv],), aq_bar[pv], ((0, 0), (7, 1))),
            ((fqv[pv],), fq_bar[pv], ((3, 0), (11, 1)))):
        base = arrs[0]
        for gp, c in probes:
            ap = base.copy(); ap[gp, c] += eps
            am = base.copy(); am[gp, c] -= eps
            if base is aq[pv]:
                fd = (lam_full @ run_b(ap, fqv[pv], nu)
                      - lam_full @ run_b(am, fqv[pv], nu)) / (2 * eps)
            else:
                fd = (lam_full @ run_b(aq[pv], ap, nu)
                      - lam_full @ run_b(aq[pv], am, nu)) / (2 * eps)
            assert abs(fd - bar[gp, c]) < 1e-6 * max(abs(fd), 1e-10), (
                (gp, c), fd, bar[gp, c])
    fd_nu = (lam_full @ run_b(aq[pv], fqv[pv], nu + eps)
             - lam_full @ run_b(aq[pv], fqv[pv], nu - eps)) / (2 * eps)
    assert abs(fd_nu - dnu) < 1e-6 * max(abs(fd_nu), 1e-10), (fd_nu, dnu)
