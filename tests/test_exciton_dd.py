"""SP-1 B1 gates: XDD Poisson brick (λ²-form, GP ε̂ field) MMS.

Gate summary
------------
G1  MMS constant-ε̂: p1 → ≥2, p2 → ≥3 (2-D, levels 3-5)
G2  MMS variable-ε̂: same orders hold for ε̂ = 1 + 0.3·tanh((x-0.5)/0.1)
G3  Source coupling: (p̂−n̂, w) term correct (ρ ≠ 0, manufactured)
G4  λ² wiring: assembled operator scales linearly with λ² (ratio test)
"""
import numpy as np
import pytest
from scipy.sparse.linalg import splu

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.poisson import gauss_points, l2_error
from diffsim.physics.exciton_dd import assemble_xdd_poisson
from diffsim.xdd.params import XDDParams
from diffsim.diagnostics.convergence import observed_order

pytestmark = pytest.mark.tier2

# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_dm(level, p, device="cpu"):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2), device)
    return dm, mesh, cons


def _apply_dirichlet(A, b, coords, u_exact):
    """Pin all boundary nodes to u_exact; return modified (A_lil, b) in place."""
    A = A.tolil()
    bdry = np.zeros(len(coords), bool)
    for c in range(2):
        bdry |= (np.abs(coords[:, c]) < 1e-12) | (np.abs(coords[:, c] - 1.0) < 1e-12)
    for i in np.where(bdry)[0]:
        A.rows[i] = [int(i)]
        A.data[i] = [1.0]
        b[i] = u_exact(coords[i : i + 1])[0]
    return A.tocsr(), b


# Manufactured solution: φ̂ = sin(πx)sin(πy)
_phi_star = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])


def _lap_phi_star(x):
    return -2.0 * np.pi ** 2 * _phi_star(x)


def _grad_phi_star(x):
    return np.stack(
        [np.pi * np.cos(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]),
         np.pi * np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])],
        axis=1,
    )


def _source_const_eps(x, lam2, eps_hat=1.0):
    """MMS source for −∇·(λ² ε̂ ∇φ̂) = f (rhs density = 0 in these gates)."""
    return -lam2 * eps_hat * _lap_phi_star(x)


def _source_var_eps(x, lam2):
    """MMS source for −∇·(λ² ε̂(x) ∇φ̂) with ε̂(x) = 1 + 0.3·tanh((x-0.5)/0.1).

    Product rule: −λ²[ ε̂ Δφ + ∇ε̂ · ∇φ ]
    """
    z = (x[:, 0] - 0.5) / 0.1
    eps_hat = 1.0 + 0.3 * np.tanh(z)
    deps_dx = 0.3 / 0.1 * (1.0 - np.tanh(z) ** 2)
    gp = _grad_phi_star(x)
    lap = _lap_phi_star(x)
    return -lam2 * (eps_hat * lap + deps_dx * gp[:, 0])


def _eps_var(x):
    z = (x[:, 0] - 0.5) / 0.1
    return 1.0 + 0.3 * np.tanh(z)


def _solve(dm, mesh, cons, eps_fn, f_fn, lam2, device):
    """Assemble + solve the XDD Poisson system; return L2 error vs φ̂."""
    xq = gauss_points(mesh, dm.tables_by_p)
    eps_gp = {pv: eps_fn(xq[pv]) for pv in xq}
    rho_gp = {pv: f_fn(xq[pv]) for pv in xq}
    A, b = assemble_xdd_poisson(dm, eps_gp, rho_gp, lam2=lam2)
    coords = mesh.node_coords[cons.free_nodes]
    A, b = _apply_dirichlet(A, b, coords, _phi_star)
    x = splu(A.tocsc()).solve(b)
    u_all = np.asarray(cons.T @ x)
    return l2_error(dm, u_all, _phi_star)


# ──────────────────────────────────────────────────────────────────────────────
# G1 — MMS constant ε̂, p1 and p2 convergence orders
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("p,order_lo", [(1, 1.9), (2, 2.9)])
def test_mms_const_eps(p, order_lo, device):
    """G1: constant ε̂=1, λ²=1 — p1 → ≥2, p2 → ≥3 (least-squares order)."""
    lam2 = 1.0
    levels = (3, 4, 5)
    hs = [2.0 ** (-lv) for lv in levels]
    errs = []
    for lv in levels:
        dm, mesh, cons = _make_dm(lv, p, device)
        errs.append(_solve(dm, mesh, cons,
                           lambda x: np.ones(len(x)),
                           lambda x: _source_const_eps(x, lam2),
                           lam2, device))
    order = observed_order(hs, errs)
    print(f"G1 p{p}: errs {[f'{e:.2e}' for e in errs]} order {order:.2f}")
    assert order >= order_lo, (order, errs)


# ──────────────────────────────────────────────────────────────────────────────
# G2 — MMS variable ε̂(x), orders hold
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("p,order_lo", [(1, 1.9), (2, 2.9)])
def test_mms_var_eps(p, order_lo, device):
    """G2: ε̂(x)=1+0.3·tanh((x-0.5)/0.1), manufactured source — orders hold."""
    lam2 = 1.0
    levels = (3, 4, 5)
    hs = [2.0 ** (-lv) for lv in levels]
    errs = []
    for lv in levels:
        dm, mesh, cons = _make_dm(lv, p, device)
        errs.append(_solve(dm, mesh, cons,
                           _eps_var,
                           lambda x: _source_var_eps(x, lam2),
                           lam2, device))
    order = observed_order(hs, errs)
    print(f"G2 p{p}: errs {[f'{e:.2e}' for e in errs]} order {order:.2f}")
    assert order >= order_lo, (order, errs)


# ──────────────────────────────────────────────────────────────────────────────
# G3 — Source coupling: (p̂−n̂, w) term
# ──────────────────────────────────────────────────────────────────────────────

def test_source_coupling(device):
    """G3: exercises the rho_gp (p̂−n̂) branch of the load kernel with ε̂=1.

    The full manufactured source −λ²Δφ̂ is routed entirely through rho_gp
    (f_src_gp=None), so the (p̂−n̂, w) accumulation path is what drives the
    solve.  Checks the solved field against φ̂=sin(πx)sin(πy), exercising the
    coupling term B4 will later drive with carrier densities.
    """
    lam2, level, p = 1.0, 5, 1
    dm, mesh, cons = _make_dm(level, p, device)

    xq = gauss_points(mesh, dm.tables_by_p)
    # Route the full manufactured source through rho_gp; no extra body load.
    rho_gp = {pv: _source_const_eps(xq[pv], lam2) for pv in xq}
    eps_gp = {pv: np.ones(len(xq[pv])) for pv in xq}
    A, b = assemble_xdd_poisson(dm, eps_gp, rho_gp, lam2=lam2, f_src_gp=None)
    coords = mesh.node_coords[cons.free_nodes]
    A, b = _apply_dirichlet(A, b, coords, _phi_star)
    x = splu(A.tocsc()).solve(b)
    u_all = np.asarray(cons.T @ x)
    err = l2_error(dm, u_all, _phi_star)
    # Measured ~4e-4 at L5 p1; bound is 1e-3 (2.5× headroom).
    assert err < 1e-3, f"G3 coupling error too large: {err:.2e}"


# ──────────────────────────────────────────────────────────────────────────────
# G4 — λ² linear scaling (ratio test)
# ──────────────────────────────────────────────────────────────────────────────

def test_lambda2_scaling(device):
    """G4: assembled stiffness scales linearly with λ² (two values, ratio)."""
    level, p = 4, 1
    dm, mesh, cons = _make_dm(level, p, device)
    xq = gauss_points(mesh, dm.tables_by_p)
    eps_gp = {pv: np.ones(len(xq[pv])) for pv in xq}
    rho_gp = {pv: np.zeros(len(xq[pv])) for pv in xq}

    lam2_a, lam2_b = 1.0, 3.7
    A_a, _ = assemble_xdd_poisson(dm, eps_gp, rho_gp, lam2=lam2_a)
    A_b, _ = assemble_xdd_poisson(dm, eps_gp, rho_gp, lam2=lam2_b)

    # A_b should be exactly (lam2_b / lam2_a) * A_a in every nonzero entry
    ratio = lam2_b / lam2_a
    diff = (A_b - ratio * A_a)
    rel = abs(diff).max() / (abs(A_a).max() * ratio)
    assert rel < 1e-12, f"G4 λ² scaling broken: rel={rel:.2e}"


# ──────────────────────────────────────────────────────────────────────────────
# G5 — XDDParams.scales().lambda2 wiring
# ──────────────────────────────────────────────────────────────────────────────

def test_xdd_params_lambda2_wiring(device):
    """G5: assemble_xdd_poisson accepts XDDParams and uses scales().lambda2."""
    level, p = 4, 1
    dm, mesh, cons = _make_dm(level, p, device)
    xq = gauss_points(mesh, dm.tables_by_p)
    eps_gp = {pv: np.ones(len(xq[pv])) for pv in xq}
    rho_gp = {pv: np.zeros(len(xq[pv])) for pv in xq}

    params = XDDParams()
    lam2 = params.scales().lambda2
    A_params, _ = assemble_xdd_poisson(dm, eps_gp, rho_gp, params=params)
    A_direct, _ = assemble_xdd_poisson(dm, eps_gp, rho_gp, lam2=lam2)

    diff = (A_params - A_direct)
    rel = abs(diff).max() / max(abs(A_direct).max(), 1e-30)
    assert rel < 1e-12, f"G5 params wiring mismatch: rel={rel:.2e}"
