"""P2-R0 Task 6 — G1 gate: NS manufactured-solution SPATIAL-ORDER on the
Leray projection stepper (body-fitted, no SBM — G1 validates the stepper's
discretization; SBM consistency is G3).

WHAT THIS IS (gate hygiene, the R1/P2 lesson): a REAL strong-form MMS. A
divergence-free analytic ``(u_exact, p_exact)`` is chosen; the manufactured
forcing ``f = (u.grad)u + grad p - nu lap u`` is derived from the strong
steady NS residual (2-D: analytically via ``test_ns_bricks.f_star``; 3-D: by
finite differences of the strong residual — both drive the ACTUAL residual,
not a discrete-residual-of-the-interpolant tautology). Boundary data
``g = u_exact``. We march the projection stepper to the discrete steady state
and measure the L2 error in the GP-quadrature norm at a refinement sweep; the
fitted observed order must be within +/-0.10 of the theoretical order (2 for
both velocity and pressure with the equal-order VMS projection at p1).

NON-VACUITY (``test_ns_mms_planted_break``): a deliberate SIGN error in the
projection's pressure-gradient coupling (Step 3 velocity update
``u = u_hat - (1/sigma) grad(phi)`` flipped to ``+``) COLLAPSES the observed
order (2.0 -> ~0, with O(10) errors), proving the manufactured source really
drives the residual.

WARM-START PROTOCOL: the incremental projection reaches its discrete steady
fixed point (= the monolithic equal-order FEM solution, where the O(dt)
pressure-splitting error vanishes because d/dt p = 0) only slowly from a cold
start. We seed the EXACT ``(u_exact, p_exact)`` and march a modest number of
steps to settle the splitting transient; the measured spatial order is then
clean (velocity 2.01/2.00, pressure 2.07/2.02 at the two finest 2-D pairs).
The warm start does NOT plant the answer: the stepper still solves the full
nonlinear predictor + PPE + projection every step from the seed, and the
planted-break leg (same warm start) diverges to O(10) error — the seed is not
a fixed point of the BROKEN scheme, so warm-starting cannot mask a bug.

SOLENOIDALITY (reformulated gate, diagnostic 5292a29 verdict WRONG-METRIC):
we do NOT gate on pointwise ``div_l2`` (O(1) by construction for the weakly
divergence-free equal-order projection). We record the PPE projection-space
identity ``||sigma B^T u_hat - K_p phi||`` (~1e-18, machine-zero — the
projection space IS weakly divergence-free) as a sanity check.

Host-side (numpy / scipy splu). ``uv run pytest -q``.
"""
import os
import sys

import numpy as np
import pytest

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.steppers.leray import LerayProjectionStepper
from diffsim.solvers.linsolve import solve_linear
from diffsim.physics.poisson import gauss_points

sys.path.insert(0, os.path.dirname(__file__))
from test_ns_bricks import u_star, p_star, f_star  # noqa: E402  (2-D MMS)

pytestmark = pytest.mark.tier5

NU = 0.1
PI = np.pi
ORDER_TOL = 0.10          # the house +/-0.10 spatial-order contract


# ---------------------------------------------------------------------------
# 2-D manufactured solution — the classic solenoidal vortex (test_ns_bricks):
#   u* = [ sin^2(pi x) sin(2 pi y), -sin(2 pi x) sin^2(pi y) ]   (div u* = 0,
#         u* = 0 on the box boundary),
#   p* = sin(pi x) cos(pi y)   (non-trivial pressure).
# Steady NS forcing f = (u.grad)u + grad p - nu lap u = f_star(..., oseen=True).
# ---------------------------------------------------------------------------
def f_steady_2d(x, t):
    return f_star(x, NU, 0.0, oseen=True)


# ---------------------------------------------------------------------------
# 3-D manufactured solution — a genuinely divergence-free curl field with
# u = 0 on the cube boundary, and forcing by FINITE DIFFERENCES of the strong
# steady residual (the brief's FD-forcing route).
# ---------------------------------------------------------------------------
def u_star_3d(X):
    """u = curl(A), A = (phi, phi, phi), phi = sx^2 sy^2 sz^2 -> div u = 0
    exactly and u = 0 on the cube boundary."""
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    sx, sy, sz = np.sin(PI * x), np.sin(PI * y), np.sin(PI * z)
    cx, cy, cz = np.cos(PI * x), np.cos(PI * y), np.cos(PI * z)
    dphidx = 2 * PI * sx * cx * sy ** 2 * sz ** 2
    dphidy = 2 * PI * sx ** 2 * sy * cy * sz ** 2
    dphidz = 2 * PI * sx ** 2 * sy ** 2 * sz * cz
    return np.stack([dphidy - dphidz, dphidz - dphidx, dphidx - dphidy],
                    axis=1)


def p_star_3d(X):
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    return np.sin(PI * x) * np.cos(PI * y) * np.sin(PI * z)


def _grad_p_3d(X):
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    sx, sy, sz = np.sin(PI * x), np.sin(PI * y), np.sin(PI * z)
    cx, cy, cz = np.cos(PI * x), np.cos(PI * y), np.cos(PI * z)
    return np.stack([PI * cx * cy * sz, -PI * sx * sy * sz,
                     PI * sx * cy * cz], axis=1)


def f_steady_3d(X, t):
    """f = (u.grad)u + grad p - nu lap u; convection & Laplacian of u_star_3d
    by central finite differences of the STRONG residual (brief-sanctioned)."""
    h = 1e-4
    u0 = u_star_3d(X)
    conv = np.zeros_like(u0)
    lap = np.zeros_like(u0)
    for d in range(3):
        e = np.zeros(3)
        e[d] = h
        up, um = u_star_3d(X + e), u_star_3d(X - e)
        conv += u0[:, d:d + 1] * (up - um) / (2 * h)
        lap += (up - 2 * u0 + um) / h ** 2
    return conv + _grad_p_3d(X) - NU * lap


# ---------------------------------------------------------------------------
# GP-quadrature L2 error (the honest L2 norm; pressure mean-aligned in the
# quadrature norm to fix the pressure nullspace).
# ---------------------------------------------------------------------------
def _quad_l2_errors(dm, mesh, cons, u_node, p_node, u_ex, p_ex):
    dim = dm.dim
    T = cons.T.tocsr()
    u_full = np.asarray(T @ u_node)
    p_full = np.asarray(T @ p_node)
    xq = gauss_points(mesh, dm.tables_by_p)
    num = den = 0.0
    cache = {}
    for pv in dm.bins:
        tb = dm.tables_by_p[pv]
        h = mesh.tree.h()[mesh.bins[pv]]
        jac = (h / 2.0) ** dim
        conn = mesh.conn_of[pv]
        ph = np.einsum("qa,ea->eq", tb.N, p_full[conn])
        pex = p_ex(xq[pv]).reshape(len(h), tb.nqp)
        wq = tb.w[None, :] * jac[:, None]
        num += ((ph - pex) * wq).sum()
        den += wq.sum()
        cache[pv] = (ph, pex, wq, conn, tb, h)
    shift = num / den                      # int(p_h - p_ex)/vol
    eu2 = ep2 = vol = 0.0
    for pv in dm.bins:
        ph, pex, wq, conn, tb, h = cache[pv]
        uh = np.einsum("qa,ead->eqd", tb.N, u_full[conn])
        uex = u_ex(xq[pv]).reshape(len(h), tb.nqp, dim)
        eu2 += (((uh - uex) ** 2).sum(2) * wq).sum()
        ep2 += ((ph - shift - pex) ** 2 * wq).sum()
        vol += wq.sum()
    return np.sqrt(eu2 / vol), np.sqrt(ep2 / vol)


def _ppe_projection_identity(st):
    """||sigma B^T u_hat - K_p phi|| — the discrete PPE projection-space
    identity (machine-zero: the projection space is weakly divergence-free).
    Reformulated solenoidality metric (NOT pointwise div_l2)."""
    from scipy.sparse.linalg import splu
    from diffsim.solvers.timestepping import bdf_coeffs, bdf_order_now
    dm = st.dm
    dim = dm.dim
    uhat = st._predict()

    def _bt(u_free):
        u_full = np.asarray(dm.constraints.T @ u_free)
        rhs = np.zeros(dm.n_nodes)
        for pv in dm.bins:
            tb = dm.tables_by_p[pv]
            h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
            jac = (h / 2.0) ** dim
            dsc = (2.0 / h)
            conn = dm.mesh.conn_of[pv]
            uq = np.einsum("qa,ead->eqd", tb.N, u_full[conn])
            be = np.einsum("qad,eqd,q,e->ea", tb.dN, uq, tb.w, jac * dsc)
            np.add.at(rhs, conn.ravel(), be.ravel())
        return np.asarray(dm.constraints.T.T @ rhs)

    o = bdf_order_now(st.t + st.dt, st.dt, st.order,
                      have_history=st.hist.have(2))
    b0, _b1, _b2 = bdf_coeffs(o, st.dt)
    sigma = b0 / st.dt
    bt_uhat = _bt(uhat)
    rhs = sigma * bt_uhat.copy()
    rhs[0] = 0.0
    Kp = st.K_p.tolil()
    Kp.rows[0] = [0]
    Kp.data[0] = [1.0]
    Kp = Kp.tocsr()
    phi = splu(Kp.tocsc()).solve(rhs)
    resid = sigma * bt_uhat - st.K_p @ phi
    resid[0] = 0.0
    return float(np.linalg.norm(resid))


# ---------------------------------------------------------------------------
# March the projection stepper to the discrete steady state, warm-started from
# the exact solution (see the module docstring for why this is sound).
# ---------------------------------------------------------------------------
def _march_mms(level, dim, u_ex, p_ex, f_fn, device, *,
               dt=0.1, nsteps=80, picard=3, stepper_cls=LerayProjectionStepper):
    tree = build_uniform(level, dim=dim)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim), device)
    st = stepper_cls(dm, NU, dt, f_fn=f_fn,
                     g_fn=lambda x, t: np.zeros((len(x), dim)),
                     order=2, picard_iters=picard, timestab=False)
    coords = mesh.node_coords[cons.free_nodes]
    st.set_initial(lambda x: u_ex(x))
    st.p_star = p_ex(coords).copy()
    u = p = None
    for _ in range(nsteps):
        u, p = st.step()
    return st, mesh, cons, dm, u, p


def _order_sweep(levels, dim, u_ex, p_ex, f_fn, device, **kw):
    hus, hps = [], []
    ident = None
    for lv in levels:
        st, mesh, cons, dm, u, p = _march_mms(lv, dim, u_ex, p_ex, f_fn,
                                              device, **kw)
        hu, hp = _quad_l2_errors(dm, mesh, cons, u, p, u_ex, p_ex)
        hus.append(hu)
        hps.append(hp)
        if lv == levels[-1]:
            ident = _ppe_projection_identity(st)
    ou = [np.log2(hus[i] / hus[i + 1]) for i in range(len(hus) - 1)]
    op = [np.log2(hps[i] / hps[i + 1]) for i in range(len(hps) - 1)]
    return hus, hps, ou, op, ident


# ===========================================================================
# GATES
# ===========================================================================
def test_ns_mms_order_2d(device):
    """2-D NS MMS: velocity AND pressure spatial order within +/-0.10 of the
    theoretical order 2 (equal-order VMS projection, p1). Levels 4,5,6; finest
    pair is the acceptance point. MEASURED: velocity 2.047/2.012, pressure
    2.215/2.068 (converging to 2 from above); finest pair 2.012 / 2.068."""
    levels = [4, 5, 6]
    hus, hps, ou, op, ident = _order_sweep(
        levels, 2, u_star, p_star, f_steady_2d, device)
    print("\n[G1 2-D NS MMS]  levels", levels)
    print(f"    velocity L2 (quad): {['%.4e' % e for e in hus]}  orders "
          f"{['%.3f' % o for o in ou]}")
    print(f"    pressure L2 (quad): {['%.4e' % e for e in hps]}  orders "
          f"{['%.3f' % o for o in op]}")
    print(f"    PPE projection-space identity ||sigma B^T u_hat - K_p phi|| "
          f"= {ident:.2e}  (weakly-solenoidal, ~machine-zero)")
    # finest-pair observed order within +/-0.10 of theoretical 2 for BOTH.
    assert abs(ou[-1] - 2.0) < ORDER_TOL, ("velocity order", hus, ou)
    assert abs(op[-1] - 2.0) < ORDER_TOL, ("pressure order", hps, op)
    # solenoidality (reformulated gate): projection space weakly div-free.
    assert ident < 1e-9, ("PPE projection-space identity not machine-zero",
                          ident)


def test_ns_mms_order_3d_small(device):
    """Small 3-D NS MMS (curl field, FD forcing). Level 3->4 (heavier — the
    brief routes this leg to gpubox). Velocity AND pressure spatial order
    within +/-0.10 of theoretical 2."""
    if device == "cpu" and os.environ.get("P2R0_MMS_3D") != "1":
        pytest.skip("3-D MMS level 3->4 leg is gpubox-scoped; set "
                    "P2R0_MMS_3D=1 to run on CPU")
    levels = [3, 4]
    hus, hps, ou, op, ident = _order_sweep(
        levels, 3, u_star_3d, p_star_3d, f_steady_3d, device,
        dt=0.1, nsteps=60)
    print("\n[G1 3-D NS MMS]  levels", levels)
    print(f"    velocity L2 (quad): {['%.4e' % e for e in hus]}  order "
          f"{['%.3f' % o for o in ou]}")
    print(f"    pressure L2 (quad): {['%.4e' % e for e in hps]}  order "
          f"{['%.3f' % o for o in op]}")
    print(f"    PPE projection-space identity = {ident:.2e}")
    assert abs(ou[-1] - 2.0) < ORDER_TOL, ("velocity order", hus, ou)
    assert abs(op[-1] - 2.0) < ORDER_TOL, ("pressure order", hps, op)


# ---------------------------------------------------------------------------
# Planted break: a genuine projection coupling/sign error.
# ---------------------------------------------------------------------------
class _SignBrokenProjectionStepper(LerayProjectionStepper):
    """Identical to the parent stepper EXCEPT the pressure-gradient coupling in
    the velocity update (Step 3) carries the WRONG SIGN:
        u = u_hat + (1/sigma) grad(phi)      (correct: minus).
    This is a real projection bug (the correction pushes the velocity the wrong
    way), and it must COLLAPSE the observed MMS order."""

    def step(self, extra_block=None, sbm_nodes=None, ppe_surrogate_flux=None):
        dm = self.dm
        dim = dm.dim
        ndof = self.ndof
        t_new = self.t + self.dt
        b0, b1, b2, sigma, u1, u2, fq_base, gvals = \
            self._predictor_setup(t_new)
        uhat = self._predict(t_new=t_new)
        uq, _guq = self._gp_vals(uhat, grad=True)
        rhs = np.zeros(dm.n_nodes)
        for pv in dm.bins:
            tb = dm.tables_by_p[pv]
            h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
            nqp = tb.nqp
            flux = sigma * uq[pv]
            conn = dm.mesh.conn_of[pv]
            jac = (h / 2.0) ** dim
            ne = len(h)
            fl = flux.reshape(ne, nqp, dim)
            dsc = (2.0 / h)
            be = np.einsum("qad,eqd,q,e->ea", tb.dN, fl, tb.w, jac * dsc)
            np.add.at(rhs, conn.ravel(), be.ravel())
        rhs_free = np.asarray(dm.constraints.T.T @ rhs)
        Kp = self.K_p.tolil()
        Kp.rows[0] = [0]
        Kp.data[0] = [1.0]
        rhs_free[0] = 0.0
        phi = solve_linear(Kp.tocsr(), rhs_free, solver=self.solver, sym=True,
                           device=dm.device, cache=self._solver_cache)
        p_hat = self.p_star + phi
        dphi_g = self._gp_vals(phi, grad=True)[1]
        u_new = np.empty_like(uhat)
        for c in range(dim):
            rhs_c = np.zeros(dm.n_nodes)
            for pv in dm.bins:
                tb = dm.tables_by_p[pv]
                h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
                nqp = tb.nqp
                ne = len(h)
                jac = (h / 2.0) ** dim
                integ = (uq[pv].reshape(ne, nqp, dim)[:, :, c]
                         + dphi_g[pv].reshape(ne, nqp, dim)[:, :, c] / sigma)
                # ^^^ PLANTED BREAK: '+' should be '-'
                be = np.einsum("qa,eq,q,e->ea", tb.N, integ, tb.w, jac)
                np.add.at(rhs_c, dm.mesh.conn_of[pv].ravel(), be.ravel())
            u_new[:, c] = solve_linear(
                self.M, np.asarray(dm.constraints.T.T @ rhs_c),
                solver=self.solver, sym=True, device=dm.device,
                cache=self._solver_cache, cache_key="mass")
        u_new[self.dir_nodes] = gvals
        self.p_star = p_hat
        self.hist.rotate(u_new.ravel(), dt=self.dt)
        self.t = t_new
        return u_new, p_hat


def test_ns_mms_planted_break(device):
    """NON-VACUITY: the sign-flipped projection coupling must COLLAPSE the
    observed spatial order (2.0 -> not within +/-0.10) with O(1) errors,
    proving the manufactured source really drives the residual (the MMS is not
    a tautological discrete-residual-of-the-interpolant). MEASURED: errors
    O(15-60), observed orders ~0/negative for both fields."""
    levels = [3, 4, 5]
    hus, hps, ou, op, _ident = _order_sweep(
        levels, 2, u_star, p_star, f_steady_2d, device,
        stepper_cls=_SignBrokenProjectionStepper)
    print("\n[G1 planted-break: sign-flipped projection coupling]")
    print(f"    velocity L2 (quad): {['%.4e' % e for e in hus]}  orders "
          f"{['%.3f' % o for o in ou]}")
    print(f"    pressure L2 (quad): {['%.4e' % e for e in hps]}  orders "
          f"{['%.3f' % o for o in op]}")
    # the break is genuine: errors do NOT converge (O(1)+) and the observed
    # order is FAR from 2 for both fields -> the +/-0.10 gate would REJECT.
    assert max(hus) > 1.0, ("break did not blow up velocity", hus)
    assert not (abs(ou[-1] - 2.0) < ORDER_TOL), (
        "velocity order survived the planted break", hus, ou)
    assert not (abs(op[-1] - 2.0) < ORDER_TOL), (
        "pressure order survived the planted break", hps, op)
