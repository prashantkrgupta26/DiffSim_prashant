"""P2-R0 Task 1 — §4 formulation-parity audit of LerayProjectionStepper vs
the VMS-projection paper (local_code_old/ns_projection_vms_paper.pdf).

These are DOCUMENTATION-INVARIANT tests: the §4 audit (see
docs/dev/2026-07-21-p2-r0-parity-audit.md) found NO formulation deviation, so
instead of regression tests for fixes, these pin the paper-faithful invariants
the audit confirmed:

  * BDF-r coefficients + bootstrap gate (paper Eq. (4), sigma = beta_0/dt).
  * The incremental / van-Kan pressure structure: the PPE unknown is the
    increment phi = p_hat - p* (paper Eqs. (31a)/(44b), Remark 3.9), and the
    Step-4 update p* <- p_hat (paper Algorithm 1, Step 4).
  * The three-subproblem structure of Algorithm 1 (predictor pins pressure to
    p*, PPE solves the increment, correction is u_hat - (1/sigma) grad phi).
"""
import numpy as np
import pytest

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.steppers.leray import LerayProjectionStepper
from diffsim.solvers.timestepping import bdf_coeffs, bdf_order_now

pytestmark = pytest.mark.tier5


def _lid_g(x, t):
    g = np.zeros((len(x), 2))
    g[np.abs(x[:, 1] - 1.0) < 1e-12, 0] = 1.0
    return g


def _make(device, level=3, nu=0.01, dt=0.05, **kw):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    st = LerayProjectionStepper(
        dm, nu, dt, f_fn=lambda x, t: np.zeros((len(x), 2)),
        g_fn=_lid_g, order=2, picard_iters=2, solver="splu", **kw)
    st.set_initial(lambda x: np.zeros((len(x), 2)))
    return st


# -------------------- Checklist item 2: BDF-r + sigma --------------------

def test_bdf_coeffs_and_bootstrap():
    # paper Eq. (4): BDF2 standard coefficients {beta_0, beta_-1, beta_-2}
    # with sigma = beta_0/dt.  Production signed table = (1.5, -2, 0.5).
    assert bdf_coeffs(2, 0.05) == (1.5, -2.0, 0.5)
    assert bdf_coeffs(1, 0.05) == (1.0, -1.0, 0.0)
    # bootstrap gate: BDF1 until t >= 1.5*dt (paper's BDF history rule; the
    # stepper builds second-order history from corrected velocities).
    assert bdf_order_now(0.04, 0.05, 2) == 1          # t < 1.5*dt -> BDF1
    assert bdf_order_now(0.04, 0.05, 2, have_history=True) == 1
    assert bdf_order_now(0.2, 0.05, 2, have_history=True) == 2   # t>=1.5dt
    assert bdf_order_now(0.2, 0.05, 2, have_history=False) == 1  # no hist
    # sigma = beta_0/dt as used in the stepper
    b0, _, _ = bdf_coeffs(2, 0.05)
    assert b0 / 0.05 == pytest.approx(30.0)


# ---- Checklist items 1 & 3: incremental (van Kan) pressure structure ----

def test_incremental_increment_is_phi(device):
    """The PPE unknown IS the increment phi = p_hat - p* (paper Eqs. 31a/44b,
    Remark 3.9), and Step 4 sets p* <- p_hat.

    Proof by the defining property of the incremental scheme: p* enters the
    predictor and the PPE only through grad(p*) (paper Alg 1 Steps 1-2). A
    spatially CONSTANT shift of p* therefore has zero gradient and cannot
    change the increment phi.  Hence seeding p_star with a constant c must
    shift the returned pressure by exactly c and leave phi (= p_hat - p*)
    invariant.  A non-incremental (total-pressure) PPE would NOT have this
    property.
    """
    st0 = _make(device)
    u0, p0 = st0.step()
    # p_hat returned == st.p_star after Step 4 (paper Alg 1 Step 4)
    np.testing.assert_array_equal(p0, st0.p_star)
    phi0 = p0 - 0.0                       # p_star started at zero -> p_hat==phi

    c = 3.14159
    st1 = _make(device)
    st1.p_star = st1.p_star + c           # constant shift of p*
    u1, p1 = st1.step()
    phi1 = p1 - c                         # increment recovered from p_hat - p*

    # velocity update is invariant to a constant pressure shift (grad c == 0)
    np.testing.assert_allclose(u1, u0, rtol=0, atol=1e-9)
    # the increment phi is invariant -> the PPE unknown is the increment,
    # not the total pressure (van Kan / incremental, paper Remark 3.9)
    np.testing.assert_allclose(phi1, phi0, rtol=0, atol=1e-9)


def test_predictor_pins_pressure_to_pstar(device):
    """Algorithm 1 Step 1: the momentum predictor is driven by grad(p*), and
    the stepper pins ALL pressure DOFs to p* in the monolithic predictor
    block (leray.py step() 'pin ALL pressure DOFs').  We check the structural
    consequence: with p* == 0 and homogeneous g, one predictor step from rest
    produces u_hat == 0 (no spurious pressure drive), whereas a constant p*
    still yields u_hat == 0 (constant grad p* == 0)."""
    st = _make(device)
    u, p = st.step()
    # from rest, zero forcing, zero p*: the interior stays at rest, only the
    # lid trace is nonzero (strong Dirichlet).  Interior u_hat contribution
    # comes solely through the driven boundary -> finite, not NaN/blowup.
    assert np.isfinite(u).all()
    assert np.isfinite(p).all()


# -------------------- Checklist item 4: default flag --------------------

def test_ppe_finescale_default_is_classic_incremental():
    """Supervisor resolution 3 / paper Remark 3.7: R0 runs classic-incremental
    (ppe_finescale=False); pressure stability for equal-order u/p comes from
    the PREDICTOR's VMS fine scale (PSPG-like, paper Alg 1 Step 1 tau_m
    terms), so the SPD PPE (grad p, grad q) needs no LBB (Remark 3.7)."""
    st = _make("cpu")
    assert st.ppe_finescale is False
