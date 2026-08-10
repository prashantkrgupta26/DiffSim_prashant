# tests/test_chns_scaffold.py
import numpy as np
import pytest
from diffsim.physics.chns import mix_props, capillary_gp, tau_m_gp


def test_mix_props_pure_phases():
    rho, eta, nc = mix_props(np.array([1.0, -1.0]), 1000.0, 1.0, 100.0, 0.1)
    assert np.allclose(rho, [1000.0, 1.0]) and np.allclose(eta, [100.0, 0.1])
    assert nc == 0


def test_mix_props_clamps_overshoot():
    # phi = -1.2 overshoots: raw rho = -0.1*999.5+500.5 < 1.0 -> clamped to floor
    rho, eta, nc = mix_props(np.array([-1.2]), 1000.0, 1.0, 100.0, 0.1)
    assert rho[0] >= 1e-3 * 1.0 and nc == 1


def test_capillary_zero_for_uniform_phi():
    f = capillary_gp(np.array([0.7]), np.zeros((1, 2)), Cn=0.01, We=10.0)
    assert np.allclose(f, 0.0)


def test_tau_m_local_viscosity_matters():
    t_lo = tau_m_gp(np.zeros((1, 2)), np.ones(1), np.array([0.1]), 0.05, 1e-2, 35.0)
    t_hi = tau_m_gp(np.zeros((1, 2)), np.ones(1), np.array([100.0]), 0.05, 1e-2, 35.0)
    assert t_hi[0] < t_lo[0]  # stiffer viscosity -> smaller tau


def test_ch_advection_translates_blob():
    """A tanh phi blob translates under uniform prescribed velocity u=(0.5,0).
    High Pe (tiny Onsager mobility) so diffusion is negligible over t=0.2.
    Expected x-centroid shift ~0.1; tolerance 20% (diffuse smearing allowed).
    Solenoidal u assumed — convective form: r_phi_i += Na*(u.grad_phi_i)*dJxW.
    """
    import warp as wp
    wp.init()
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh
    from diffsim.physics.multiphase import MultiPhaseStepper
    from diffsim import default_device

    # --- mesh: level-5 2-D uniform, same as the smallest S0 gate ---
    level = 5
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    device = default_device()
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)

    # --- build adv_gp: [n_elem, nqp, 2] with u=(0.5, 0) everywhere ---
    pv = next(iter(dm.bins))            # single polynomial-degree bin
    b = dm.bins[pv]
    n_elem = len(mesh.conn_of[pv])
    nqp = b["nqp"]
    adv = np.zeros((n_elem, nqp, 2))
    adv[..., 0] = 0.5                  # uniform u_x = 0.5

    # --- chi_aa: M=1, n_sp=2 — minimal symmetric 2x2 ---
    chi_aa = np.zeros((2, 2))
    chi_aa[0, 1] = chi_aa[1, 0] = 2.0

    # --- tanh blob centred at (0.3, 0.5) ---
    xc = mesh.node_coords[cons.free_nodes]
    w = 0.08
    phi0 = 0.5 * (1.0 - np.tanh((np.sqrt((xc[:, 0] - 0.3)**2
                                          + (xc[:, 1] - 0.5)**2) - 0.12) / w))
    phi0 = np.clip(phi0, 1e-3, 1.0 - 1e-3)

    # --- stepper: M=1, K=0, tiny mobility (high Pe), small kappa ---
    st = MultiPhaseStepper(
        dm, M=1, K=0, chi_aa=chi_aa,
        onsager=[[1e-6]],          # tiny: diffusion negligible
        kappa=[1e-8],
        dt=2e-3,
        adv_gp=adv,
    )
    st.set_initial([lambda x: phi0])

    # --- x-centroid of the phi>0.3 region ---
    def centroid_x(stepper):
        phi_v = stepper.phi(0)
        mask = phi_v > 0.3
        if not mask.any():
            return float("nan")
        return float(np.average(xc[mask, 0], weights=phi_v[mask]))

    c0 = centroid_x(st)
    st.march(t_end=0.2)
    c1 = centroid_x(st)
    shift = c1 - c0
    print(f"[adv test] centroid shift: {shift:.4f} (expected ~0.10)")
    assert abs(shift - 0.1) < 0.02, (
        f"Blob did not translate correctly: shift={shift:.4f}, "
        f"expected 0.10 ± 0.02"
    )


# ---------------------------------------------------------------------------
# Task 3: per-GP variable rho/eta/body-force in linear NS assembly
# ---------------------------------------------------------------------------

def _make_cavity_dm(level=3):
    """Minimal mesh+DeviceMesh for a 2-D cavity test (copies test_ns_bricks
    harness: uniform p=1 grid, build_constraints, basis_tables)."""
    import warp as wp
    wp.init()
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh
    from diffsim import default_device

    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    device = default_device()
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    return dm, mesh, cons


def test_variable_viscosity_ns_stokes_layers():
    """Two-viscosity channel: lower half eta=1, upper half eta=10.
    Lid-driven Stokes (sigma=0, zero advection): velocity in the stiff
    (high-eta) layer must be smaller than in the soft (low-eta) layer
    at matched vertical depths from each respective edge.

    Harness copies the smallest cavity solve in test_ns_bricks.py's _solve().
    """
    from scipy.sparse.linalg import splu
    from diffsim.api.ns_bricks import assemble_linear_ns
    from diffsim.physics.poisson import gauss_points

    dm, mesh, cons = _make_cavity_dm(level=4)
    ndof = 3
    dim = 2

    xq = gauss_points(mesh, dm.tables_by_p)

    # Zero advection (Stokes), zero body force except lid BC
    aq_by_bin = {pv: np.zeros((len(xq[pv]), dim)) for pv in xq}
    dq_by_bin = {pv: np.zeros(len(xq[pv])) for pv in xq}
    fq_by_bin = {pv: np.zeros((len(xq[pv]), dim)) for pv in xq}

    # Per-bin per-GP viscosity arrays: eta=1 for y<0.5, eta=10 for y>=0.5
    nu_q_by_bin = {}
    rho_q_by_bin = {}
    for pv in xq:
        y_coords = xq[pv][:, 1]  # [ne*nqp]
        nu_q = np.where(y_coords < 0.5, 1.0, 10.0)
        nu_q_by_bin[pv] = nu_q.reshape(len(dm.bins[pv]["eids"]),
                                        dm.bins[pv]["nqp"])
        rho_q_by_bin[pv] = np.ones_like(nu_q_by_bin[pv])

    # Constant reference nu (unused when per-GP path active; pass nu=1.0)
    A, b = assemble_linear_ns(dm, aq_by_bin, dq_by_bin, fq_by_bin, nu=1.0,
                               sigma=0.0,
                               nu_q_by_bin=nu_q_by_bin,
                               rho_q_by_bin=rho_q_by_bin)

    # Apply lid-driven BC: u_x=1 on the top wall (y=1), no-slip elsewhere
    free_coords = mesh.node_coords[cons.free_nodes]
    boundary = mesh.boundary_nodes[cons.free_nodes]
    n_free = cons.T.shape[1]

    rows = []
    rhs_vals = {}
    for i in np.where(boundary)[0]:
        rows.append(i * ndof)
        rows.append(i * ndof + 1)
        # lid: top wall (y close to 1.0) -> u_x = 1, else 0
        if abs(free_coords[i, 1] - 1.0) < 1e-9:
            rhs_vals[i * ndof] = 1.0
    rows.append(0 * ndof + 2)  # pressure pin

    A = A.tolil()
    for r in rows:
        A.rows[r] = [r]
        A.data[r] = [1.0]
    A = A.tocsr()
    b[rows] = 0.0
    for dof_idx, val in rhs_vals.items():
        b[dof_idx] = val

    x = splu(A.tocsc()).solve(b).reshape(n_free, ndof)
    u_x = x[:, 0]

    # Also solve with constant nu=1 for comparison
    A_ref, b_ref = assemble_linear_ns(dm, aq_by_bin, dq_by_bin, fq_by_bin,
                                       nu=1.0, sigma=0.0)
    A_ref = A_ref.tolil()
    for r in rows:
        A_ref.rows[r] = [r]
        A_ref.data[r] = [1.0]
    A_ref = A_ref.tocsr()
    b_ref[rows] = 0.0
    for dof_idx, val in rhs_vals.items():
        b_ref[dof_idx] = val
    x_ref = splu(A_ref.tocsc()).solve(b_ref).reshape(n_free, ndof)
    u_x_ref = x_ref[:, 0]

    # Physical check 1: the solutions must differ (variable viscosity changes physics)
    max_diff = float(np.abs(u_x - u_x_ref).max())
    assert max_diff > 1e-4, (
        f"Variable-viscosity solution is identical to constant-nu: max_diff={max_diff:.3e}"
    )

    # Physical check 2: lower soft layer (eta=1, y~0.25) has LARGER |u_x| than
    # upper stiff layer (eta=10, y~0.75) in 2-D lid-driven cavity.
    # In the low-viscosity region the flow circulates more freely (measured).
    lower_mask = (free_coords[:, 1] > 0.15) & (free_coords[:, 1] < 0.35)
    upper_mask = (free_coords[:, 1] > 0.65) & (free_coords[:, 1] < 0.85)
    assert lower_mask.any() and upper_mask.any(), "No nodes in expected regions"

    u_lower = float(np.mean(np.abs(u_x[lower_mask])))
    u_upper = float(np.mean(np.abs(u_x[upper_mask])))

    # Soft (low eta=1) lower layer circulates faster than stiff (eta=10) upper layer
    assert u_lower > u_upper, (
        f"Variable-viscosity physics wrong: u_lower={u_lower:.4f} (eta=1) should "
        f"exceed u_upper={u_upper:.4f} (eta=10); stiff layer damps circulation."
    )
    # Also sanity: non-zero flow and finite
    assert np.isfinite(u_x).all()
    assert np.abs(u_x).max() > 1e-6, "Solution is essentially zero"


def test_variable_coefficient_none_path_identical():
    """Assemble the same small cavity system twice:
      1) all-None kwargs (original call signature),
      2) uniform nu_q/rho_q arrays equal to the constant nu.
    Assert the resulting sparse matrices and RHS vectors are allclose
    (rtol=1e-12). This is the bit-identical guarantee made testable.
    """
    from diffsim.api.ns_bricks import assemble_linear_ns
    from diffsim.physics.poisson import gauss_points

    dm, mesh, cons = _make_cavity_dm(level=3)
    ndof = 3
    dim = 2
    nu_const = 0.1

    xq = gauss_points(mesh, dm.tables_by_p)
    aq_by_bin = {pv: np.zeros((len(xq[pv]), dim)) for pv in xq}
    dq_by_bin = {pv: np.zeros(len(xq[pv])) for pv in xq}
    fq_by_bin = {pv: np.zeros((len(xq[pv]), dim)) for pv in xq}

    # Reference: all-None (original signature)
    A_ref, b_ref = assemble_linear_ns(dm, aq_by_bin, dq_by_bin, fq_by_bin,
                                       nu=nu_const, sigma=0.5)

    # Variable path: uniform arrays equal to the constant
    nu_q_by_bin = {}
    rho_q_by_bin = {}
    for pv in xq:
        n_elem = len(dm.bins[pv]["eids"])
        nqp = dm.bins[pv]["nqp"]
        nu_q_by_bin[pv] = np.full((n_elem, nqp), nu_const)
        rho_q_by_bin[pv] = np.ones((n_elem, nqp))

    A_var, b_var = assemble_linear_ns(dm, aq_by_bin, dq_by_bin, fq_by_bin,
                                       nu=nu_const, sigma=0.5,
                                       nu_q_by_bin=nu_q_by_bin,
                                       rho_q_by_bin=rho_q_by_bin)

    # Matrices must be allclose (rtol 1e-12)
    diff = (A_ref - A_var)
    nnz_ref = A_ref.nnz
    assert diff.nnz == 0 or np.abs(diff.data).max() < 1e-12 * np.abs(A_ref.data).max(), (
        f"Matrix mismatch: max|diff|={np.abs(diff.data).max():.3e}, "
        f"max|A_ref|={np.abs(A_ref.data).max():.3e}"
    )
    assert np.allclose(b_ref, b_var, rtol=1e-12, atol=0), (
        f"RHS mismatch: max|diff|={np.abs(b_ref - b_var).max():.3e}"
    )
