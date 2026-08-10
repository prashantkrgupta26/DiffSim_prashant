# tests/test_chns_scaffold.py
import numpy as np
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
