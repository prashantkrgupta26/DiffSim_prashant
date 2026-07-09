"""M4 track (c) gates: Wodo evaporating film — Landau-frame + evaporation
consistency. The load-bearing invariant is the sign pairing (advection
+K theta/h, top flux +K phi_i): physical solute content h * Int(phi_i)
is conserved EXACTLY per step (machine precision), while the film
thins and the mapped fractions enrich."""
import numpy as np
import pytest

from diffsim.octree.build import build_uniform, Octree
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.wodo_film import WodoFilmStepper

pytestmark = pytest.mark.tier2


def test_wodo_film_evaporation(device):
    tree0 = build_uniform(5, dim=2)
    keep = tree0.centers()[:, 0] < 4 / 32
    tree = Octree(tree0.keys[keep], tree0.levels[keep], dim=2,
                  periodic=tree0.periodic)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    st = WodoFilmStepper(dm, chi=(1.0, 0.3, 0.3), N=(5.0, 5.0, 1.0),
                         M=(0.225, 0.0, 0.225), kappa=(2e-4, 2e-4),
                         k_e=1.0, dt=1e-3)
    rng = np.random.default_rng(3)
    st.set_initial(lambda x: 0.2 + 0.01 * rng.standard_normal(len(x)),
                   lambda x: 0.2 + 0.01 * rng.standard_normal(len(x)))

    width = mesh.node_coords[:, 0].max()

    def phi_int(vec):                     # Int phi dtheta (mapped mean)
        v, _ = st._gp(vec)
        m = 0.0
        for pv, b in dm.bins.items():
            h = mesh.tree.h()[mesh.bins[pv]]
            ne = len(mesh.conn_of[pv])
            wq = np.tile(dm.tables_by_p[pv].w, ne) \
                * np.repeat((h / 2) ** 2, b["nqp"])
            m += float((wq * v[pv]).sum())
        return m / width

    P1_0, P2_0 = phi_int(st.hist[0][0]), phi_int(st.hist[0][1])
    c1_0, c2_0 = st.h_curr * P1_0, st.h_curr * P2_0    # physical content
    reason = st.march(h_min=0.8, phis_stop=0.05, max_steps=200)
    P1, P2 = phi_int(st.x[0::4]), phi_int(st.x[2::4])
    c1, c2 = st.h_curr * P1, st.h_curr * P2
    print(f"wodo: reason={reason} h={st.h_curr:.3f} t={st.t:.4f} "
          f"Phi_p {P1_0:.4f}->{P1:.4f} content drift "
          f"({abs(c1 - c1_0) / c1_0:.1e},{abs(c2 - c2_0) / c2_0:.1e})")
    assert reason == "h_min"
    assert st.h_curr <= 0.8 < 1.0                      # film thinned
    assert abs(c1 - c1_0) / c1_0 < 1e-12               # solute conserved
    assert abs(c2 - c2_0) / c2_0 < 1e-12
    assert P1 > P1_0 * 1.15 and P2 > P2_0 * 1.15       # enrichment
    ps = 1.0 - st.x[0::4] - st.x[2::4]
    assert ps.min() > -0.05                            # simplex ~respected


# ---------------------------------------------------------------------
# M4 device-bound gates: use_device_assembly=True trajectory parity
# < 1e-11 vs the host COO+scipy path.
# ---------------------------------------------------------------------
def _march_fixed_dt(st, nsteps, dt):
    """March at FIXED dt via _attempt + manual commit (removes the
    Appendix-A heuristic from the parity comparison)."""
    xs = []
    for _ in range(nsteps):
        p1n, p2n = st.hist[0]
        K = max(st.k_e * st._top_phis_avg(p1n, p2n), 0.0)
        x, iters, ok = st._attempt(dt, K)
        assert ok, iters
        st.x = x
        st.hist = [(x[0::4].copy(), x[2::4].copy()), st.hist[0]]
        st.t += dt
        st.h_curr -= dt * K
        xs.append(x.copy())
    return xs


def test_wodo_device_parity_spinodal(device):
    """GATE (i): ternary-spinodal trajectory parity < 1e-11 over 10
    march steps (tests/test_ternary_ch.py's config: chi=(6,.8,.8),
    M=(1,-.2,1), kappa=8e-4, dt=0.005, BDF1). With k_e=0 and
    N=(1,1,1) the film stepper reduces EXACTLY to TernaryCHStepper's
    equations. TEST-DESIGN NOTE (measured): at fixed dt=0.005 the
    quench-onset attempt does not converge in 50 Newton iterations
    (Newton wanders chaotically -> any two linear solvers land in
    different wells: O(1) 'parity' regardless of correctness). The
    PRODUCTION path — march() + the Appendix-A reject/retry — never
    commits a non-converged attempt, so parity through march() is the
    well-posed gate; both paths must also take the IDENTICAL
    accept/reject ladder. FLAKE NOTE (measured 2026-07-09): cuDSS is not
    run-to-run bitwise reproducible, and the quench-onset attempt sits on
    the 50-iteration convergence knife edge, so the ladders occasionally
    differ by one reject (1/6 standalone, 2/2 under full-suite GPU load)
    with per-step parity a clean 1.3-1.9e-13 whenever they match. A
    ladder mismatch triggers ONE full retry of both paths: a real parity
    break fails consistently; the cuDSS coin flip does not."""
    def run(dev_asm):
        tree = build_uniform(5, dim=2)
        mesh = build_mesh(tree, p=1)
        cons = build_constraints(mesh)
        dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2),
                                  device)
        st = WodoFilmStepper(dm, chi=(6.0, 0.8, 0.8), N=(1.0, 1.0, 1.0),
                             M=(1.0, -0.2, 1.0), kappa=(8e-4, 8e-4),
                             k_e=0.0, dt=0.005,
                             use_device_assembly=dev_asm)
        rng = np.random.default_rng(4)
        st.set_initial(
            lambda x: 0.35 + 0.02 * rng.standard_normal(len(x)),
            lambda x: 0.35 + 0.02 * rng.standard_normal(len(x)))
        rec = []
        st.march(h_min=0.0, phis_stop=-1.0, max_steps=10,
                 callback=lambda s, K, dt, it:
                 rec.append((dt, it, s.x.copy())))
        return rec, st.n_reject

    for attempt in range(2):
        rec_h, rej_h = run(False)
        rec_d, rej_d = run(True)
        if rej_h == rej_d and len(rec_h) == len(rec_d):
            break
    assert rej_h == rej_d and len(rec_h) == len(rec_d)
    assert all(a[:2] == b[:2] for a, b in zip(rec_h, rec_d)), \
        "accept/reject or iteration ladder diverged"
    errs = [np.abs(a[2] - b[2]).max() / max(np.abs(a[2]).max(), 1e-30)
            for a, b in zip(rec_h, rec_d)]
    print(f"wodo spinodal parity: {len(rec_h)} accepted steps "
          f"({rej_h} rejects, identical ladders), step-wise rel err "
          f"max {max(errs):.2e}")
    assert max(errs) < 1e-11, errs


def test_wodo_device_parity_film(device):
    """Device-bound parity on the ACTUAL film physics (evaporation
    advection + top-face flux + var-mobility + b-regularizer — the
    fig67 gate configuration sans noise): 10 steps < 1e-11, and the
    conservation identity holds on the device path."""
    tree0 = build_uniform(5, dim=2)
    keep = tree0.centers()[:, 0] < 4 / 32
    tree = Octree(tree0.keys[keep], tree0.levels[keep], dim=2,
                  periodic=tree0.periodic)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)

    def run(dev_asm):
        dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2),
                                  device)
        st = WodoFilmStepper(dm, chi=(1.0, 0.3, 0.3), N=(5.0, 5.0, 1.0),
                             M=(0.225, 0.0, 0.225), kappa=(2e-4, 2e-4),
                             k_e=1.0, dt=1e-3, var_mob=True, b_reg=1e-3,
                             use_device_assembly=dev_asm)
        rng = np.random.default_rng(3)
        st.set_initial(
            lambda x: 0.2 + 0.01 * rng.standard_normal(len(x)),
            lambda x: 0.2 + 0.01 * rng.standard_normal(len(x)))
        return st, _march_fixed_dt(st, 10, 1e-3)

    st_h, xs_h = run(False)
    st_d, xs_d = run(True)
    errs = [np.abs(a - b).max() / max(np.abs(a).max(), 1e-30)
            for a, b in zip(xs_h, xs_d)]
    print(f"wodo film parity: step-wise rel err max {max(errs):.2e}; "
          f"h_curr {st_h.h_curr:.6f} vs {st_d.h_curr:.6f}")
    assert max(errs) < 1e-11, errs
    assert abs(st_h.h_curr - st_d.h_curr) < 1e-13
