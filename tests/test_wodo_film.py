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
    """March at FIXED dt via _attempt + the canonical commit (removes
    the Appendix-A heuristic from the parity comparison; _commit keeps
    the BDF2 two-level history consistent — retrofit G2)."""
    xs = []
    for _ in range(nsteps):
        p1n, p2n = st.hist[0]
        K = max(st.k_e * st._top_phis_avg(p1n, p2n), 0.0)
        x, iters, ok = st._attempt(dt, K)
        assert ok, iters
        st._commit(x, dt, K)
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


# ---------------------------------------------------------------------
# Retrofit G1 (2026-07-13): basis-generic film — quadrature-built top-face
# mass (multiphase A4a pattern), p == 1 assert lifted.  Measured at the
# retrofit: p1 trajectory parity vs the pre-change hardcoded-P1 code
# 5.9e-16 over 10 steps (face-mass ulp noise only), h_curr bit-identical.
# Retrofit G2 (2026-07-13): tstep="bdf2" — VARIABLE-COEFFICIENT BDF2
# (multiphase A4b sigma/hist rewiring, zero kernel change).  Measured at
# the retrofit: bdf1 default trajectory BIT-IDENTICAL to pre-G2 (0.0 dev).
# ---------------------------------------------------------------------
def _strip_dm(p, level, width_cells, device):
    tree0 = build_uniform(level, dim=2)
    keep = tree0.centers()[:, 0] < width_cells / 2 ** level
    tree = Octree(tree0.keys[keep], tree0.levels[keep], dim=2,
                  periodic=tree0.periodic)
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    return DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2),
                                device)


def _mk_smooth(dm, k_e, tstep, dt=1e-3):
    st = WodoFilmStepper(dm, chi=(1.0, 0.3, 0.3), N=(5.0, 5.0, 1.0),
                         M=(0.225, 0.0, 0.225), kappa=(2e-4, 2e-4),
                         k_e=k_e, dt=dt, tstep=tstep)
    st.set_initial(lambda x: 0.2 + 0.02 * np.cos(np.pi * x[:, 1]),
                   lambda x: 0.2 - 0.02 * np.cos(np.pi * x[:, 1]))
    return st


def test_wodo_bdf2_temporal_order(device):
    """G2 gate: BDF2 temporal order ~2 at k_e = 0 (h frozen — the
    explicit h-update is O(dt) by construction, a recorded limit), BDF1
    control ~1, and the BDF1-vs-BDF2 consistency e_bdf2(dt) <
    e_bdf1(dt/2).  Measured at the retrofit: BDF2 2.01/2.13
    (errs 5.4e-8/1.3e-8/3.1e-9), BDF1 0.99/1.00, consistency 158x.
    Locks follow the A4b pattern (orders > 1.7).  Cross cell: p2 x
    BDF2 (measured 1.98)."""
    T = 0.064

    def run(tstep, dt, p=1):
        st = _mk_smooth(_strip_dm(p, 4, 2, device), 0.0, tstep, dt=dt)
        _march_fixed_dt(st, round(T / dt), dt)
        assert abs(st.t - T) < 1e-12
        return st.x.copy()

    ref = run("bdf2", 2.5e-4)
    e2 = [np.abs(run("bdf2", dt) - ref).max()
          for dt in (4e-3, 2e-3, 1e-3)]
    e1 = [np.abs(run("bdf1", dt) - ref).max()
          for dt in (4e-3, 2e-3, 1e-3)]
    o2 = [np.log2(e2[i] / e2[i + 1]) for i in range(2)]
    o1 = [np.log2(e1[i] / e1[i + 1]) for i in range(2)]
    print(f"wodo BDF2 errs {['%.2e' % e for e in e2]} orders "
          f"{['%.2f' % o for o in o2]}; BDF1 orders "
          f"{['%.2f' % o for o in o1]}")
    assert min(o2) > 1.7, (e2, o2)
    assert max(o1) < 1.3, (e1, o1)              # mechanism live
    assert e2[1] < e1[2], (e2[1], e1[2])        # BDF2(dt) < BDF1(dt/2)
    ref_p2 = run("bdf2", 2.5e-4, p=2)
    e2p = [np.abs(run("bdf2", dt, p=2) - ref_p2).max()
           for dt in (4e-3, 2e-3)]
    o2p = np.log2(e2p[0] / e2p[1])
    print(f"wodo p2 x BDF2 order {o2p:.2f}")
    assert o2p > 1.7, (e2p, o2p)


def test_wodo_bdf2_adaptive_dt_and_rejects(device):
    """G2 gate (directive 2026-07-13): (a) ADAPTIVE-dt order study —
    prescribed alternating (dt0, dt0/2) sequence exercises r = 2 and
    r = 0.5 variable coefficients on every step; BDF2 order ~2 where
    the scheme without variable coefficients sits at ~1 (measured
    1.96/1.97 vs BDF1 0.99/0.99).  (b) reject-consistency: DISCARDED
    attempts (the Appendix-A ladder's rejects) between accepted steps
    leave the accepted trajectory BIT-IDENTICAL (measured 0.0).
    (c) device-assembly x BDF2 parity (measured 6.3e-16, lock 1e-11).
    (d) BDF2 is deterministic-only (noise asserted off)."""
    T = 0.06

    def run_fix(tstep, dt):
        st = _mk_smooth(_strip_dm(1, 4, 2, device), 0.0, tstep, dt=dt)
        _march_fixed_dt(st, round(T / dt), dt)
        return st.x.copy()

    def run_var(tstep, dt0):
        st = _mk_smooth(_strip_dm(1, 4, 2, device), 0.0, tstep, dt=dt0)
        for _ in range(round(T / (1.5 * dt0))):
            for dtk in (dt0, dt0 / 2):
                x, iters, ok = st._attempt(dtk, 0.0)
                assert ok, iters
                st._commit(x, dtk, 0.0)
        assert abs(st.t - T) < 1e-12, st.t
        return st.x.copy()

    ref = run_fix("bdf2", 2.5e-4)
    ev2 = [np.abs(run_var("bdf2", d) - ref).max()
           for d in (4e-3, 2e-3, 1e-3)]
    ev1 = [np.abs(run_var("bdf1", d) - ref).max()
           for d in (4e-3, 2e-3, 1e-3)]
    ov2 = [np.log2(ev2[i] / ev2[i + 1]) for i in range(2)]
    ov1 = [np.log2(ev1[i] / ev1[i + 1]) for i in range(2)]
    print(f"wodo ADAPTIVE-dt BDF2 orders {['%.2f' % o for o in ov2]}; "
          f"BDF1 orders {['%.2f' % o for o in ov1]}")
    assert min(ov2) > 1.7, (ev2, ov2)
    assert max(ov1) < 1.3, (ev1, ov1)

    def run_rej(inject):
        dm = _strip_dm(1, 4, 2, device)
        st = WodoFilmStepper(dm, chi=(1.0, 0.3, 0.3), N=(5.0, 5.0, 1.0),
                             M=(0.225, 0.0, 0.225), kappa=(2e-4, 2e-4),
                             k_e=1.0, dt=1e-3, tstep="bdf2")
        rng = np.random.default_rng(3)
        st.set_initial(
            lambda x: 0.2 + 0.01 * rng.standard_normal(len(x)),
            lambda x: 0.2 + 0.01 * rng.standard_normal(len(x)))
        xs = []
        for k in range(8):
            p1n, p2n = st.hist[0]
            K = max(st.k_e * st._top_phis_avg(p1n, p2n), 0.0)
            if inject and k in (2, 5):
                st._attempt(4e-3, K)     # rejected attempt, DISCARDED
            x, iters, ok = st._attempt(1e-3, K)
            assert ok
            st._commit(x, 1e-3, K)
            xs.append(x.copy())
        return xs

    xa, xb = run_rej(False), run_rej(True)
    rd = max(np.abs(a - b).max() for a, b in zip(xa, xb))
    print(f"wodo reject-consistency dev {rd:.1e}")
    assert rd == 0.0, rd

    def run_dev(dev_asm):
        dm = _strip_dm(1, 5, 4, device)
        st = WodoFilmStepper(dm, chi=(1.0, 0.3, 0.3), N=(5.0, 5.0, 1.0),
                             M=(0.225, 0.0, 0.225), kappa=(2e-4, 2e-4),
                             k_e=1.0, dt=1e-3, tstep="bdf2",
                             use_device_assembly=dev_asm)
        rng = np.random.default_rng(3)
        st.set_initial(
            lambda x: 0.2 + 0.01 * rng.standard_normal(len(x)),
            lambda x: 0.2 + 0.01 * rng.standard_normal(len(x)))
        return _march_fixed_dt(st, 5, 1e-3)

    perr = max(np.abs(a - b).max() / max(np.abs(a).max(), 1e-30)
               for a, b in zip(run_dev(False), run_dev(True)))
    print(f"wodo device x BDF2 parity {perr:.1e}")
    assert perr < 1e-11, perr

    # BDF2 + FDT noise is a rejected config: the deterministic-only guard
    # now raises a typed ConfigError (was a bare assert before the
    # critical-eval hardening). Accept either so the gate is robust to -O.
    from diffsim.errors import ConfigError
    with pytest.raises((ConfigError, AssertionError)):
        WodoFilmStepper(_strip_dm(1, 3, 2, device), tstep="bdf2",
                        noise=1e-3)


def test_wodo_face_mass_generic(device):
    """The quadrature-built 1-D edge mass reproduces the analytic
    consistent masses per degree: p1 le[[1/3,1/6],[1/6,1/3]] (1-ulp
    class), p2 the Simpson-consistent le/30 [[4,2,-1],[2,16,2],[-1,2,4]].
    Guards the builder itself — a wrong face mass leaks solute at rate
    2*K*Phi (the SIGN NOTE), caught by the conservation gates below."""
    for p, ref in ((1, np.array([[2.0, 1.0], [1.0, 2.0]]) / 6.0),
                   (2, np.array([[4.0, 2.0, -1.0], [2.0, 16.0, 2.0],
                                 [-1.0, 2.0, 4.0]]) / 30.0)):
        tree = build_uniform(3, dim=2)
        mesh = build_mesh(tree, p=p)
        cons = build_constraints(mesh)
        dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2),
                                  device)
        st = WodoFilmStepper(dm, k_e=0.0, dt=1e-3)
        le = mesh.tree.h()[0]
        dev = np.abs(st.top_face_M / le - ref[None]).max()
        print(f"p{p} face-mass dev vs analytic: {dev:.2e}")
        assert dev < 1e-15, (p, dev)


def test_wodo_film_p2(device):
    """G1 capability gate: the film at p = 2 — (a) the conservation
    identity d/dt[h Int phi dtheta] = 0 holds at machine precision
    (mechanism-bearing: it requires the CONSISTENT p2 face mass paired
    with the p2 volume advection; a p1-shaped face term leaks), (b) the
    film thins and enriches, (c) host-vs-device-assembly parity < 1e-11
    (cross-matrix cell p2 x device path; measured 3.0e-16)."""
    tree0 = build_uniform(5, dim=2)
    keep = tree0.centers()[:, 0] < 4 / 32
    tree = Octree(tree0.keys[keep], tree0.levels[keep], dim=2,
                  periodic=tree0.periodic)
    mesh = build_mesh(tree, p=2)
    cons = build_constraints(mesh)

    def mk(dev_asm):
        dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(2, dim=2),
                                  device)
        st = WodoFilmStepper(dm, chi=(1.0, 0.3, 0.3), N=(5.0, 5.0, 1.0),
                             M=(0.225, 0.0, 0.225), kappa=(2e-4, 2e-4),
                             k_e=1.0, dt=1e-3,
                             use_device_assembly=dev_asm)
        rng = np.random.default_rng(3)
        st.set_initial(
            lambda x: 0.2 + 0.01 * rng.standard_normal(len(x)),
            lambda x: 0.2 + 0.01 * rng.standard_normal(len(x)))
        return st

    # (c) 5-step host/device parity at p2
    st_h = mk(False)
    st_d = mk(True)
    xs_h = _march_fixed_dt(st_h, 5, 1e-3)
    xs_d = _march_fixed_dt(st_d, 5, 1e-3)
    perr = max(np.abs(a - b).max() / max(np.abs(a).max(), 1e-30)
               for a, b in zip(xs_h, xs_d))
    print(f"p2 host/device parity: {perr:.2e}")
    assert perr < 1e-11
    assert abs(st_h.h_curr - st_d.h_curr) < 1e-13

    # (a)+(b) conservation + enrichment over a march (host path)
    st = mk(False)
    dm = st.dm
    width = mesh.node_coords[:, 0].max()

    def phi_int(vec):
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
    c1_0, c2_0 = st.h_curr * P1_0, st.h_curr * P2_0
    reason = st.march(h_min=0.9, phis_stop=0.05, max_steps=200)
    P1, P2 = phi_int(st.x[0::4]), phi_int(st.x[2::4])
    c1, c2 = st.h_curr * P1, st.h_curr * P2
    print(f"wodo p2: reason={reason} h={st.h_curr:.4f} content drift "
          f"({abs(c1 - c1_0) / c1_0:.1e},{abs(c2 - c2_0) / c2_0:.1e})")
    assert reason == "h_min" and st.h_curr <= 0.9
    assert abs(c1 - c1_0) / c1_0 < 1e-12       # solute conserved at p2
    assert abs(c2 - c2_0) / c2_0 < 1e-12
    assert P1 > P1_0 * 1.05 and P2 > P2_0 * 1.05


def test_wodo_p2_beats_p1(device):
    """G1 measured-order gate (the A4a Richardson pattern): smooth
    deterministic film run, fixed dt common-mode, reference L6-p2 —
    quadratic beats linear at the SAME h.  Measured at the retrofit:
    L4-p1 err 2.02e-3 vs L4-p2 1.93e-4 (10.5x); lock e_p2 < 0.5 e_p1."""
    def run(level, p, nsteps=50, dt=1e-3):
        tree0 = build_uniform(level, dim=2)
        keep = tree0.centers()[:, 0] < 2 / 2 ** level
        tree = Octree(tree0.keys[keep], tree0.levels[keep], dim=2,
                      periodic=tree0.periodic)
        mesh = build_mesh(tree, p=p)
        cons = build_constraints(mesh)
        dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2),
                                  device)
        st = WodoFilmStepper(dm, chi=(1.0, 0.3, 0.3),
                             N=(5.0, 5.0, 1.0),
                             M=(0.225, 0.0, 0.225), kappa=(2e-4, 2e-4),
                             k_e=1.0, dt=dt)
        st.set_initial(lambda x: 0.2 + 0.02 * np.cos(np.pi * x[:, 1]),
                       lambda x: 0.2 - 0.02 * np.cos(np.pi * x[:, 1]))
        _march_fixed_dt(st, nsteps, dt)
        coords = mesh.node_coords[cons.free_nodes]
        on_col = np.abs(coords[:, 0]) < 1e-12
        ys = coords[on_col, 1]
        vals = st.x[0::4][on_col]
        order = np.argsort(ys)
        return ys[order], vals[order]

    y_ref, v_ref = run(6, 2)
    on_l4 = np.abs(y_ref * 16 - np.round(y_ref * 16)) < 1e-12
    yr, vr = y_ref[on_l4], v_ref[on_l4]
    errs = {}
    for p in (1, 2):
        y, v = run(4, p)
        sel = np.abs(y * 16 - np.round(y * 16)) < 1e-12
        yy, vv = y[sel], v[sel]
        idx = np.searchsorted(yr, yy)
        errs[p] = np.abs(vv - vr[idx]).max()
    print(f"wodo Richardson: L4-p1 {errs[1]:.2e} vs L4-p2 {errs[2]:.2e} "
          f"(ratio {errs[2] / errs[1]:.3f})")
    assert errs[2] < 0.5 * errs[1], errs


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
