"""M4 track (c) gates: Wodo evaporating film — Landau-frame + evaporation
consistency. The load-bearing invariant is the sign pairing (advection
+K theta/h, top flux +K phi_i): physical solute content h * Int(phi_i)
is conserved EXACTLY per step (machine precision), while the film
thins and the mapped fractions enrich."""
import numpy as np
import pytest
import warp as wp

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


# ---------------------------------------------------------------------
# Task #37 (v1.4) G1 stage-parity gates: device-resident GP fields.
# Solver-free (no nvmath/cuDSS needed), so they run on Mac CPU AND box
# GPU.  TOLERANCE CONTRACT: the ported stage swaps numpy einsum
# (vectorized accumulation) for a sequential per-basis-function device
# loop, so CPU bit-equality is IMPOSSIBLE across the two arithmetics
# (measured 3.6e-15 abs on random fields) — the gate is the
# _assert_scatter_equal GPU band (rtol 1e-13 / atol 1e-14) on both
# devices: an indexing defect would produce O(1) diffs.
# ---------------------------------------------------------------------
def _mk_film_dev(device, tstep="bdf1", noise=0.0, p=1,
                 gp_residency="persistent"):
    tree0 = build_uniform(5, dim=2)
    keep = tree0.centers()[:, 0] < 4 / 32
    tree = Octree(tree0.keys[keep], tree0.levels[keep], dim=2,
                  periodic=tree0.periodic)
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2), device)
    st = WodoFilmStepper(dm, chi=(1.0, 0.3, 0.3), N=(5.0, 5.0, 1.0),
                         M=(0.225, 0.0, 0.225), kappa=(2e-4, 2e-4),
                         k_e=1.0, dt=1e-3, var_mob=True, b_reg=1e-3,
                         noise=noise, tstep=tstep,
                         use_device_assembly=True,
                         gp_residency=gp_residency)
    rng = np.random.default_rng(3)
    st.set_initial(
        lambda x: 0.2 + 0.01 * rng.standard_normal(len(x)),
        lambda x: 0.2 + 0.01 * rng.standard_normal(len(x)))
    return st


def _close(a, b, label):
    a, b = np.asarray(a), np.asarray(b)
    np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-14,
                               err_msg=label)


@pytest.mark.parametrize("p", [1, 2])
def test_wodo_gp_device_parity(device, p):
    """G1 stage parity: the persistent device GP buffers
    (_gp_eval_device -> gp_multifield) match the host _gp values and
    gradients for all 4 fields; the device BDF history (_bdf_time_device
    -> gp_vals + axpby) matches _bdf_time for BDF1 AND variable-
    coefficient BDF2; sigma is bit-equal; the identity-Tc elision
    (_full_of) is bit-equal to the spmv."""
    st = _mk_film_dev(device, p=p)
    st._init_device_assembly()
    rng = np.random.default_rng(11)
    x = st.x + 1e-3 * rng.standard_normal(len(st.x))
    st._gp_eval_device(x)
    fields = [st._gp(x[i::4]) for i in range(4)]
    for pv in st.dm.bins:
        vals = st._vals_dev[pv].numpy().copy()
        grads = st._grads_dev[pv].numpy().copy()
        for i in range(4):
            _close(vals[:, i], fields[i][0][pv], f"vals f{i} p{p}")
            _close(grads[:, i, :], fields[i][1][pv], f"grads f{i} p{p}")
    # BDF1 history
    dt = 1e-3
    sig_d = st._bdf_time_device(dt)
    sig_h, h1, h2 = st._bdf_time(dt)
    assert sig_d == sig_h
    for pv in st.dm.bins:
        hk = st._hist_dev[pv].numpy().copy()
        _close(hk[:, 0], h1[pv], "hist1 bdf1")
        _close(hk[:, 1], h2[pv], "hist2 bdf1")
    # identity-Tc elision is bit-equal
    p1n = st.hist[0][0]
    assert np.array_equal(st._full_of(p1n), np.asarray(st.Tc @ p1n))
    # variable-coefficient BDF2 (r != 1) with a distinct second level
    st2 = _mk_film_dev(device, tstep="bdf2", p=p)
    st2._init_device_assembly()
    rng = np.random.default_rng(4)
    st2.hist2 = (st2.hist[0][0] * 0.9
                 + 1e-3 * rng.standard_normal(len(st2.hist[0][0])),
                 st2.hist[0][1] * 1.1)
    st2.dt_prev = 4e-4
    sig_d = st2._bdf_time_device(dt)
    sig_h, h1, h2 = st2._bdf_time(dt)
    assert abs(sig_d - sig_h) < 1e-15 * abs(sig_h)
    for pv in st2.dm.bins:
        hk = st2._hist_dev[pv].numpy().copy()
        _close(hk[:, 0], h1[pv], "hist1 bdf2")
        _close(hk[:, 1], h2[pv], "hist2 bdf2")
    # the held-reference cache: a second call with the SAME history
    # must not re-upload (the #35 reviewer-minor contract)
    ref = st2._hist_up_ref
    st2._bdf_time_device(dt)
    assert st2._hist_up_ref is ref is st2.hist[0]


def test_wodo_device_fill_parity(device):
    """G1 (assembled system): one Newton-iterate fill through
    _fill_device_system with (a) the device GP path vs (b) the SAME
    persistent buffers overwritten by host-_gp values (the pre-v1.4
    data path) — vals_d/F_d agree to the few-ULP band.  Isolates
    exactly the ported stage; no linear solver involved.  Also checks
    the noise path: nonzero q buffers uploaded once per attempt enter
    the fill identically."""
    st = _mk_film_dev(device, noise=1e-3)
    st._init_device_assembly()
    rng = np.random.default_rng(7)
    x = st.x + 1e-3 * rng.standard_normal(len(st.x))
    dt = 1e-3
    K = 0.31
    sigma = st._bdf_time_device(dt)
    minv = 1.0 / st.h_curr
    mlat = 1.0 / st.lat_scale
    mvert = st.y_comp * minv
    coef = K * minv * st.y_comp
    # per-attempt uploads (noise + flux), as _attempt_device does
    rho = st.noise * np.sqrt(2.0 / dt)
    for pv, b in st.dm.bins.items():
        ngp = len(st.mesh.conn_of[pv]) * b["nqp"]
        st._upload_dev(st._q_dev[pv][0],
                       rho * st._nrng.standard_normal((ngp, st.dm.dim)))
        st._upload_dev(st._q_dev[pv][1],
                       rho * st._nrng.standard_normal((ngp, st.dm.dim)))
    st._upload_dev(st._flux_vals_dev, -coef * st._flux_base)
    args = (x, sigma, coef, K, minv, mlat, mvert, -1.0, -1.0, 0)

    # (a) device GP path
    st._gp_eval_device(x)
    st._fill_device_system(*args)
    vals_dev = st._asm.vals_d.numpy().copy()
    F_dev = st._asm.F_d.numpy().copy()

    # (b) host reference values pushed into the same buffers
    fields = [st._gp(x[i::4]) for i in range(4)]
    for pv in st.dm.bins:
        st._upload_dev(st._vals_dev[pv],
                       np.stack([fields[i][0][pv] for i in range(4)],
                                axis=1))
        st._upload_dev(st._grads_dev[pv],
                       np.stack([fields[i][1][pv] for i in range(4)],
                                axis=1))
    _, h1, h2 = st._bdf_time(dt)
    for pv in st.dm.bins:
        st._upload_dev(st._hist_dev[pv],
                       np.stack([h1[pv], h2[pv]], axis=1))
    st._fill_device_system(*args)
    vals_ref = st._asm.vals_d.numpy().copy()
    F_ref = st._asm.F_d.numpy().copy()

    dv = np.abs(vals_dev - vals_ref).max()
    df = np.abs(F_dev - F_ref).max()
    print(f"wodo fill parity: |dA|={dv:.2e} |dF|={df:.2e}")
    _close(vals_dev, vals_ref, "vals_d")
    _close(F_dev, F_ref, "F_d")


class _SpluDeviceFilm(WodoFilmStepper):
    """Test double (Task #37 reviewer finding): the FULL v1.4 device
    path — real per-attempt noise draws/uploads, device GP fields,
    multi-batch fill, flux adds — with ONLY the linear solve swapped
    for the same host splu the reference run uses.  Any host/device
    trajectory difference is then purely the ported DATA path (no
    solver-tolerance blur, no nvmath/cuDSS dependency; CPU-warp
    blockch inners were measured at ~7 s/step of pure launch
    orchestration, so this keeps the test in the seconds class)."""

    def _solve_device_blockch(self, asm):
        import scipy.sparse as _sp
        from scipy.sparse.linalg import splu
        A = _sp.csr_matrix(
            (asm.vals_d.numpy().copy(), asm.indices, asm.indptr),
            shape=(asm.Nfull, asm.Nfull))
        return splu(A.tocsc()).solve(asm.F_d.numpy().copy())


def test_wodo_device_noise_trajectory_parity(device):
    """End-to-end host-vs-device trajectory parity with ACTIVE Langevin
    noise (Task #37 reviewer finding): the fill-parity test above
    uploads the q buffers itself, so a q1/q2 swap or a rho error
    inside _attempt_device would pass every other committed test —
    this one drives the REAL per-attempt noise path (same host RNG
    stream on both sides, the noise_seed contract) through 5 fixed-dt
    steps of the actual film physics (evaporation + top flux +
    var-mob + b_reg).  The device side is _SpluDeviceFilm (identical
    splu solve both sides -> the standard 1e-11 parity lock) and is
    forced MULTI-BATCH: nb_cap = 7 does not divide ne = 128, so a
    batch-offset defect in the s0/s1 slicing of the persistent
    GP/noise buffers shifts whole batches and is caught at
    O(noise)."""
    tree0 = build_uniform(5, dim=2)
    keep = tree0.centers()[:, 0] < 4 / 32
    tree = Octree(tree0.keys[keep], tree0.levels[keep], dim=2,
                  periodic=tree0.periodic)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)

    def run(dev_asm):
        dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2),
                                  device)
        cls = _SpluDeviceFilm if dev_asm else WodoFilmStepper
        st = cls(dm, chi=(1.0, 0.3, 0.3), N=(5.0, 5.0, 1.0),
                 M=(0.225, 0.0, 0.225), kappa=(2e-4, 2e-4),
                 k_e=1.0, dt=1e-3, var_mob=True, b_reg=1e-3,
                 noise=1e-3, noise_seed=5,
                 linsolver="blockch" if dev_asm else "splu",
                 use_device_assembly=dev_asm)
        rng = np.random.default_rng(3)
        st.set_initial(
            lambda x: 0.2 + 0.01 * rng.standard_normal(len(x)),
            lambda x: 0.2 + 0.01 * rng.standard_normal(len(x)))
        if dev_asm:
            st._init_device_assembly()
            for k_bin, (pv, b, ne, nbf, _g) in enumerate(st._asm._bins):
                assert ne % 7 != 0 and ne > 7   # partial last batch
                nl = 4 * nbf
                st._batch_bufs[k_bin] = (
                    7, wp.zeros((7, nl, nl), dtype=wp.float64,
                                device=device),
                    wp.zeros((7, nl), dtype=wp.float64, device=device))
        return st, _march_fixed_dt(st, 5, 1e-3)

    st_h, xs_h = run(False)
    st_d, xs_d = run(True)
    errs = [np.abs(a - b).max() / max(np.abs(a).max(), 1e-30)
            for a, b in zip(xs_h, xs_d)]
    print(f"wodo noise trajectory parity (multi-batch): step-wise rel "
          f"err max {max(errs):.2e}; h {st_h.h_curr:.6f} vs "
          f"{st_d.h_curr:.6f}")
    assert max(errs) < 1e-11, errs
    assert abs(st_h.h_curr - st_d.h_curr) < 1e-13


# -- Task #41: batch-local GP-eval fallback (rung-c memory) --------------
def _mk_bl_stepper(device, cls, gp_residency, noise, mesh, cons):
    """One _SpluDeviceFilm/WodoFilmStepper in the given gp_residency
    mode, initial state fixed by seed (the run() helper below), the #37
    multi-batch nb_cap=7 forced so partial batches are exercised."""
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    st = cls(dm, chi=(1.0, 0.3, 0.3), N=(5.0, 5.0, 1.0),
             M=(0.225, 0.0, 0.225), kappa=(2e-4, 2e-4), k_e=1.0,
             dt=1e-3, var_mob=True, b_reg=1e-3, noise=noise, noise_seed=5,
             linsolver="blockch", use_device_assembly=True,
             gp_residency=gp_residency)
    rng = np.random.default_rng(3)
    st.set_initial(
        lambda x: 0.2 + 0.01 * rng.standard_normal(len(x)),
        lambda x: 0.2 + 0.01 * rng.standard_normal(len(x)))
    st._init_device_assembly()
    for k_bin, (pv, b, ne, nbf, _g) in enumerate(st._asm._bins):
        assert ne % 7 != 0 and ne > 7           # partial last batch
        nl = 4 * nbf
        st._batch_bufs[k_bin] = (
            7, wp.zeros((7, nl, nl), dtype=wp.float64, device=device),
            wp.zeros((7, nl), dtype=wp.float64, device=device))
    return st


@pytest.mark.parametrize("noise", [0.0, 1e-3])
def test_wodo_gp_residency_parity(device, noise):
    """Task #41 G1: batch_local GP eval produces the SAME assembled
    system as persistent — the memory-layout fallback is a pure
    residency change, not a numerics change.  Drives the FULL
    _SpluDeviceFilm device data path (real noise draws, device GP,
    multi-batch fill, flux) for 5 fixed-dt steps in BOTH residency
    modes with the identical splu solve and identical RNG stream, and
    asserts the per-step trajectory is bit-equal on CPU / few-ULP on
    GPU.  nb_cap=7 (not a divisor of ne) forces partial batches, so a
    batch-offset defect in the batch-local eval/upload is caught."""
    tree0 = build_uniform(5, dim=2)
    keep = tree0.centers()[:, 0] < 4 / 32
    tree = Octree(tree0.keys[keep], tree0.levels[keep], dim=2,
                  periodic=tree0.periodic)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)

    def run(gp_residency):
        st = _mk_bl_stepper(device, _SpluDeviceFilm, gp_residency,
                            noise, mesh, cons)
        assert st._gp_batch_local == (gp_residency == "batch_local")
        return st, _march_fixed_dt(st, 5, 1e-3)

    st_p, xs_p = run("persistent")
    st_b, xs_b = run("batch_local")
    if str(device).startswith("cpu"):
        for i, (a, b) in enumerate(zip(xs_p, xs_b)):
            assert np.array_equal(a, b), (
                f"batch_local != persistent bit-for-bit at step {i}: "
                f"max |d| = {np.abs(a - b).max():.2e}")
        assert st_p.h_curr == st_b.h_curr
    else:
        errs = [np.abs(a - b).max() / max(np.abs(a).max(), 1e-30)
                for a, b in zip(xs_p, xs_b)]
        assert max(errs) < 1e-13, errs
        assert abs(st_p.h_curr - st_b.h_curr) < 1e-14
    print(f"gp_residency parity (noise={noise}): "
          f"h {st_p.h_curr:.6f} vs {st_b.h_curr:.6f}")


@pytest.mark.parametrize("tstep", ["bdf1", "bdf2"])
def test_wodo_gp_residency_fill_parity(device, tstep):
    """Task #41 G1 (assembled system, solver-free): one Newton-iterate
    fill of asm.vals_d / asm.F_d in batch_local mode equals the
    persistent fill to few-ULP (bit-equal on CPU) — for BOTH BDF1 and
    variable-coefficient BDF2 (the batch-local history axpby path).
    Noise active so the per-batch noise-slice upload is exercised too."""
    noise = 0.0 if tstep == "bdf2" else 1e-3   # BDF2 is deterministic-only

    def fill(gp_residency):
        st = _mk_film_dev(device, tstep=tstep, noise=noise,
                          gp_residency=gp_residency)
        if tstep == "bdf2":
            rng = np.random.default_rng(4)
            st.hist2 = (st.hist[0][0] * 0.9
                        + 1e-3 * rng.standard_normal(len(st.hist[0][0])),
                        st.hist[0][1] * 1.1)
            st.dt_prev = 4e-4
        st._init_device_assembly()
        rng = np.random.default_rng(7)
        x = st.x + 1e-3 * rng.standard_normal(len(st.x))
        dt, K = 1e-3, 0.31
        sigma = st._bdf_time_device(dt)
        minv, mlat = 1.0 / st.h_curr, 1.0 / st.lat_scale
        mvert = st.y_comp * minv
        coef = K * minv * st.y_comp
        st._q_host = None
        if noise != 0.0:
            rho = noise * np.sqrt(2.0 / dt)
            st._q_host = {}
            for pv, b in st.dm.bins.items():
                ngp = len(st.mesh.conn_of[pv]) * b["nqp"]
                q0 = rho * st._nrng.standard_normal((ngp, st.dm.dim))
                q1 = rho * st._nrng.standard_normal((ngp, st.dm.dim))
                if st._gp_batch_local:
                    st._q_host[pv] = (np.ascontiguousarray(q0),
                                      np.ascontiguousarray(q1))
                else:
                    st._upload_dev(st._q_dev[pv][0], q0)
                    st._upload_dev(st._q_dev[pv][1], q1)
        st._upload_dev(st._flux_vals_dev, -coef * st._flux_base)
        st._gp_eval_device(x)
        st._fill_device_system(x, sigma, coef, K, minv, mlat, mvert,
                               -1.0, -1.0, 0)
        return (st._asm.vals_d.numpy().copy(), st._asm.F_d.numpy().copy())

    vp, fp = fill("persistent")
    vb, fb = fill("batch_local")
    if str(device).startswith("cpu"):
        assert np.array_equal(vp, vb), np.abs(vp - vb).max()
        assert np.array_equal(fp, fb), np.abs(fp - fb).max()
    else:
        _close(vb, vp, f"vals_d {tstep}")
        _close(fb, fp, f"F_d {tstep}")


def test_wodo_gp_residency_auto_cpu(device):
    """Task #41: gp_residency='auto' resolves to persistent on CPU (no
    free-VRAM query — host RAM is not the constrained resource the
    fallback targets), and the config knob validates its allowed set."""
    st = _mk_film_dev(device, gp_residency="auto")
    st._init_device_assembly()
    if str(device).startswith("cpu"):
        assert st._gp_batch_local is False       # auto -> persistent
    from diffsim.errors import ConfigError
    tree0 = build_uniform(5, dim=2)
    keep = tree0.centers()[:, 0] < 4 / 32
    tree = Octree(tree0.keys[keep], tree0.levels[keep], dim=2,
                  periodic=tree0.periodic)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    with pytest.raises(ConfigError):
        WodoFilmStepper(dm, use_device_assembly=True,
                        gp_residency="nonsense")


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
