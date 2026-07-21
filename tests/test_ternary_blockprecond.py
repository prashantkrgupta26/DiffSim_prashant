"""G4 gates: the blockch per-pair two-factor Schur preconditioner on the
ternary 4-dof CH block (phi1,mu1,phi2,mu2) and the Wodo film
(solvers/linsolve.py _blockch_pairs; meta {"sigma","ndof","pairs"}).

MEASURED VERDICT (2026-07-09, L5 2-D uniform, dt=0.005, BDF1): the
planned spinodal gate config chi=(6.0,0.8,0.8), M=(1.0,-0.2,1.0),
kappa=8e-4 is OUT OF CONTRACT for the pairwise form, and the failure is
STRUCTURAL, not an implementation artifact:

  * two-factor pairs AND the per-pair exact-Schur escalation FAIL on the
    FIRST Newton solve of step 1 (outer lgmres 100 cycles + fallback 40,
    no convergence; reproduced on dumped Newton iterates 0-2).
  * even EXACT per-pair block-diagonal solves (splu on each (phi_i,mu_i)
    2x2 subsystem — the ceiling of ANY pairwise preconditioner) need
    127-161 outer lgmres cycles (binary G1 contract: 1-4).
  * diagnosis: the FH curvature block at the IC (phi1=phi2=0.35) is
    [[d11,d12],[d12,d22]] = [[4.59,7.73],[7.73,4.59]], eigenvalues
    {+12.3, -3.14}. At chi12=6 the unstable spinodal direction lives
    ENTIRELY in the DROPPED d12 cross block (d12 = 1/phi_s + chi12
    - chi1s - chi2s EXCEEDS the kept d11) — a pairwise preconditioner is
    blind to the indefiniteness that drives the physics. This is design
    law 1 (never drop the destabilizing curvature) one level up: for
    chi12-driven quenches the destabilizer is the CROSS curvature, which
    the per-pair recipe cannot carry. Follow-up (G4b): a species-coupled
    Schur that carries d12 across the pairs.

CONTRACT BOUNDARY (measured, same mesh/dt/IC): the pairwise form is in
contract while the d12-carried indefiniteness is marginal —
  chi12=3.0, M12=-0.2 (unstable eig -0.14, the marginal quench of
    test_ternary_ch's comment): parity 2.4e-11 vs splu, outer its 1-4,
    IDENTICAL numbers with host and device ("blockch_dev") inners;
  chi12=-1.73, M12=0 (d12 ~ 0 at the IC): parity 8.5e-12, outer 1-4;
  Wodo film evaporation (chi12=1.0, the paper physics: advection +
    mapped metric + top-flux rows): 179 solves over a full h 1->0.8
    march, outer its 1-4, ZERO fallbacks/rejects, solute drift 1.8e-15.

The sigma-scaling of the boundary, CONFIRMED at chi12=6 through the
film's march(): each failing blockch solve signals divergence to the
Appendix-A ladder, dt collapses 0.005 -> 7.8e-5 (three 0.25x rejects),
and there the pairwise form CONVERGES at outer its 3-4 — right at the
dt < ~1e-4 threshold predicted by d12/(2 sqrt(kap sigma)) ~ 1 (the
dropped-coupling-to-mass ratio at the resonant wavenumber sigma =
m kap k^4). The march SURVIVES, but at a ~64x dt penalty vs splu's
ladder [(5e-3, 5 its), (6.25e-3, 7), (2e-3, 6)] — out of production
contract, and splu-vs-blockch ladder parity is structurally ill-posed
there.

TOLERANCE NOTE (the parity-gate lesson, docs/dev/2026-07-09-parity-gate-
nondeterminism.md): parity asserted at 1e-7, ~3.5 decades above the
measured 1e-11-class; iteration bounds carry >= 2x headroom over the
measured maxima."""
import numpy as np
import pytest

from diffsim.octree.build import build_uniform, Octree
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.ternary_ch import TernaryCHStepper
from diffsim.physics.wodo_film import WodoFilmStepper

pytestmark = pytest.mark.tier3


class _RecCache(dict):
    """Solver cache that logs every ('blockch_iters', key) record — the
    per-step read of the binary gates misses intra-step Newton iterates."""

    def __init__(self):
        super().__init__()
        self.iters = []

    def __setitem__(self, k, v):
        if isinstance(k, tuple) and k[0] == "blockch_iters":
            self.iters.append(v[0])
        super().__setitem__(k, v)


def _dm(level, device):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    return DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)


def _ternary(dm, solver, chi12):
    st = TernaryCHStepper(dm, chi=(chi12, 0.8, 0.8), M=(1.0, -0.2, 1.0),
                          kappa=(8e-4, 8e-4), dt=0.005, order=1,
                          linsolver=solver)
    rng = np.random.default_rng(4)
    st.set_initial(lambda x: 0.35 + 0.02 * rng.standard_normal(len(x)),
                   lambda x: 0.35 + 0.02 * rng.standard_normal(len(x)))
    st._solver_cache = _RecCache()
    return st


@pytest.mark.parametrize("solver", ["blockch", "blockch_dev"])
def test_ternary_blockch_marginal_parity(solver, device):
    """Ternary parity at the pairwise-contract boundary: chi12=3.0 (a
    REAL marginal spinodal, unstable eig -0.14) with the M12=-0.2 and
    d12 cross blocks active. Measured (host): parity 2.4e-11, outer its
    1-4 over 19 solves. NOTE: the originally planned chi12=6.0 deep
    quench is out of contract — see the module docstring and
    test_ternary_blockch_deep_quench_verdict."""
    dm = _dm(5, device)
    sts = _ternary(dm, "splu", 3.0)
    stb = _ternary(dm, solver, 3.0)
    rels = []
    for _ in range(6):
        p1r, p2r = sts.step()
        p1b, p2b = stb.step()
        rels.append(max(
            np.abs(p1r - p1b).max() / max(np.abs(p1r).max(), 1e-30),
            np.abs(p2r - p2b).max() / max(np.abs(p2r).max(), 1e-30)))
    its = stb._solver_cache.iters
    mx = max(i % 1000 for i in its)
    print(f"ternary blockch[{solver}] chi12=3: parity {max(rels):.2e}, "
          f"outer its {its}")
    assert max(rels) < 1e-7, rels
    assert mx <= 8, its
    assert not any(i >= 1000 for i in its), ("fallback engaged in the "
                                             "converged regime", its)


def test_ternary_blockch_deep_quench_verdict(device):
    """HONEST FAILURE PIN — the planned G4 spinodal gate (chi12=6.0).
    Measured 2026-07-09: the first blockch Newton solve fails BOTH the
    two-factor pairwise form (100 outer lgmres cycles) and the per-pair
    exact-Schur escalation (40 cycles); exact per-pair block-diagonal
    solves need 127-161 cycles. See the module docstring for the full
    diagnosis (the -3.14 unstable curvature direction lives in the
    dropped d12 cross block). blockch_dev shares the identical outer
    operator (device inners only relocate the inner Krylov solves), so
    the verdict transfers. This pin flips when a species-coupled Schur
    (G4b) lands — at that point promote chi12=6 into the parity gate."""
    dm = _dm(5, device)
    stb = _ternary(dm, "blockch", 6.0)
    with pytest.raises(RuntimeError, match="blockch fallback"):
        stb.step()


def test_film_blockch_reduction_parity(device):
    """Film reduction gate at the contract boundary: WodoFilmStepper
    with k_e=0, N=(1,1,1) reduces EXACTLY to the ternary equations;
    chi12=3.0 keeps the quench inside the pairwise contract (at the
    planned chi12=6.0 the blockch ladder collapses dt to 7.8e-5 before
    converging — measured, see module docstring — so splu/blockch
    ladders are structurally incomparable there). march() carries the
    accept/reject ladder; a ladder mismatch triggers one full retry
    (the GPU-nondeterminism lesson). Measured: identical splu/blockch
    ladders (5 accepted / 0 rejects, dt 5e-3 -> 1.2e-2), parity max
    9.3e-12, outer its 1-4 over 17 solves, 0 fallbacks — numbers
    identical with host and device inners."""
    def run(solver):
        dm = _dm(5, device)
        st = WodoFilmStepper(dm, chi=(3.0, 0.8, 0.8), N=(1.0, 1.0, 1.0),
                             M=(1.0, -0.2, 1.0), kappa=(8e-4, 8e-4),
                             k_e=0.0, dt=0.005, linsolver=solver)
        rng = np.random.default_rng(4)
        st.set_initial(
            lambda x: 0.35 + 0.02 * rng.standard_normal(len(x)),
            lambda x: 0.35 + 0.02 * rng.standard_normal(len(x)))
        st._solver_cache = _RecCache()
        rec = []
        st.march(h_min=0.0, phis_stop=-1.0, max_steps=5,
                 callback=lambda s, K, dt, it:
                 rec.append((dt, it, s.x.copy())))
        return rec, st.n_reject, st._solver_cache.iters

    for attempt in range(2):
        rec_s, rej_s, _ = run("splu")
        rec_b, rej_b, its = run("blockch")
        if rej_s == rej_b and len(rec_s) == len(rec_b):
            break
    assert rej_s == rej_b and len(rec_s) == len(rec_b)
    assert all(a[:2] == b[:2] for a, b in zip(rec_s, rec_b)), \
        "accept/reject or iteration ladder diverged"
    errs = [np.abs(a[2] - b[2]).max() / max(np.abs(a[2]).max(), 1e-30)
            for a, b in zip(rec_s, rec_b)]
    mx = max(i % 1000 for i in its)
    print(f"film reduction chi12=3: {len(rec_s)} accepted steps "
          f"({rej_b} rejects), parity max {max(errs):.2e}, "
          f"outer its {its}")
    assert max(errs) < 1e-7, errs
    assert mx <= 8, its
    assert not any(i >= 1000 for i in its), ("fallback engaged", its)


def test_film_blockch_device_setup_parity(device):
    """G5 gate: blockch with the DEVICE-RESIDENT setup
    (use_device_assembly=True -> linsolve.blockch_pairs_device: pair
    W1/W2/mass values built by a fill kernel on the assembler's slot-map
    CSR, device inner Krylov, no host matrix) vs the host-setup blockch
    trajectory, on the evaporation config. Measured (2026-07-09, 30
    march steps): trajectory parity 3.03e-14; accepted/reject ladders
    identical (30/0 both, dt sequences equal); the OUTER-iteration
    ladder wobbles by +-1 in 6/91 solves (device inner-stack rounding —
    the 7badd7b GPU-nondeterminism class; invisible to the dt heuristic,
    which only tests iters < 20). Bitwise ladder equality is therefore
    NOT asserted; solve count, accept/reject ladder, parity and
    iteration bounds are. Cost note (measured): at this tiny 2-D size
    the device-setup path is ~45x slower than host-setup (latency-bound
    inner launches) — it exists for the 3-D sizes where a host CSR
    cannot (G5)."""
    def run(dev_asm, nsteps=10):
        tree0 = build_uniform(5, dim=2)
        keep = tree0.centers()[:, 0] < 4 / 32
        tree = Octree(tree0.keys[keep], tree0.levels[keep], dim=2,
                      periodic=tree0.periodic)
        mesh = build_mesh(tree, p=1)
        cons = build_constraints(mesh)
        dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2),
                                  device)
        st = WodoFilmStepper(dm, chi=(1.0, 0.3, 0.3), N=(5.0, 5.0, 1.0),
                             M=(0.225, 0.0, 0.225), kappa=(2e-4, 2e-4),
                             k_e=1.0, dt=1e-3, linsolver="blockch",
                             use_device_assembly=dev_asm)
        rng = np.random.default_rng(3)
        st.set_initial(
            lambda x: 0.2 + 0.01 * rng.standard_normal(len(x)),
            lambda x: 0.2 + 0.01 * rng.standard_normal(len(x)))
        st._solver_cache = _RecCache()
        rec = []
        st.march(h_min=0.8, phis_stop=0.05, max_steps=nsteps,
                 callback=lambda s, K, dt, it:
                 rec.append((dt, s.x.copy())))
        return rec, st.n_reject, st._solver_cache.iters

    rec_h, rej_h, its_h = run(False)
    rec_d, rej_d, its_d = run(True)
    assert rej_h == rej_d and len(rec_h) == len(rec_d)
    # dt compared RELATIVELY, not bitwise: march clamps dt_eff =
    # min(dt, dh_cap/K) and K is a continuous function of the state, so
    # the 1e-14 trajectory difference perturbs the clamped dt in its
    # last ulps (measured: exact dt equality fails once the clamp binds)
    assert all(abs(a[0] - b[0]) < 1e-9 * a[0]
               for a, b in zip(rec_h, rec_d)), "dt ladder diverged"
    assert len(its_h) == len(its_d), "solve count diverged"
    errs = [np.abs(a[1] - b[1]).max() / max(np.abs(a[1]).max(), 1e-30)
            for a, b in zip(rec_h, rec_d)]
    mx = max(i % 1000 for i in its_d)
    fb = sum(1 for i in its_d if i >= 1000)
    wobble = sum(1 for a, b in zip(its_h, its_d) if a != b)
    print(f"film device-setup parity: {len(rec_h)} steps, parity max "
          f"{max(errs):.2e}, outer wobble {wobble}/{len(its_d)} solves, "
          f"dev max outer {mx}, fallbacks {fb}")
    assert max(errs) < 1e-9, errs
    assert mx <= 8, its_d
    assert fb == 0, ("fallback engaged", its_d)


def test_film_blockch_device_resident_apply_parity(device, monkeypatch):
    """Task #42 G1: the OPT-IN device-resident preconditioner apply
    (DIFFSIM_PRECOND_DEV_APPLY=1 — r/z stay on device through the whole
    blockch apply chain) is iterate-identical to the default host-transfer
    device apply.  Both sides use the device-resident #37 SETUP; only the
    APPLY residency differs, so any trajectory difference is the residency
    change alone.  CUDA-only (the device-resident path auto-gates off on
    CPU); asserts few-ULP parity + identical outer-iteration ladders + no
    exact-Schur fallback (a degraded preconditioner would trip it)."""
    if not str(device).startswith("cuda"):
        pytest.skip("device-resident apply is CUDA-only")

    def run(dev_apply):
        monkeypatch.setenv("DIFFSIM_PRECOND_DEV_APPLY",
                           "1" if dev_apply else "0")
        tree0 = build_uniform(5, dim=2)
        keep = tree0.centers()[:, 0] < 4 / 32
        tree = Octree(tree0.keys[keep], tree0.levels[keep], dim=2,
                      periodic=tree0.periodic)
        mesh = build_mesh(tree, p=1)
        cons = build_constraints(mesh)
        dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2),
                                  device)
        st = WodoFilmStepper(dm, chi=(1.0, 0.3, 0.3), N=(5.0, 5.0, 1.0),
                             M=(0.225, 0.0, 0.225), kappa=(2e-4, 2e-4),
                             k_e=1.0, dt=1e-3, linsolver="blockch",
                             use_device_assembly=True)
        assert st.precond_dev_apply == dev_apply    # knob wired
        rng = np.random.default_rng(3)
        st.set_initial(
            lambda x: 0.2 + 0.01 * rng.standard_normal(len(x)),
            lambda x: 0.2 + 0.01 * rng.standard_normal(len(x)))
        st._solver_cache = _RecCache()
        rec = []
        st.march(h_min=0.8, phis_stop=0.05, max_steps=10,
                 callback=lambda s, K, dt, it: rec.append((dt, s.x.copy())))
        return rec, st._solver_cache.iters

    rec_host, its_host = run(False)
    rec_dev, its_dev = run(True)
    assert len(rec_host) == len(rec_dev)
    errs = [np.abs(a[1] - b[1]).max() / max(np.abs(a[1]).max(), 1e-30)
            for a, b in zip(rec_host, rec_dev)]
    wobble = sum(1 for a, b in zip(its_host, its_dev) if a != b)
    fb = sum(1 for i in its_dev if i >= 1000)
    print(f"device-resident apply parity: {len(rec_host)} steps, "
          f"max rel err {max(errs):.2e}, outer wobble {wobble}, "
          f"fallbacks {fb}")
    assert max(errs) < 1e-9, errs
    assert fb == 0, ("fallback engaged on device-resident apply", its_dev)


def test_film_blockch_device_outer_fgmres_parity(device, monkeypatch):
    """Task #49 G1: the device-resident OUTER FGMRES
    (DIFFSIM_PRECOND_DEV_OUTER=1 — the whole preconditioned solve is one
    device-resident region: outer Krylov vecops + preconditioner apply +
    matvec) is trajectory-equivalent to the default host-scipy-lgmres
    outer (device-resident #37 SETUP on both sides; only the OUTER changes).
    CUDA-only (the device outer auto-gates off on CPU).  The device FGMRES
    is right-preconditioned flexible GMRES(30) — the flexible analogue of
    scipy lgmres with the LGMRES augmentation off — so the iterate matches
    host to few-ULP where the outer-check cadence agrees; the outer count
    may shift +-1 by the augmentation difference (documented, task-49 G1),
    so convergence-history EQUIVALENCE (parity + accept/reject ladder +
    bounded iters + no fallback) is asserted, not bitwise iters."""
    if not str(device).startswith("cuda"):
        pytest.skip("device-outer FGMRES is CUDA-only")

    def run(dev_outer):
        monkeypatch.setenv("DIFFSIM_PRECOND_DEV_OUTER",
                           "1" if dev_outer else "0")
        tree0 = build_uniform(5, dim=2)
        keep = tree0.centers()[:, 0] < 4 / 32
        tree = Octree(tree0.keys[keep], tree0.levels[keep], dim=2,
                      periodic=tree0.periodic)
        mesh = build_mesh(tree, p=1)
        cons = build_constraints(mesh)
        dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2),
                                  device)
        st = WodoFilmStepper(dm, chi=(1.0, 0.3, 0.3), N=(5.0, 5.0, 1.0),
                             M=(0.225, 0.0, 0.225), kappa=(2e-4, 2e-4),
                             k_e=1.0, dt=1e-3, linsolver="blockch",
                             use_device_assembly=True)
        assert st.precond_dev_outer == dev_outer     # knob wired
        rng = np.random.default_rng(3)
        st.set_initial(
            lambda x: 0.2 + 0.01 * rng.standard_normal(len(x)),
            lambda x: 0.2 + 0.01 * rng.standard_normal(len(x)))
        st._solver_cache = _RecCache()
        rec = []
        st.march(h_min=0.8, phis_stop=0.05, max_steps=10,
                 callback=lambda s, K, dt, it: rec.append((dt, s.x.copy())))
        return rec, st.n_reject, st._solver_cache.iters

    rec_host, rej_host, its_host = run(False)
    rec_dev, rej_dev, its_dev = run(True)
    assert rej_host == rej_dev and len(rec_host) == len(rec_dev)
    assert all(abs(a[0] - b[0]) < 1e-9 * a[0]
               for a, b in zip(rec_host, rec_dev)), "dt ladder diverged"
    assert len(its_host) == len(its_dev), "solve count diverged"
    errs = [np.abs(a[1] - b[1]).max() / max(np.abs(a[1]).max(), 1e-30)
            for a, b in zip(rec_host, rec_dev)]
    wobble = sum(1 for a, b in zip(its_host, its_dev) if a != b)
    mx = max(i % 1000 for i in its_dev)
    fb = sum(1 for i in its_dev if i >= 1000)
    print(f"device-outer FGMRES parity: {len(rec_host)} steps, max rel "
          f"err {max(errs):.2e}, outer wobble {wobble}, dev max outer "
          f"{mx}, fallbacks {fb}")
    assert max(errs) < 1e-9, errs
    assert mx <= 12, its_dev
    assert fb == 0, ("fallback engaged on device outer", its_dev)


def test_film_blockch_evaporation(device):
    """The full film physics through the pairwise preconditioner —
    test_wodo_film.py::test_wodo_film_evaporation's config with
    linsolver="blockch": frame advection, mapped-gradient anisotropy and
    the top-flux enrichment rows all ride in through the extracted
    blocks. Same physics gates as the splu test (h thins, solute
    conserved, enrichment). Measured: 179 solves to h=0.8, outer its
    1-4, zero fallbacks, zero rejects, content drift 1.8e-15."""
    tree0 = build_uniform(5, dim=2)
    keep = tree0.centers()[:, 0] < 4 / 32
    tree = Octree(tree0.keys[keep], tree0.levels[keep], dim=2,
                  periodic=tree0.periodic)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    st = WodoFilmStepper(dm, chi=(1.0, 0.3, 0.3), N=(5.0, 5.0, 1.0),
                         M=(0.225, 0.0, 0.225), kappa=(2e-4, 2e-4),
                         k_e=1.0, dt=1e-3, linsolver="blockch")
    rng = np.random.default_rng(3)
    st.set_initial(lambda x: 0.2 + 0.01 * rng.standard_normal(len(x)),
                   lambda x: 0.2 + 0.01 * rng.standard_normal(len(x)))
    st._solver_cache = _RecCache()

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
    its = st._solver_cache.iters
    mx = max(i % 1000 for i in its)
    fb = sum(1 for i in its if i >= 1000)
    print(f"film blockch evaporation: reason={reason} h={st.h_curr:.3f} "
          f"t={st.t:.4f} Phi_p {P1_0:.4f}->{P1:.4f} content drift "
          f"({abs(c1 - c1_0) / c1_0:.1e},{abs(c2 - c2_0) / c2_0:.1e}); "
          f"{len(its)} solves, max outer {mx}, fallbacks {fb}")
    assert reason == "h_min"
    assert st.h_curr <= 0.8 < 1.0                      # film thinned
    assert abs(c1 - c1_0) / c1_0 < 1e-10               # solute conserved
    assert abs(c2 - c2_0) / c2_0 < 1e-10
    assert P1 > P1_0 * 1.15 and P2 > P2_0 * 1.15       # enrichment
    ps = 1.0 - st.x[0::4] - st.x[2::4]
    assert ps.min() > -0.05                            # simplex ~respected
    assert mx <= 8, its
    assert fb == 0, ("fallback engaged on the evaporation march", its)
