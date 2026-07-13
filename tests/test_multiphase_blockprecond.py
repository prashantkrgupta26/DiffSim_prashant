"""B-track gates: blockch on the full multiphase (M, K) system
(linsolve `pairs` + `ac` meta; MultiPhaseStepper linsolver="blockch" /
"blockch_dev", both assembly modes; docs/dev/2026-07-13-blockch-mpf.md).

MEASURED (2026-07-13, L5 2-D, RTX 6000 Ada; scratchpad b1_gates /
b2_smoke logs — the dev note carries the full panel):

  * one-solve parity on the identical assembled film system (S3b-class
    (M=2, K=1) film + FDT noise, frozen theta): rel|dx| 3.1e-11 vs
    splu, relres 4.5e-11, outer its 3;
  * film march (10 steps, splu vs blockch, same seeds): parity max
    5.3e-14, identical ladders (0 rejects both), outer its <= 4
    (mean 3.0), 0 fallbacks;
  * device-assembly march (device splu vs blockch_dev zero-copy on the
    slot-map CSR): parity 2.0e-14, outer <= 3, 0 fallbacks — identical
    numbers with the block-masked kron(G, mask) pattern;
  * blockmask exactness (film config): masked nnz ratio 0.611 (22/36
    live blocks), rel max|dA| 3.4e-16 vs superset values, |dr| 5.6e-17;
  * the AC extension is load-bearing: KWC theta rows (fig6-class
    seeded crystallites) and frozen bookkeeping rows both ride the
    extracted diagonal-block solves; the frozen-theta residual is
    EXACTLY zero -> the device inners short-circuit (the BiCGStab
    0/0-breakdown fix, measured relres nan without it).

TOLERANCE NOTE (the parity-gate lesson): parity locked at 1e-7,
~3 decades above the measured 1e-11..1e-14 class; iteration bounds
carry >= 2x headroom over measured maxima.

DEEP-QUENCH BOUNDARY (G4 carry-over, measured at (M, K)): the
pairwise form's chi12-class boundary transfers to the multiphase
system — see test_mpf_blockch_deepq_* and the module-level numbers in
the dev note.  A solvent-pinned IC (retained fractions summing to 1:
phi_s at the log regularization floor) puts the 1/phi_s CROSS
curvature in the dropped d12 block and stalls the two-factor form
into the exact-Schur escalation (measured: > 20 min at L5) — keep a
trace fraction, as every production config does."""
import numpy as np
import pytest

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.multiphase import MultiPhaseStepper

pytestmark = pytest.mark.tier3


def _dm(level, device, periodic="lat"):
    per = {"lat": (True, False), "all": (True, True),
           "none": None}[periodic]
    tree = build_uniform(level, dim=2, periodic=per)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    return DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2),
                                device)


class _RecCache(dict):
    """Logs every ('blockch_iters', key) record (per Newton solve)."""

    def __init__(self):
        super().__init__()
        self.iters = []

    def __setitem__(self, k, v):
        if isinstance(k, tuple) and k[0] == "blockch_iters":
            self.iters.append(v)
        super().__setitem__(k, v)


def _film(dm, linsolver, noise=5e-3, tstep="bdf1", assembly="host",
          block_sparse=False):
    """The S3b-class film production config (bench make_stepper
    family): (M=2, K=1), r14 + fastmode_n Vignes + ls_drop + film
    k_e = 0.1 + FDT noise, frozen theta."""
    chi_aa = np.zeros((3, 3))
    chi_aa[0, 1] = chi_aa[1, 0] = 1.0
    chi_aa[0, 2] = chi_aa[2, 0] = 0.7248
    chi_aa[1, 2] = chi_aa[2, 1] = 0.3
    chi_ca = np.zeros((3, 3))
    chi_ca[0, 1] = chi_ca[0, 2] = 1.6
    Dslf = np.array([[1e-2, 1e-3, 0.5],
                     [1e-4, 1e-4, 1e-2],
                     [1e-2, 1e-3, 1.0]])
    st = MultiPhaseStepper(
        dm, M=2, K=1, chi_aa=chi_aa, chi_ac=chi_ca.T.copy(),
        chi_ca=chi_ca, N=[65.4, 87.0, 1.0], mob="fastmode_n",
        D_self=Dslf, ls_drop=(1e-6, 0.97, 35.0), kappa=[2e-4] * 2,
        T=333.0, dt=1e-4, bulk="r14", b_reg=1e-3,
        dsig=[8.0], dh=[-40.0], Tm=[402.0], eps2=[4e-3],
        L_psi=[65.4], noise_psi=noise,
        noise_damp=(1e-2, 0.85, 15.0), clip_psi=False,
        newton_tol=1e-8, newton_max=50, linsolver=linsolver,
        line_search=True, noise_seed=11, tstep=tstep,
        film=dict(k_e=0.1), assembly=assembly,
        block_sparse=block_sparse)
    rng = np.random.default_rng(1011)
    nf = st.nfree
    icf = 0.10 + 0.01 * rng.standard_normal(nf)
    icp = 0.05 + 0.01 * rng.standard_normal(nf)
    st.set_initial([lambda x: icf, lambda x: icp],
                   [lambda x: np.zeros(len(x))],
                   [lambda x: np.zeros(len(x))])
    return st


def _deepq(dm, linsolver, chi12):
    """The G4 boundary config at (M=2, K=0): ternary FH quench with
    the M12 = -0.2 transport coupling (test_ternary_blockprecond's
    _ternary, multiphase layout)."""
    chi_aa = np.zeros((3, 3))
    chi_aa[0, 1] = chi_aa[1, 0] = chi12
    chi_aa[0, 2] = chi_aa[2, 0] = 0.8
    chi_aa[1, 2] = chi_aa[2, 1] = 0.8
    st = MultiPhaseStepper(
        dm, M=2, K=0, chi_aa=chi_aa, N=[1.0, 1.0, 1.0],
        onsager=[[1.0, -0.2], [-0.2, 1.0]], kappa=[8e-4] * 2,
        dt=0.005, bulk="p1", mob="const", linsolver=linsolver,
        newton_tol=1e-9, newton_max=50)
    rng = np.random.default_rng(4)
    nf = st.nfree
    i1 = 0.35 + 0.02 * rng.standard_normal(nf)
    i2 = 0.35 + 0.02 * rng.standard_normal(nf)
    st.set_initial([lambda x: i1, lambda x: i2])
    return st


def _parity(sa, sb):
    return max(np.abs(sa.x[i::sa.ndof] - sb.x[i::sb.ndof]).max()
               / max(np.abs(sa.x[i::sa.ndof]).max(), 1e-30)
               for i in range(sa.ndof))


def test_mpf_blockch_one_solve_parity(device):
    """Same linear system, splu vs the blockch route through
    stepper._solve — plus the VERIFY-CONSTRUCTED-OBJECTS check: the
    meta reached the solver (('blockch_iters', 'mpf') recorded)."""
    from scipy.sparse.linalg import splu
    d = _dm(5, device)
    s1 = _film(d, "splu")
    s2 = _film(d, "blockch")
    A, r = s1._debug_assemble(1e-4)
    s2._debug_assemble(1e-4)        # align RNG + freeze sigma/m_ref
    x_ref = splu(A.tocsc()).solve(r)
    x_bc = s2._solve(A, r)
    assert np.isfinite(x_bc).all()
    it = s2._solver_cache.get(("blockch_iters", "mpf"))
    assert it is not None, "blockch meta/iters never reached the solver"
    assert it[0] % 1000 <= 8 and it[0] < 1000, it
    rel = np.abs(x_bc - x_ref).max() / max(np.abs(x_ref).max(), 1e-30)
    res = np.linalg.norm(A @ x_bc - r) / np.linalg.norm(r)
    print(f"mpf blockch one-solve: parity {rel:.2e} relres {res:.2e} "
          f"iters {it}")
    assert rel < 1e-7 and res < 1e-7
    meta = s2._solver_cache[("blockch_meta", "mpf")]
    assert len(meta["pairs"]) == 2 and len(meta["ac"]) == 1
    assert all(p["m"] > 0 for p in meta["pairs"])


@pytest.mark.parametrize("solver", ["blockch", "blockch_dev"])
def test_mpf_blockch_film_march_parity(solver, device):
    """8-step film march, splu vs blockch (host + device inners):
    trajectory parity, identical ladders, outer bound, no fallback.
    Measured: parity 5.3e-14, outer <= 4, 0 rejects/fallbacks."""
    da, db = _dm(5, device), _dm(5, device)
    sa = _film(da, "splu")
    sb = _film(db, solver)
    rec = _RecCache()
    sb._solver_cache = rec
    for st in (sa, sb):
        st.march(t_end=1e9, max_steps=8)
    assert sa.n_reject == sb.n_reject == 0
    outs = [v[0] % 1000 for v in rec.iters]
    assert not any(v[0] >= 1000 for v in rec.iters), rec.iters
    par = _parity(sa, sb)
    print(f"mpf film[{solver}]: parity {par:.2e} outer max "
          f"{max(outs)} over {len(outs)} solves")
    assert par < 1e-7
    assert max(outs) <= 8


def test_mpf_blockch_device_march_parity(device):
    """B2: device assembly, splu (fixed-pattern pull) vs blockch_dev
    (zero-copy blockch_pairs_device).  Measured: parity 2.0e-14,
    outer <= 3, 0 fallbacks."""
    da, db = _dm(5, device), _dm(5, device)
    sa = _film(da, "splu", assembly="device")
    sb = _film(db, "blockch_dev", assembly="device")
    rec = _RecCache()
    sb._solver_cache = rec
    for st in (sa, sb):
        st.march(t_end=1e9, max_steps=8)
    assert sa.n_reject == sb.n_reject == 0
    assert rec.iters, "blockch_pairs_device never recorded iters"
    outs = [v[0] % 1000 for v in rec.iters]
    assert not any(v[0] >= 1000 for v in rec.iters)
    par = _parity(sa, sb)
    print(f"mpf device blockch: parity {par:.2e} outer max {max(outs)}")
    assert par < 1e-7
    assert max(outs) <= 8


def test_mpf_blockmask_exactness_and_parity(device):
    """B5-i: the kron(G, blockmask) pattern.  (i) one-assembly
    exactness vs the superset values (measured 3.4e-16, lock 1e-13,
    >= 100x); (ii) march parity blockch_dev-on-masked vs splu-on-
    superset; (iii) the mask is a strict subset with the predicted
    live count (22/36 at (M=2, K=1) fastmode_n frozen)."""
    d = _dm(5, device)
    su = _film(d, "splu", assembly="device")
    sm = _film(d, "splu", assembly="device", block_sparse=True)
    A_u, r_u = su._debug_assemble(1e-4)
    A_m, r_m = sm._debug_assemble(1e-4)
    mask = sm._block_mask()
    assert int(mask.sum()) == 22 and mask.size == 36
    assert A_m.nnz < A_u.nnz
    A_u.eliminate_zeros()
    A_m.eliminate_zeros()
    dA = abs(A_u - A_m).max() / max(abs(A_u).max(), 1e-30)
    assert dA < 1e-13, dA
    assert np.abs(r_u - r_m).max() < 1e-13
    # march parity on the masked pattern
    da, db = _dm(5, device), _dm(5, device)
    sa = _film(da, "splu", assembly="device")
    sb = _film(db, "blockch_dev", assembly="device",
               block_sparse=True)
    for st in (sa, sb):
        st.march(t_end=1e9, max_steps=6)
    par = _parity(sa, sb)
    print(f"mpf blockmask: dA {dA:.2e} parity {par:.2e} "
          f"nnz {A_m.nnz}/{A_u.nnz}")
    assert par < 1e-7


def test_mpf_blockch_kwc_one_solve_parity(device):
    """KWC theta rows (Ats torque + (p+pf)-weighted theta block) at a
    SEEDED-CRYSTALLITE state — the AC machinery with pk ~ 1 and theta
    gradients live (at theta = const the theta rows are downstream-
    decoupled with zero residual and the gate would be vacuous —
    measured).  fig6-class 16390 pars + 15 grains, one Newton system,
    splu vs blockch.  A full KWC march is a multi-minute affair even
    at L5 (panel-measured) — the one-solve gate keeps the row physics
    covered at test cost."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    import test_multiphase_s2 as S2
    from scipy.sparse.linalg import splu
    d = _dm(5, device, "all")
    pars = S2._fig6_pars(128e-9)
    a_nd = np.pi * S2.EPSGRAIN / (S2.U0 * 128e-9)

    def mk(solver):
        st = MultiPhaseStepper(d, dt=1e-3, newton_tol=1e-6,
                               newton_max=60, linsolver=solver,
                               b_reg=3e-4, line_search=True,
                               noise_seed=11,
                               alpha_th=[a_nd, a_nd],
                               beta_th=[0.1 * a_nd, 0.1 * a_nd],
                               L_th=[pars["L_psi"][1]] * 2,
                               kg_delta=0.2, **pars)
        rng = np.random.default_rng(4)
        nf = len(st.free_coords)
        psi_seed, ctr = S2._seed_discs(
            st.free_coords, 15, r0=4.5 / 128.0, w=1.5 / 128.0,
            seed=111, min_sep=3.5 * 4.5 / 128.0)
        th_seed = np.random.default_rng(5).uniform(0.0, 1.0, 15)
        dd = st.free_coords[:, None, :2] - ctr[None, :, :]
        dd -= np.round(dd)
        th = th_seed[np.argmin((dd ** 2).sum(axis=2), axis=1)]
        p1 = 0.45 * 0.98 * (1 - psi_seed) + 0.97 * psi_seed \
            + 1e-3 * rng.standard_normal(nf)
        p2 = 0.55 * 0.98 * (1 - psi_seed) + 0.01 * psi_seed \
            + 1e-3 * rng.standard_normal(nf)
        st.set_initial([lambda x: p1, lambda x: p2],
                       [lambda x: psi_seed,
                        lambda x: np.zeros(len(x))],
                       [lambda x: th, lambda x: th])
        return st

    assert mk("splu").theta_mode == "kwc"
    s1, s2 = mk("splu"), mk("blockch")
    A, r = s1._debug_assemble(1e-3)
    s2._debug_assemble(1e-3)
    x_ref = splu(A.tocsc()).solve(r)
    x_bc = s2._solve(A, r)
    assert np.isfinite(x_bc).all()
    it = s2._solver_cache[("blockch_iters", "mpf")]
    rel = np.abs(x_bc - x_ref).max() / max(np.abs(x_ref).max(), 1e-30)
    res = np.linalg.norm(A @ x_bc - r) / np.linalg.norm(r)
    print(f"mpf kwc one-solve: parity {rel:.2e} relres {res:.2e} "
          f"iters {it}")
    assert rel < 1e-6 and res < 1e-7
    assert it[0] % 1000 <= 66 and it[0] < 1000, it
    assert len(s2._solver_cache[("blockch_meta", "mpf")]["ac"]) == 2


def test_mpf_blockch_bdf2_march_parity(device):
    """A4b BDF2 rides the same meta (sigma = a/dt from _attempt_ctx):
    deterministic film march, splu vs blockch."""
    da, db = _dm(5, device), _dm(5, device)
    sa = _film(da, "splu", noise=0.0, tstep="bdf2")
    sb = _film(db, "blockch", noise=0.0, tstep="bdf2")
    rec = _RecCache()
    sb._solver_cache = rec
    for st in (sa, sb):
        st.march(t_end=1e9, max_steps=8)
    assert sa.n_reject == sb.n_reject == 0
    outs = [v[0] % 1000 for v in rec.iters]
    par = _parity(sa, sb)
    print(f"mpf bdf2: parity {par:.2e} outer max {max(outs)}")
    assert par < 1e-7
    assert max(outs) <= 8
    assert not any(v[0] >= 1000 for v in rec.iters)


def test_mpf_blockch_deepq_marginal_parity(device):
    """The G4 contract boundary at (M, K): chi12 = 3.0 (marginal
    spinodal, M12 = -0.2 cross transport active) — in contract."""
    da, db = _dm(5, device, "none"), _dm(5, device, "none")
    sa = _deepq(da, "splu", 3.0)
    sb = _deepq(db, "blockch", 3.0)
    rec = _RecCache()
    sb._solver_cache = rec
    for st in (sa, sb):
        st.march(t_end=1e9, max_steps=6)
    par = _parity(sa, sb)
    outs = [v[0] % 1000 for v in rec.iters]
    print(f"mpf deepq3: parity {par:.2e} outer max {max(outs)} "
          f"rej {sa.n_reject}/{sb.n_reject}")
    assert sa.n_reject == sb.n_reject
    assert par < 1e-7
    assert max(outs) <= 8
    assert not any(v[0] >= 1000 for v in rec.iters)


def test_mpf_blockch_deepq_boundary_ladder(device):
    """HONEST BOUNDARY PIN (G4 carry-over): chi12 = 6.0 — the
    d12-driven deep quench where per-pair preconditioning is
    structurally blind.  The (M, K) wiring must keep the march ALIVE
    through the reject ladder (failed solves signal divergence as
    nan, dt collapses — the cudss contract), not crash.  Measured
    behavior recorded in the dev note; this gate asserts the CONTRACT
    (march survives, rejects observed, dt collapsed) rather than a
    parity that is ill-posed here."""
    d = _dm(5, device, "none")
    st = _deepq(d, "blockch", 6.0)
    st._solver_cache = _RecCache()
    st.march(t_end=1e9, max_steps=6, dt_min=1e-7)
    print(f"mpf deepq6: t {st.t:.3e} rej {st.n_reject} dt {st.dt:.2e}")
    assert st.t > 0.0, "no step ever accepted"
    assert st.n_reject > 0 or st.dt <= 0.005, \
        "expected the boundary to engage the ladder"
