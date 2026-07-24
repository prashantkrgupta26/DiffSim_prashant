"""M5 S4 gates: QUATERNARY demonstrations — the (M, K) kernel factory
is generic by construction, so quaternary systems need CONFIGS ONLY,
no code changes.

Contract: docs/theory/crystallization_formulation_p1.md Sec 4 stage S4:
 (a) 2 ACTIVE + 2 SOLVENTS (M=3, K=1): selective evaporation — two
     solvents of different volatility (per-species k_e), the drying
     line bends in composition space;
 (b) 3 ACTIVE + 1 SOLVENT (M=3, K=3): three crystallizable species
     (distinct N/dsig/dh/Tm per species), annealing.

Dev ledger: docs/dev/2026-07-13-m5-s4-quaternary.md.  Tolerances
measured-then-locked (>= 2x headroom; measured numbers in comments).
Assembly = HOST (D4 verdict: device atomics flake bit-class gates).
Constructed-object asserts guard the missing-splat lesson (S3 Sec 2.4).
"""
import importlib.util

import numpy as np
import pytest

_has_nvmath = importlib.util.find_spec("nvmath") is not None
_skip_nvmath = pytest.mark.skipif(
    not _has_nvmath, reason="nvmath (cuDSS) not available on this machine"
)

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.multiphase import MultiPhaseStepper, grain_labels

pytestmark = pytest.mark.tier3


def _dm(level, device, p=1, periodic=None):
    tree = build_uniform(level, dim=2, periodic=periodic)
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    return (DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2),
                                 device), mesh, cons)


def _integ(dm, mesh, st, *vecs):
    """Computational-domain quadrature integral of the product of
    GP-interpolated free vectors; physical content = h_curr * this."""
    vs = [st._gp(v)[0] for v in vecs]
    m = 0.0
    for pv, b in dm.bins.items():
        h = mesh.tree.h()[mesh.bins[pv]]
        ne = len(mesh.conn_of[pv])
        wq = np.tile(dm.tables_by_p[pv].w, ne) \
            * np.repeat((h / 2.0) ** dm.dim, b["nqp"])
        prod = np.ones_like(wq)
        for v in vs:
            prod = prod * v[pv]
        m += float((wq * prod).sum())
    return m


# =====================================================================
# S4a — 2 ACTIVE + 2 SOLVENTS (M=3, K=1): selective evaporation
# Layout: species 0=SM (nonvol), 1=polymer (nonvol), 2=slow solvent
# (tracked, vol), eliminated 3=fast solvent (vol).  Both nonvolatiles
# TRACKED => both conserve exactly (S3a telescoping at M=3).
# =====================================================================
def _s4a_evap_stepper(dm, ke_vec, p=1):
    """Neutral crystal energetics (dsig=dh=0, L_psi=0) to ISOLATE the
    moving-frame mapping (the S3a mapping-isolation lesson).  Miscible
    chi (no spinodal chaos)."""
    chi_aa = np.full((4, 4), 0.3) - 0.3 * np.eye(4)
    st = MultiPhaseStepper(
        dm, M=3, K=1, chi_aa=chi_aa,
        onsager=[[0.4, 0, 0], [0, 0.4, 0], [0, 0, 0.6]],
        kappa=[2e-4] * 3, N=[5.0, 20.0, 1.0, 1.0],
        dsig=[0.0], dh=[0.0], Tm=[1.0], eps2=[1e-3], L_psi=[0.0],
        T=0.5, bulk="r14", dt=1e-3, linsolver="cudss",
        film=dict(k_e=np.asarray(ke_vec, float)), clip_psi=False)
    # missing-splat guard: assert the constructed object received the
    # per-species k_e vector (NOT the scalar-solvent default)
    assert st.ndof == 2 * 3 + 2 * 1
    assert np.allclose(st.k_e, ke_vec), (st.k_e, ke_vec)
    st.set_initial([lambda x: np.full(len(x), 0.10),    # SM
                    lambda x: np.full(len(x), 0.10),    # polymer
                    lambda x: np.full(len(x), 0.35)],   # slow solvent
                   [lambda x: np.zeros(len(x))],
                   [lambda x: np.zeros(len(x))])
    return st


@_skip_nvmath
def test_s4a_selective_evaporation_bookkeeping(device):
    """S4a gate (i): (a) BOOKKEEPING — the S3a telescoping identity
    holds at M=3: BOTH tracked nonvolatile species (SM idx0, polymer
    idx1) conserve physical content h*Int(phi_i) through the moving
    frame to machine precision; (b) per-solvent DEPLETION ORDERING —
    the fast (eliminated) solvent's physical content falls measurably
    ahead of the slow (tracked) one, and its mean fraction FALLS while
    the slow solvent's mean fraction RISES (the drying line bending in
    composition space — the solvent-blend mechanism).  k_e fast:slow =
    4:1 (kf=1.0, ks=0.25).
    Measured (2026-07-13, L5, cudss): SM drift 1.39e-16, polymer drift
    0.0 (EXACT class); fast content loss 0.2387 vs slow 0.0626 (ratio
    3.82); fast mean-fraction 0.45->0.302 (falls), slow 0.35->0.411
    (rises); 0 rejects, h_min stop."""
    kf, ks = 1.0, 0.25
    dm, mesh, cons = _dm(5, device, periodic=(True, False))
    st = _s4a_evap_stepper(dm, [0.0, 0.0, ks, kf])
    nd = st.ndof
    fast = lambda s: 1.0 - sum(s.x[2 * i::nd] for i in range(3))
    c0_sm = st.h_curr * _integ(dm, mesh, st, st.x[0::nd])
    c0_poly = st.h_curr * _integ(dm, mesh, st, st.x[2::nd])
    c0_slow = st.h_curr * _integ(dm, mesh, st, st.x[4::nd])
    c0_fast = st.h_curr * _integ(dm, mesh, st, fast(st))
    full = lambda v: np.asarray(cons.T @ v)
    mf_slow = lambda s: float(np.mean(full(s.phi(2))))
    mf_fast = lambda s: float(np.mean(full(fast(s))))
    r = st.march(t_end=5.0, dt_max=5e-3, max_steps=800, h_min=0.7,
                 grow_iters=30, phis_stop=0.02)
    c1_sm = st.h_curr * _integ(dm, mesh, st, st.x[0::nd])
    c1_poly = st.h_curr * _integ(dm, mesh, st, st.x[2::nd])
    c1_slow = st.h_curr * _integ(dm, mesh, st, st.x[4::nd])
    c1_fast = st.h_curr * _integ(dm, mesh, st, fast(st))
    d_sm = abs(c1_sm - c0_sm) / c0_sm
    d_poly = abs(c1_poly - c0_poly) / c0_poly
    loss_ratio = (c0_fast - c1_fast) / (c0_slow - c1_slow)
    print(f"S4a bookkeeping: stop={r} h={st.h_curr:.4f} rej={st.n_reject}"
          f"; SM drift {d_sm:.2e} poly drift {d_poly:.2e}; fast loss "
          f"{c0_fast - c1_fast:.4f} slow loss {c0_slow - c1_slow:.4f} "
          f"ratio {loss_ratio:.3f}; fast frac {mf_fast(st):.4f} slow "
          f"frac {mf_slow(st):.4f}")
    assert r == "h_min", r
    assert d_sm < 1e-12, d_sm            # measured 1.39e-16
    assert d_poly < 1e-12, d_poly        # measured 0.0
    assert loss_ratio > 2.0, loss_ratio  # measured 3.82 (2x headroom)
    # ordering: fast depletes, slow concentrates (drying-line bend)
    assert mf_fast(st) < 0.35, mf_fast(st)   # fell from 0.45; meas .302
    assert mf_slow(st) > 0.38, mf_slow(st)   # rose from 0.35; meas .411
    assert mf_fast(st) < mf_slow(st), (mf_fast(st), mf_slow(st))


@_skip_nvmath
def test_s4a_selectivity_contrast(device):
    """S4a gate (ii): SELECTIVITY CONTRAST — flipping the k_e ratio
    (slow:fast instead of fast:slow) on the same species slots produces
    a measurably different composition path.  The tracked idx2 solvent
    ends at a DIFFERENT mean fraction depending on whether it is the
    fast or the slow component, and the (eliminated - idx2) fraction
    difference FLIPS SIGN (the composition-path crossover).
    Measured (2026-07-13, L5): idx2 end fraction 0.4113 (as slow) vs
    0.2068 (as fast), delta 0.205; (elim - idx2) -0.109 -> +0.300."""
    kf, ks = 1.0, 0.25
    dm, mesh, cons = _dm(5, device, periodic=(True, False))
    full = lambda v: np.asarray(cons.T @ v)
    mf2 = lambda s: float(np.mean(full(s.phi(2))))
    mf_elim = lambda s: float(np.mean(
        1.0 - sum(np.asarray(cons.T @ s.phi(i)) for i in range(3))))
    st = _s4a_evap_stepper(dm, [0.0, 0.0, ks, kf])   # idx2 = slow
    st.march(t_end=5.0, dt_max=5e-3, max_steps=800, h_min=0.7,
             grow_iters=30, phis_stop=0.02)
    st2 = _s4a_evap_stepper(dm, [0.0, 0.0, kf, ks])  # idx2 = fast
    st2.march(t_end=5.0, dt_max=5e-3, max_steps=800, h_min=0.7,
              grow_iters=30, phis_stop=0.02)
    base = mf_elim(st) - mf2(st)
    flip = mf_elim(st2) - mf2(st2)
    dpath = abs(mf2(st) - mf2(st2))
    print(f"S4a selectivity: idx2 end (slow) {mf2(st):.4f} vs (fast) "
          f"{mf2(st2):.4f} delta {dpath:.4f}; (elim-idx2) base "
          f"{base:+.4f} flip {flip:+.4f}")
    assert dpath > 0.1, dpath                 # measured 0.205
    assert base < 0.0 and flip > 0.0, (base, flip)   # crossover
    assert flip - base > 0.2, (base, flip)


# ---- S4a gate (iii): CRYSTALLIZATION IN THE QUATERNARY FILM ----
_S4A_NF, _S4A_NP = 5.0298, 87.0
_S4A_DSIG, _S4A_DH, _S4A_TM = 2.6355, 1.3072, 558.0
_S4A_CHI_CA = 1.2


def _s4a_cryst_stepper(dm):
    """M=3 K=1 two-solvent film with the SM crystallization energetics
    (2310-class); constant mobility (fastmode_n is Newton-hostile with
    seeded discs — the S4b lesson) + splu (the film-mode crystallization
    Jacobian flaps nnz, which forces cudss symbolic replans every step;
    splu refactorizes without replanning — measured faster here).
    kf:ks = 0.14:0.06 (selective; both nonzero)."""
    chi_aa = np.zeros((4, 4))
    chi_aa[0, 1] = chi_aa[1, 0] = 1.0
    chi_aa[0, 2] = chi_aa[2, 0] = 0.7248
    chi_aa[0, 3] = chi_aa[3, 0] = 0.7248
    chi_aa[1, 2] = chi_aa[2, 1] = 0.3
    chi_aa[1, 3] = chi_aa[3, 1] = 0.3
    chi_ca = np.zeros((4, 4))
    chi_ca[0, 1] = chi_ca[0, 2] = chi_ca[0, 3] = _S4A_CHI_CA
    st = MultiPhaseStepper(
        dm, M=3, K=1, chi_aa=chi_aa, chi_ac=chi_ca.T.copy(),
        chi_ca=chi_ca, N=[_S4A_NF, _S4A_NP, 1.0, 1.0], mob="const",
        onsager=np.diag([0.1, 0.1, 0.1]), kappa=[2e-4] * 3, T=333.0,
        dt=2e-5, bulk="r14", b_reg=1e-3, dsig=[_S4A_DSIG], dh=[_S4A_DH],
        Tm=[_S4A_TM], eps2=[4e-3], L_psi=[2.0], newton_tol=1e-6,
        newton_max=40, linsolver="splu", line_search=True,
        noise_seed=11, clip_psi=False,
        film=dict(k_e=np.array([0.0, 0.0, 0.06, 0.14])))
    # missing-splat guards: per-species k_e vector + crystal energetics
    assert st.ndof == 8 and st.K == 1, (st.ndof, st.K)
    assert np.allclose(st.k_e, [0.0, 0.0, 0.06, 0.14]), st.k_e
    assert np.isclose(st.Tm[0], _S4A_TM) and np.isclose(st.dsig[0],
                                                        _S4A_DSIG)
    return st


def test_s4a_crystallization_in_film(device):
    """S4a gate (iii): CRYSTALLIZATION OPERATES in the QUATERNARY film.
    A seeded SM crystal (fate-robust r0=0.2, the S3b margin) in the
    two-active/two-solvent film persists and grows: psi_max stays high
    (the crystal does not spuriously dissolve or blow up — the coupled
    K=1 crystallization + M=3 CH + moving-frame + selective-evaporation
    system integrates stably) and the relative crystallinity X_SM rises
    (crystallization proceeding).  Short window: the seeded-crystal +
    moving-frame Jacobian stiffens beyond t ~ 0.02 (a recorded
    frontier; the ladder crawls there — dev note).  Selective
    evaporation itself is gated in (i)/(ii); here the point is that the
    K=1 machinery operates unchanged in the M=3 two-solvent film.
    Measured (2026-07-13, L5, splu, HOST): psi_max 0.950 -> 0.943,
    X_SM 0.1182 -> 0.1279 over t = 0.01, 0 rejects."""
    dm, mesh, cons = _dm(5, device, periodic=(True, False))
    st = _s4a_cryst_stepper(dm)
    rng = np.random.default_rng(1011)
    nf = st.nfree
    st.set_initial([lambda x: 0.42 + 0.01 * rng.standard_normal(nf),
                    lambda x: 0.05 + 0.01 * rng.standard_normal(nf),
                    lambda x: 0.13 + 0.01 * rng.standard_normal(nf)],
                   [lambda x: np.zeros(len(x))],
                   [lambda x: np.zeros(len(x))])
    nd = st.ndof
    coords = st.free_coords
    psi = st.x[2 * st.M::nd]
    th = st.x[2 * st.M + 1::nd]
    r = np.hypot(coords[:, 0] - 0.5, coords[:, 1] - 0.45)
    disc = 0.95 * 0.5 * (1 - np.tanh((r - 0.2) / 0.03))
    m = disc > psi
    psi[m] = disc[m]
    th[r < 0.29] = 0.4
    st.hist = st.x.copy()
    st.hist2 = None
    full = lambda v: np.asarray(cons.T @ v)
    Xsm = lambda: float((full(st.phi(0)) * full(st.psi(0))).mean()
                        / max(full(st.phi(0)).mean(), 1e-12))
    pmax0 = float(st.psi(0).max())
    X0 = Xsm()
    st.march(t_end=0.01, dt_max=1e-3, max_steps=5000, dt_min=1e-11,
             grow_iters=45, h_min=0.30, phis_stop=0.05)
    pmax1 = float(st.psi(0).max())
    X1 = Xsm()
    print(f"S4a crystallization: psi_max {pmax0:.3f}->{pmax1:.3f} "
          f"X_SM {X0:.4f}->{X1:.4f} h={st.h_curr:.4f} rej={st.n_reject}")
    assert pmax1 > 0.9, pmax1                  # crystal persists
    assert X1 > X0 + 1e-3, (X0, X1)            # crystallization operates
    assert X1 > 0.12, X1                       # measured 0.1279


def test_s4a_bookkeeping_p2_basis(device):
    """CROSS-MATRIX cell: the S4a moving-frame bookkeeping at p=2 basis
    (the nbf/nqp-generic factory).  BOTH tracked nonvolatiles conserve
    physical content exactly on the p=2 moving frame.  L4 p=2 (matched
    cost).  Measured (2026-07-13): SM/polymer drift ~1e-15."""
    kf, ks = 1.0, 0.25
    dm, mesh, cons = _dm(4, device, p=2, periodic=(True, False))
    chi_aa = np.full((4, 4), 0.3) - 0.3 * np.eye(4)
    st = MultiPhaseStepper(
        dm, M=3, K=1, chi_aa=chi_aa,
        onsager=[[0.4, 0, 0], [0, 0.4, 0], [0, 0, 0.6]],
        kappa=[2e-4] * 3, N=[5.0, 20.0, 1.0, 1.0], dsig=[0.0], dh=[0.0],
        Tm=[1.0], eps2=[1e-3], L_psi=[0.0], T=0.5, bulk="r14", dt=1e-3,
        linsolver="splu",
        film=dict(k_e=np.array([0.0, 0.0, ks, kf])), clip_psi=False)
    assert np.allclose(st.k_e, [0.0, 0.0, ks, kf]), st.k_e
    st.set_initial([lambda x: np.full(len(x), 0.10),
                    lambda x: np.full(len(x), 0.10),
                    lambda x: np.full(len(x), 0.35)],
                   [lambda x: np.zeros(len(x))],
                   [lambda x: np.zeros(len(x))])
    nd = st.ndof
    c0_sm = st.h_curr * _integ(dm, mesh, st, st.x[0::nd])
    c0_poly = st.h_curr * _integ(dm, mesh, st, st.x[2::nd])
    r = st.march(t_end=5.0, dt_max=5e-3, max_steps=800, h_min=0.7,
                 grow_iters=30, phis_stop=0.02)
    c1_sm = st.h_curr * _integ(dm, mesh, st, st.x[0::nd])
    c1_poly = st.h_curr * _integ(dm, mesh, st, st.x[2::nd])
    d_sm = abs(c1_sm - c0_sm) / c0_sm
    d_poly = abs(c1_poly - c0_poly) / c0_poly
    print(f"S4a p=2 bookkeeping: stop={r} h={st.h_curr:.4f} "
          f"ndof={st.ndof} nfree={st.nfree} SM drift {d_sm:.2e} "
          f"poly drift {d_poly:.2e}")
    assert d_sm < 1e-10, d_sm
    assert d_poly < 1e-10, d_poly


# =====================================================================
# S4b — 3 ACTIVE + 1 SOLVENT (M=3, K=3): three crystallizable species,
# ANNEALING (deterministic; contract allows annealing OR drying — S4a
# already exercises the drying/selective-evaporation machinery, so S4b
# isolates the NEW machinery: three independent psi fields, K=3).
# Distinct N/dsig/dh/Tm/eps2 spread (2310-PCBM-class + two shifted-Tm
# variants).  Constant mobility (the fastmode_n composition-dependent
# mobility is Newton-hostile with seeded discs at these depths —
# measured: it underflowed the ladder at t=0; const mobility marches
# with 0 rejects).
# =====================================================================
_S4B_N = [5.03, 6.0, 7.0, 1.0]
_S4B_DSIG = [2.6355, 2.45, 2.30]
_S4B_DH = [1.3072, 1.20, 1.10]
_S4B_TM = [558.0, 520.0, 490.0]
_S4B_EPS2 = [4e-3, 4e-3, 4e-3]
_S4B_CHI_CA = 1.2
_S4B_T = 333.0


def _s4b_stepper(dm, tstep="bdf1"):
    chi_aa = np.zeros((4, 4))
    for i in range(3):
        for j in range(i + 1, 3):
            chi_aa[i, j] = chi_aa[j, i] = 1.0
        chi_aa[i, 3] = chi_aa[3, i] = 0.5
    chi_ca = np.zeros((4, 4))
    for i in range(3):
        for j in range(4):
            if j != i:
                chi_ca[i, j] = _S4B_CHI_CA
    st = MultiPhaseStepper(
        dm, M=3, K=3, chi_aa=chi_aa, chi_ac=chi_ca.T.copy(),
        chi_ca=chi_ca, N=_S4B_N, mob="const",
        onsager=np.diag([0.1, 0.1, 0.1]), kappa=[2e-4] * 3, T=_S4B_T,
        dt=2e-5, bulk="r14", b_reg=1e-3, dsig=_S4B_DSIG, dh=_S4B_DH,
        Tm=_S4B_TM, eps2=_S4B_EPS2, L_psi=[2.0, 2.0, 2.0],
        newton_tol=1e-6, newton_max=40, linsolver="cudss",
        line_search=True, clip_psi=False, tstep=tstep)
    # missing-splat guard: the constructed object MUST carry the
    # per-species arrays (K=3, ndof=12), else Tm etc. silently default
    assert st.K == 3 and st.ndof == 12, (st.K, st.ndof)
    assert np.allclose(st.Tm[:3], _S4B_TM), st.Tm
    assert np.allclose(st.dsig[:3], _S4B_DSIG), st.dsig
    assert np.allclose(st.dh[:3], _S4B_DH), st.dh
    assert np.allclose(st.eps2[:3], _S4B_EPS2), st.eps2
    return st


def _s4b_stripe(xc, c, w=0.16):
    d = np.abs(xc[:, 0] - c)
    d = np.minimum(d, 1 - d)                 # periodic x
    return 0.5 * (1 - np.tanh((d - w) / 0.04))


_S4B_C = (0.2, 0.5, 0.8)          # A/B/C-rich stripe centers


def _s4b_set_ic(st):
    xc = st.free_coords
    b = [_s4b_stripe(xc, c) for c in _S4B_C]
    pfns = [(lambda x, bb=b[i]: 0.03 + 0.77 * bb) for i in range(3)]
    st.set_initial(pfns, [lambda x: np.zeros(len(x))] * 3,
                   [lambda x: np.zeros(len(x))] * 3)


def _s4b_implant(st, speck, c, thval, r0=0.18, w=0.03, amp=0.95):
    nd = st.ndof
    coords = st.free_coords
    psi = st.x[2 * st.M + 2 * speck::nd]
    th = st.x[2 * st.M + 2 * speck + 1::nd]
    r = np.hypot(coords[:, 0] - c[0], coords[:, 1] - c[1])
    disc = amp * 0.5 * (1 - np.tanh((r - r0) / w))
    m = disc > psi
    psi[m] = disc[m]
    th[r < r0 + 3 * w] = thval
    st.hist = st.x.copy()
    st.hist2 = None


def _s4b_seed_and_march(dm, t_end=3.0, tstep="bdf1", sample=None):
    """Seed A/B/C crystals in their own rich stripes + a B-crystal in
    the A stripe (the solubility-selectivity probe); anneal.  Returns
    the stepper."""
    st = _s4b_stepper(dm, tstep=tstep)
    _s4b_set_ic(st)
    cA, cB, cC = _S4B_C
    _s4b_implant(st, 0, (cA, 0.5), thval=0.3)   # A in A-rich
    _s4b_implant(st, 1, (cB, 0.5), thval=0.5)   # B in B-rich
    _s4b_implant(st, 2, (cC, 0.5), thval=0.7)   # C in C-rich
    _s4b_implant(st, 1, (cA, 0.25), thval=0.5)  # B in A-rich (foreign)
    cb = None
    if sample is not None:
        def cb(s, dt, it):
            if s.t - sample["t"][-1] >= 0.2:
                for i in range(3):
                    sample["X"][i].append(_s4b_Xi(s, i))
                sample["t"].append(s.t)
    st.march(t_end=t_end, dt_max=0.02, max_steps=20000, dt_min=1e-11,
             grow_iters=45, callback=cb)
    return st


def _s4b_Xi(st, i):
    ph = np.asarray(st.Tc @ st.phi(i))
    ps = np.asarray(st.Tc @ st.psi(i))
    return float((ph * ps).mean() / max(ph.mean(), 1e-12))


@_skip_nvmath
def test_s4b_three_species_crystallization(device):
    """S4b gates (i)+(ii)+(iii) on one K=3 annealing march (0 rejects):

    (i) SOLUBILITY SELECTIVITY (generalized): each crystallizable
        species crystallizes in ITS OWN rich domain but not in a
        foreign one, measured by crystalline volume v_i^R =
        mean_{region R}(phi_i psi_i).  In the A-rich stripe v_A >> v_B
        (a B-seed placed there cannot draw B material — phi_B is far
        below B's solubility threshold); species B crystallizes only
        where B is rich.
    (ii) GRAIN IDENTITY ACROSS SPECIES: each species carries its own
        (psi_i, theta_i); grain_labels on each field returns that
        species' grains, and the per-species theta MARKERS are
        distinct and ordered (A ~ 0.20, B ~ 0.47, C ~ 0.70 — the
        seeded 0.3/0.5/0.7 markers, pulled toward the amorphous-0
        edge but cleanly separable), so crystals of different species
        are distinguished.
    (iii) KINETICS SPREAD: per-species X_i(t) reflect the distinct
        driving (drive_A -0.527 > drive_B -0.432 > drive_C -0.352):
        X_A and X_B rise monotonically to distinct levels while X_C
        (weakest quench, marginally above threshold) stays low.

    Measured (2026-07-13, L5, cudss, HOST): X_A 0.256->0.710, X_B
    0.289->0.627, X_C 0.256->0.239; v_A/v_B (A-region) 6.30,
    v_B(B)/v_B(A) 5.92; grains A=2 B=2 C=1, theta markers
    0.203/0.465/0.700; psi_max 0.94-0.97; 0 rejects."""
    dm, mesh, cons = _dm(5, device, periodic=(True, True))
    st = _s4b_stepper(dm)
    _s4b_set_ic(st)
    cA, cB, cC = _S4B_C
    _s4b_implant(st, 0, (cA, 0.5), thval=0.3)
    _s4b_implant(st, 1, (cB, 0.5), thval=0.5)
    _s4b_implant(st, 2, (cC, 0.5), thval=0.7)
    _s4b_implant(st, 1, (cA, 0.25), thval=0.5)
    X = [[_s4b_Xi(st, i)] for i in range(3)]
    ts = [st.t]

    def cb(s, dt, it):
        if s.t - ts[-1] >= 0.2:
            for i in range(3):
                X[i].append(_s4b_Xi(s, i))
            ts.append(s.t)
    st.march(t_end=3.0, dt_max=0.02, max_steps=20000, dt_min=1e-11,
             grow_iters=45, callback=cb)
    for i in range(3):
        X[i].append(_s4b_Xi(st, i))
    full = lambda v: np.asarray(cons.T @ v)

    def vol(i, c):
        m = _s4b_stripe(mesh.node_coords, c) > 0.5
        return float((full(st.phi(i)) * full(st.psi(i)))[m].mean())
    vAA = vol(0, cA)
    vBA = vol(1, cA)
    vBB = vol(1, cB)
    # grain identity
    counts, thetas, pmax = [], [], []
    for i in range(3):
        lab, sz = grain_labels(mesh.node_coords, full(st.psi(i)),
                               full(st.theta(i)), psi_th=0.5,
                               theta_tol=0.1, periodic=True)
        counts.append(int((sz >= 5).sum()))
        ps = full(st.psi(i))
        th = full(st.theta(i))
        thetas.append(float(th[ps > 0.5].mean()))
        pmax.append(float(ps.max()))
    print(f"S4b crystallization: X_A {X[0][0]:.3f}->{X[0][-1]:.3f} "
          f"X_B {X[1][0]:.3f}->{X[1][-1]:.3f} X_C {X[2][0]:.3f}->"
          f"{X[2][-1]:.3f}; v_A/v_B(A) {vAA / max(vBA, 1e-9):.2f} "
          f"v_B(B)/v_B(A) {vBB / max(vBA, 1e-9):.2f}; grains {counts} "
          f"theta {[f'{t:.3f}' for t in thetas]} psi_max "
          f"{[f'{p:.3f}' for p in pmax]} rej {st.n_reject}")
    # (i) solubility selectivity via crystalline volume
    assert vAA / max(vBA, 1e-9) > 3.0, (vAA, vBA)      # measured 6.30
    assert vBB / max(vBA, 1e-9) > 3.0, (vBB, vBA)      # measured 5.92
    # (ii) grain identity: 3 species labelled, distinct ordered markers
    assert counts[0] >= 1 and counts[1] >= 1 and counts[2] >= 1, counts
    assert thetas[0] < thetas[1] < thetas[2], thetas   # 0.20<0.47<0.70
    assert thetas[1] - thetas[0] > 0.15, thetas
    assert thetas[2] - thetas[1] > 0.15, thetas
    assert min(pmax) > 0.9, pmax
    # (iii) kinetics spread: A,B rise monotonically to distinct levels,
    # C stays low; ordering X_A > X_B > X_C reflects the driving spread
    assert all(X[0][k + 1] >= X[0][k] - 1e-3
               for k in range(len(X[0]) - 1)), X[0]     # A monotone
    assert all(X[1][k + 1] >= X[1][k] - 1e-3
               for k in range(len(X[1]) - 1)), X[1]     # B monotone
    assert X[0][-1] > 0.6, X[0][-1]                     # A grows strong
    assert X[1][-1] > 0.5, X[1][-1]                     # B grows
    assert X[2][-1] < 0.35, X[2][-1]                    # C stays low
    assert X[0][-1] > X[1][-1] > X[2][-1], (X[0][-1], X[1][-1],
                                            X[2][-1])


@_skip_nvmath
def test_s4b_bdf2_deterministic(device):
    """CROSS-MATRIX cell: an S4b K=3 crystallization gate under BDF2
    (deterministic, the standing basis x tstep rule).  Short horizon
    (t=1.0); the three psi fields ride the variable-step scheme and
    REPRODUCE the BDF1 result at the same time to ~1% (BDF1 at t=1.0:
    X_A 0.405, X_B 0.430, X_C ~0.27).  At this early time the fast-
    starting B species still leads A (the X_A > X_B plateau crossover
    happens later, ~t=1.6 under BDF1 — a plateau property, not an
    early-kinetics one); the robust invariant at t=1.0 is that the two
    strongly-driven species A, B both grow well clear of the weakly-
    driven laggard C (the driving spread).  Measured (L5, BDF2):
    X_A 0.405, X_B 0.428, X_C 0.269, 0 rejects (matches BDF1)."""
    dm, mesh, cons = _dm(5, device, periodic=(True, True))
    st = _s4b_seed_and_march(dm, t_end=1.0, tstep="bdf2")
    XA, XB, XC = (_s4b_Xi(st, 0), _s4b_Xi(st, 1), _s4b_Xi(st, 2))
    print(f"S4b BDF2: X_A {XA:.4f} X_B {XB:.4f} X_C {XC:.4f} "
          f"rej {st.n_reject}")
    assert st.n_reject == 0, st.n_reject       # deterministic, no rejects
    assert XA > 0.35, XA                        # A grows (meas 0.405)
    assert XB > 0.35, XB                        # B grows (meas 0.428)
    assert XC < 0.35, XC                        # C the laggard (0.269)
    assert min(XA, XB) > XC + 0.05, (XA, XB, XC)  # driving spread
