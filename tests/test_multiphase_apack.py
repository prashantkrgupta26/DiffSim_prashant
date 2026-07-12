"""M5 (a)-pack gates: A1 temperature field, A2 substrate surface
energy, A3 anisotropic crystal growth (diffsim.physics.multiphase;
formulation contract docs/theory/flow_film_formulation_p2.md Track A,
measured ledger docs/dev/2026-07-12-m5-apack.md).  Tolerances
measured-then-locked (>= 2x headroom; measured numbers in the
assertion comments)."""
import numpy as np
import pytest

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.multiphase import (MultiPhaseStepper,
                                        TemperatureField)

pytestmark = pytest.mark.tier3


def _dm(level, device, p=1):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    return (DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2),
                                 device), mesh, cons)


def _disc(c, r0, w):
    return lambda x: 0.5 * (1.0 - np.tanh(
        (np.sqrt((x[:, 0] - c[0]) ** 2
                 + (x[:, 1] - c[1]) ** 2) - r0) / w))


def _bdry(coords):
    b = np.zeros(len(coords), bool)
    for cc in range(2):
        b |= (np.abs(coords[:, cc]) < 1e-12) | \
             (np.abs(coords[:, cc] - 1) < 1e-12)
    return b


# ---------------------------------------------------------------------
# A1(i) — T-field MMS (segregated TemperatureField, BDF1)
# ---------------------------------------------------------------------
def test_a1_tfield_mms(device):
    """Manufactured T* = 0.5 + 0.3 cos(pi x) cos(pi y) e^{-t} on the
    scalar diffusion equation rho_cp T_t = div(k_th grad T) + s,
    Dirichlet everywhere from T*; dt = 1e-3 x 4 steps so the spatial
    error dominates.  P1 elements: expect order ~2."""
    rho_cp, k_th, dt = 2.0, 0.5, 1e-3
    pi = np.pi
    Ts = lambda x, t: 0.5 + 0.3 * np.cos(pi * x[:, 0]) \
        * np.cos(pi * x[:, 1]) * np.exp(-t)
    # s = rho_cp T_t - k_th lap T = (-rho_cp + 2 k_th pi^2)(T* - 0.5)
    src = lambda x, t: (-rho_cp + 2.0 * k_th * pi ** 2) \
        * (Ts(x, t) - 0.5)
    errs = []
    for lv in (4, 5):
        dm, mesh, cons = _dm(lv, device)
        coords = mesh.node_coords[cons.free_nodes]
        tf = TemperatureField(dm, rho_cp=rho_cp, k_th=k_th, src_fn=src,
                              dirichlet=np.where(_bdry(coords))[0],
                              g_fn=Ts)
        Tn = Ts(coords, 0.0)
        t = 0.0
        for _ in range(4):
            Tn = tf.attempt(Tn, dt, t + dt)
            t += dt
        from diffsim.physics.poisson import l2_error
        errs.append(l2_error(dm, np.asarray(cons.T @ Tn),
                             lambda x: Ts(x, t)))
    order = np.log2(errs[0] / errs[1])
    print(f"A1 T-field MMS: errs {[f'{e:.2e}' for e in errs]} "
          f"order {order:.2f}")
    assert order > 1.8, (errs, order)


# ---------------------------------------------------------------------
# A1(ii) — ISOTHERMAL REGRESSION PARITY (field == scalar, distribution)
# ---------------------------------------------------------------------
def _a1_stepper(dm, T_mode, jitter, T=333.0):
    """S1-class binary crystallization (the s1b PCBM-class r14 config):
    seeded psi disc in an undercooled uniform blend, deterministic
    (no noise), FIXED dt (attempt counts stay aligned between modes —
    ladder fate flips are the house FP-fate lesson, gated in S0)."""
    chi_aa = np.array([[0.0, 0.7248], [0.7248, 0.0]])
    chi_ca = np.array([[0.0, 1.0836], [0.0, 0.0]])
    kw = {}
    if T_mode == "field":
        kw = dict(T_mode="field", T_field=dict(T0=T))
    st = MultiPhaseStepper(
        dm, M=1, K=1, chi_aa=chi_aa, chi_ac=chi_ca.T.copy(),
        chi_ca=chi_ca, N=[5.0298, 1.0], onsager=[[0.1]], kappa=[2e-4],
        dsig=[2.6355], dh=[1.3072], Tm=[558.0], eps2=[1e-3],
        L_psi=[5.0], alpha_th=[0.0], beta_th=[0.0], L_th=[5.0],
        T=T, dt=2e-3, bulk="r14", newton_tol=1e-7, newton_max=60, **kw)
    st.set_initial([lambda x: np.full(len(x), 0.6) + jitter],
                   [_disc((0.5, 0.5), 0.15, 0.02)],
                   [lambda x: np.zeros(len(x))])
    return st


def test_a1_isothermal_parity(device):
    """T_mode='field' with uniform T0 and insulated BCs must reproduce
    the scalar march: the only difference is the segregated T solve
    (LU of a uniform field -> ~1e-16-relative T nonuniformity feeding
    the per-GP drive) plus the in-kernel vs host drive arithmetic.
    Parity across a march is a DISTRIBUTION -> 8 jittered-IC repeats,
    40 fixed steps (t = 0.08, active seeded growth); the lock sits
    >= 2x above the measured tail."""
    dm, mesh, cons = _dm(5, device)
    nf = len(mesh.node_coords[cons.free_nodes])
    devs = []
    for rep in range(8):
        jit = 1e-3 * np.random.default_rng(rep).standard_normal(nf)
        sts = _a1_stepper(dm, "scalar", jit)
        stf = _a1_stepper(dm, "field", jit)
        d_run = 0.0
        for _ in range(40):
            xs = sts.step()
            xf = stf.step()
            d_run = max(d_run, float(np.abs(xs - xf).max()))
        devs.append(d_run)
        # the T field itself must stay uniform to LU roundoff
        assert np.abs(stf.T_nodes - 333.0).max() < 1e-9
    print(f"A1 isothermal parity (8 repeats, 40 steps): per-run max "
          f"{[f'{e:.2e}' for e in devs]}, tail {max(devs):.3e}")
    # MEASURED (2026-07-12, 8 repeats): per-run max 8.66e-15..9.10e-15
    # — FP-identical class (the ~1e-16-relative LU-uniform T feeds the
    # per-GP drive; Newton stopping points coincide).  Locked 5e-14
    # (5.5x above the tail).
    assert max(devs) < 5e-14, devs


# ---------------------------------------------------------------------
# A1(iii) — GRADIENT DEMO: linear T ramp -> spatially graded kinetics
# ---------------------------------------------------------------------
def test_a1_gradient_demo(device):
    """Linear T ramp across the box (Dirichlet x = 0: 300 K, x = 1:
    700 K; Tm = 558 K crosses at x = 0.645): the cold-side seed GROWS,
    the hot-side seed (above Tm) MELTS OUT -> spatially graded
    crystallization, asserted as the half-box psi-mean contrast.
    SEED PLACEMENT (measured threshold scan, 2026-07-12, this config,
    r0 = 0.1/0.13 discs, t = 0.3): growth flips to melt between
    T = 390 (1.06-1.09x) and T = 420 (0.88-0.96x) — kinetics + the
    Gibbs-Thomson penalty put the effective growth threshold ~130 K
    below Tm at this seed size, so the cold seed sits at x = 0.15
    (T ~ 360, measured 1.22-1.25x scalar growth) and the hot seed at
    x = 0.85 (T ~ 640 > Tm, melts)."""
    dm, mesh, cons = _dm(6, device)
    coords = mesh.node_coords[cons.free_nodes]
    lr = (np.abs(coords[:, 0]) < 1e-12) | \
         (np.abs(coords[:, 0] - 1) < 1e-12)
    Tl, Tr = 300.0, 700.0
    ramp = lambda x: Tl + (Tr - Tl) * x[:, 0]
    chi_aa = np.array([[0.0, 0.7248], [0.7248, 0.0]])
    chi_ca = np.array([[0.0, 1.0836], [0.0, 0.0]])
    st = MultiPhaseStepper(
        dm, M=1, K=1, chi_aa=chi_aa, chi_ac=chi_ca.T.copy(),
        chi_ca=chi_ca, N=[5.0298, 1.0], onsager=[[0.1]], kappa=[2e-4],
        dsig=[2.6355], dh=[1.3072], Tm=[558.0], eps2=[1e-3],
        L_psi=[5.0], alpha_th=[0.0], beta_th=[0.0], L_th=[5.0],
        dt=2e-3, bulk="r14", newton_tol=1e-7, newton_max=60,
        T_mode="field",
        T_field=dict(T0=ramp, dirichlet=np.where(lr)[0],
                     g=lambda x, t: Tl + (Tr - Tl) * x[:, 0]))
    st.set_initial(
        [lambda x: np.full(len(x), 0.6)],
        [lambda x: np.maximum(_disc((0.15, 0.5), 0.13, 0.02)(x),
                              _disc((0.85, 0.5), 0.13, 0.02)(x))],
        [lambda x: np.zeros(len(x))])
    a0 = float(np.mean(st.psi(0)[coords[:, 0] < 0.5] > 0.5))
    st.march(t_end=0.6, dt_max=0.02, max_steps=500, dt_min=1e-7)
    # the ramp is the exact discrete steady state: T must hold it
    assert np.abs(st.T_nodes - ramp(coords)).max() < 1e-8
    left = float(np.mean(st.psi(0)[coords[:, 0] < 0.5]))
    right = float(np.mean(st.psi(0)[coords[:, 0] >= 0.5]))
    al = float(np.mean(st.psi(0)[coords[:, 0] < 0.5] > 0.5))
    pr = float(st.psi(0)[coords[:, 0] >= 0.5].max())
    print(f"A1 gradient demo: psi-mean left {left:.4f} vs right "
          f"{right:.5f} (ratio {left / max(right, 1e-12):.1f}x); "
          f"cold-half area {a0:.4f} -> {al:.4f}; hot-side psi_max "
          f"{pr:.4f}")
    # MEASURED (2026-07-12, t = 0.6): psi-mean left 0.1678 vs right
    # 0.00055 (ratio 304.6x); cold-half area 0.1048 -> 0.1649 (1.57x);
    # hot-side psi_max 0.0165.  Locks >= 2x headroom on each margin.
    assert al > a0 * 1.2, (a0, al)       # cold side grows (1.57x)
    assert pr < 0.3, pr                  # hot side melts out (0.0165)
    assert left > 10.0 * max(right, 1e-12), (left, right)   # 304.6x


# ---------------------------------------------------------------------
# A1(iv) — D(T) Arrhenius mobility hook
# ---------------------------------------------------------------------
def test_a1_dt_hook(device):
    """(a) T = T_ref: the hook is exactly inert (mT = exp(0) = 1).
    (b) T != T_ref: one-step CH |dphi| ratio hook-on/off matches the
    Arrhenius factor exp(-Ea (1/T - 1/T_ref)) (const-mob path; the
    fastmode/_n paths scale the same per-GP mT through the Vignes D)."""
    dm, mesh, cons = _dm(5, device)
    chi_aa = np.array([[0.0, 0.7248], [0.7248, 0.0]])
    chi_ca = np.array([[0.0, 1.0836], [0.0, 0.0]])
    Ea, T = 2000.0, 333.0
    mT = np.exp(-Ea * (1.0 / T - 1.0 / 400.0))
    common = dict(M=1, K=1, chi_aa=chi_aa, chi_ac=chi_ca.T.copy(),
                  chi_ca=chi_ca, N=[5.0298, 1.0], onsager=[[0.1]],
                  kappa=[2e-4], dsig=[2.6355], dh=[1.3072], Tm=[558.0],
                  eps2=[1e-3], L_psi=[1e-6], T=T, dt=1e-4, bulk="r14",
                  newton_tol=1e-9, newton_max=60)
    pert = lambda x: 0.5 + 0.05 * np.cos(2 * np.pi * x[:, 0]) \
        * np.cos(2 * np.pi * x[:, 1])
    dphi = {}
    for dtk in (None, (Ea, T), (Ea, 400.0)):
        st = MultiPhaseStepper(dm, D_T=dtk, **common)
        st.set_initial([pert], [lambda x: np.zeros(len(x))],
                       [lambda x: np.zeros(len(x))])
        p0 = st.phi(0).copy()
        st.step()
        dphi[dtk] = np.abs(st.phi(0) - p0).max()
    r_inert = dphi[(Ea, T)] / dphi[None]
    r_hook = dphi[(Ea, 400.0)] / dphi[None]
    print(f"A1 D(T) hook: inert ratio {r_inert:.15f} (=1), active "
          f"ratio {r_hook:.4f} vs Arrhenius mT = {mT:.4f}")
    assert abs(r_inert - 1.0) < 1e-12, r_inert
    assert 0.7 * mT < r_hook < 1.4 * mT, (r_hook, mT)


# ---------------------------------------------------------------------
# A2 — SUBSTRATE SURFACE ENERGY (wall free-energy natural term)
# ---------------------------------------------------------------------
def _a2_stepper(dm, g1, h1=0.0, jit=0.0):
    """Miscible binary (M=1, K=0; chi = 1.5 < chi_c = 2 at N = (1,1)):
    the uniform phi = 0.5 state is bulk-stable AND the exact g = 0
    steady state, so any stratification is the WALL term's doing.
    g scaled to kappa (the natural BC slope is -f_w'/kappa): g = 0.005,
    kappa = 1e-3 -> boundary-layer contrast O(0.1) without saturating
    the [1e-3, 1-1e-3] projection (measured phi_max 0.647)."""
    chi_aa = np.array([[0.0, 1.5], [1.5, 0.0]])
    st = MultiPhaseStepper(dm, M=1, K=0, chi_aa=chi_aa, N=[1.0, 1.0],
                           onsager=[[1.0]], kappa=[1e-3], dt=1e-3,
                           newton_tol=1e-8, newton_max=40,
                           wall_g=None if g1 is None else [g1],
                           wall_h=[h1] if h1 else None)
    st.set_initial([lambda x: np.full(len(x), 0.5) + jit])
    return st


def test_a2_wall_enrichment_and_sign_flip(device):
    """(i) energetic consistency: g < 0 (wall attracts species 0)
    enriches the substrate band vs the g = 0 baseline (uniform 0.5 is
    the exact baseline steady state); (ii) SIGN FLIP: g > 0 depletes
    and the vertical stratification reverses (deterministic, no
    noise).  Near-wall excess = mean phi over the y < 1/16 band minus
    the bulk 0.5.  MEASURED (2026-07-12, L6, t = 0.5, 0 rejects):
    g = -0.005 -> bottom +0.07821, top -0.004723 (mass balance);
    g = +0.005 -> EXACT mirror -0.07821/+0.004723 (the phi -> 1-phi
    symmetry of this config).  Locks >= 2x headroom."""
    dm, mesh, cons = _dm(6, device)
    coords = mesh.node_coords[cons.free_nodes]
    bot = coords[:, 1] < 1.0 / 16.0
    top = coords[:, 1] > 1.0 - 1.0 / 16.0
    exc = {}
    for g in (-0.005, 0.005):
        st = _a2_stepper(dm, g)
        r = st.march(t_end=0.5, dt_max=0.02, max_steps=300, dt_min=1e-9)
        assert r == "t_end", r
        exc[g] = (float(st.phi(0)[bot].mean() - 0.5),
                  float(st.phi(0)[top].mean() - 0.5))
    print(f"A2 wall: g=-0.005 bottom/top excess {exc[-0.005][0]:+.4f}/"
          f"{exc[-0.005][1]:+.5f}; g=+0.005 {exc[0.005][0]:+.4f}/"
          f"{exc[0.005][1]:+.5f}")
    eb_n, et_n = exc[-0.005]
    eb_p, et_p = exc[0.005]
    assert eb_n > 0.03, exc          # measured +0.07821 (2.6x)
    assert eb_p < -0.03, exc         # sign flip, measured -0.07821
    assert eb_n > et_n and eb_p < et_p, exc   # stratification flips
    assert abs(eb_n + eb_p) < 0.5 * abs(eb_n), exc  # measured 0 exact


def test_a2_wall_quadratic_and_regression(device):
    """(iii) g = 0 REGRESSION: explicit wall_g = [0.0] takes the
    UNMODIFIED path (wall_on False, no face assembly) — 3-step
    trajectory parity vs the no-arg stepper from a jittered IC.
    Plus the quadratic h_i term: f_w' = g + 2 h phi vanishes at
    phi* = -g/(2h) — the wall RESTORES phi toward phi*, from ABOVE
    (phi* = 0.125 < 0.5: depletion) and from BELOW (phi* = 0.75 >
    0.5: enrichment); exercises the -2 h Mw Jacobian block both ways.
    MEASURED (2026-07-12, L5, t = 0.5, 0 rejects): parity 0.0 exactly
    (same code path); phi* = 0.125 -> bottom excess -0.13045;
    phi* = 0.75 -> +0.08867."""
    dm, mesh, cons = _dm(5, device)
    coords = mesh.node_coords[cons.free_nodes]
    rng = np.random.default_rng(7)
    jit = 1e-3 * rng.standard_normal(len(coords))
    stA = _a2_stepper(dm, None, jit=jit)     # no wall args at all
    stB = _a2_stepper(dm, 0.0, jit=jit)      # explicit g = 0
    assert not stA.wall_on and not stB.wall_on
    errs = []
    for _ in range(3):
        xa = stA.step()
        xb = stB.step()
        errs.append(float(np.abs(xa - xb).max()))
    print(f"A2 g=0 regression parity: {[f'{e:.2e}' for e in errs]}")
    assert max(errs) < 1e-13, errs   # measured 0.0 exactly

    bot = coords[:, 1] < 1.0 / 16.0
    exc = {}
    for g1 in (-0.005, -0.03):       # phi* = 0.125 / 0.75
        st = _a2_stepper(dm, g1, h1=0.02)
        r = st.march(t_end=0.5, dt_max=0.02, max_steps=300, dt_min=1e-9)
        assert r == "t_end", r
        exc[g1] = float(st.phi(0)[bot].mean() - 0.5)
    print(f"A2 quadratic restoring: phi*=0.125 excess {exc[-0.005]:+.4f}"
          f" (pull-down), phi*=0.75 excess {exc[-0.03]:+.4f} (pull-up)")
    assert exc[-0.005] < -0.05, exc  # measured -0.13045 (2.6x)
    assert exc[-0.03] > +0.04, exc   # measured +0.08867 (2.2x)


# ---------------------------------------------------------------------
# A3 — ANISOTROPIC CRYSTAL GROWTH
# ---------------------------------------------------------------------
def _a3_stepper(dm, delta, theta0, m=2.0, T=333.0):
    """s1b-class PCBM r14 config, frozen-marker theta (KWC coeffs 0 —
    the A3 contract; theta holds the orientation as pure bookkeeping),
    single centered seed, deterministic (no noise)."""
    chi_aa = np.array([[0.0, 0.7248], [0.7248, 0.0]])
    chi_ca = np.array([[0.0, 1.0836], [0.0, 0.0]])
    kw = {} if delta is None else dict(delta_a=[delta], m_a=[m])
    st = MultiPhaseStepper(
        dm, M=1, K=1, chi_aa=chi_aa, chi_ac=chi_ca.T.copy(),
        chi_ca=chi_ca, N=[5.0298, 1.0], onsager=[[0.1]], kappa=[2e-4],
        dsig=[2.6355], dh=[1.3072], Tm=[558.0], eps2=[1e-3],
        L_psi=[5.0], alpha_th=[0.0], beta_th=[0.0], L_th=[5.0],
        T=T, dt=2e-3, bulk="r14", newton_tol=1e-7, newton_max=60, **kw)
    st.set_initial([lambda x: np.full(len(x), 0.6)],
                   [_disc((0.5, 0.5), 0.08, 0.02)],
                   [lambda x: np.full(len(x), theta0)])
    return st


def _shape_weighted(st):
    """psi-WEIGHTED second-moment tensor: aspect ratio
    sqrt(l_max/l_min) + principal-axis angle in [0, pi).  Weighted
    moments smooth the ~170-node pixelation that makes the
    THRESHOLDED estimator axis-degenerate at weak anisotropy
    (measured: thresholded axis 45.0 deg at delta = 0.05 from
    Ixx ~ Iyy; weighted axis 29.89 deg at every delta > 0)."""
    w = np.clip(st.psi(0), 0.0, None)
    pts = st.free_coords[:, :2]
    W = w.sum()
    c = (pts * w[:, None]).sum(0) / W
    d = pts - c
    Ixx = (w * d[:, 0] ** 2).sum() / W
    Iyy = (w * d[:, 1] ** 2).sum() / W
    Ixy = (w * d[:, 0] * d[:, 1]).sum() / W
    tr, det = Ixx + Iyy, Ixx * Iyy - Ixy ** 2
    dsc = np.sqrt(max(tr ** 2 / 4 - det, 0.0))
    l1, l2 = tr / 2 + dsc, tr / 2 - dsc
    ang = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)
    return float(np.sqrt(l1 / max(l2, 1e-30))), float(ang % np.pi)


def test_a3_delta_zero_regression(device):
    """delta_a = 0 REGRESSION: the factory compiles the ISOTROPIC
    kernel (aniso = False -> identical cache key), so parity vs the
    no-delta stepper is same-kernel/same-inputs — assembly-atomics
    class only.  MEASURED (2026-07-12, L5, 3 steps): 0.0 exactly."""
    dm, mesh, cons = _dm(5, device)
    stA = _a3_stepper(dm, None, 0.0)
    stB = _a3_stepper(dm, 0.0, 0.0)
    assert not stA.aniso and not stB.aniso
    errs = []
    for _ in range(3):
        xa = stA.step()
        xb = stB.step()
        errs.append(float(np.abs(xa - xb).max()))
    print(f"A3 delta=0 regression parity: {[f'{e:.2e}' for e in errs]}")
    assert max(errs) < 1e-13, errs   # measured 0.0 exactly


def test_a3_single_seed_anisotropy(device):
    """Single seed at fixed theta = 30 deg, m = 2 (2-fold: needle
    class): aspect ratio increases MONOTONICALLY in delta_a with a
    floor on the contrast, and the grown long axis aligns with theta
    (orientation locking), including at a second theta = 60 deg (grid
    independence).  MEASURED (2026-07-12, L6, t = 0.5, 0 rejects at
    every delta): ARw = 1.0000 / 1.0284 / 1.0726 / 1.1506 at delta =
    0 / 0.02 / 0.05 / 0.1; axis 29.89 deg at ALL delta > 0 (theta
    30); axis 60.11 deg at theta 60, delta 0.1 — |axis error|
    0.11 deg.  Locks >= 2x headroom."""
    dm, mesh, cons = _dm(6, device)
    th0 = np.pi / 6.0
    res = {}
    for delta in (0.0, 0.02, 0.05, 0.1):
        st = _a3_stepper(dm, delta, th0)
        r = st.march(t_end=0.5, dt_max=0.02, max_steps=500,
                     dt_min=1e-8)
        assert r == "t_end", (delta, r)
        res[delta] = _shape_weighted(st)
    st = _a3_stepper(dm, 0.1, np.pi / 3.0)
    r = st.march(t_end=0.5, dt_max=0.02, max_steps=500, dt_min=1e-8)
    assert r == "t_end", r
    ar60, ax60 = _shape_weighted(st)
    print(f"A3 anisotropy: ARw {[f'{res[d][0]:.4f}' for d in res]}, "
          f"axis(0.02/0.05/0.1) "
          f"{[f'{np.degrees(res[d][1]):.2f}' for d in (0.02, 0.05, 0.1)]}"
          f" deg vs theta 30; theta=60 run: ARw {ar60:.4f} axis "
          f"{np.degrees(ax60):.2f} deg")
    # monotone AR (measured gaps 0.0284 / 0.0442 / 0.0780)
    assert res[0.0][0] < res[0.02][0] < res[0.05][0] < res[0.1][0], res
    # contrast floor (measured 0.1506; 2.15x)
    assert res[0.1][0] - res[0.0][0] > 0.07, res
    # orientation locking (measured |err| 0.11 deg; locked 2 deg)
    for d in (0.02, 0.05, 0.1):
        e = abs(res[d][1] - th0)
        assert min(e, np.pi - e) < np.radians(2.0), (d, res[d])
    e60 = abs(ax60 - np.pi / 3.0)
    assert min(e60, np.pi - e60) < np.radians(2.0), ax60
