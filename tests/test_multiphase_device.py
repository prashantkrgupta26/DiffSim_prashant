"""M5 D-track gates: DEVICE-SIDE ASSEMBLY for the multiphase brick
(multiphase.MultiPhaseStepper assembly="device" — slot-map CSR scatter,
wodo v1.2 model; see the DEVICE ASSEMBLY section of the module
docstring and _init_device_assembly for the weak-form -> scatter map).

D1 gates:
  (i)  assembled-matrix parity host vs device at the SAME Newton
       iterate (same state, same attempt-frozen RNG draws) across the
       (M, K) x p x film/wall/aniso/T-field/theta-mode/noise/BDF2
       config matrix.  The device pattern is the fixed element-graph
       superset while the host T^T K T prunes exact zeros, so the
       comparison canonicalizes via the sparse difference.
       MEASURED (2026-07-12, L3/L4 2-D, 9 configs): rel max|dA|
       2.6e-17..4.1e-16, rel max|dr| 1.6e-15..4.3e-15 (pure
       FP-ordering: device GP eval + atomic scatter vs host einsum +
       COO dedup).  The pruning is real: e.g. r14_fmn nnz 42362 host
       vs 82944 device (fixed superset).  LOCK 1e-13 (>= 23x headroom
       on the worst measured).
  (ii) one-step solution parity (fixed dt, splu both paths — isolates
       assembly).  MEASURED max|dx| 2.2e-16..3.6e-15 vs device-repeat
       atomics spread up to 2.7e-15 (film + line-search config); state
       scale O(0.6).  LOCK 1e-12 (>= 280x).

D2 gates:
  march parity vs host on an S1-class config (observable level,
  distribution lock) + cuDSS plan stability: the device pattern nnz
  NEVER flaps across noise steps (the S2 plan-flapping regression),
  asserted via the plan counter after a noisy march.
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
from diffsim.physics.multiphase import MultiPhaseStepper

pytestmark = pytest.mark.tier3


def _dm(level, device, p=1, periodic=None, dim=2):
    tree = build_uniform(level, dim=dim, periodic=periodic)
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    return DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=dim),
                                device)


def _disc(c, r0, w):
    return lambda x: 0.5 * (1.0 - np.tanh(
        (np.sqrt((x[:, 0] - c[0]) ** 2 + (x[:, 1] - c[1]) ** 2) - r0)
        / w))


def _chi3(a01=1.2, a0s=0.6, a1s=0.3):
    c = np.zeros((3, 3))
    c[0, 1] = c[1, 0] = a01
    c[0, 2] = c[2, 0] = a0s
    c[1, 2] = c[2, 1] = a1s
    return c


def _chi4():
    c = np.zeros((4, 4))
    pairs = {(0, 1): 1.1, (0, 2): 0.8, (0, 3): 0.5, (1, 2): 0.9,
             (1, 3): 0.4, (2, 3): 0.35}
    for (i, j), v in pairs.items():
        c[i, j] = c[j, i] = v
    return c


# ---------------------------------------------------------------------
# config matrix: name -> (level, p, periodic, stepper kwargs, IC spec)
# covers (M, K) in {(1,0), (2,1), (3,2)}, p in {1, 2}, film/wall/
# aniso/T-field/dirichlet/noise/KWC-vs-frozen-theta/BDF2
# ---------------------------------------------------------------------
def _configs():
    cfgs = {}
    cfgs["m1k0_const"] = dict(
        level=4, p=1, periodic=None,
        kw=dict(M=1, K=0, chi_aa=np.array([[0.0, 2.2], [2.2, 0.0]]),
                N=[1.0, 1.0], kappa=[1e-3], dt=1e-4, bulk="p1",
                mob="const"),
        ic=dict(means=[0.45]))
    cfgs["m2k1_r14_fmn_periodic"] = dict(
        level=4, p=1, periodic=(True, True),
        kw=dict(M=2, K=1, chi_aa=_chi3(), chi_ca=_chi3(0.0, 0.0, 0.0)
                + np.array([[0, 1.4, 1.4], [0, 0, 0], [0, 0, 0]]),
                chi_ac=(np.array([[0, 1.4, 1.4], [0, 0, 0],
                                  [0, 0, 0]])).T.copy(),
                N=[5.0, 10.0, 1.0], mob="fastmode_n",
                D_self=np.array([[1e-2, 1e-3, 0.5],
                                 [1e-4, 1e-4, 1e-2],
                                 [1e-2, 1e-3, 1.0]]),
                ls_drop=(1e-6, 0.97, 35.0), kappa=[2e-4] * 2,
                dsig=[1.0], dh=[3.0], Tm=[1.0], T=0.9, eps2=[4e-3],
                L_psi=[2.0], dt=1e-4, bulk="r14", b_reg=1e-3,
                clip_psi=False),
        ic=dict(means=[0.2, 0.2], psi=[_disc((0.5, 0.5), 0.2, 0.05)],
                theta=[lambda x: 0.7 * np.ones(len(x))]))
    cfgs["m3k2_kwc_p1bulk"] = dict(
        level=4, p=1, periodic=None,
        kw=dict(M=3, K=2, chi_aa=_chi4(),
                N=[2.0, 3.0, 4.0, 1.0],
                onsager=np.diag([1.0, 0.8, 0.6]),
                kappa=[1e-3] * 3, dsig=[1.0, 0.8], dh=[-0.5, -0.4],
                Tm=[1.0, 1.0], T=0.8, eps2=[1e-4, 2e-4],
                L_psi=[1.0, 1.5], alpha_th=[0.05, 0.05],
                beta_th=[1e-3, 1e-3], L_th=[5.0, 5.0],
                dt=1e-4, bulk="p1", mob="const"),
        ic=dict(means=[0.25, 0.25, 0.2],
                psi=[_disc((0.35, 0.5), 0.15, 0.05),
                     _disc((0.7, 0.5), 0.12, 0.05)],
                theta=[lambda x: 0.4 * np.ones(len(x)),
                       lambda x: 1.1 * np.ones(len(x))]))
    cfgs["m2k1_p2"] = dict(
        level=3, p=2, periodic=None,
        kw=dict(M=2, K=1, chi_aa=_chi3(), N=[3.0, 3.0, 1.0],
                kappa=[1e-3] * 2, dsig=[1.0], dh=[-0.5], Tm=[1.0],
                eps2=[1e-4], L_psi=[1.0], dt=1e-4, bulk="p1",
                mob="const"),
        ic=dict(means=[0.3, 0.3], psi=[_disc((0.5, 0.5), 0.2, 0.08)],
                theta=[lambda x: 0.5 * np.ones(len(x))]))
    cfgs["m2k1_film_wall_ls"] = dict(
        level=4, p=1, periodic=(True, False),
        kw=dict(M=2, K=1, chi_aa=_chi3(1.0, 0.7248, 0.3),
                N=[5.0, 10.0, 1.0], mob="fastmode_n",
                D_self=np.array([[1e-2, 1e-3, 0.5],
                                 [1e-4, 1e-4, 1e-2],
                                 [1e-2, 1e-3, 1.0]]),
                kappa=[2e-4] * 2, dsig=[1.0], dh=[3.0], Tm=[1.0],
                T=0.9, eps2=[4e-3], L_psi=[2.0], dt=1e-4, bulk="r14",
                b_reg=1e-3, clip_psi=False, line_search=True,
                wall_g=[-0.2, 0.1], wall_h=[0.05, 0.0],
                film=dict(k_e=0.5, h0=1.0)),
        ic=dict(means=[0.15, 0.1],
                psi=[_disc((0.5, 0.35), 0.15, 0.05)],
                theta=[lambda x: 0.7 * np.ones(len(x))]))
    cfgs["m2k1_aniso_tfield"] = dict(
        level=4, p=1, periodic=None,
        kw=dict(M=2, K=1, chi_aa=_chi3(), N=[3.0, 3.0, 1.0],
                kappa=[1e-3] * 2, dsig=[1.0], dh=[3.0], Tm=[1.0],
                eps2=[4e-3], L_psi=[1.0], dt=1e-4, bulk="r14",
                clip_psi=False, delta_a=[0.3], m_a=[4.0],
                T_mode="field", T_field=dict(T0=0.9),
                D_T=(1.5, 1.0)),
        ic=dict(means=[0.3, 0.3], psi=[_disc((0.5, 0.5), 0.2, 0.05)],
                theta=[lambda x: 0.3 * np.ones(len(x))]))
    cfgs["m2k1_dirichlet"] = dict(
        level=4, p=1, periodic=None, dirichlet="boundary",
        kw=dict(M=2, K=1, chi_aa=_chi3(), N=[3.0, 3.0, 1.0],
                kappa=[1e-3] * 2, dsig=[1.0], dh=[-0.5], Tm=[1.0],
                eps2=[1e-4], L_psi=[1.0], dt=1e-4, bulk="p1",
                mob="const"),
        ic=dict(means=[0.3, 0.3], psi=[_disc((0.5, 0.5), 0.2, 0.08)],
                theta=[lambda x: 0.5 * np.ones(len(x))]))
    cfgs["m2k1_noise"] = dict(
        level=4, p=1, periodic=(True, True),
        kw=dict(M=2, K=1, chi_aa=_chi3(), N=[3.0, 3.0, 1.0],
                kappa=[1e-3] * 2, dsig=[1.0], dh=[-0.5], Tm=[1.0],
                eps2=[1e-4], L_psi=[1.0], dt=1e-4, bulk="p1",
                mob="const", noise_psi=5e-3, noise_phi=1e-3,
                noise_damp=(1e-2, 0.85, 15.0), noise_seed=7),
        ic=dict(means=[0.3, 0.3], psi=[_disc((0.5, 0.5), 0.2, 0.08)],
                theta=[lambda x: 0.5 * np.ones(len(x))]))
    cfgs["m2k1_bdf2"] = dict(
        level=4, p=1, periodic=None,
        kw=dict(M=2, K=1, chi_aa=_chi3(), N=[3.0, 3.0, 1.0],
                kappa=[1e-3] * 2, dsig=[1.0], dh=[-0.5], Tm=[1.0],
                eps2=[1e-4], L_psi=[1.0], dt=1e-4, bulk="p1",
                mob="const", tstep="bdf2"),
        ic=dict(means=[0.3, 0.3], psi=[_disc((0.5, 0.5), 0.2, 0.08)],
                theta=[lambda x: 0.5 * np.ones(len(x))]))
    return cfgs


def _mk(cfg, device, assembly):
    dm = _dm(cfg["level"], device, p=cfg["p"],
             periodic=cfg["periodic"])
    kw = dict(cfg["kw"])
    if cfg.get("dirichlet") == "boundary":
        coords = dm.mesh.node_coords[dm.constraints.free_nodes]
        bdry = ((np.abs(coords[:, 0]) < 1e-12)
                | (np.abs(coords[:, 0] - 1.0) < 1e-12))
        nd = 2 * kw["M"] + 2 * kw["K"]
        g_fns = [None] * nd
        g_fns[0] = lambda x, t: 0.3 * np.ones(len(x))
        kw.update(dirichlet=np.where(bdry)[0], g_fns=g_fns)
    st = MultiPhaseStepper(dm, linsolver="splu", newton_tol=1e-8,
                           assembly=assembly, **kw)
    ic = cfg["ic"]
    rng = np.random.default_rng(3)
    nf = st.nfree
    phi_fns = []
    for m in ic["means"]:
        v = m + 0.02 * rng.standard_normal(nf)
        phi_fns.append(lambda x, v=v: v)
    st.set_initial(phi_fns, ic.get("psi"), ic.get("theta"))
    return st


def _sync_state(src, dst):
    """Copy the committed march state (host stepper -> device stepper)
    so both assemble at the SAME Newton iterate."""
    dst.x = src.x.copy()
    dst.hist = src.hist.copy()
    dst.hist2 = None if src.hist2 is None else src.hist2.copy()
    dst.dt_prev = src.dt_prev
    dst.t = src.t
    dst.dt = src.dt
    if src.film_on:
        dst.h_curr = src.h_curr
    if src.T_mode == "field":
        dst.T_nodes = src.T_nodes.copy()


@pytest.mark.parametrize("name", sorted(_configs()))
def test_d1_matrix_parity(name, device):
    """D1 gate (i): assembled matrix + rhs parity at the same Newton
    iterate.  The host CSR prunes exact zeros (T^T K T), the device
    CSR is the fixed superset pattern — compare the canonicalized
    sparse difference.  Locks from measured (module docstring)."""
    cfg = _configs()[name]
    sh = _mk(cfg, device, "host")
    sd = _mk(cfg, device, "device")
    # land on a nontrivial committed state (mu, psi, T all live),
    # then sync it into the device stepper and align the RNG streams
    sh.step()
    if cfg["kw"].get("tstep") == "bdf2":
        sh.step()               # exercise the true BDF2 branch
    _sync_state(sh, sd)
    sh._nrng = np.random.default_rng(77)
    sd._nrng = np.random.default_rng(77)
    Ah, rh = sh._debug_assemble(sh.dt)
    Ad, rd = sd._debug_assemble(sd.dt)
    dA = Ad - Ah
    rel_A = (np.abs(dA.data).max() / np.abs(Ah.data).max()
             if dA.nnz else 0.0)
    rsc = max(1.0, np.abs(rh).max())
    rel_r = np.abs(rd - rh).max() / rsc
    assert rel_A < 1e-13, \
        f"{name}: rel max|A_dev - A_host| = {rel_A:.3e} (lock 1e-13)"
    assert rel_r < 1e-13, \
        f"{name}: rel max|r_dev - r_host| = {rel_r:.3e} (lock 1e-13)"


@pytest.mark.parametrize("name", ["m1k0_const", "m2k1_r14_fmn_periodic",
                                  "m2k1_film_wall_ls", "m2k1_noise",
                                  "m2k1_bdf2"])
def test_d1_one_step_parity(name, device):
    """D1 gate (ii): one fixed-dt step, host vs device (splu both —
    isolates assembly).  MEASURED host-vs-device max|dx| 2.2e-16..
    7.6e-15 (atomics-spread class); LOCK 1e-12."""
    cfg = _configs()[name]
    sh = _mk(cfg, device, "host")
    sd = _mk(cfg, device, "device")
    xh = sh.step()
    xd = sd.step()
    d = np.abs(xh - xd).max()
    assert d < 1e-12, f"{name}: one-step max|dx| = {d:.3e} (lock 1e-12)"
    if cfg["kw"].get("noise_psi", 0.0) or cfg["kw"].get("noise_phi",
                                                        0.0):
        # equal draw counts: the same second step stays locked
        xh2, xd2 = sh.step(), sd.step()
        d2 = np.abs(xh2 - xd2).max()
        assert d2 < 1e-12, f"{name}: step-2 max|dx| = {d2:.3e}"


@_skip_nvmath
def test_d2_march_parity_and_plan_stability(device):
    """D2 gates: (a) march parity vs host on an S1-class seeded-growth
    config at OBSERVABLE level (psi area, phi mean/extremes — the
    house FP-chaos rule: distribution lock, measured device-repeat
    spread sets the scale); (b) cuDSS plan stability on the device
    path — the fixed pattern must plan EXACTLY ONCE across a noisy
    march (the S2 nnz-flapping regression gate).
    MEASURED (L5, noisy march to t = 8e-3, noise_psi 5e-3): psi-area
    host 0.123047 == device == device-repeat (thresholded observable,
    atomics-robust here); psi_max |h - d| 6.7e-16 (repeat spread
    2.2e-16); phi-mean |h - d| 2.8e-17 (conservation); plan count 1,
    nnz 331776 fixed.  LOCKS: area 5e-3, phi mean 1e-12, psi_max 5e-3
    (distribution-class headroom, the house FP-chaos rule)."""
    cfg = _configs()["m2k1_r14_fmn_periodic"]
    cfg = dict(cfg)
    cfg["level"] = 5
    kw = dict(cfg["kw"])
    kw.update(noise_psi=5e-3, noise_seed=11)
    cfg["kw"] = kw

    def run(assembly, solver):
        st = _mk(cfg, device, assembly)
        st.linsolver = solver
        st.dt = 2e-4
        st.march(t_end=8e-3, dt_max=2e-4)
        phi0 = st.phi(0)
        return st, float(np.mean(st.psi(0) > 0.5)), \
            float(np.mean(phi0)), float(st.psi(0).max())

    sh, area_h, m_h, pmax_h = run("host", "splu")
    sd, area_d, m_d, pmax_d = run("device", "cudss")
    sd2, area_d2, m_d2, pmax_d2 = run("device", "cudss")
    spread = abs(area_d - area_d2)
    da = abs(area_d - area_h)
    assert sd._n_dev_plans == 1, \
        f"cuDSS device plan count {sd._n_dev_plans} != 1 (nnz flap?)"
    assert sd._asm.nnz == sd2._asm.nnz
    assert abs(m_d - m_h) < 1e-12, \
        f"phi_0 mean drift host-vs-device {abs(m_d - m_h):.3e}"
    assert da < 5e-3, (f"psi area host {area_h:.5f} vs device "
                       f"{area_d:.5f} (|d| = {da:.2e}, repeat spread "
                       f"{spread:.2e}, lock 5e-3)")
    assert abs(pmax_d - pmax_h) < 5e-3, \
        f"psi_max host {pmax_h:.5f} vs device {pmax_d:.5f}"
