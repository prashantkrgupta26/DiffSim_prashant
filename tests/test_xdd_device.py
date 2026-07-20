"""Task #35 gates: XDD device assembly + device closures.

G1 (parity): the device-assembled Newton system (node-major, warp kernel +
    blockmasked node-pattern scatter, device closures) matches the host path
    (field-major scipy bmat + numpy closures) — structure EXACT (pre-Dirichlet
    pattern == permuted kron(G, blockmask)), values ≤ few-ULP at
    rtol=1e-13 / atol=1e-14 (the `_assert_scatter_equal` contract from
    tests/test_index_widening.py, adapted: host-vs-device arithmetic reorders
    sums even on CPU, so the CUDA tolerance band applies on both devices).
    Covered configurations: steady no-BC, electrode Dirichlet, supg=0,
    log-carrier mode + BDF history + generation + MMS source.
G2 (MMS): the R0 coupled-MMS ladder (G_B4_1) green with assembly="device"
    (±0.10 order contract) — run here at p1 on CPU/CUDA via the device fixture;
    the stock ladder covers the host path.
G5 (gradients): forward (J·v vs FD) AND adjoint (Jᵀ·w vs FD of wᵀR)
    directional checks on the B4/E-c gate config with device closures on,
    ≤ 1e-6.  (No warp tape exists for these kernels — enable_backward=False by
    design; the XDD adjoint contract is the assembled-Jacobian transpose, per
    the A3 closure-derivative contract.  Recorded in the task report.)

Knob: assembly="auto" (device on CUDA, host on CPU) | "host" | "device";
device requires identity constraints (uniform mesh).
"""
import numpy as np
import pytest
import scipy.sparse as sp

from diffsim.physics.exciton_system import (
    XDDSystem, NDOF, IPHI, IN, IP, IXD, IXA, _gp_value,
    bilayer_electrode_bcs,
)
from tests.test_exciton_system import (
    _make_dm, _make_system_for_jac, _random_state, _dist_bilayer,
    _mms_solve_ladder,
)

pytestmark = pytest.mark.tier2


# ── helpers ──────────────────────────────────────────────────────────────────

def _perm(nf):
    """field-major index f·nf+i  →  node-major index i·NDOF+f."""
    perm = np.empty(NDOF * nf, np.int64)
    for f in range(NDOF):
        perm[f * nf + np.arange(nf)] = np.arange(nf) * NDOF + f
    return perm


def _host_system(sysm, state):
    """The R0 host composition (field-major reference)."""
    sysm._current_state = state
    return sysm._reduce_and_eliminate(sysm.jacobian_full(state),
                                      sysm.residual_full(state))


def _device_system(sysm, state):
    """The device composition, permuted back to field-major for comparison."""
    sysm._current_state = state
    Ad, rd = sysm.device_assembler().assemble(state, need_matrix=True)
    perm = _perm(sysm.n_free)
    return Ad[perm][:, perm].tocsr(), rd[perm]


def _assert_system_equal(Ah, rh, Ad, rd):
    """G1 value contract: few-ULP (rtol=1e-13, atol=1e-14 — the
    _assert_scatter_equal CUDA band; host-vs-device reassociation applies on
    CPU too, so the band is used on both devices)."""
    np.testing.assert_allclose(rd, rh, rtol=1e-13, atol=1e-14)
    D = (Ad - Ah).tocoo()
    if D.nnz == 0:
        return
    # entrywise |diff| <= atol + rtol*|host| on the union pattern
    Hd = np.asarray(Ah[D.row, D.col]).ravel()
    ok = np.abs(D.data) <= 1e-14 + 1e-13 * np.abs(Hd)
    assert ok.all(), (
        f"G1 value mismatch: worst |diff|={np.abs(D.data).max():.3e} at "
        f"|host|={np.abs(Hd)[np.argmax(np.abs(D.data))]:.3e}")


def _jac_system(level=3, device="cpu", supg=1.0, carrier_vars="primal",
                zeta=1e-3):
    """The B4/E-c gate config (real A3 closures, bilayer dist) with an
    explicit carrier_vars/supg override."""
    from diffsim.physics.exciton_closures import (
        LangevinRecombination, OnsagerBraunDissociation)
    from diffsim.xdd.params import XDDParams
    from diffsim.physics.poisson import gauss_points

    dm, mesh, cons = _make_dm(level, 1, device)
    xq = gauss_points(mesh, dm.tables_by_p)
    params = XDDParams()
    dist_gp = {pv: _dist_bilayer(xq[pv]) for pv in xq}
    one = {pv: np.ones(len(xq[pv])) for pv in xq}
    mu_n = {pv: np.full(len(xq[pv]), 1.0) for pv in xq}
    mu_p = {pv: np.full(len(xq[pv]), 0.8) for pv in xq}
    mu_xd = {pv: np.full(len(xq[pv]), 0.3) for pv in xq}
    mu_xa = {pv: np.full(len(xq[pv]), 0.25) for pv in xq}
    langevin = LangevinRecombination(params, strategy="sum", zeta=zeta,
                                     spatial="uniform")
    onsager = OnsagerBraunDissociation(params, width=params.interface_thk)
    sysm = XDDSystem(
        dm, lam2=1.0, eps_gp=one, mu_n_gp=mu_n, mu_p_gp=mu_p,
        mu_xd_gp=mu_xd, mu_xa_gp=mu_xa, dist_gp=dist_gp,
        langevin=langevin, onsager=onsager,
        tau_inv_d=1.0, tau_inv_a=1.0, supg=supg,
        carrier_vars=carrier_vars, assembly="device")
    return sysm, dm, mesh, cons, xq


# ══════════════════════════════════════════════════════════════════════════════
# G1 — parity: structure exact, values few-ULP
# ══════════════════════════════════════════════════════════════════════════════

def test_g1_structure_blockmask(device):
    """Pre-Dirichlet structure EXACT: the device kron(G, blockmask) pattern,
    permuted to field-major, equals the host bmat pattern (which drops the
    four structurally-zero blocks)."""
    sysm, dm, mesh, cons, xq = _jac_system(level=3, device=device)
    state = _random_state(dm, seed=0)
    Ah, rh = _host_system(sysm, state)
    Ad, rd = _device_system(sysm, state)
    Ah = Ah.tocsr()
    Ah.sort_indices()
    Ad.sort_indices()
    assert Ah.nnz == Ad.nnz, (Ah.nnz, Ad.nnz)
    assert np.array_equal(Ah.indptr, Ad.indptr)
    assert np.array_equal(Ah.indices, Ad.indices)
    # nnz/dof budget sanity (recorded properly in the G4 report)
    bud = sysm.device_assembler().budget()
    assert 15.0 < bud["nnz_per_dof"] < 45.0, bud


@pytest.mark.parametrize("seed", [0, 1])
def test_g1_parity_steady(seed, device):
    """G1 values, steady, closures active, no Dirichlet."""
    sysm, dm, mesh, cons, xq = _jac_system(level=3, device=device)
    state = _random_state(dm, seed=seed)
    Ah, rh = _host_system(sysm, state)
    Ad, rd = _device_system(sysm, state)
    _assert_system_equal(Ah.tocsr(), rh, Ad, rd)


def test_g1_parity_supg_off(device):
    """G1 with supg=0 (the τ-terms must vanish exactly on both paths)."""
    sysm, dm, mesh, cons, xq = _jac_system(level=3, device=device, supg=0.0)
    state = _random_state(dm, seed=2)
    Ah, rh = _host_system(sysm, state)
    Ad, rd = _device_system(sysm, state)
    _assert_system_equal(Ah.tocsr(), rh, Ad, rd)


def test_g1_parity_dirichlet(device):
    """G1 with electrode Dirichlet rows (device strong-row plan vs host LIL
    surgery).  Structure differs by design on the pinned rows (host prunes the
    row to [diag], device keeps explicit zeros), so values are compared on the
    union pattern; the r rows carry g−u on both."""
    sysm, dm, mesh, cons, xq = _jac_system(level=3, device=device)
    bilayer_electrode_bcs(sysm, mesh, cons, Eg_hat=4.0, V_app_hat=0.0,
                          minority_ln=-4.0)
    state = _random_state(dm, seed=3)
    Ah, rh = _host_system(sysm, state)
    Ad, rd = _device_system(sysm, state)
    _assert_system_equal(Ah.tocsr(), rh, Ad, rd)


def test_g1_parity_log_bdf_gen_mms(device):
    """G1 in the full production configuration: log-carrier mode (column
    chain-rule scaling), BDF history, generation sources, MMS nodal source,
    electrode Dirichlet (log-space g−u rows)."""
    sysm, dm, mesh, cons, xq = _jac_system(level=3, device=device,
                                           carrier_vars="log")
    bilayer_electrode_bcs(sysm, mesh, cons, Eg_hat=4.0, V_app_hat=0.0,
                          minority_ln=-4.0)
    rng = np.random.default_rng(11)
    gd = {pv: 0.5 + rng.random(len(xq[pv])) for pv in xq}
    ga = {pv: 0.2 + rng.random(len(xq[pv])) for pv in xq}
    sysm.set_generation(gd, ga)
    sysm.mms_source = {f: rng.standard_normal(dm.n_nodes)
                       for f in range(NDOF)}
    state = _random_state(dm, seed=4)
    prev = _random_state(dm, seed=5)
    sysm.sigma = 1.0 / 0.05
    hist_full = {f: sysm.sigma * prev[f] for f in range(NDOF)}
    hist_full[IPHI] = np.zeros(dm.n_nodes)
    sysm.hist = {f: _gp_value(dm, hist_full[f]) for f in range(NDOF)}
    Ah, rh = _host_system(sysm, state)
    Ad, rd = _device_system(sysm, state)
    _assert_system_equal(Ah.tocsr(), rh, Ad, rd)


def test_g1_no_closures(device):
    """G1 with langevin=None / onsager=None (the zero-factor guard path)."""
    from diffsim.physics.poisson import gauss_points
    dm, mesh, cons = _make_dm(3, 1, device)
    xq = gauss_points(mesh, dm.tables_by_p)
    dist_gp = {pv: _dist_bilayer(xq[pv]) for pv in xq}
    one = {pv: np.ones(len(xq[pv])) for pv in xq}
    mu = {pv: np.full(len(xq[pv]), 0.5) for pv in xq}
    sysm = XDDSystem(
        dm, lam2=1.0, eps_gp=one, mu_n_gp=mu, mu_p_gp=mu,
        mu_xd_gp=mu, mu_xa_gp=mu, dist_gp=dist_gp,
        langevin=None, onsager=None, tau_inv_d=1.0, tau_inv_a=1.0,
        supg=1.0, assembly="device")
    state = _random_state(dm, seed=6)
    Ah, rh = _host_system(sysm, state)
    Ad, rd = _device_system(sysm, state)
    _assert_system_equal(Ah.tocsr(), rh, Ad, rd)


# ══════════════════════════════════════════════════════════════════════════════
# Newton-solve equivalence + G2 MMS ladder with device assembly
# ══════════════════════════════════════════════════════════════════════════════

def test_newton_solve_equivalence(device):
    """solve_newton with assembly='device' converges to the SAME state as the
    host path on the WELL-POSED coupled-MMS problem (G_B4_1 config; the
    end-to-end routing check: node-major solve + du permutation + Dirichlet
    flips + MMS source on device)."""
    from tests.test_exciton_system import (
        _coupled_mms_system, _mms_fields, _mms_strong_source_gp,
        _mms_source_nodal, _mms_dirichlet_all)
    fields = _mms_fields()
    states = {}
    for mode in ("host", "device"):
        sysm, dm, mesh, cons, xq = _coupled_mms_system(3, 1, device,
                                                       assembly=mode)
        src_gp = _mms_strong_source_gp(sysm, dm, xq)
        sysm.mms_source = _mms_source_nodal(sysm, dm, xq, src_gp)
        _mms_dirichlet_all(sysm, mesh, cons, fields)
        coords = dm.mesh.node_coords
        rng = np.random.default_rng(17)
        ic = {}
        for f in range(NDOF):
            base = fields[f](coords)
            ic[f] = base + 0.05 * rng.standard_normal(dm.n_nodes)
        ic[IN] = np.abs(ic[IN]) + 0.05
        ic[IP] = np.abs(ic[IP]) + 0.05
        st, info = sysm.solve_newton(ic, max_iter=12)
        assert info["converged"], (mode, info)
        states[mode] = st
    for f in range(NDOF):
        num = np.linalg.norm(states["device"][f] - states["host"][f])
        den = max(np.linalg.norm(states["host"][f]), 1e-30)
        assert num / den < 1e-8, (f, num / den)


def test_g2_mms_orders_device(device):
    """G2: the R0 coupled-MMS ladder (G_B4_1, p1) with device assembly ON —
    orders ≥ 2−0.10 for all five fields (the ±0.10 contract)."""
    from diffsim.diagnostics.convergence import observed_order
    levels = (3, 4)
    hs = [2.0 ** (-lv) for lv in levels]
    errs = _mms_solve_ladder(1, device, levels=levels, assembly="device")
    labels = ["phi", "n", "p", "Xd", "Xa"]
    for f in range(NDOF):
        order = observed_order(hs, errs[f])
        print(f"G2 device-assembly p1 {labels[f]}: "
              f"errs {[f'{e:.2e}' for e in errs[f]]} order {order:.2f}")
        assert order >= 1.9, (labels[f], order, errs[f])


# ══════════════════════════════════════════════════════════════════════════════
# G5 — gradient contract with device closures on
# ══════════════════════════════════════════════════════════════════════════════

def test_g5_gradients_device(device):
    """G5: forward (J·v vs central FD of R) and adjoint (Jᵀ·w vs central FD of
    wᵀR) directional checks with the DEVICE-assembled system, ≤ 1e-6, on the
    B4/E-c gate config (dissociation-field coupling active)."""
    sysm, dm, mesh, cons, xq = _jac_system(level=3, device=device)
    state = _random_state(dm, seed=0)
    sysm._current_state = state

    n = dm.n_nodes

    def flat(st):
        v = np.empty(n * NDOF)
        for f in range(NDOF):
            v[f::NDOF] = st[f]
        return v

    def unflat(v):
        return {f: np.ascontiguousarray(v[f::NDOF]) for f in range(NDOF)}

    def res(v):
        _, r = sysm.device_assembler().assemble(unflat(v), need_matrix=False)
        return r

    A, r0 = sysm.device_assembler().assemble(state, need_matrix=True)
    u0 = flat(state)
    eps = 1e-7
    rng = np.random.default_rng(42)
    worst_fwd = worst_adj = 0.0
    for _ in range(3):
        v = rng.standard_normal(len(u0))
        v /= np.linalg.norm(v)
        w = rng.standard_normal(len(u0))
        w /= np.linalg.norm(w)
        fd = (res(u0 + eps * v) - res(u0 - eps * v)) / (2.0 * eps)
        Jv = A @ v
        rel_f = np.linalg.norm(fd - Jv) / max(np.linalg.norm(Jv), 1e-30)
        # adjoint: (Jᵀw)·v must equal the FD directional derivative of wᵀR
        JTw_v = float((A.T @ w) @ v)
        fd_wr = float(w @ fd)
        rel_a = abs(JTw_v - fd_wr) / max(abs(JTw_v), 1e-30)
        print(f"G5 device: fwd rel={rel_f:.3e}  adj rel={rel_a:.3e}")
        worst_fwd = max(worst_fwd, rel_f)
        worst_adj = max(worst_adj, rel_a)
    assert worst_fwd < 1e-6, worst_fwd
    assert worst_adj < 1e-6, worst_adj


# ══════════════════════════════════════════════════════════════════════════════
# Knob behavior
# ══════════════════════════════════════════════════════════════════════════════

def test_knob_auto_and_errors(device):
    """auto = device on CUDA / host on CPU; invalid value raises; the
    resolved mode is recorded on the system."""
    sysm, dm, mesh, cons, xq = _make_system_for_jac(level=3, p=1,
                                                    device=device)
    if str(device).startswith("cuda"):
        assert sysm.assembly == "device" and sysm._assembly_device
    else:
        assert sysm.assembly == "host" and not sysm._assembly_device
    with pytest.raises(ValueError):
        XDDSystem(dm, lam2=1.0, eps_gp=sysm.eps_gp, mu_n_gp=sysm.mu_n_gp,
                  mu_p_gp=sysm.mu_p_gp, mu_xd_gp=sysm.mu_xd_gp,
                  mu_xa_gp=sysm.mu_xa_gp, dist_gp=sysm.dist_gp,
                  assembly="bogus")
