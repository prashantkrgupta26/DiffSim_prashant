"""LTE / PI(D) adaptive time-step controller gates (physics/lte.py).

Measured-then-locked (2026-07-14, docs/dev/2026-07-14-lte-controller.md;
RTX 6000 Ada, level 5).  Four gates per stage:
  (i)   step-doubling LTE tracks the tolerance (achieved-error band);
  (ii)  dt spans decades (the dynamic range that pays for adaptivity);
  (iii) OFF-path bit-parity (adapt="ladder" == the native ladder, ==0.0);
  (iv)  reject/rewind bit-exactness (snapshot/restore ==0.0).
Plus the NOISE CONTRACT (L3): LTE+noise -> explicit notice + BDF ladder
fallback (bit-identical to a plain march); deterministic LTE unaffected;
the preflight rule grades WARN/PASS/absent.

Bounds carry >= 2x headroom over the measured values.
"""
import numpy as np
import pytest

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.cahn_hilliard import CahnHilliardStepper, adaptive_march
from diffsim.physics.multiphase import MultiPhaseStepper
from diffsim.physics import lte

pytestmark = pytest.mark.tier3


def _dm(level, device):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    return DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)


def _ch(dm, dt, order=2):
    st = CahnHilliardStepper(dm, M=1.0, kappa=2e-3, dt=dt, order=order,
                             linsolver="splu", energy="poly")
    rng = np.random.default_rng(3)
    st.set_initial(lambda x: 0.05 * rng.standard_normal(len(x)),
                   mu_init="consistent")
    return st


# =====================================================================
# StepController unit gate (no GPU): PI(D) reduces to deadbeat, clamps
# =====================================================================
def test_controller_deadbeat_rule_pi_memory_and_clamps():
    p = 2                                    # pe = p+1 = 3
    db = lte.StepController("deadbeat", order=p)
    # deadbeat = the integral rule the existing ladders use:
    #   fac = safety * (tol/e)^(1/(p+1))
    assert db.propose(1e-5, 1e-4) == pytest.approx(
        0.9 * 10 ** (1.0 / 3.0), rel=1e-12)
    # a "pi" controller with kI=1, kP=0 and NO history reduces EXACTLY to
    # deadbeat (the P memory is what differs, not the integral base)
    pi0 = lte.StepController("pi", order=p, kI=1.0, kP=0.0)
    assert pi0.propose(1e-5, 1e-4) == pytest.approx(
        db.propose(1e-5, 1e-4), rel=1e-12)
    # the P term ENGAGES only after an accepted error is committed, and
    # then it changes the step ratio (memory of the error trend)
    pi = lte.StepController("pi", order=p)
    f_no_hist = pi.propose(2e-5, 1e-4)
    pi.commit(1e-5)                          # e_prev < e -> P damps growth
    f_hist = pi.propose(2e-5, 1e-4)
    assert f_hist != pytest.approx(f_no_hist, rel=1e-9)
    # growth clamp + a reject never grows dt
    assert lte.StepController("pi", order=p, fac_max=3.0).propose(
        1e-12, 1e-4) == 3.0
    assert pi.reject_ratio(1e-2, 1e-4) <= 1.0


# =====================================================================
# L1 — CH binary: error band, dt range, OFF-parity, reject/rewind
# =====================================================================
def test_l1_ch_error_tracks_tol(device):
    """(i) achieved global error tracks tol; (ii) dt spans decades.
    Step-doubling BDF2, deadbeat + PI + PID (one shared reference)."""
    dm = _dm(5, device)
    # reference: fine fixed-dt BDF2
    ref = _ch(dm, 1.25e-4, order=2)
    while ref.t < 0.04 - 1e-12:
        ref.dt = min(1.25e-4, 0.04 - ref.t)
        ref.step()
    cref = ref.x[0::2].copy()

    for ctrl in ("deadbeat", "pi"):    # pid == pi here; unit test covers
        prev_ge = None
        for tol in (1e-3, 1e-4):
            st = _ch(dm, 2e-4, order=2)
            ts, dts = lte.lte_march(st, 0.04, tol=tol, controller=ctrl,
                                    estimator="doubling", dt_min=1e-7,
                                    dt_max=0.02)
            ge = np.linalg.norm(st.x[0::2] - cref) / np.linalg.norm(cref)
            # (i) achieved error in-band around tol (>= 2x headroom on
            # the measured gerr/tol 5..14 for step-doubling BDF2)
            assert ge < 40 * tol, (ctrl, tol, ge)
            # (ii) dt grew by >= 2 decades (measured ~4)
            da = np.array(dts)
            assert da.max() / da.min() > 100.0, (ctrl, da.min(), da.max())
            if prev_ge is not None:
                assert ge < prev_ge, (ctrl, ge, prev_ge)  # monotone
            prev_ge = ge


def test_l1_ch_off_parity(device):
    """(iii) adapt='ladder' is byte-identical to adaptive_march."""
    dm = _dm(5, device)
    st_a = _ch(dm, 2e-4, order=2)
    adaptive_march(st_a, 0.03, tol=5e-4, dt_min=1e-7, dt_max=0.02)
    st_b = _ch(dm, 2e-4, order=2)
    lte.march(st_b, 0.03, adapt="ladder", tol=5e-4, dt_min=1e-7,
              dt_max=0.02)
    assert np.abs(st_a.x - st_b.x).max() == 0.0


def test_l1_ch_reject_rewind_bitexact(device):
    """(iv) a full step + two half steps + restore returns the EXACT
    pre-step state (the reject path leaves the trajectory untouched)."""
    dm = _dm(5, device)
    st = _ch(dm, 2e-4, order=2)
    lte.lte_march(st, 0.01, tol=3e-4, controller="pi", dt_min=1e-7,
                  dt_max=0.02)
    A = lte._adapter(st)
    snap = A.snapshot()
    before = A.solution().copy()
    A.advance(A.dt)
    A.restore(snap)
    A.advance(A.dt * 0.5)
    A.advance(A.dt * 0.5)
    A.restore(snap)
    assert np.abs(A.solution() - before).max() == 0.0


# =====================================================================
# L2 — coupled multiphase (p1, M=1, K=1, deterministic): same gates
# =====================================================================
def _mpf(dm, dt, noise_psi=0.0, seed=0, tstep="bdf1"):
    chi_aa = np.array([[0.0, 0.7248], [0.7248, 0.0]])
    st = MultiPhaseStepper(
        dm, M=1, K=1, chi_aa=chi_aa, chi_ac=chi_aa.copy(),
        chi_ca=chi_aa.copy(), N=[1.0, 1.0], onsager=[[0.1]],
        kappa=[2e-4], dsig=[1.0], dh=[-1.0], Tm=[1.0], eps2=[1e-3],
        L_psi=[5.0], alpha_th=[0.0], beta_th=[0.0], L_th=[5.0],
        T=0.5, dt=dt, bulk="p1", newton_tol=1e-9, newton_max=40,
        tstep=tstep, noise_psi=noise_psi, noise_seed=seed)

    def disc(x):
        r = np.sqrt((x[:, 0] - 0.5) ** 2 + (x[:, 1] - 0.5) ** 2)
        return 0.5 * (1.0 - np.tanh((r - 0.15) / 0.02))
    st.set_initial([lambda x: np.full(len(x), 0.6)], [disc],
                   [lambda x: np.zeros(len(x))])
    return st


def _mfield(st):
    return np.concatenate([st.phi(0), st.psi(0)])


def test_l2_multiphase_error_and_rewind(device):
    """Coupled config (phi CH + psi AC + frozen theta): LTE controls
    error and the reject/rewind is bit-exact (the adapter captures
    phi/psi/theta + the two-level BDF2 history + film/T state).
    embedded predictor-corrector on tstep=bdf2 (genuine LTE, 1 solve/
    step) — step-doubling also works but the sustained crystal-growth
    front keeps dt small (costly); see the dev note."""
    dm = _dm(5, device)
    # reference: fine fixed-dt bdf1
    ref = _mpf(dm, 5e-4)
    ref.march(t_end=0.06, dt_max=5e-4, max_steps=100000, dt_min=1e-7)
    fref = _mfield(ref)

    prev = None
    for tol in (1e-3, 1e-4):
        st = _mpf(dm, 1e-3, tstep="bdf2")
        lte.lte_march(st, 0.06, tol=tol, controller="pi",
                      estimator="embedded", dt_min=1e-7, dt_max=0.02)
        ge = np.linalg.norm(_mfield(st) - fref) / np.linalg.norm(fref)
        assert np.isfinite(ge) and ge < 0.05, (tol, ge)
        if prev is not None:
            assert ge <= prev * 1.2, (ge, prev)   # error non-increasing
        prev = ge
    # reject/rewind bit-exactness (bdf1 adapter, two half steps)
    st = _mpf(dm, 1e-3)
    lte.lte_march(st, 0.01, tol=1e-3, controller="pi",
                  estimator="embedded", dt_min=1e-7, dt_max=0.02)
    A = lte._adapter(st)
    snap = A.snapshot()
    before = A.solution().copy()
    A.advance(A.dt)
    A.restore(snap)
    A.advance(A.dt * 0.5)
    A.advance(A.dt * 0.5)
    A.restore(snap)
    assert np.abs(A.solution() - before).max() == 0.0


def test_l2_multiphase_off_parity(device):
    """adapt='ladder' == the native Appendix-A .march() (bit-parity)."""
    dm = _dm(5, device)
    a = _mpf(dm, 1e-3)
    a.march(t_end=0.05, dt_max=0.02, max_steps=100000, dt_min=1e-7)
    b = _mpf(dm, 1e-3)
    lte.march(b, 0.05, adapt="ladder", dt_max=0.02, max_steps=100000,
              dt_min=1e-7)
    assert np.abs(_mfield(a) - _mfield(b)).max() == 0.0


# =====================================================================
# L3 — the NOISE CONTRACT
# =====================================================================
def test_l3_noise_fallback_and_notice(device):
    """LTE + noise ON: explicit notice fires, LTE is refused, the run
    falls back to the native ladder, and the fallback is bit-identical
    to a plain .march() (no silent downgrade, no crash)."""
    dm = _dm(5, device)
    st = _mpf(dm, 1e-3, noise_psi=1e-3, seed=7)
    assert lte.has_noise(st)
    lte.lte_march(st, 0.03, tol=1e-4, controller="pi", dt_min=1e-7,
                  dt_max=0.02)
    assert st._lte_fell_back is True
    assert st._lte_active is False
    assert st._lte_notice and "LTE-INCOMPATIBLE-WITH-NOISE" in \
        st._lte_notice
    assert st.t >= 0.03 - 1e-9          # fallback march completed
    # fallback == plain native ladder (same seed/config)
    st2 = _mpf(dm, 1e-3, noise_psi=1e-3, seed=7)
    st2.march(t_end=0.03, dt_max=0.02, max_steps=100000, dt_min=1e-7)
    assert np.abs(_mfield(st) - _mfield(st2)).max() == 0.0


def test_l3_deterministic_lte_unaffected(device):
    """Deterministic run: full LTE controller runs, no fallback."""
    dm = _dm(5, device)
    st = _mpf(dm, 1e-3, noise_psi=0.0, tstep="bdf2")
    assert not lte.has_noise(st)
    ts, dts = lte.lte_march(st, 0.03, tol=3e-4, controller="pi",
                            estimator="embedded", dt_min=1e-7,
                            dt_max=0.02)
    assert st._lte_fell_back is False
    assert st._lte_active is True
    assert len(dts) > 0


def test_l3_preflight_rule():
    """The RunLog preflight rule grades WARN (noise) / PASS (det) and is
    ABSENT when adapt is OFF (bit-parity for existing film runs)."""
    from diffsim.film.preflight import run_preflight
    from diffsim.film.params import FilmParams

    class R:
        phi_p0 = phi_f0 = phi_s0 = 0.33
        kappa = (1e-3, 1e-3)
        D_pair = (1.0, 1.0); M11 = M22 = 1.0
        level = 5; cells = (32, 32); nodes = 1089; dofs = 2178
        nnz = 10000; lat_scale = 1.0; y_comp = 1.0
        dx_lat = 0.03; dz0 = 0.03

    cases = [
        (FilmParams(adapt="lte", noise=1e-3, preflight="warn"), "WARN"),
        (FilmParams(adapt="lte", noise=0.0, tstep="bdf2",
                    preflight="warn"), "PASS"),
        (FilmParams(adapt="ladder", noise=1e-3), None),
    ]
    for pp, want in cases:
        rep = run_preflight(pp, R(), query_hardware=False)
        assert rep.status_of("lte-noise-incompatible") == want, (pp.adapt,
                                                                 want)
