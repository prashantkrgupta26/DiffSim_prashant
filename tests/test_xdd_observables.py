"""SP-1 Block D gates: XDD instrument-twin observables.

The twin OUTPUTS (spec §4; plan D1–D3) — consistent-flux J–V, J(t)+Nirmal
features+IRF, PL/TRPL+quench.  Pure functions over states/traces; the marching
gates reuse XDDRun (no parallel infrastructure).

Gate summary
------------
D1 (flux-balance) — G_D_1 : contact_flux_pair through the PUBLIC API reproduces
    |Jny−Jpy|/J < 1% on the Block-C bilayer dark steady state.
D1 (J–V format)   — G_D_2 : build_jv_curve / write_jv_curve column layout, Jsc,
    FF extraction, Voc bracket on a synthetic sign-changing curve.
D2 (feature-match)— G_D_3 : nirmal_features on analytic traces match closed-form
    values (triangle time features + Fourier-partial FFT features; hand values
    in the test).
D2 (IRF)          — G_D_4 : irf_convolve of a δ-like trace returns the kernel
    (printed check); Gaussian kernel unit-area; bandwidth low-pass smokes.
D3 (PL decay-rate)— G_D_5 : uniform-slab STEADY_PULSE march — PL(t) ∝ e^(−t̂/τ̂_eff)
    with 1/τ̂_eff = 1/τ̂_r + 1/τ̂_nr + k̂; fitted rate matches to 1e-3 rel (the
    interface-free limit).  Plus pl_species_integral / pl_quench_ratio units.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.poisson import gauss_points
from diffsim.physics.exciton_system import (
    XDDSystem, NDOF, IPHI, IN, IP, IXD, IXA, bilayer_electrode_bcs,
)
from diffsim.physics.exciton_closures import (
    LangevinRecombination, OnsagerBraunDissociation, RegionMobility, Generation,
)
from diffsim.xdd.params import XDDParams
from diffsim.xdd.run import (
    XDDRun, STEADY_STATE_JV, STEADY_PULSE, _march_to_steady, log_linear_ic,
)
from diffsim.xdd.observables import (
    contact_flux_pair, to_mA_per_cm2, report_current, designated_current,
    JVPoint, JV_COLUMNS, build_jv_curve, write_jv_curve,
    extract_jsc_ff, voc_bracket,
    capture_trace, nirmal_features, gaussian_kernel, irf_convolve,
    apply_bandwidth,
    pl_species_integral, pl_trace, pl_quench_ratio, fit_decay_rate,
    _tau_r_inv_hat,
)

pytestmark = pytest.mark.tier2

_DIST_SCALE = 4e-9


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures (mirror test_xdd_run.py)
# ══════════════════════════════════════════════════════════════════════════════

def _make_dm(level, p, device="cpu"):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2), device)
    return dm, mesh, cons


def _resolvable_bilayer(level=3, p=1, device="cpu", zeta=1e-3,
                        mu=0.5, lam2=1e-1, Eg_hat=4.0, carrier_vars="log",
                        params=None):
    """Marchable symmetric bilayer (Block C's _resolvable_bilayer)."""
    dm, mesh, cons = _make_dm(level, p, device)
    xq = gauss_points(mesh, dm.tables_by_p)
    params = params or XDDParams()
    s = params.scales()

    dist_gp = {pv: (xq[pv][:, 1] - 0.5) * _DIST_SCALE for pv in xq}
    mu_n = {}; mu_p = {}; mu_xd = {}; mu_xa = {}; eps = {}
    for pv in xq:
        n = len(dist_gp[pv])
        mu_n[pv] = np.full(n, mu); mu_p[pv] = np.full(n, mu)
        mu_xd[pv] = np.full(n, mu); mu_xa[pv] = np.full(n, mu)
        eps[pv] = np.ones(n)

    langevin = LangevinRecombination(params, strategy="sum", zeta=zeta,
                                     spatial="uniform")
    onsager = OnsagerBraunDissociation(params, width=params.interface_thk)
    tau_inv = s.t0 / params.tau_x_donor

    sysm = XDDSystem(
        dm, lam2=lam2, eps_gp=eps, mu_n_gp=mu_n, mu_p_gp=mu_p,
        mu_xd_gp=mu_xd, mu_xa_gp=mu_xa, dist_gp=dist_gp,
        langevin=langevin, onsager=onsager,
        tau_inv_d=tau_inv, tau_inv_a=tau_inv, supg=1.0,
        carrier_vars=carrier_vars)
    return sysm, dm, mesh, cons, params, s, Eg_hat


def _uniform_slab(level=3, p=1, device="cpu", mu=0.5, lam2=1e-1,
                  Eg_hat=4.0, tau_x=1e-9, k_hat=0.0, params=None):
    """Interface-FREE uniform slab for the D3 analytic decay check.

    NO Onsager dissociation, NO Langevin recombination, uniform mobilities and
    a spatially-CONSTANT dissociation sink k̂ (injected as a fixed scalar added
    to the exciton σ_tot via a constant kd/ka closure surrogate).  In this limit
    the exciton row is  ∂t X̂ = −(1/τ̂_x + k̂)X̂ + Ĝ  (diffusion negligible for a
    volume-integrated mode at V̂=0 symmetric), so after light-off
    ∫X̂ dV̂ ∝ e^(−t̂ (1/τ̂_x + k̂)) and PL ∝ e^(−t̂/τ̂_eff),
    1/τ̂_eff = 1/τ̂_r + 1/τ̂_nr + k̂ = 1/τ̂_x + k̂ (since 1/τ̂_r+1/τ̂_nr = 1/τ̂_x).
    """
    dm, mesh, cons = _make_dm(level, p, device)
    xq = gauss_points(mesh, dm.tables_by_p)
    params = params or XDDParams()
    s = params.scales()

    dist_gp = {pv: np.zeros(len(xq[pv])) for pv in xq}   # everywhere "interface"
    mu_n = {}; mu_p = {}; mu_xd = {}; mu_xa = {}; eps = {}
    for pv in xq:
        n = len(xq[pv])
        mu_n[pv] = np.full(n, mu); mu_p[pv] = np.full(n, mu)
        # tiny exciton mobility → diffusion negligible, pure decay mode
        mu_xd[pv] = np.full(n, 1e-6); mu_xa[pv] = np.full(n, 1e-6)
        eps[pv] = np.ones(n)

    # No recombination, no dissociation (interface-free).  Constant k̂ sink is
    # folded into τ_inv below (1/τ̂_x + k̂) so the decay rate is analytic.
    tau_inv = s.t0 / tau_x + k_hat

    sysm = XDDSystem(
        dm, lam2=lam2, eps_gp=eps, mu_n_gp=mu_n, mu_p_gp=mu_p,
        mu_xd_gp=mu_xd, mu_xa_gp=mu_xa, dist_gp=dist_gp,
        langevin=None, onsager=None,
        tau_inv_d=tau_inv, tau_inv_a=tau_inv, supg=1.0,
        carrier_vars="log")
    return sysm, dm, mesh, cons, params, s, Eg_hat, tau_inv


# ══════════════════════════════════════════════════════════════════════════════
# G_D_1 — flux-balance through the PUBLIC observable API (D1)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
def test_gate_d1_public_flux_balance():
    """G_D_1: contact_flux_pair (PUBLIC API) reproduces |Jny−Jpy|/J < 1%.

    The Block-C self-consistency test, now through the public observable:
    march the symmetric resolvable bilayer to dark steady, then the public
    contact_flux_pair must balance to < 1% (and equal run._flux_pair — it wraps
    it).  Also checks the mA/cm² re-dimensionalisation (J0·0.1) and the
    designated/report current conventions.
    """
    sysm, dm, mesh, cons, params, s, Eg_hat = _resolvable_bilayer(
        level=3, p=1, carrier_vars="log")
    z = {pv: np.zeros(len(sysm.dist_gp[pv])) for pv in dm.bins}
    sysm.set_generation(z, z)
    bilayer_electrode_bcs(sysm, mesh, cons, Eg_hat=Eg_hat, V_app_hat=0.0,
                          h_axis=1, minority_ln=-Eg_hat)
    ic = log_linear_ic(mesh, Eg_hat, -Eg_hat, h_axis=1)
    st, info = _march_to_steady(
        sysm, ic, dt0_hat=1e-6, dt_max_hat=1e-1, max_steps=500,
        time_stepping_tol=1e-4, flux_floor=1e-3, bdf2=False,
        stage_name="d1", h_axis=1, newton_kw={"max_iter": 20})
    assert info["criterion_fired"], "G_D_1: march did not converge"

    Jny, Jpy = contact_flux_pair(sysm, st, h_axis=1)
    J = report_current(Jny, Jpy)
    imbalance = abs(Jny - Jpy) / max(J, 1e-3)
    print(f"\nG_D_1: Jny={Jny:.4e} Jpy={Jpy:.4e} J(min)={J:.4e} "
          f"imbalance={imbalance:.3e}")
    assert np.isfinite(Jny) and np.isfinite(Jpy)
    assert imbalance < 0.01, f"G_D_1: imbalance {imbalance:.3e} >= 1%"

    # public wraps run._flux_pair identically
    from diffsim.xdd.run import _flux_pair
    j2 = _flux_pair(sysm, st, h_axis=1)
    assert Jny == pytest.approx(j2[0]) and Jpy == pytest.approx(j2[1])

    # mA/cm² re-dimensionalisation = J0 · 0.1
    J_mA = to_mA_per_cm2(J, s.J0)
    assert J_mA == pytest.approx(J * s.J0 * 0.1, rel=1e-12)

    # designated (anode) == |Jny|; softmin ≤ min
    assert designated_current(Jny, Jpy, "anode") == pytest.approx(abs(Jny))
    sm = designated_current(Jny, Jpy, "softmin", beta=200.0)
    assert sm <= min(abs(Jny), abs(Jpy)) + 1e-9


# ══════════════════════════════════════════════════════════════════════════════
# G_D_2 — J–V curve format, Jsc/FF, Voc bracket (D1)
# ══════════════════════════════════════════════════════════════════════════════

def test_gate_d2_jv_curve_format(tmp_path):
    """G_D_2: J–V writer column layout + Jsc/FF/Voc extraction.

    Synthetic sweep_history (V̂, march_info) with a photocurrent that changes
    sign between V=0.5 and V=0.6 (a Voc there).  Checks:
      • build_jv_curve produces JVPoint rows with the CPU column set
        (V_app, t̂, t, J, Jny, Jpy) and t = t̂·t0.
      • write_jv_curve emits the header = JV_COLUMNS and one row per point.
      • extract_jsc_ff: Jsc = |Jny(V=0)|, Voc in (0.5,0.6), 0<FF<1.
      • voc_bracket returns the sign-change interval.
    """
    t0 = 2.0e-9
    # anode photocurrent Jny sweeps +0.30 (V=0, generating: signed −0.30) up
    # through 0 near V≈0.55.  designated signed current = −|Jny|.
    biases = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8]
    Jny_vals = [0.30, 0.24, 0.12, 0.04, -0.05, -0.20]  # |·| used; sign flips
    sweep = []
    for V, jn in zip(biases, Jny_vals):
        sweep.append((V, {"jny": jn, "jpy": jn * 1.001, "t_hat": 3.0}))

    curve = build_jv_curve(sweep, t0=t0)
    assert len(curve) == len(biases)
    assert all(isinstance(p, JVPoint) for p in curve)
    assert curve[0].t == pytest.approx(3.0 * t0)
    assert curve[0].Jny == pytest.approx(0.30)
    # min convention for J
    assert curve[0].J == pytest.approx(min(abs(0.30), abs(0.30 * 1.001)))

    # write + read back header/rows
    fpath = os.path.join(tmp_path, "j_v_curve.txt")
    write_jv_curve(fpath, curve)
    lines = open(fpath).read().splitlines()
    assert lines[0].split() == list(JV_COLUMNS)
    assert len(lines) == 1 + len(curve)

    res = extract_jsc_ff(curve)
    print(f"\nG_D_2: Jsc={res['Jsc']:.4f} Voc={res['Voc']:.4f} "
          f"Pmax={res['Pmax']:.4f} FF={res['FF']:.4f}")
    assert res["Jsc"] == pytest.approx(0.30, rel=1e-9)   # |Jny(V=0)|
    assert 0.5 < res["Voc"] < 0.6, f"Voc {res['Voc']} not in (0.5,0.6)"
    assert 0.0 < res["FF"] < 1.0, f"FF {res['FF']} not a valid fill factor"

    br = voc_bracket(curve)
    assert br is not None and br[0] == pytest.approx(0.5) and br[1] == pytest.approx(0.6)

    # dark/monotone curve → no bracket, FF NaN
    dark = build_jv_curve([(V, {"jny": 0.1, "jpy": 0.1, "t_hat": 1.0})
                           for V in [0.0, 0.2, 0.4]], t0=t0)
    assert voc_bracket(dark) is None
    assert math.isnan(extract_jsc_ff(dark)["FF"])


# ══════════════════════════════════════════════════════════════════════════════
# G_D_3 — Nirmal feature vector vs closed-form values (D2)
# ══════════════════════════════════════════════════════════════════════════════

def test_gate_d3_nirmal_features_analytic():
    """G_D_3: nirmal_features on analytic traces match hand-computed values.

    TRIANGLE trace y(t) = 1 − |t − 1| on t∈[0,2] (2001 samples):
      peak = 1 (at t=1);   left_min = right_min = 0.
      left_amp = right_amp = avg_amp = 1;  asymmetry = 0.
      rise_time  (10%→90% on the linear slope-1 rising edge) = 0.9−0.1 = 0.8.
      fall_time  (90%→10% on the linear slope-1 falling edge) = 1.9−1.1 = 0.8.
      cycle_time = t[-1]−t[0] = 2;  phase_shift = (1/2)·2π = π.
      time_avg ≈ 0.5 (triangle mean).

    SINE trace y = sin(2π t) over 4 whole periods:
      fundamental bin = 4; THD ≈ 0 (single tone); only ODD harmonic order
      present (even sum = 0 → odd/even ratio = +∞-class huge).

    FOURIER-PARTIAL square y = sin + (1/3)sin(3·) + (1/5)sin(5·):
      THD = √((1/3)²+(1/5)²) = 0.388730… ;  harmonic_decay_slope ≈ −3.03
      (log|H_m| vs log m over m∈{1,3,5} = orders {1,2,3} in the ladder).
    """
    # -- triangle time features --
    t = np.linspace(0.0, 2.0, 2001)
    y = 1.0 - np.abs(t - 1.0)
    f = nirmal_features(t, y)
    print("\nG_D_3 TRIANGLE:")
    for k in ("peak_value", "left_min_value", "right_min_value",
              "time_avg_value", "left_amplitude", "right_amplitude",
              "avg_amplitude", "rise_time", "fall_time", "cycle_time",
              "phase_shift", "asymmetry"):
        print(f"  {k:18s} {f[k]:.6f}")
    assert f["peak_value"] == pytest.approx(1.0, abs=1e-9)
    assert f["left_min_value"] == pytest.approx(0.0, abs=1e-9)
    assert f["right_min_value"] == pytest.approx(0.0, abs=1e-9)
    assert f["left_amplitude"] == pytest.approx(1.0, abs=1e-9)
    assert f["right_amplitude"] == pytest.approx(1.0, abs=1e-9)
    assert f["avg_amplitude"] == pytest.approx(1.0, abs=1e-9)
    assert f["rise_time"] == pytest.approx(0.8, abs=2e-3)
    assert f["fall_time"] == pytest.approx(0.8, abs=2e-3)
    assert f["cycle_time"] == pytest.approx(2.0, abs=1e-9)
    assert f["phase_shift"] == pytest.approx(math.pi, abs=1e-3)
    assert f["asymmetry"] == pytest.approx(0.0, abs=1e-9)
    assert f["time_avg_value"] == pytest.approx(0.5, abs=1e-3)

    # -- sine FFT features --
    N = 4
    t2 = np.linspace(0.0, N, N * 256, endpoint=False)
    y2 = np.sin(2.0 * np.pi * t2)
    f2 = nirmal_features(t2, y2)
    print(f"G_D_3 SINE: h1={f2['h1_amplitude']:.4e} thd={f2['thd_ratio']:.3e} "
          f"oe={f2['odd_even_ratio']:.3e} slope={f2['harmonic_decay_slope']:.4f}")
    assert f2["thd_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert f2["h1_amplitude"] == pytest.approx(N * 256 / 2, rel=1e-6)  # rfft mag
    assert f2["odd_even_ratio"] > 1e6  # even harmonic sum ≈ 0

    # -- Fourier-partial square: closed-form THD + slope --
    y3 = (np.sin(2 * np.pi * t2)
          + (1 / 3) * np.sin(3 * 2 * np.pi * t2)
          + (1 / 5) * np.sin(5 * 2 * np.pi * t2))
    f3 = nirmal_features(t2, y3)
    thd_exp = math.sqrt((1 / 3) ** 2 + (1 / 5) ** 2)
    print(f"G_D_3 SQUARE3: thd={f3['thd_ratio']:.5f} (exp {thd_exp:.5f}) "
          f"slope={f3['harmonic_decay_slope']:.4f}")
    assert f3["thd_ratio"] == pytest.approx(thd_exp, rel=1e-4)
    # Present harmonics are at ladder orders {1,3,5} with amps {1,1/3,1/5}
    # (the even orders are float-noise, excluded).  log(1/m) vs log(m) is a
    # perfect −1 slope:
    exp_slope = np.polyfit(np.log([1.0, 3.0, 5.0]),
                           np.log([1.0, 1 / 3, 1 / 5]), 1)[0]
    assert exp_slope == pytest.approx(-1.0, abs=1e-9)
    assert f3["harmonic_decay_slope"] == pytest.approx(-1.0, rel=1e-3)


# ══════════════════════════════════════════════════════════════════════════════
# G_D_4 — IRF convolution + bandwidth (D2)
# ══════════════════════════════════════════════════════════════════════════════

def test_gate_d4_irf_and_bandwidth():
    """G_D_4: IRF of a δ-like trace returns the kernel (printed check).

    A unit spike convolved (mode='same') with a normalised kernel reproduces the
    kernel centred at the spike — the defining IRF property.  Also: Gaussian
    kernel is unit-area; an arbitrary kernel works; bandwidth low-pass reduces
    high-frequency content (a step's overshoot is smoothed).
    """
    k = gaussian_kernel(dt=1.0, sigma=2.0)
    assert k.sum() == pytest.approx(1.0, rel=1e-12)
    assert len(k) % 2 == 1  # symmetric, odd length

    spike = np.zeros(41); spike[20] = 1.0
    out = irf_convolve(spike, k)
    c = len(k) // 2
    seg = out[20 - c:20 - c + len(k)]
    max_diff = float(np.max(np.abs(seg - k)))
    print(f"\nG_D_4: IRF(δ) vs kernel max|diff| = {max_diff:.3e}; "
          f"kernel sum = {k.sum():.6f}, len = {len(k)}")
    assert max_diff < 1e-12, "G_D_4: IRF of a δ did not return the kernel"

    # arbitrary (non-Gaussian) kernel also returns itself for a δ
    kk = np.array([0.1, 0.2, 0.4, 0.2, 0.1]); kk = kk / kk.sum()
    out2 = irf_convolve(spike, kk)
    seg2 = out2[20 - 2:20 + 3]
    assert np.max(np.abs(seg2 - kk)) < 1e-12

    # bandwidth low-pass: a sharp step gets smoothed (final value preserved)
    t = np.linspace(0, 1, 200)
    step = np.where(t < 0.5, 0.0, 1.0)
    filt = apply_bandwidth(t, step, bandwidth_hz=5.0, t0=1.0)
    assert filt[-1] == pytest.approx(1.0, abs=0.05)     # settles to step value
    # rise is gradual: value right after the edge is below the raw step
    i_edge = int(np.searchsorted(t, 0.5)) + 1
    assert filt[i_edge] < 1.0
    # infinite bandwidth = passthrough
    assert np.allclose(apply_bandwidth(t, step, 0.0), step)


# ══════════════════════════════════════════════════════════════════════════════
# G_D_5 — PL uniform-slab analytic decay (D3), through a real STEADY_PULSE march
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
def test_gate_d5_pl_slab_decay():
    """G_D_5: uniform-slab PL(t) ∝ e^(−t̂/τ̂_eff), 1/τ̂_eff = 1/τ̂_r+1/τ̂_nr+k̂.

    Interface-free slab (no Onsager, no Langevin): after light-off the exciton
    volume-integral relaxes as ∫X̂ dV̂ ∝ e^(−t̂ (1/τ̂_x + k̂)) and PL = ∫X̂/τ̂_r ∝
    the same exponential.  With 1/τ̂_x = 1/τ̂_r + 1/τ̂_nr the target decay rate is
    1/τ̂_eff = 1/τ̂_r + 1/τ̂_nr + k̂.  Runs an actual STEADY_PULSE march and fits
    the dark-relaxation PL(t); the fitted rate must match τ_inv (= 1/τ̂_x + k̂,
    here k̂=0 so = 1/τ̂_x) to 1e-3 rel.

    q_r_donor = 0.5 exercises the τ_r/τ_nr split (τ_r = 2τ_x): PL uses 1/τ̂_r,
    but the DECAY RATE is set by the total sink 1/τ̂_x — the test asserts BOTH
    (the split is a PL amplitude scale, not a decay-rate change).
    """
    # τ_x = 1e-6 s → τ̂_x ≈ 0.517 (rate ≈ 1.934 nondim): an O(1) decay rate the
    # log-dt march resolves with a fixed fine dt cap (dt_max ≪ τ̂_x).
    params = XDDParams().replace(q_r_donor=0.5, q_r_acceptor=0.5)
    sysm, dm, mesh, cons, params, s, Eg_hat, tau_inv = _uniform_slab(
        level=3, p=1, tau_x=1e-6, k_hat=0.0, params=params)

    # dt̂_max = 1e-3 (fixed fine): λΔt̂ ≈ 1.9e-3 so BDF1's O(Δt) rate error is
    # ~0.1% (< the 1e-3 gate); ~500 steps span ~1 lifetime (decays to ~1/e).
    gen = Generation(params=params, profile="constant", waveform="cw")
    runner = XDDRun(
        sysm, mesh, cons, params=params, strategy=STEADY_PULSE,
        Eg_hat=Eg_hat, V_sweep=[0.0], generation=gen, G_max_hat=1.0,
        dt0_hat=1e-3, dt_max_hat=1e-3, time_stepping_tol=1e-6,
        flux_floor=1e-3, max_steps_per_stage=800, bdf2=False,
        minority_ln=-Eg_hat, h_axis=1, newton_kw={"max_iter": 20})

    result = runner.run()
    dark = result["histories"]["dark_relax"]
    # NOTE: the steady criterion need NOT fire here — a slab in continuous
    # exponential decay holds ex_norm ≈ Δt̂/τ̂_x > tol at fixed dt̂ (the field is
    # never "steady" until fully quenched).  The gate is on the decay RATE, not
    # steadiness; we require enough decaying samples to fit.

    # PL(t) from the post-processed dark-relaxation trace
    pl = pl_trace(sysm, dark["step_history"], params, t0=s.t0)
    assert np.all(np.isfinite(pl["PL_D"]))
    assert len(pl["PL_D"]) >= 20, "G_D_5: too few dark-relax samples to fit"
    assert pl["PL_D"][-1] < 0.5 * pl["PL_D"][0], "G_D_5: PL did not decay"

    # radiative-rate amplitude scale: 1/τ̂_r = q_r / τ̂_x  → 0.5·(1/τ̂_x)
    tr_inv = _tau_r_inv_hat(params, "donor")
    assert tr_inv == pytest.approx(0.5 * s.t0 / params.tau_x_donor, rel=1e-9)

    # fit the dark-relaxation decay rate over the well-resolved decay window
    # (skip the first BDF startup step; take samples while PL is above 1e-4 of
    # its start so the log-fit is not corrupted by the near-floor tail).
    pl0 = pl["PL_D"][0]
    keep = pl["PL_D"] > 1e-4 * pl0
    lam = fit_decay_rate(pl["t_hat"][keep], pl["PL_D"][keep], skip=1)
    target = tau_inv    # = 1/τ̂_x + k̂ (k̂=0 here)
    rel = abs(lam - target) / target
    print(f"\nG_D_5: fitted PL decay rate={lam:.6e} target(1/τ̂_eff)={target:.6e}"
          f" rel_err={rel:.3e}  (1/τ̂_r amp scale={tr_inv:.4e})")
    assert rel < 1e-3, (
        f"G_D_5: PL decay rate {lam:.6e} vs target {target:.6e} "
        f"off by {rel:.3e} (> 1e-3)")


# ══════════════════════════════════════════════════════════════════════════════
# G_D_6 — PL species integral + quench ratio units (D3)
# ══════════════════════════════════════════════════════════════════════════════

def test_gate_d6_pl_integral_and_quench():
    """G_D_6: pl_species_integral = ∫X̂/τ̂_r dV̂ and quench-ratio units.

    On a uniform X̂_D = c slab the integral is c·(1/τ̂_r)·V̂ (V̂ = domain volume
    = 1 in nondim device units).  Checks the analytic value, the scalar
    spectral-weight hook, and pl_quench_ratio = blend/neat.
    """
    params = XDDParams().replace(q_r_donor=0.5)
    sysm, dm, mesh, cons, params, s, Eg_hat, tau_inv = _uniform_slab(
        level=3, p=1, params=params)
    z = {pv: np.zeros(len(sysm.dist_gp[pv])) for pv in dm.bins}
    sysm.set_generation(z, z)

    c = 0.037
    n = dm.n_nodes
    state = {IPHI: np.zeros(n), IN: np.full(n, 0.1), IP: np.full(n, 0.1),
             IXD: np.full(n, c), IXA: np.zeros(n)}

    tr_inv = _tau_r_inv_hat(params, "donor")
    pl = pl_species_integral(sysm, state, tr_inv, IXD)
    # domain volume = 1 (unit square nondim); ∫c dV = c
    expected = c * tr_inv * 1.0
    print(f"\nG_D_6: ∫X̂_D/τ̂_r = {pl:.6e} expected {expected:.6e} "
          f"(1/τ̂_r={tr_inv:.4e})")
    assert pl == pytest.approx(expected, rel=1e-6)

    # spectral weight scales linearly
    pl_w = pl_species_integral(sysm, state, tr_inv, IXD, weight=0.25)
    assert pl_w == pytest.approx(0.25 * pl, rel=1e-9)

    # quench ratio
    assert pl_quench_ratio(0.3, 1.2) == pytest.approx(0.25)
    assert math.isnan(pl_quench_ratio(0.3, 0.0))


# ══════════════════════════════════════════════════════════════════════════════
# G_D_7 — J(t) trace capture from a march (D2)
# ══════════════════════════════════════════════════════════════════════════════

def test_gate_d7_capture_trace():
    """G_D_7: capture_trace assembles (t̂,t,J,Jny,Jpy) from a step_history."""
    t0 = 1.5e-9
    hist = [
        {"t_hat": 1e-6, "jny": 0.10, "jpy": 0.11, "accepted": True},
        {"t_hat": 2e-6, "jny": 0.20, "jpy": 0.19, "accepted": True},
        {"t_hat": 3e-6, "jny": 0.05, "jpy": 0.05, "accepted": False},  # rejected
        {"t_hat": 3e-6, "jny": 0.30, "jpy": 0.28, "accepted": True},
    ]
    tr = capture_trace(hist, t0=t0, contact="report")
    assert len(tr["t_hat"]) == 3           # rejected step dropped
    assert tr["t"][0] == pytest.approx(1e-6 * t0)
    assert tr["J"][0] == pytest.approx(min(0.10, 0.11))
    assert tr["J"][2] == pytest.approx(min(0.30, 0.28))
    tra = capture_trace(hist, t0=t0, contact="anode")
    assert tra["J"][1] == pytest.approx(0.20)
