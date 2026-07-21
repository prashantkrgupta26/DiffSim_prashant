"""XDD instrument-twin observables layer (SP-1 Block D).

The instrument-twin OUTPUTS that Block E's parity/perf gates consume — pure
functions over converged states and marched traces (R1 will differentiate them,
so there is NO hidden mutation of the system here).  Spec §4 (the twin table);
plan D1–D3.

Three groups:

D1 — consistent-flux J + J–V machinery
    contact_flux_pair()   the bulk-adjacent (Jny,Jpy) nondim contact pair,
                          promoting run._flux_pair to the public API.
    to_mA_per_cm2()       re-dimension J0·0.1 (CPU units convention).
    report_current()      J = min(|Jny|,|Jpy|) — parity REPORTING only.
    designated_current()  smooth/designated-contact variant for gradients.
    JVPoint / build_jv_curve()   the CPU `j_v_curve` column layout.
    extract_jsc_ff()      Jsc (J at V=0) + fill factor.
    voc_bracket()         sign-change bracket of J(V)=0 for a Voc root solve.

D2 — J(t) + features + IRF
    capture_trace()       (t̂, t[s], J) trace from a march step_history.
    nirmal_features()     the EXACT Nirmal feature vector (time + FFT features).
    irf_convolve()        IRF convolution (Gaussian σ or arbitrary kernel).
    apply_bandwidth()     first-order bandwidth (RC) low-pass on a J(t) trace.

D3 — PL observables
    pl_trace()            PL_i(t) = ∫ X̂_i / τ̂_r,i dV per species (nodal/GP
                          quadrature integral), optional scalar spectral weight.
    pl_quench_ratio()     steady blend-vs-neat PL quench ratio.
    fit_decay_rate()      log-linear fit of an exponential decay (slab check).

Units.  Every current is nondimensional unless a `_mA` suffix or an explicit
J0 re-dimensionalisation is applied.  Nondim time t̂; dimensional time t = t̂·t0.
PL is reported nondim (∫X̂/τ̂_r dV̂) — an amplitude, spectrally weightable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from diffsim.xdd.run import _flux_pair, _post_process


# ══════════════════════════════════════════════════════════════════════════════
# D1 — consistent-flux J + J–V machinery
# ══════════════════════════════════════════════════════════════════════════════

def contact_flux_pair(sysm, state, h_axis: int = 1) -> Tuple[float, float]:
    """Public bulk-adjacent consistent-flux contact pair (Jny, Jpy), nondim.

    Promotes run._flux_pair (the DDJscBulk-equivalent discretely-conservative
    contact residual: steady σ=0, SUPG=0 carrier stiffness summed over the
    contact Dirichlet nodes) to the public observable API.  Jny = net electron
    current into the anode (ĥ=0); Jpy = net hole current into the cathode (ĥ=1).
    Pure function of the converged `state` (no system mutation).

    Returns
    -------
    (Jny, Jpy) : float, float   nondim (multiply by J0 for A/m²).
    """
    return _flux_pair(sysm, state, h_axis=h_axis)


def to_mA_per_cm2(J_nondim: float, J0: float) -> float:
    """Re-dimension a nondim current to mA/cm².

    J[A/m²] = J_nondim · J0;  mA/cm² = A/m² · 0.1  (1 A/m² = 0.1 mA/cm²).
    The combined factor J0·0.1 is the CPU convention (DDJscBulk × 0.1).
    """
    return J_nondim * J0 * 0.1


def report_current(Jny: float, Jpy: float) -> float:
    """CPU parity-REPORTING current  J = min(|Jny|, |Jpy|).

    NONSMOOTH (a hard min); used ONLY for parity reporting.  Gradients must use
    `designated_current` (a fixed contact) or a smooth-min surrogate.
    """
    return min(abs(Jny), abs(Jpy))


def designated_current(Jny: float, Jpy: float, contact: str = "anode",
                       beta: Optional[float] = None) -> float:
    """Differentiable current for gradients.

    contact="anode"   → |Jny|         (designated electron contact)
    contact="cathode" → |Jpy|         (designated hole contact)
    contact="softmin" → smooth-min  −(1/β) log(e^{−β|Jny|}+e^{−β|Jpy|})
                        (β = beta, default 50) — a C^∞ surrogate for min that
                        R1 can backprop through where the hard min is nonsmooth.

    The designated-contact forms are the spec §4 gradient convention (the min
    is kept only for parity reporting).
    """
    an, ca = abs(Jny), abs(Jpy)
    if contact == "anode":
        return an
    if contact == "cathode":
        return ca
    if contact == "softmin":
        b = 50.0 if beta is None else float(beta)
        m = min(an, ca)                       # shift for numerical stability
        return m - (1.0 / b) * math.log(
            math.exp(-b * (an - m)) + math.exp(-b * (ca - m)))
    raise ValueError(f"designated_current: unknown contact {contact!r}")


@dataclass(frozen=True)
class JVPoint:
    """One row of the CPU `j_v_curve` file (spec §4 / D1 column layout).

    Columns: V_app [V], t̂ [-], t [s], J, Jny, Jpy  (all nondim currents; a
    dimensional column is produced by `build_jv_curve(..., J0=...)`).
    """
    V_app: float     # [V]   applied voltage
    t_hat: float     # [-]   nondim time at which the bias reached steady
    t: float         # [s]   dimensional time  = t̂ · t0
    J: float         # [-]   report_current(Jny,Jpy) (nondim; min convention)
    Jny: float       # [-]   electron-into-anode contact current
    Jpy: float       # [-]   hole-into-cathode contact current


# CPU j_v_curve column header (exact column layout for the writer).
JV_COLUMNS = ("V_app", "t_hat", "t", "J", "Jny", "Jpy")


def build_jv_curve(sweep_history, *, t0: float, V_scale: float = 1.0,
                   J0: Optional[float] = None) -> List[JVPoint]:
    """Assemble a J–V curve (list of JVPoint) from a run_voltage_sweep history.

    Parameters
    ----------
    sweep_history : list of (V_hat, march_info)
        Each march_info carries `jny`, `jpy`, `t_hat` (from _march_to_steady).
    t0 : float
        Nondim time scale [s] (params.scales().t0) → dimensional t column.
    V_scale : float
        Multiplier from nondim V̂ to volts (= phi0).  Default 1 (already volts).
    J0 : float or None
        If given, currents are re-dimensioned to mA/cm² (× J0·0.1); else nondim.

    Returns
    -------
    list of JVPoint
    """
    rows: List[JVPoint] = []
    for V_hat, info in sweep_history:
        jny = float(info["jny"]); jpy = float(info["jpy"])
        t_hat = float(info.get("t_hat", 0.0))
        J = report_current(jny, jpy)
        if J0 is not None:
            jny = to_mA_per_cm2(jny, J0)
            jpy = to_mA_per_cm2(jpy, J0)
            J = to_mA_per_cm2(J, J0)
        rows.append(JVPoint(
            V_app=V_hat * V_scale, t_hat=t_hat, t=t_hat * t0,
            J=J, Jny=jny, Jpy=jpy))
    return rows


def write_jv_curve(path: str, curve: Sequence[JVPoint]) -> None:
    """Write a J–V curve to a whitespace-delimited file (CPU column layout).

    Header line = JV_COLUMNS; one row per JVPoint.  Pure I/O (deterministic).
    """
    with open(path, "w") as f:
        f.write(" ".join(JV_COLUMNS) + "\n")
        for r in curve:
            f.write(f"{r.V_app:.8e} {r.t_hat:.8e} {r.t:.8e} "
                    f"{r.J:.8e} {r.Jny:.8e} {r.Jpy:.8e}\n")


def _signed_current(pt: JVPoint) -> float:
    """Photocurrent sign convention: J(V) with the generating photocurrent < 0.

    The consistent-flux Jny is POSITIVE at short circuit (photogenerated
    electrons extracted at the anode) and crosses zero at Voc, going negative
    past it (forward injection dominates).  A power-generating photodiode J–V is
    conventionally drawn with J < 0 in that quadrant, so we negate the
    SIGN-PRESERVING Jny (not |Jny|): J_signed = −Jny.  This makes the V=0 value
    −Jsc (<0) and produces the sign change that brackets Voc.
    """
    return -pt.Jny


def extract_jsc_ff(curve: Sequence[JVPoint]) -> dict:
    """Extract Jsc and fill factor from a light J–V curve.

    Jsc  = |J(V=0)|            (short-circuit current; interpolated if V=0 absent)
    Voc  = V where J(V)=0      (linear interpolation across the sign change)
    Pmax = max over the curve of |V · J|  in the generating quadrant (V∈[0,Voc])
    FF   = Pmax / (Jsc · Voc)

    Uses the signed photocurrent convention (generating photocurrent negative).
    Returns a dict {Jsc, Voc, Pmax, FF} (FF/Voc = NaN if no sign change is
    bracketed — a dark or monotone curve).
    """
    if len(curve) < 2:
        raise ValueError("extract_jsc_ff needs >= 2 J–V points")
    V = np.array([p.V_app for p in curve], float)
    J = np.array([_signed_current(p) for p in curve], float)
    order = np.argsort(V)
    V = V[order]; J = J[order]

    # Jsc = |J at V=0| (interpolate)
    Jsc = abs(float(np.interp(0.0, V, J)))

    # Voc: first sign change of J(V)
    Voc = float("nan")
    for i in range(len(V) - 1):
        if J[i] == 0.0:
            Voc = V[i]; break
        if J[i] * J[i + 1] < 0.0:
            # linear interp for the root
            Voc = V[i] - J[i] * (V[i + 1] - V[i]) / (J[i + 1] - J[i])
            break

    Pmax = 0.0
    FF = float("nan")
    if math.isfinite(Voc) and Voc > 0 and Jsc > 0:
        # power = |V·J| in the generating quadrant V in [0, Voc]
        for p in curve:
            if 0.0 <= p.V_app <= Voc:
                power = abs(p.V_app * _signed_current(p))
                Pmax = max(Pmax, power)
        FF = Pmax / (Jsc * Voc)
    return {"Jsc": Jsc, "Voc": Voc, "Pmax": Pmax, "FF": FF}


def voc_bracket(curve: Sequence[JVPoint]) -> Optional[Tuple[float, float]]:
    """Return a (V_lo, V_hi) bracket enclosing the Voc root J(V)=0, or None.

    A helper for an implicit-function-theorem Voc solve (R1): finds the first
    adjacent pair of biases where the signed photocurrent changes sign.
    """
    V = [p.V_app for p in curve]
    J = [_signed_current(p) for p in curve]
    order = np.argsort(V)
    V = [V[i] for i in order]; J = [J[i] for i in order]
    for i in range(len(V) - 1):
        if J[i] == 0.0:
            return (V[i], V[i])
        if J[i] * J[i + 1] < 0.0:
            return (V[i], V[i + 1])
    return None


# ══════════════════════════════════════════════════════════════════════════════
# D2 — J(t) traces, Nirmal feature vector, IRF convolution
# ══════════════════════════════════════════════════════════════════════════════

def capture_trace(step_history, *, t0: float, contact: str = "report",
                  h_axis: int = 1) -> dict:
    """Capture a J(t) trace from a march's step_history (output cadence).

    Each accepted step of _march_to_steady records (t_hat, jny, jpy).  This
    assembles them into arrays at the march's natural output cadence.

    Parameters
    ----------
    step_history : list of step dicts (accepted steps carry jny, jpy, t_hat).
    t0 : float                nondim time scale [s].
    contact : str
        "report" → J = min(|Jny|,|Jpy|);  "anode" → |Jny|;  "cathode" → |Jpy|.
    h_axis : int              (unused; carried for API symmetry).

    Returns
    -------
    dict {t_hat[N], t[N], J[N], Jny[N], Jpy[N]}  (numpy arrays).
    """
    steps = [r for r in step_history if r.get("accepted", True)]
    t_hat = np.array([r["t_hat"] for r in steps], float)
    jny = np.array([r["jny"] for r in steps], float)
    jpy = np.array([r["jpy"] for r in steps], float)
    if contact == "report":
        J = np.minimum(np.abs(jny), np.abs(jpy))
    elif contact == "anode":
        J = np.abs(jny)
    elif contact == "cathode":
        J = np.abs(jpy)
    else:
        raise ValueError(f"capture_trace: unknown contact {contact!r}")
    return {"t_hat": t_hat, "t": t_hat * t0, "J": J, "Jny": jny, "Jpy": jpy}


def _rise_time(t: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    """Time for a rising signal to go from lo·amp to hi·amp of its peak.

    Interpolates the first up-crossings of the two thresholds (measured from
    the left minimum to the peak).  Returns NaN if not resolvable.
    """
    ymin = float(np.min(y)); ymax = float(np.max(y))
    amp = ymax - ymin
    if amp <= 0:
        return float("nan")
    ipk = int(np.argmax(y))
    lo_v = ymin + lo * amp
    hi_v = ymin + hi * amp
    t_lo = _first_crossing(t[:ipk + 1], y[:ipk + 1], lo_v, rising=True)
    t_hi = _first_crossing(t[:ipk + 1], y[:ipk + 1], hi_v, rising=True)
    if t_lo is None or t_hi is None:
        return float("nan")
    return t_hi - t_lo


def _fall_time(t: np.ndarray, y: np.ndarray, hi: float, lo: float) -> float:
    """Time for a falling signal to go from hi·amp to lo·amp of its peak.

    Measured from the peak toward the right minimum.
    """
    ymin = float(np.min(y)); ymax = float(np.max(y))
    amp = ymax - ymin
    if amp <= 0:
        return float("nan")
    ipk = int(np.argmax(y))
    lo_v = ymin + lo * amp
    hi_v = ymin + hi * amp
    t_hi = _first_crossing(t[ipk:], y[ipk:], hi_v, rising=False)
    t_lo = _first_crossing(t[ipk:], y[ipk:], lo_v, rising=False)
    if t_lo is None or t_hi is None:
        return float("nan")
    return t_lo - t_hi


def _first_crossing(t, y, level, rising=True):
    """Linear-interpolated first crossing of `level` in the (t,y) segment."""
    for i in range(len(y) - 1):
        a, b = y[i], y[i + 1]
        if rising and a <= level <= b and b != a:
            return t[i] + (level - a) * (t[i + 1] - t[i]) / (b - a)
        if (not rising) and a >= level >= b and b != a:
            return t[i] + (level - a) * (t[i + 1] - t[i]) / (b - a)
    return None


def _skewness(y: np.ndarray) -> float:
    """Fisher-Pearson skewness of the sample distribution of values."""
    m = float(np.mean(y))
    s = float(np.std(y))
    if s == 0:
        return 0.0
    return float(np.mean((y - m) ** 3) / s ** 3)


def nirmal_features(t: np.ndarray, y: np.ndarray, *,
                    period: Optional[float] = None) -> dict:
    """The Nirmal light-modulation J(t) feature vector (plan D2, spec §4).

    Time-domain features (all on the trace y(t)):
      peak_value        max(y)
      left_min_value    min over the rising segment (t < argmax)
      right_min_value   min over the falling segment (t > argmax)
      time_avg_value    mean(y)
      left_amplitude    peak − left_min
      right_amplitude   peak − right_min
      avg_amplitude     (left_amplitude + right_amplitude)/2
      rise_time         10%→90% amplitude time on the rising edge
      fall_time         90%→10% amplitude time on the falling edge
      cycle_time        `period` if supplied else t[-1] − t[0]
      phase_shift       (t_peak − t[0]) / cycle_time · 2π   [rad]
      skewness_left     Fisher skewness of the rising-segment values
      skewness_right    Fisher skewness of the falling-segment values
      asymmetry         (left_amplitude − right_amplitude) / avg_amplitude

    FFT features (real FFT of the mean-subtracted trace):
      h1_amplitude          |first harmonic| (largest non-DC bin magnitude)
      thd_ratio             √(Σ_{k≥2}|H_k|²) / |H_1|   total harmonic distortion
      odd_even_ratio        Σ_odd|H_k| / Σ_even|H_k|   (k≥2 for even)
      harmonic_decay_slope  slope of log|H_k| vs log k  (k=1..K linear fit)

    Pure function of (t, y).  NaN where a feature is not resolvable (flat trace,
    <2 harmonics, etc.).
    """
    t = np.asarray(t, float); y = np.asarray(y, float)
    n = len(y)
    ipk = int(np.argmax(y))
    peak = float(y[ipk])

    left = y[:ipk + 1]
    right = y[ipk:]
    left_min = float(np.min(left)) if len(left) else peak
    right_min = float(np.min(right)) if len(right) else peak
    time_avg = float(np.mean(y))

    left_amp = peak - left_min
    right_amp = peak - right_min
    avg_amp = 0.5 * (left_amp + right_amp)

    rise = _rise_time(t, y, 0.10, 0.90)
    fall = _fall_time(t, y, 0.90, 0.10)

    if period is not None:
        cycle = float(period)
    else:
        cycle = float(t[-1] - t[0]) if n > 1 else float("nan")
    if cycle and math.isfinite(cycle) and cycle != 0:
        phase = ((t[ipk] - t[0]) / cycle) * 2.0 * math.pi
    else:
        phase = float("nan")

    skew_l = _skewness(left) if len(left) > 2 else 0.0
    skew_r = _skewness(right) if len(right) > 2 else 0.0

    asym = (left_amp - right_amp) / avg_amp if avg_amp != 0 else float("nan")

    # -- FFT features --
    # The fundamental is the largest non-DC magnitude bin (bin index k0);
    # harmonics H_m are then the bins at m·k0.  Keying off the fundamental (not
    # bin 1) makes the features period-agnostic: a signal with N whole periods
    # over the window has its fundamental at bin N, and THD/odd-even/slope are
    # measured on the true harmonic ladder m·k0.
    yac = y - np.mean(y)
    H = np.abs(np.fft.rfft(yac))
    h1 = float("nan"); thd = float("nan")
    oe = float("nan"); slope = float("nan")
    if len(H) >= 3:
        ac = H[1:]                                   # drop DC
        k0 = int(np.argmax(ac)) + 1                  # fundamental bin index
        if k0 >= 1:
            max_order = (len(H) - 1) // k0           # how many harmonics fit
            orders = np.arange(1, max_order + 1)
            harm = H[orders * k0]                    # H_1, H_2, ... at m·k0
            h1 = float(harm[0])
            # a harmonic counts as "present" only above float-noise relative to
            # the fundamental (rfft leaks ~1e-16·h1 into the null bins; without
            # this threshold those pollute the odd/even ratio and slope fit).
            present = harm > 1e-9 * max(h1, 1e-300)
            if h1 > 0 and len(harm) >= 2:
                rest = harm[1:]
                thd = float(np.sqrt(np.sum(rest ** 2)) / h1)
            odd = float(np.sum(harm[present & (orders % 2 == 1)]))
            even = float(np.sum(harm[present & (orders % 2 == 0)]))
            # even == 0 with odd > 0 → all energy in odd harmonics (→ ∞);
            # both zero (flat) → NaN.
            if even > 0:
                oe = odd / even
            elif odd > 0:
                oe = float("inf")
            else:
                oe = float("nan")
            pos = present
            if np.count_nonzero(pos) >= 2:
                lk = np.log(orders[pos].astype(float))
                lh = np.log(harm[pos])
                slope = float(np.polyfit(lk, lh, 1)[0])

    return {
        "peak_value": peak,
        "left_min_value": left_min,
        "right_min_value": right_min,
        "time_avg_value": time_avg,
        "left_amplitude": left_amp,
        "right_amplitude": right_amp,
        "avg_amplitude": avg_amp,
        "rise_time": rise,
        "fall_time": fall,
        "cycle_time": cycle,
        "phase_shift": phase,
        "skewness_left": skew_l,
        "skewness_right": skew_r,
        "asymmetry": asym,
        "h1_amplitude": h1,
        "thd_ratio": thd,
        "odd_even_ratio": oe,
        "harmonic_decay_slope": slope,
    }


def gaussian_kernel(dt: float, sigma: float, n_sigma: float = 4.0) -> np.ndarray:
    """Unit-area Gaussian IRF kernel sampled at spacing dt, width σ.

    Support = ±n_sigma·σ (odd length so it is symmetric about the centre).
    Normalised to sum 1 so convolution preserves the trace integral.
    """
    if sigma <= 0:
        return np.array([1.0])
    half = max(1, int(math.ceil(n_sigma * sigma / dt)))
    x = np.arange(-half, half + 1) * dt
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return k


def irf_convolve(y: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve a trace y with a (normalised) IRF kernel, same length out.

    `mode='same'` centred convolution — a δ-like trace (single unit spike)
    returns the kernel itself (centred at the spike).  Differentiable (a linear
    operator on y), so R1 can propagate through it.

    For a Gaussian IRF use `gaussian_kernel(dt, sigma)` to build `kernel`; any
    arbitrary (measured) kernel array is accepted directly.
    """
    y = np.asarray(y, float); kernel = np.asarray(kernel, float)
    return np.convolve(y, kernel, mode="same")


def apply_bandwidth(t: np.ndarray, y: np.ndarray, bandwidth_hz: float,
                    *, t0: float = 1.0) -> np.ndarray:
    """First-order (RC) bandwidth low-pass on a J(t) trace.

    Models instrument bandwidth as a single-pole RC filter with cutoff
    `bandwidth_hz`: τ_RC = 1/(2π f_c).  Discrete one-pole IIR
        y_f[i] = y_f[i-1] + α (y[i] − y_f[i-1]),  α = Δt/(τ_RC + Δt).
    `t` is nondim; the dimensional Δt = Δt̂·t0 sets the physical cutoff.
    bandwidth_hz ≤ 0 → passthrough (infinite bandwidth).
    """
    y = np.asarray(y, float)
    if bandwidth_hz <= 0 or len(y) < 2:
        return y.copy()
    tau_rc = 1.0 / (2.0 * math.pi * bandwidth_hz)   # [s]
    out = np.empty_like(y)
    out[0] = y[0]
    tt = np.asarray(t, float) * t0                  # dimensional time [s]
    for i in range(1, len(y)):
        dt = tt[i] - tt[i - 1]
        alpha = dt / (tau_rc + dt) if (tau_rc + dt) > 0 else 1.0
        out[i] = out[i - 1] + alpha * (y[i] - out[i - 1])
    return out


# ══════════════════════════════════════════════════════════════════════════════
# D3 — PL observables
# ══════════════════════════════════════════════════════════════════════════════

def _tau_r_inv_hat(params, species: str) -> float:
    """Nondim radiative decay rate 1/τ̂_r,i = t0 / τ_r,i for a species.

    τ_r,i = τ_x,i / q_r,i (the A3 radiative/non-radiative split); with q_r=1,
    τ_r = τ_x so 1/τ̂_r = 1/τ̂_x.  Species = "donor" | "acceptor".
    """
    s = params.scales()
    if species == "donor":
        tau_r = params.tau_r_donor
    elif species == "acceptor":
        tau_r = params.tau_r_acceptor
    else:
        raise ValueError(f"_tau_r_inv_hat: unknown species {species!r}")
    return s.t0 / tau_r


def pl_species_integral(sysm, state, tau_r_inv_hat: float, field_index: int,
                        weight: float = 1.0) -> float:
    """∫ w · X̂_i / τ̂_r,i dV̂  — the radiative PL amplitude of one species.

    GP-quadrature volume integral of the radiative recombination rate
    X̂_i / τ̂_r,i (spec §4: PL_i = ∫ X̂_i/τ̂_r,i dV), optionally scaled by a
    scalar spectral weight `weight` (a hook for per-species spectral weighting;
    full spectra land later).  Pure function of the state.

    `field_index` ∈ {IXD=3, IXA=4}; `tau_r_inv_hat` = 1/τ̂_r,i from
    `_tau_r_inv_hat`.
    """
    from diffsim.physics.exciton_system import IXD
    dm = sysm.dm
    h_all = dm.mesh.tree.h()
    cl = sysm._closures(state)             # evaluate GP fields once
    total = 0.0
    for pv, b in dm.bins.items():
        eids = dm.mesh.bins[pv]
        he = h_all[eids]
        jac = (0.5 * he) ** dm.dim
        nqp = b["nqp"]
        wt = dm.tables_by_p[pv].w
        ne = len(eids)
        dV = np.tile(wt, ne) * np.repeat(jac, nqp)
        # X̂_i at GPs for this field (donor→xd_gp, acceptor→xa_gp)
        x_gp = cl["xd_gp"][pv] if field_index == IXD else cl["xa_gp"][pv]
        total += float(np.sum(dV * x_gp))
    return weight * tau_r_inv_hat * total


def pl_trace(sysm, step_history, params, *, t0: float,
             weight_donor: float = 1.0, weight_acceptor: float = 1.0,
             recompute_states: Optional[Sequence] = None) -> dict:
    """PL_i(t) traces per species from a STEADY_PULSE (or any) march.

    When the march was run with post_process=True the step records carry the
    volume integrals ∫X̂_D and ∫X̂_A (`int_xd`, `int_xa`); PL_i(t) = weight_i ·
    (1/τ̂_r,i) · ∫X̂_i dV̂ per step.  This is the differentiable TRPL observable
    the spec's STEADY_PULSE protocol produces (radiative rate, not the total
    ∫X̂/τ̂_x which includes non-radiative decay).

    Parameters
    ----------
    sysm : XDDSystem   (unused when the trace has post_process; kept for API).
    step_history : list of step dicts (accepted; carry `post_process`).
    params : XDDParams   (for τ_r via q_r).
    t0 : float           nondim time scale [s].
    weight_donor, weight_acceptor : float   scalar spectral weights.

    Returns
    -------
    dict {t_hat[N], t[N], PL_D[N], PL_A[N], PL[N]}  (PL = PL_D + PL_A).
    """
    tr_d = _tau_r_inv_hat(params, "donor")
    tr_a = _tau_r_inv_hat(params, "acceptor")
    steps = [r for r in step_history
             if r.get("accepted", True) and "post_process" in r]
    if not steps:
        raise ValueError("pl_trace: step_history has no post_process records "
                         "(run the march with post_process=True)")
    t_hat = np.array([r["t_hat"] for r in steps], float)
    ixd = np.array([r["post_process"]["int_xd"] for r in steps], float)
    ixa = np.array([r["post_process"]["int_xa"] for r in steps], float)
    PL_D = weight_donor * tr_d * ixd
    PL_A = weight_acceptor * tr_a * ixa
    return {"t_hat": t_hat, "t": t_hat * t0,
            "PL_D": PL_D, "PL_A": PL_A, "PL": PL_D + PL_A}


def pl_quench_ratio(pl_blend: float, pl_neat: float) -> float:
    """Steady PL-quench ratio  PL_blend / PL_neat.

    The classic quench observable: the ratio of the blend's steady PL to a
    τ-only neat-film reference (no interface dissociation).  < 1 (quenched)
    when the interface drains excitons.  `pl_neat` is the neat-film ∫X̂/τ̂_r
    reference amplitude.
    """
    if pl_neat == 0:
        return float("nan")
    return pl_blend / pl_neat


def fit_decay_rate(t: np.ndarray, y: np.ndarray, *,
                   skip: int = 0) -> float:
    """Fit a single-exponential decay rate λ from y(t) ≈ A e^(−λ t).

    Log-linear least squares: λ = −slope of ln y vs t (positive y only, after
    skipping the first `skip` samples to drop a startup transient).  Used by the
    D3 slab check (PL ∝ e^(−t̂/τ̂_eff), 1/τ̂_eff = 1/τ̂_r + 1/τ̂_nr + k̂).

    Returns the decay rate in the same (nondim) inverse-time units as t.
    """
    t = np.asarray(t, float)[skip:]
    y = np.asarray(y, float)[skip:]
    m = y > 0
    if np.count_nonzero(m) < 2:
        return float("nan")
    slope = np.polyfit(t[m], np.log(y[m]), 1)[0]
    return -float(slope)


# ══════════════════════════════════════════════════════════════════════════════
# R1 — differentiable steady QoI faces (steady adjoint RHS seeds)
# ══════════════════════════════════════════════════════════════════════════════

class SteadyCurrentQoI:
    """Differentiable designated contact current as a steady QoI.

    ``value`` = ``designated_current(Jny, Jpy, contact)`` (a fixed contact, NOT
    the nonsmooth min).  The adjoint flows from ``dJ_du``, the flat reduced
    field-major seed (5·n_free,) consumed as the Mode-A steady-adjoint RHS
    (Task 4 passes a matching flat reduced λ).

    Design (see task-3 brief / ``run._flux_pair``).  With the designated
    contact ``Jc = Σ_{i∈wall} (Kc @ ĉ)[i]`` where ``Kc = _carrier_block(σ=0,
    SUPG=0)`` (electrons at the anode, holes at the cathode):

      • ∂Jc/∂ĉ  = Kcᵀ e_wall           (Kc is φ̂-dependent but ĉ-independent).
      • ∂Jc/∂φ̂ = the CONSERVATIVE drift cross-term — the SAME weighted-stiffness
        block the residual Jacobian uses (exciton_system ``build_jacobian``):
        ``∂(∫ sign·μ̂ ĉ (∇φ̂·∇N_a))/∂φ̂ = _poisson_block(sign·μ̂·ĉ, lam2=1)``,
        NOT the (∇ĉ·∇N_b)N_a advection-mass block.

    ``designated_current`` returns |Jc|, so the seed carries an overall
    sign(Jc).  Assembled dR is transposed and reduced with ``sysm.T.T`` per
    field into a flat (5·n_free,) vector.
    """

    def __init__(self, contact: str = "anode", h_axis: int = 1):
        if contact not in ("anode", "cathode"):
            raise ValueError(
                "SteadyCurrentQoI: contact must be 'anode' or 'cathode' "
                f"(got {contact!r}); the nonsmooth min is not differentiable.")
        self.contact = contact
        self.h_axis = h_axis

    def value(self, sysm, state) -> float:
        jny, jpy = contact_flux_pair(sysm, state, h_axis=self.h_axis)
        return designated_current(jny, jpy, contact=self.contact)

    def dJ_du(self, sysm, state) -> np.ndarray:
        """∂J/∂u as a flat (5·n_free,) reduced field-major seed (adjoint RHS)."""
        from diffsim.physics.exciton_system import (
            _carrier_block, _poisson_block, NDOF, IPHI, IN, IP)

        dm = sysm.dm
        cl = sysm._closures(state)
        coords = dm.mesh.node_coords
        hc = coords[:, self.h_axis]
        lo, hi = hc.min(), hc.max()
        if self.contact == "anode":
            wall = np.where(np.abs(hc - lo) < 1e-9)[0]
            mu_gp, fld, sign = sysm.mu_n_gp, IN, -1.0
            c_gp = cl["n_gp"]
        else:
            wall = np.where(np.abs(hc - hi) < 1e-9)[0]
            mu_gp, fld, sign = sysm.mu_p_gp, IP, +1.0
            c_gp = cl["p_gp"]

        # ĉ-flux Galerkin operator Kc (σ=0, SUPG=0) — the _flux_pair operator.
        aq = sysm._aq(cl["gradphi"], mu_gp, sign)
        Kc = _carrier_block(dm, aq, mu_gp, 0.0, 0.0, 0.0)
        e = np.zeros(dm.n_nodes)
        e[wall] = 1.0

        # ∂(Σ_wall Kc@ĉ)/∂ĉ = Kcᵀ e   (Kc independent of ĉ).
        dJ_dc = Kc.T @ e
        # ∂(Σ_wall Kc@ĉ)/∂φ̂ = conservative drift cross-term (weighted stiffness):
        #   ∂/∂φ̂ ∫ sign·μ̂ ĉ (∇φ̂·∇N_a) = _poisson_block(sign·μ̂·ĉ, lam2=1).
        cdrift = {pv: sign * mu_gp[pv] * c_gp[pv] for pv in dm.bins}
        Kphi = _poisson_block(dm, cdrift, 1.0)
        dJ_dphi = Kphi.T @ e

        # designated_current = |Jc| → overall sign(Jc).
        jny, jpy = contact_flux_pair(sysm, state, h_axis=self.h_axis)
        Jc = jny if self.contact == "anode" else jpy
        sgn = 1.0 if Jc >= 0 else -1.0

        dR = {f: np.zeros(dm.n_nodes) for f in range(NDOF)}
        dR[fld] = sgn * dJ_dc
        dR[IPHI] = sgn * dJ_dphi
        return np.concatenate(
            [np.asarray(sysm.T.T @ dR[f]) for f in range(NDOF)])

    def dJ_dp(self, sysm, state, control) -> np.ndarray:
        """Explicit ∂J/∂p for a control — 0 for this pure-state QoI.

        Returned for forward-compat req 2 (the QoI exposes both the dJ/du seed
        and the ∂J/∂p sensitivity face; Task 4's driver builds the full
        sensitivity rows from these plus the adjoint).
        """
        return np.zeros(control.size)


class JVMisfitQoI:
    """Full-curve J–V misfit  Σ_V (J_model(V) − J_data(V))²  as a steady QoI.

    R1 keeps the SINGLE-BIAS reduced form: the driver holds one operating bias
    and this evaluates the one misfit term there (the full sweep is "sum the
    single-bias adjoint over sweep points" — each bias contributes an
    independent adjoint solve with this same seed at that bias).

    value  = (J_model − J_data)²
    dJ/du  = 2 (J_model − J_data) · dJ_model/du   (chain rule through the
             differentiable designated current).
    """

    def __init__(self, target_current: float, contact: str = "anode",
                 h_axis: int = 1):
        self.target = float(target_current)
        self._cur = SteadyCurrentQoI(contact=contact, h_axis=h_axis)

    def value(self, sysm, state) -> float:
        return (self._cur.value(sysm, state) - self.target) ** 2

    def dJ_du(self, sysm, state) -> np.ndarray:
        r = self._cur.value(sysm, state) - self.target
        return 2.0 * r * self._cur.dJ_du(sysm, state)

    def dJ_dp(self, sysm, state, control) -> np.ndarray:
        return np.zeros(control.size)


# ══════════════════════════════════════════════════════════════════════════════
# R1 — differentiable TRANSIENT QoI faces (trajectory functionals; Mode B)
# ══════════════════════════════════════════════════════════════════════════════

def _params_of(sysm):
    """Best-effort XDDParams handle for τ̂_r (attached by the fixture/driver)."""
    p = getattr(sysm, "params", None)
    if p is None:
        raise ValueError("TRPLMisfitQoI needs params: pass params=... or set "
                         "sysm.params")
    return p


def _pl_integral_grad_nodal(sysm, field_index, tau_r_inv_hat, weight):
    """∂(∫ w·X̂/τ̂_r dV)/∂X̂_nodal = weight·(1/τ̂_r)·(Nᵀ dV) — the GP-quadrature
    adjoint of the nodal→GP interpolation used by ``pl_species_integral``.

    Returns the (n_nodes,) nodal cotangent of the LINEAR PL-integral functional;
    contracting it with a nodal δX̂ reproduces the exact directional derivative of
    ``pl_species_integral`` (verified independently by the FD gate)."""
    dm = sysm.dm
    h_all = dm.mesh.tree.h()
    g = np.zeros(dm.n_nodes)
    for pv, b in dm.bins.items():
        conn = dm.mesh.conn_of[pv]
        N = dm.tables_by_p[pv].N               # [nqp, nbf]
        eids = dm.mesh.bins[pv]; he = h_all[eids]
        jac = (0.5 * he) ** dm.dim; nqp = b["nqp"]
        wt = dm.tables_by_p[pv].w; ne = len(eids)
        dV = (np.tile(wt, ne) * np.repeat(jac, nqp)).reshape(ne, nqp)
        # contribution to node a: Σ_q dV[e,q] N[q,a]
        ge = np.einsum("eq,qa->ea", dV, N)     # [ne, nbf]
        np.add.at(g, conn.ravel(), ge.ravel())
    return weight * tau_r_inv_hat * g


class TRPLMisfitQoI:
    """TRPL decay misfit  Σ_n (PL(tₙ) − PL_data(tₙ))²  as a TRAJECTORY QoI.

    ``PL(tₙ) = Σ_i weight_i·(1/τ̂_r,i)·∫X̂_i dV̂`` (the R0 ``pl_species_integral``),
    a LINEAR functional of the nodal X̂ field, so the per-step exciton seed is
    exact and cheap.  ``dJ_dx_list`` produces the per-step reduced seeds
    ``∂J/∂xₙ`` (flat field-major, 5·n_free per step) that Mode-B
    (``XDDTransientAdjoint.gradient``) consumes; nonzero only on the exciton
    (IXD/IXA) fields since PL depends only on X̂.
    """

    def __init__(self, target_decay, weight_donor=1.0, weight_acceptor=1.0,
                 params=None):
        self.target = [float(x) for x in target_decay]
        self.wd = float(weight_donor); self.wa = float(weight_acceptor)
        self._params = params

    def _pl(self, sysm, state):
        from diffsim.physics.exciton_system import IXD, IXA
        p = self._params if self._params is not None else _params_of(sysm)
        trd = _tau_r_inv_hat(p, "donor"); tra = _tau_r_inv_hat(p, "acceptor")
        pl_d = pl_species_integral(sysm, state, trd, IXD, weight=self.wd)
        pl_a = pl_species_integral(sysm, state, tra, IXA, weight=self.wa)
        return pl_d + pl_a

    def value(self, sysm, steps) -> float:
        return sum((self._pl(sysm, s["state"]) - self.target[n]) ** 2
                   for n, s in enumerate(steps))

    def dJ_dx_list(self, sysm, steps):
        from diffsim.physics.exciton_system import IXD, IXA, NDOF
        p = self._params if self._params is not None else _params_of(sysm)
        trd = _tau_r_inv_hat(p, "donor"); tra = _tau_r_inv_hat(p, "acceptor")
        gD = _pl_integral_grad_nodal(sysm, IXD, trd, self.wd)   # (n_nodes,)
        gA = _pl_integral_grad_nodal(sysm, IXA, tra, self.wa)
        seeds = []
        for n, s in enumerate(steps):
            r = self._pl(sysm, s["state"]) - self.target[n]
            seed = {f: np.zeros(sysm.dm.n_nodes) for f in range(NDOF)}
            seed[IXD] = 2.0 * r * gD
            seed[IXA] = 2.0 * r * gA
            seeds.append(np.concatenate([np.asarray(sysm.T.T @ seed[f])
                                         for f in range(NDOF)]))
        return seeds


class JtMisfitQoI:
    """J(t) light-modulation misfit  Σ_n (J_model(tₙ) − J_data(tₙ))²  as a
    TRAJECTORY QoI.

    Mirrors ``TRPLMisfitQoI`` but the per-step observable is the differentiable
    designated contact current (``SteadyCurrentQoI(contact).value`` /
    ``dJ_du`` evaluated at each recorded state — the Task-3 current-seed math).
    ``dJ_dx_list`` produces the per-step reduced seeds
    ``∂J/∂xₙ = 2 (J_model(tₙ) − J_data(tₙ)) · ∂J_model/∂xₙ`` (flat field-major,
    5·n_free per step) that Mode-B consumes.
    """

    def __init__(self, target_Jt, contact="anode", h_axis=1):
        self.target = [float(x) for x in target_Jt]
        self._cur = SteadyCurrentQoI(contact=contact, h_axis=h_axis)

    def value(self, sysm, steps) -> float:
        return sum((self._cur.value(sysm, s["state"]) - self.target[n]) ** 2
                   for n, s in enumerate(steps))

    def dJ_dx_list(self, sysm, steps):
        seeds = []
        for n, s in enumerate(steps):
            r = self._cur.value(sysm, s["state"]) - self.target[n]
            seeds.append(2.0 * r * self._cur.dJ_du(sysm, s["state"]))
        return seeds
