r"""User-switchable LTE / PI(D) adaptive time-step controller for the
CH / ternary_ch / AC / multiphase phase-field family.

WHY THIS MODULE (Baskar 2026-07-14).  The steppers already carry
variable-coefficient BDF2 (the A4b retrofit) and two OFF-by-default
"grow/shrink" ladders: the step-doubling DEADBEAT driver
``cahn_hilliard.adaptive_march`` (CH/ternary) and the iteration-count
``MultiPhaseStepper.march`` (multiphase).  Both pick dt from a single
sample of the current step — a pure INTEGRAL (deadbeat) rule
``dt_new = dt * (tol/err)**(1/(p+1))``.  A deadbeat controller has no
memory, so on a stiff quench it churns (accept/reject oscillation).

This module adds a PROPER local-truncation-error controller with PI(D)
step selection to a user TOLERANCE, on TOP of the existing schemes:

    - the LTE estimate is the embedded low-order/step-doubling difference
      (one dt-step vs two dt/2-steps from the same state): for a scheme
      of order p the Richardson estimate of the local error is
      ``err ~ |x_full - x_half| / (2**p - 1)`` (tstep-agnostic — p is the
      stepper's own order, BDF1 or BDF2);
    - a PI(D) digital-filter gain sets the next step
      ``dt_new = dt * safety * (tol/err)**(kI/(p+1))
                                * (err_prev/err)**(kP/(p+1))``
      (the kP term is the memory that damps the deadbeat oscillation;
      kP = kD = 0 reproduces the deadbeat rule exactly);
    - accept/reject with BIT-EXACT rewind: a rejected step restores the
      snapshot byte-for-byte, so it cannot perturb the accepted
      trajectory.

SWITCHABLE, OFF BY DEFAULT.  Nothing here runs unless the user calls
``lte_march`` (or ``march(..., adapt="lte")``).  The existing ladders and
``step()`` are untouched — OFF-path bit-parity is structural.

NOISE CONTRACT (non-negotiable; module ruling recorded for Baskar).
Strong/pathwise LTE is ILL-POSED under FDT (stochastic) forcing: the
step-doubling difference of two noisy realizations is dominated by the
independent noise draws, not the truncation error, so "tol" has no
deterministic meaning.  If the user requests LTE while noise is ON the
framework must NOT silently downgrade and must NOT crash: ``lte_march``
emits an EXPLICIT notice (printed, stashed on ``stepper._lte_notice``,
logged to a RunLog if one is passed, and surfaced by the film preflight
rule ``lte-noise-incompatible``) and FALLS BACK to the fixed-order BDF
integrator under the existing ladder.  BDF2 is the deterministic
recommendation, but the steppers forbid BDF2 + noise at construction
(the FDT weak order under BDF2 is out of scope), so a run that KEEPS its
noise falls back to the noise-valid BDF1 ladder; the notice states this.
Deterministic runs get the full LTE controller.
"""
import numpy as np

# LTE ~ dt**(p+1); the accept test divides the raw difference by (2**p-1).
_TINY = 1e-30


# =====================================================================
# PI(D) digital-filter step-size controller
# =====================================================================
class StepController:
    r"""Multiplicative PI(D) step-size controller (Soederlind digital
    filter / Gustafsson predictive form), normalized by the error order.

    For a scheme of order p the local error scales as dt**(p+1), so the
    integral exponent base is 1/(p+1) (== the deadbeat rule the existing
    ladders use).  The controller keeps the last accepted error(s) and
    forms the step ratio

        fac = safety * (tol/e)**(kI/pe)
                     * (e_prev/e)**(kP/pe)                [P: memory]
                     * (e_prev**2/(e*e_prev2))**(kD/pe)   [D: 2nd memory]

    with pe = p + 1, clamped to [fac_min, fac_max].  kP = kD = 0 (mode
    "deadbeat") reproduces the existing ladders' pure-integral rule
    exactly; "pi" adds the proportional memory that damps stiff-quench
    accept/reject oscillation; "pid" adds the second difference for the
    residual oscillation PI leaves (Baskar's PID option).

    Gains are the dimensionless multipliers on 1/pe; defaults are the
    house measured-then-locked values (see docs/dev/2026-07-14-lte-
    controller.md).  reject_ratio shrinks dt from the SAME local sample
    on a reject (never grows), and rejected errors do NOT enter the
    accepted-error memory (Gustafsson: the filter tracks the accepted
    sequence)."""

    _DEFAULTS = {
        "deadbeat": (1.0, 0.0, 0.0),
        "pi":       (0.7, 0.4, 0.0),
        "pid":      (0.7, 0.4, 0.1),
    }

    def __init__(self, mode="pi", order=1, safety=0.9,
                 fac_min=0.2, fac_max=5.0, kI=None, kP=None, kD=None):
        assert mode in self._DEFAULTS, mode
        self.mode = mode
        self.order = int(order)
        self.pe = self.order + 1
        self.safety = float(safety)
        self.fac_min = float(fac_min)
        self.fac_max = float(fac_max)
        dI, dP, dD = self._DEFAULTS[mode]
        self.kI = dI if kI is None else float(kI)
        self.kP = dP if kP is None else float(kP)
        self.kD = dD if kD is None else float(kD)
        self.reset()

    def reset(self):
        self.e_prev = None
        self.e_prev2 = None

    def _clamp(self, fac):
        return float(min(self.fac_max, max(self.fac_min, fac)))

    def propose(self, err, tol):
        """Step ratio for the NEXT step given this step's error estimate.
        Uses the deadbeat (integral-only) rule until enough accepted
        history exists for the P (and D) memory terms."""
        e = max(float(err), _TINY)
        t = max(float(tol), _TINY)
        fac = self.safety * (t / e) ** (self.kI / self.pe)
        if self.mode != "deadbeat" and self.e_prev is not None \
                and self.kP != 0.0:
            fac *= (self.e_prev / e) ** (self.kP / self.pe)
        if self.mode == "pid" and self.e_prev2 is not None \
                and self.kD != 0.0:
            fac *= (self.e_prev * self.e_prev
                    / (e * self.e_prev2)) ** (self.kD / self.pe)
        return self._clamp(fac)

    def reject_ratio(self, err, tol):
        """Shrink ratio on a reject: the local integral rule, capped at 1
        (a reject never grows dt)."""
        e = max(float(err), _TINY)
        t = max(float(tol), _TINY)
        fac = self.safety * (t / e) ** (1.0 / self.pe)
        return float(min(1.0, max(self.fac_min, fac)))

    def commit(self, err):
        """Push an ACCEPTED error into the filter memory."""
        self.e_prev2 = self.e_prev
        self.e_prev = max(float(err), _TINY)


# =====================================================================
# stepper adapters — snapshot / restore / advance / solution
# =====================================================================
# The controller drives any stepper through a thin adapter so the loop
# stays basis- AND tstep-agnostic.  Two families:
#   _MixedAdapter       CH (binary), ternary, AC  (.order, .step()).
#   _MultiphaseAdapter  MultiPhaseStepper (.tstep, ._attempt, film/T).
# Both expose: order, snapshot()/restore() (BIT-EXACT), advance(dt)->ok,
# solution() (the CONSERVED-field vector used for the error norm),
# get/set dt, t.

class _MixedAdapter:
    """CH / ternary / AC.  Solution vector lives in ``.x`` (CH, ternary)
    or ``.hist[0]`` (AC); histories are a small python list.  Newton has
    no convergence flag here, so advance() reports ok via finiteness."""

    def __init__(self, st):
        self.st = st
        self.order = int(st.order)
        self._has_x = hasattr(st, "x")
        # conserved-dof layout for the error norm:
        #   CH binary (c, mu) -> c = x[0::2];
        #   ternary (p1,mu1,p2,mu2) -> [p1, p2] = x[0::4], x[2::4];
        #   AC scalar c -> hist[0] (whole vector).
        self._kind = "ac"
        if self._has_x:
            n = st.x.size
            self._kind = "ternary" if (
                n % 4 == 0 and getattr(st, "cons", None) is not None
                and n == st.cons.T.shape[1] * 4) else "ch"

    def snapshot(self):
        st = self.st
        hist = [self._copy(h) for h in st.hist]
        x = st.x.copy() if self._has_x else None
        return (x, hist, st.t, st.dt_prev)

    def restore(self, snap):
        st = self.st
        x, hist, t, dt_prev = snap
        if self._has_x:
            st.x = x.copy()
        st.hist = [self._copy(h) for h in hist]
        st.t = t
        st.dt_prev = dt_prev

    @staticmethod
    def _copy(h):
        if isinstance(h, tuple):
            return tuple(a.copy() for a in h)
        return h.copy()

    def advance(self, dt):
        st = self.st
        st.dt = dt
        st.step()
        return bool(np.isfinite(self.solution()).all())

    def solution(self):
        st = self.st
        if not self._has_x:
            return st.hist[0]
        if self._kind == "ternary":
            return np.concatenate([st.x[0::4], st.x[2::4]])
        return st.x[0::2]

    def snap_fields(self, snap):
        """(x_n, x_{n-1}, dt_prev) in the solution-field layout from a
        PRE-step snapshot, for the embedded predictor.  For CH/AC the
        history IS the field; ternary stores (p1, p2) tuples.  x_{n-1} is
        None on the BDF1 bootstrap (hist[1] duplicates hist[0])."""
        _x, hist, _t, dt_prev = snap
        h0, h1 = hist[0], hist[1]
        if self._kind == "ternary":
            xn = np.concatenate([h0[0], h0[1]])
            xm = np.concatenate([h1[0], h1[1]])
        else:
            xn, xm = h0, h1
        return xn, xm, dt_prev

    @property
    def dt(self):
        return self.st.dt

    @dt.setter
    def dt(self, v):
        self.st.dt = v

    @property
    def t(self):
        return self.st.t


class _MultiphaseAdapter:
    """MultiPhaseStepper.  Uses ``_attempt(dt)`` (which does NOT mutate
    committed state and returns a convergence flag) and mirrors
    ``step()``'s commit, so a Newton failure becomes a clean reject.
    Order is 1 (bdf1) or 2 (bdf2).  Snapshot captures the two-level
    history, the film height and the segregated T field."""

    def __init__(self, st):
        self.st = st
        self.order = 2 if st.tstep == "bdf2" else 1

    def snapshot(self):
        st = self.st
        return (st.x.copy(),
                st.hist.copy(),
                None if st.hist2 is None else st.hist2.copy(),
                st.t, st.dt_prev,
                getattr(st, "h_curr", None),
                None if getattr(st, "T_nodes", None) is None
                else st.T_nodes.copy())

    def restore(self, snap):
        st = self.st
        (x, hist, hist2, t, dt_prev, h_curr, T_nodes) = snap
        st.x = x.copy()
        st.hist = hist.copy()
        st.hist2 = None if hist2 is None else hist2.copy()
        st.t = t
        st.dt_prev = dt_prev
        if h_curr is not None:
            st.h_curr = h_curr
        if T_nodes is not None:
            st.T_nodes = T_nodes.copy()

    def advance(self, dt):
        st = self.st
        x, iters, ok = st._attempt(dt)
        if not ok:
            return False
        # mirror MultiPhaseStepper.step()'s commit exactly
        st.x = x
        if st.tstep == "bdf2":
            st.hist2 = st.hist
            st.dt_prev = dt
        st.hist = x.copy()
        st.t += dt
        if st.T_mode == "field":
            st.T_nodes = st._T_pend
        if st.film_on:
            st.h_curr -= dt * st._K_pend
        return bool(np.isfinite(self.solution()).all())

    def solution(self):
        return self._fields(self.st.x)

    def _fields(self, x):
        st = self.st
        cols = [x[2 * i::st.ndof] for i in range(st.M)]
        cols += [x[2 * st.M + 2 * k::st.ndof] for k in range(st.K)]
        return np.concatenate(cols)

    def snap_fields(self, snap):
        """(x_n, x_{n-1}, dt_prev) in the conserved-field layout from a
        PRE-step snapshot for the embedded predictor.  hist2 is None on
        the BDF1 bootstrap (BDF1 keeps no x_{n-1})."""
        (_x, hist, hist2, _t, dt_prev, _h, _T) = snap
        xn = self._fields(hist)
        xm = None if hist2 is None else self._fields(hist2)
        return xn, xm, dt_prev

    @property
    def dt(self):
        return self.st.dt

    @dt.setter
    def dt(self, v):
        self.st.dt = v

    @property
    def t(self):
        return self.st.t


def _adapter(stepper):
    if hasattr(stepper, "tstep") and hasattr(stepper, "_attempt"):
        return _MultiphaseAdapter(stepper)
    return _MixedAdapter(stepper)


# =====================================================================
# noise contract
# =====================================================================
def _noise_amp(stepper):
    """(noise_psi, noise_phi) — 0 for steppers without FDT noise
    (CH/AC/ternary have no noise channel)."""
    return (float(getattr(stepper, "noise_psi", 0.0)),
            float(getattr(stepper, "noise_phi", 0.0)))


def has_noise(stepper):
    a, b = _noise_amp(stepper)
    return a > 0.0 or b > 0.0


def lte_noise_notice(stepper):
    """The explicit incompatibility notice (single source of truth; the
    preflight rule reuses the same wording)."""
    npsi, nphi = _noise_amp(stepper)
    return (
        "LTE-INCOMPATIBLE-WITH-NOISE: adaptive LTE step control was "
        f"requested but FDT noise is ON (noise_psi={npsi:g}, "
        f"noise_phi={nphi:g}). Strong/pathwise LTE is ill-posed under "
        "stochastic forcing (the step-doubling difference is dominated "
        "by the independent noise draws, not the truncation error), so "
        "the tolerance has no deterministic meaning. LTE is DISABLED. "
        "Falling back to the fixed-order BDF integrator under the "
        "existing reject/grow ladder (BDF2 is the deterministic "
        "recommendation; because this run keeps noise ON, and BDF2+noise "
        "is out of scope, the fallback marches the noise-valid BDF1 "
        "ladder). Turn noise off to get the full LTE controller.")


def _report_notice(stepper, notice, runlog=None, verbose=True):
    stepper._lte_notice = notice
    if verbose:
        print("[lte] " + notice, flush=True)
    if runlog is not None:
        for meth in ("event", "note", "warn"):
            fn = getattr(runlog, meth, None)
            if callable(fn):
                fn(notice)
                break


# =====================================================================
# the driver
# =====================================================================
def _embedded_err(A, dt, snap, x_new):
    """Embedded predictor-corrector LTE estimate (the "BDF1-vs-BDF2
    difference" of the spec), ZERO extra solves.  The BDF-p corrector
    x_new is compared against the polynomial PREDICTOR extrapolated from
    the committed history (x_n, x_{n-1}) that the snapshot holds:

        r = dt / dt_prev,  x_pred = x_n + r (x_n - x_{n-1})   (linear),

    the order-1 extrapolant.  ||x_new - x_pred|| is the classic
    predictor-corrector (Milne-device) local error indicator; it scales
    as dt**2 (the predictor's error dominates), so the controller steps
    on the LOW-order estimate while marching the HIGH-order BDF2 solution
    — the standard efficient "estimate low, integrate high" strategy.
    On the BDF1 bootstrap (no x_{n-1}) the predictor degrades to x_n
    (order 0).  Returns the RELATIVE-L2 estimate."""
    xn, xm, dt_prev = A.snap_fields(snap)
    if xm is None or dt_prev is None or dt_prev <= 0.0:
        x_pred = xn                                  # bootstrap: x_n
    else:
        r = dt / dt_prev
        x_pred = xn + r * (xn - xm)
    num = float(np.linalg.norm(x_new - x_pred))
    den = max(float(np.linalg.norm(x_new)), _TINY)
    return num / den


def lte_march(stepper, t_end, tol=1e-4, controller="pi",
              estimator="doubling", dt_min=1e-8, dt_max=None, safety=0.9,
              fac_min=0.2, fac_max=5.0, max_steps=200000,
              kI=None, kP=None, kD=None, runlog=None,
              verbose=False, on_accept=None):
    r"""LTE-controlled adaptive march with PI(D) step selection.

    estimator="doubling" (DEFAULT, genuine LTE): step-doubling Richardson
    estimate ``err = ||x_full - x_half|| / ||x_half|| / (2**p - 1)`` (one
    dt vs two dt/2 from the same state; THREE solves per step) — the
    accepted state is the more accurate two-half-step solution.  This is
    the codebase's established production-march pattern and the estimate
    that TRACKS the tolerance on stiff phase-field quenches (measured
    achieved-error ~ tol; dt grows through coarsening).

    estimator="embedded" (CHEAP, conservative): the predictor-corrector
    ("BDF1-vs-BDF2") indicator — ONE BDF solve per step, error read from
    the history extrapolation (``_embedded_err``).  The accepted state is
    the ordinary BDF trajectory.  MEASURED CAVEAT: on stiff quenches the
    dt^2 increment indicator is dominated by the fast physics, so it
    OVER-RESOLVES (achieved error << tol); use it on smooth/slow regimes
    or when the 3x doubling cost is prohibitive.  Multiphase BDF1 keeps
    no x_{n-1} so its embedded estimate degrades to increment-control —
    prefer doubling (or tstep="bdf2") there.

    Accept iff err < tol (or dt at the floor).  On accept the PI(D)
    controller proposes the next dt; on reject the state is rewound
    BIT-EXACTLY and dt is shrunk from the local sample.

    NOISE: if FDT noise is on, LTE is refused with an explicit notice and
    the run falls back to the fixed-order ladder (module docstring / the
    NOISE CONTRACT).  Returns (t_list, dt_list) of ACCEPTED steps; on the
    noise fallback returns the fallback march's (t_list, dt_list)."""
    assert estimator in ("embedded", "doubling"), estimator
    A = _adapter(stepper)
    # The embedded (linear-predictor) indicator scales as dt**2 -> the
    # step-control error order is 1 (pe = 2) regardless of the BDF order;
    # step-doubling controls the scheme's own order p.
    ctrl_order = 1 if estimator == "embedded" else A.order
    if isinstance(controller, StepController):
        ctrl = controller
    else:
        ctrl = StepController(mode=controller, order=ctrl_order,
                              safety=safety, fac_min=fac_min,
                              fac_max=fac_max, kI=kI, kP=kP, kD=kD)

    # -- NOISE CONTRACT: refuse LTE, report, fall back -----------------
    if has_noise(stepper):
        notice = lte_noise_notice(stepper)
        _report_notice(stepper, notice, runlog=runlog, verbose=True)
        stepper._lte_active = False
        stepper._lte_fell_back = True
        return _bdf_fallback_march(stepper, t_end, dt_min=dt_min,
                                   dt_max=dt_max, max_steps=max_steps)

    stepper._lte_active = True
    stepper._lte_fell_back = False
    stepper._lte_notice = None
    ctrl.reset()

    p = A.order
    denom = (2 ** p - 1)
    ts, dts = [], []
    n_reject = 0
    eps = 1e-14
    for _ in range(max_steps):
        if A.t >= t_end - eps:
            break
        dt = min(A.dt, t_end - A.t)
        if dt <= 0.0:
            break
        at_floor = dt <= dt_min * (1.0 + 1e-9)
        snap = A.snapshot()

        if estimator == "embedded":
            ok = A.advance(dt)               # ONE BDF solve (commits)
            if ok:
                err = _embedded_err(A, dt, snap, A.solution())
        else:                                # step-doubling (3 solves)
            ok1 = A.advance(dt)
            x_full = A.solution().copy()
            A.restore(snap)
            A.dt = dt * 0.5
            ok_a = A.advance(dt * 0.5)
            ok_b = A.advance(dt * 0.5) if ok_a else False
            ok = ok1 and ok_a and ok_b
            if ok:
                num = float(np.linalg.norm(x_full - A.solution()))
                den = max(float(np.linalg.norm(A.solution())), _TINY)
                err = (num / den) / denom

        if not ok:
            # Newton failure -> reject, hard shrink (bit-exact rewind).
            A.restore(snap)
            A.dt = max(dt_min, dt * 0.5)
            n_reject += 1
            if at_floor:
                break
            continue

        if err < tol or at_floor:
            # ACCEPT (state already committed).  PI(D) sets the next dt.
            fac = ctrl.propose(err, tol)
            ctrl.commit(err)
            new_dt = dt * fac
            if dt_max is not None:
                new_dt = min(new_dt, dt_max)
            A.dt = max(dt_min, new_dt)
            ts.append(A.t)
            dts.append(dt)
            if verbose:
                print(f"  t={A.t:.5f} dt={dt:.3e} err={err:.3e} "
                      f"fac={fac:.3f}", flush=True)
            if on_accept is not None:
                on_accept(stepper, dt, err)
        else:
            # REJECT: rewind bit-exactly, shrink from the local sample.
            A.restore(snap)
            A.dt = max(dt_min, dt * ctrl.reject_ratio(err, tol))
            n_reject += 1
    stepper._lte_n_reject = n_reject
    return ts, dts


def _bdf_fallback_march(stepper, t_end, dt_min=1e-8, dt_max=None,
                        max_steps=200000):
    """The noise / non-LTE fallback: run the stepper's EXISTING ladder
    (no LTE control).  BDF2 is the deterministic recommendation; a run
    that keeps noise on stays on its (noise-valid) configured scheme —
    the notice already told the user.  Uses the native ladder so the
    fallback is byte-identical to a plain non-LTE march."""
    if hasattr(stepper, "march"):
        # MultiPhaseStepper native Appendix-A ladder.
        reason = stepper.march(t_end, max_steps=max_steps,
                               dt_min=dt_min, dt_max=dt_max)
        stepper._lte_fallback_reason = reason
        return [], []
    # CH / ternary: the existing deadbeat step-doubling ladder.
    from .cahn_hilliard import adaptive_march
    stride = 4 if (hasattr(stepper, "x") and getattr(
        stepper, "cons", None) is not None
        and stepper.x.size == stepper.cons.T.shape[1] * 4) else 2
    return adaptive_march(stepper, t_end, dt_min=dt_min,
                          dt_max=(dt_max or 0.5), stride=stride)


# =====================================================================
# convenience dispatcher: one knob (adapt=) + one tol
# =====================================================================
def march(stepper, t_end, adapt="ladder", tol=1e-4, controller="pi",
          **kw):
    r"""User-facing switch.  ``adapt="ladder"`` (DEFAULT) runs the
    existing OFF-path grow/shrink ladder unchanged (bit-parity);
    ``adapt="lte"`` turns the PI(D) LTE controller ON.  One tolerance
    knob (``tol``) plus the ``adapt`` switch — and the noise notice fires
    automatically inside ``lte_march`` when noise is on."""
    assert adapt in ("ladder", "lte"), adapt
    if adapt == "lte":
        return lte_march(stepper, t_end, tol=tol, controller=controller,
                         **kw)
    # OFF path — dispatch to the native ladder, untouched.
    if hasattr(stepper, "march"):
        return stepper.march(t_end, **kw)
    from .cahn_hilliard import adaptive_march
    return adaptive_march(stepper, t_end, tol=tol, **kw)
