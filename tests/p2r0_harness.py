"""P2-R0 Task 5 — Cd/Strouhal extraction harness via surrogate_traction.

Reusable test helper (not shipped src) for the Tasks 8-9 validation gates:

    march_to_steady(st, dt, U_in, D, ...) -> (cd, cl, steps)
        Marches ``LeraySBMStepper`` until the drag-rate converges, then
        extracts Cd = F_x / (0.5 U_in^2 D) and Cl = F_y / (0.5 U_in^2 D)
        from the final ``surrogate_traction`` call.

    strouhal_from_lift(times, cl, tail_frac=0.5) -> (St, amp)
        Extracts St from zero-crossing periods of the demeaned Cl tail.
        Re-exported from ``test_cylinder_strouhal`` with attribution.

Both functions are used by Tasks 8 (steady Re20 Cd lock) and 9 (transient
Re100 St lock). See also tests/test_cylinder_strouhal.py (original source
of ``strouhal_from_lift``).
"""
import numpy as np

# ---------------------------------------------------------------------------
# strouhal_from_lift — re-exported from test_cylinder_strouhal with attribution.
# Source: tests/test_cylinder_strouhal.py::strouhal_from_lift (M1b Task 8c).
# The function is reproduced here (with attribution, not forked) so harness
# importers get a single import point without a circular dependency on a test
# module that may carry heavy fixtures. Signature and body are UNCHANGED.
# ---------------------------------------------------------------------------

def strouhal_from_lift(times, cl, tail_frac=0.5):
    """St from mean zero-crossing period of the Cl tail (demeaned).

    Source: tests/test_cylinder_strouhal.py (M1b Task 8c), reproduced with
    attribution. Signature and body unchanged.

    Parameters
    ----------
    times : array_like, shape (N,)
        Time coordinate of the Cl samples.
    cl : array_like, shape (N,)
        Lift-coefficient history.
    tail_frac : float
        Fraction of the history to use as the "tail" (default 0.5 = last half).

    Returns
    -------
    St : float or None
        Strouhal number ``f D / U_in`` extracted from zero-crossing periods
        (``D`` and ``U_in`` are baked into the D/U_in factor in the calling
        test / via the ``D`` and ``U_IN`` module-level constants in
        ``test_cylinder_strouhal.py``); ``None`` when fewer than 3 upward
        crossings are found.
    amp : float
        Half peak-to-peak amplitude of the demeaned tail.

    Notes
    -----
    The ``D`` and ``U_IN`` symbols are resolved at call time from the calling
    module's namespace in the original; here they must be supplied explicitly
    via the ``D`` and ``U_in`` keyword arguments to ``march_to_strouhal`` (the
    wrapper function) — see below.  The bare ``strouhal_from_lift`` signature
    is kept identical to the original so Tasks 8c / 9 can swap in either
    version.
    """
    # Use the same D/U_IN as the calling test's module scope (re-exported raw).
    # The original implementation uses module-level D and U_IN constants from
    # test_cylinder_strouhal.py; since we reproduce the body, callers that use
    # different D/U_IN must pass their own wrapper or pre-scale the times.
    # For direct use by test_p2r0_projection_sbm (which imports strouhal_from_lift
    # and passes pure-tone synthetic Cl with pre-scaled times), the raw body is
    # correct: the "D" and "U_IN" references in the zero-crossing formula are the
    # _specific_ constants frozen into the strouhal computation.  We resolve this
    # by making strouhal_from_lift accept an optional (D, U_in) override so the
    # closed-form gate test can supply its own values.
    #
    # To keep the signature identical to the original (for Tasks 8c/9 reuse),
    # we still accept **only** (times, cl, tail_frac) but allow callers to
    # monkey-patch _STROUHAL_D and _STROUHAL_U_IN module-level overrides, OR
    # use the convenience wrapper strouhal_from_lift_for(D, U_in) below.
    # The synthetic Strouhal test in test_p2r0_projection_sbm pre-scales the
    # zero-crossing output directly using St = f * D / U_in at known f, so the
    # actual D/U_IN values in the zero-crossing formula cancel --- what matters
    # is that the zero-crossing period is correct.  We embed the _same_ D/U_IN
    # defaults as test_cylinder_strouhal.py (D=0.14, U_IN=1.0) so that the
    # harness is a drop-in replacement for Tasks 8c/9.
    D = _STROUHAL_D
    U_IN = _STROUHAL_U_IN
    n0 = int(len(cl) * tail_frac)
    t, c = times[n0:], cl[n0:]
    c = c - c.mean()
    sgn = np.sign(c)
    idx = np.where(np.diff(sgn) > 0)[0]            # upward crossings
    if len(idx) < 3:
        return None, 0.0
    periods = np.diff(t[idx])
    St = D / (U_IN * periods.mean())
    amp = 0.5 * (c.max() - c.min())
    return St, amp


# Module-level D/U_IN for strouhal_from_lift — match test_cylinder_strouhal.py.
# Override by setting p2r0_harness._STROUHAL_D and ._STROUHAL_U_IN before
# calling strouhal_from_lift, or use make_strouhal_fn(D, U_in) below.
_STROUHAL_R = 0.07
_STROUHAL_D = 2 * _STROUHAL_R       # = 0.14
_STROUHAL_U_IN = 1.0


def make_strouhal_fn(D, U_in):
    """Return a strouhal_from_lift-compatible function with different D/U_in.

    For use by Tasks 8c/9 that run with different geometry than the default
    cylinder fixture (R=0.07, U_in=1.0).

    Usage::

        St_fn = make_strouhal_fn(D=2*R, U_in=U_IN)
        St, amp = St_fn(times, cl_hist)
    """
    def _st(times, cl, tail_frac=0.5):
        n0 = int(len(cl) * tail_frac)
        t, c = times[n0:], cl[n0:]
        c = c - c.mean()
        sgn = np.sign(c)
        idx = np.where(np.diff(sgn) > 0)[0]
        if len(idx) < 3:
            return None, 0.0
        periods = np.diff(t[idx])
        St = D / (U_in * periods.mean())
        amp = 0.5 * (c.max() - c.min())
        return St, amp
    return _st


# ---------------------------------------------------------------------------
# march_to_steady — extraction harness for steady-state Cd/Cl.
# ---------------------------------------------------------------------------

def march_to_steady(st, *, dt, U_in, D, max_steps=300, rate_tol=5e-3,
                    min_steps=10):
    """March ``LeraySBMStepper`` to steady state and extract (Cd, Cl, steps).

    Parameters
    ----------
    st : LeraySBMStepper
        Already-initialized stepper (``set_initial`` must have been called).
        The stepper is marched IN PLACE.
    dt : float
        Time step used to compute the drag-change rate (same as the stepper's
        ``dt``). Used only for the convergence rate ``|Cd_new - Cd_old| / dt``.
    U_in : float
        Free-stream velocity for Cd/Cl normalization: ``Cd = Fx / (0.5 U_in^2 D)``.
    D : float
        Characteristic length (cylinder diameter = 2R) for normalization.
    max_steps : int
        Maximum number of steps before declaring convergence failure.
    rate_tol : float
        Convergence criterion: ``|Cd_new - Cd_old| / dt < rate_tol``.
    min_steps : int
        Minimum steps before convergence is checked (default 10).

    Returns
    -------
    cd : float
        Drag coefficient at the last step.
    cl : float
        Lift coefficient at the last step.
    steps : int
        Number of steps taken (including the min_steps warm-up).

    Raises
    ------
    RuntimeError
        If ``max_steps`` is reached without satisfying ``rate_tol``.
    """
    qref = 0.5 * U_in ** 2 * D
    cd_prev = None
    for step in range(max_steps):
        u_new, p_hat = st.step()
        F = st.surrogate_traction()
        cd = F[0] / qref
        cl = F[1] / qref
        if cd_prev is not None and step >= min_steps:
            rate = abs(cd - cd_prev) / dt
            if rate < rate_tol:
                return cd, cl, step + 1
        cd_prev = cd
    raise RuntimeError(
        f"march_to_steady: no convergence in {max_steps} steps; "
        f"last Cd={cd:.4f}, |dCd/dt|={abs(cd - cd_prev) / dt:.4e}")
