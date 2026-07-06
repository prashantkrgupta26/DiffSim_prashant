"""Time integration for M1b (Task 4): BDF tables, bootstrap, history
rotation, extrapolations — production conventions verbatim
(NSPNPInputData.h:163-201; proteus tables; conventions doc S'BDF').

Discrete time derivative:  du/dt ~ (b0 u^{n+1} + b1 u^n + b2 u^{n-1})/dt
(coefficients SIGNED as in production: BDF2 = {1.5, -2, 0.5}); the known
history part (b1 u^n + b2 u^{n-1})/dt moves to the RHS, sigma = b0/dt goes
into the operator (and into tau's transient term).

Bootstrap rule: t < 1.5*dt (or missing history) => BDF1 — the production
`if (t < 1.5*dt)` gate.

History slots follow the PRE1/PRE2/PRE3 naming (NodeData convention):
rotate() shifts PRE1->PRE2->PRE3 and installs the new field as PRE1.
"""
import numpy as np


def bdf_coeffs(order: int, dt: float, dt_prev: float | None = None):
    """(b0, b1, b2) — signed production tables. Variable-step BDF2 uses the
    NSPNPInputData formula; dt_prev=None or order 1 gives constant-step."""
    if order == 1:
        return 1.0, -1.0, 0.0
    if order == 2:
        if dt_prev is None or dt_prev == dt:
            return 1.5, -2.0, 0.5
        return ((2 * dt + dt_prev) / (dt + dt_prev),
                -(dt + dt_prev) / dt_prev,
                dt * dt / (dt_prev * (dt + dt_prev)))
    raise ValueError(f"BDF order {order} not tabulated (M8 revisits)")


def bdf_order_now(t: float, dt: float, target_order: int,
                  have_history: bool = True) -> int:
    """The production bootstrap gate: BDF1 until t >= 1.5*dt (and whenever
    second-order history is unavailable, e.g. post-restart)."""
    if target_order == 1 or not have_history or t < 1.5 * dt:
        return 1
    return target_order


def extrapolate_velocity(order: int, u_pre1, u_pre2=None):
    """Advecting-velocity extrapolation: order 1 = u^n; order 2 = 2u^n - u^{n-1}."""
    if order == 1 or u_pre2 is None:
        return np.array(u_pre1, copy=True)
    return 2.0 * np.asarray(u_pre1) - np.asarray(u_pre2)


def extrapolate_pressure(order: int, p_pre1=None, p_pre2=None):
    """proteus tables {0,1,0,0}/{0,2,-1,0}: order 0 -> 0; 1 -> p^n;
    2 -> 2p^n - p^{n-1}. (The Leray stepper's p* <- p_hat is order 1.)"""
    if order == 0 or p_pre1 is None:
        return None
    if order == 1 or p_pre2 is None:
        return np.array(p_pre1, copy=True)
    return 2.0 * np.asarray(p_pre1) - np.asarray(p_pre2)


class History:
    """PRE1/PRE2/PRE3 rotation for named fields (host-side epoch state;
    device residency is the stepper's concern)."""

    def __init__(self, nslots: int = 3):
        self.nslots = nslots
        self._slots: list = [None] * nslots
        self.dt_prev: float | None = None

    @property
    def pre1(self):
        return self._slots[0]

    @property
    def pre2(self):
        return self._slots[1]

    @property
    def pre3(self):
        return self._slots[2] if self.nslots > 2 else None

    def rotate(self, new_field, dt: float | None = None):
        self._slots = [np.array(new_field, copy=True)] + self._slots[:-1]
        if dt is not None:
            self.dt_prev = dt

    def have(self, n: int) -> bool:
        return all(s is not None for s in self._slots[:n])

    def rhs_history(self, b1: float, b2: float, dt: float):
        """(b1 u^n + b2 u^{n-1})/dt — the known part of the BDF derivative
        (moves to the RHS with a sign flip applied by the caller's form)."""
        out = b1 * self._slots[0]
        if b2 != 0.0:
            out = out + b2 * self._slots[1]
        return out / dt
