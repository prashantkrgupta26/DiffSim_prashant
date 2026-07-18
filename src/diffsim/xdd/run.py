"""XDDRun — continuation/marching protocol driver (SP-1 Block C).

Drives XDDSystem via BDF transient marching to reach steady states.
Steady states are NEVER reached by direct Newton on device configs —
always by transient marching (BDF σ=1/Δt̂ regularises each step's Newton).

Solver strategies (CPU enum values from A1):
  PULSE_GENERATION    = 0
  STEADY_STATE_JV     = 1
  TRANSIENT_JV        = 2
  STEADY_PULSE        = 3
"""
from __future__ import annotations

import io
import math
import os
import time
from typing import List, Optional

import numpy as np

from diffsim.physics.exciton_system import (
    IPHI, IN, IP, IXD, IXA, NDOF,
    bilayer_electrode_bcs,
    _carrier_block,
)


# ── Log-linear IC builder (the B5 "continuation-free" starting state) ────────

def log_linear_ic(mesh, Eg_hat: float, minority_ln: float,
                  h_axis: int = 1) -> dict:
    """Log-space linear initial condition for the bilayer march.

    u = ln n̂ and v = ln p̂ are interpolated LINEARLY in height between the
    electrode log-values (anode u=0, cathode u=minority_ln; p mirrored);
    φ̂ linear; X̂=0.  These are smooth O(10) numbers: the primal Boltzmann
    IC's e^{Ê_g}≈e⁴² carrier would make the primal it-1 residual ~1e18 (B4's
    documented divergence); the log-space IC keeps n̂ representable.

    Parameters
    ----------
    mesh : mesh object with .node_coords attribute
    Eg_hat : float  — nondim band gap (used to set phi profile)
    minority_ln : float  — log of the minority carrier BC floor (e.g. -60)
    h_axis : int  — device height axis (default 1 = y)

    Returns
    -------
    dict {IPHI, IN, IP, IXD, IXA} of full nodal vectors
    """
    coords = mesh.node_coords
    hc = coords[:, h_axis]
    lo, hi = hc.min(), hc.max()
    xi = (hc - lo) / max(hi - lo, 1e-30)   # 0 = anode, 1 = cathode
    phi_a = +0.5 * Eg_hat
    return {
        IPHI: phi_a - Eg_hat * xi,
        IN:   np.exp(minority_ln * xi),
        IP:   np.exp(minority_ln * (1.0 - xi)),
        IXD:  np.zeros(len(coords)),
        IXA:  np.zeros(len(coords)),
    }

# ── Solver strategy identifiers (CPU enum values) ────────────────────────────

PULSE_GENERATION = 0
STEADY_STATE_JV  = 1
TRANSIENT_JV     = 2
STEADY_PULSE     = 3


# ══════════════════════════════════════════════════════════════════════════════
# _XDDRunLog — lightweight 4-layer log
# ══════════════════════════════════════════════════════════════════════════════

class _XDDRunLog:
    """Lightweight 4-layer log for XDDRun.

    Does NOT import film internals.  Optional file output to outdir.

    Layers
    ------
    (a) protocol header  — log_header(info_dict)
    (b) per-step         — log_step(rec)
    (c) per-stage        — log_stage(rec)
    (d) machine-readable — records, stage_records, finalize()
    """

    def __init__(self, outdir: Optional[str] = None):
        self.outdir = outdir
        self.records: List[dict] = []
        self.stage_records: List[dict] = []
        self._buf = io.StringIO()
        self._file = None
        if outdir is not None:
            os.makedirs(outdir, exist_ok=True)
            self._file = open(os.path.join(outdir, "xdd_runlog.txt"), "w")

    def _write(self, text: str):
        self._buf.write(text)
        if self._file is not None:
            self._file.write(text)
            self._file.flush()

    # -- layer (a) -----------------------------------------------------------
    def log_header(self, info_dict: dict):
        """Write the == PROTOCOL == header from an info dict."""
        self._write("== PROTOCOL ==\n")
        for k, v in info_dict.items():
            self._write(f"  {k}: {v}\n")
        self._write("\n")

    # -- layer (b) -----------------------------------------------------------
    def log_step(self, rec: dict):
        """Log a single march step.

        rec keys: t_hat, dt_hat, newton_its, dd_norm, ex_norm, jny, jpy,
                  accepted (bool)
        """
        rec = dict(rec)
        self.records.append(rec)
        accepted_str = "OK" if rec.get("accepted", True) else "REJECT"
        self._write(
            f"  step t={rec.get('t_hat', 0):.4e} dt={rec.get('dt_hat', 0):.2e}"
            f" its={rec.get('newton_its', 0):3d}"
            f" dd={rec.get('dd_norm', 0):.2e} ex={rec.get('ex_norm', 0):.2e}"
            f" Jn={rec.get('jny', 0):.3e} Jp={rec.get('jpy', 0):.3e}"
            f" [{accepted_str}]\n"
        )

    # -- layer (c) -----------------------------------------------------------
    def log_stage(self, rec: dict):
        """Log a stage summary.

        rec keys: stage, steps, wall_s, criterion_fired, final_jny, final_jpy
        """
        rec = dict(rec)
        self.stage_records.append(rec)
        self._write(
            f"== STAGE {rec.get('stage', '?')} =="
            f" steps={rec.get('steps', 0)}"
            f" wall={rec.get('wall_s', 0):.1f}s"
            f" criterion={'FIRED' if rec.get('criterion_fired') else 'timeout'}"
            f" Jn={rec.get('final_jny', 0):.3e}"
            f" Jp={rec.get('final_jpy', 0):.3e}\n"
        )

    # -- layer (d) -----------------------------------------------------------
    def finalize(self) -> dict:
        """Return a dict summary of all logged data."""
        return {
            "n_steps": len(self.records),
            "n_stages": len(self.stage_records),
            "records": self.records,
            "stage_records": self.stage_records,
            "log_text": self._buf.getvalue(),
        }

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


# ══════════════════════════════════════════════════════════════════════════════
# _flux_pair — boundary contact flux pair (Jny, Jpy)
# ══════════════════════════════════════════════════════════════════════════════

def _flux_pair(sysm, state, h_axis: int = 1):
    """Compute (Jny, Jpy): nondim contact current pair — RESIDUAL (consistent) form.

    Jny is the net electron current INTO the anode (ĥ=0) contact; Jpy the net
    hole current INTO the cathode (ĥ=1) contact.  Both are evaluated as the
    steady-state carrier flux-divergence residual, `K_c(σ=0) @ ĉ`, summed over
    the contact's Dirichlet nodes:

        R_n = K_n(σ=0, supg=0) @ n̂      Jny = Σ_{i∈anode}   R_n[i]
        R_p = K_p(σ=0, supg=0) @ p̂      Jpy = Σ_{i∈cathode} R_p[i]

    where K_c is the SAME conservative drift-diffusion Galerkin operator the
    residual assembles (drift −n̂∇φ̂ for electrons, +p̂∇φ̂ for holes; post-fix
    convention n̂∝e^{+φ̂}).  It is DISCRETELY CONSERVATIVE: the WHOLE-domain
    residual sums to zero to machine precision (Σ_i R_c[i] ≈ 1e-16), because a
    constant test function integrates the pure-Neumann drift-diffusion form to
    zero.  (Individual interior rows do NOT vanish — R_c on interior nodes
    carries the recombination sink; only the global sum is zero.)  The two
    contact sums therefore partition that zero: Jny (anode e⁻ inflow) + the
    cathode e⁻ residual + the interior e⁻ residual = 0, and likewise for holes.
    This is the discretely-conservative contact flux; a volume Gauss-point
    integral of J_y is only consistent when J_y is discretely div-free — which
    it is NOT on a coarse mesh, giving a spurious residual that never balances.
    (The prior volume-integral `_flux_pair`, plus its OLD-convention
    Boltzmann-check docstring, were the fossils this fix replaces — Block C
    completion on the c9c499c-fixed kernels.)

    What the pair means physically.  This transport model has no thermal-
    generation term to balance Langevin recombination, so a bilayer at V̂=0
    dark carries a small but genuine recombination current (Jny≈Jpy≈O(0.2),
    NOT zero).  Jny (e⁻ into anode) and Jpy (h⁺ into cathode) are DIFFERENT
    physical currents; they balance (|Jny−Jpy|→0) only for SYMMETRIC transport
    — with μ̂_n≠μ̂_p the imbalance is the mobility-ratio asymmetry (~8%,
    mesh-independent).  So |Jny−Jpy| is a device-SYMMETRY probe, not a
    steadiness detector (the field-norm part of `_steady_criterion` detects
    steadiness); see that function's docstring.

    Uses SUPG=0 (pure Galerkin) so the flux is the exact conservative
    divergence residual, independent of the stabilisation term.

    Returns
    -------
    (Jny, Jpy) : float, float
        Nondim contact current pair (electrons into anode, holes into cathode).
    """
    dm = sysm.dm
    cl = sysm._closures(state)
    aq_n = sysm._aq(cl["gradphi"], sysm.mu_n_gp, -1.0)   # electrons: −μ̂_n∇φ̂
    aq_p = sysm._aq(cl["gradphi"], sysm.mu_p_gp, +1.0)   # holes:     +μ̂_p∇φ̂
    # Steady (σ=0), pure-Galerkin (SUPG=0) conservative carrier stiffness.
    Kn = _carrier_block(dm, aq_n, sysm.mu_n_gp, 0.0, 0.0, 0.0)
    Kp = _carrier_block(dm, aq_p, sysm.mu_p_gp, 0.0, 0.0, 0.0)
    Rn = Kn @ state[IN]      # nodal electron flux-divergence residual
    Rp = Kp @ state[IP]      # nodal hole     flux-divergence residual

    coords = dm.mesh.node_coords
    hc = coords[:, h_axis]
    lo, hi = hc.min(), hc.max()
    anode = np.where(np.abs(hc - lo) < 1e-9)[0]
    cathode = np.where(np.abs(hc - hi) < 1e-9)[0]

    Jny = float(np.sum(Rn[anode]))
    Jpy = float(np.sum(Rp[cathode]))
    return Jny, Jpy


# ══════════════════════════════════════════════════════════════════════════════
# _steady_criterion — 10-consecutive-step steady-state check
# ══════════════════════════════════════════════════════════════════════════════

def _steady_criterion(history: List[dict],
                      time_stepping_tol: float = 1e-5,
                      flux_floor: float = 1e-6):
    """Check the CPU steady-state criterion on a sliding window of step records.

    The criterion has TWO parts with DIFFERENT roles:
      • field-norm part (‖Δu‖/‖u‖)_dd + (‖Δu‖/‖u‖)_ex < time_stepping_tol —
        the actual STEADINESS detector (the state has stopped changing).
      • flux-balance part |Ĵny − Ĵpy| < 0.01·max(|Ĵ|, floor) — a device-
        SYMMETRY probe, NOT a steadiness test.  Ĵny (e⁻ into anode) and Ĵpy
        (h⁺ into cathode) are different physical currents; they balance only
        when transport is symmetric.  On an ASYMMETRIC device (μ̂_n≠μ̂_p) a
        genuinely-steady march can leave a ~mobility-ratio imbalance (~8%,
        mesh-independent) that this part will not accept — a physical property,
        not a solver failure.  Block C's marching gates use a symmetric config
        where the part is well-posed; a future asymmetric-device criterion
        should compare TOTAL current continuity (Ĵn+Ĵp, the conserved
        quantity) across slices instead.  The `max(|Ĵ|, floor)` form makes it
        relative-to-current normally and absolute-floor only as Ĵ→0.

    Parameters
    ----------
    history : list of step dicts
        Each dict must contain: dd_norm, ex_norm, jny, jpy.
    time_stepping_tol : float
        Threshold for combined field norm dd_norm + ex_norm.
    flux_floor : float
        Minimum flux scale for the relative imbalance check.

    Returns
    -------
    (converged : bool, flux_imbalance : float)
        converged is True iff the last 10 consecutive steps all satisfy BOTH
        parts above.
    """
    if len(history) < 10:
        return False, float("inf")

    window = history[-10:]
    jny_last = window[-1]["jny"]
    jpy_last = window[-1]["jpy"]
    flux_scale = max(max(abs(jny_last), abs(jpy_last)), flux_floor)
    flux_imbalance = abs(jny_last - jpy_last) / flux_scale

    for rec in window:
        if rec["dd_norm"] + rec["ex_norm"] >= time_stepping_tol:
            return False, flux_imbalance
    if flux_imbalance >= 0.01:
        return False, flux_imbalance
    return True, flux_imbalance


# ══════════════════════════════════════════════════════════════════════════════
# _dt_schedule — log-space adaptive dt controller
# ══════════════════════════════════════════════════════════════════════════════

def _dt_schedule(t_hat: float, dt_hat: float, dt0_hat: float,
                 dt_max_hat: float) -> float:
    """Next dt̂ from the log-space adaptive schedule.

    At each accepted step with current time t_hat, the schedule sets
        dt = 10^floor(log10(t_hat))  capped at dt_max_hat.
    At t=0 (before any accepted step), returns dt0_hat.

    This mimics the CPU LOGSPACE strategy: one step per decade of time.
    """
    if t_hat <= 0.0:
        return dt0_hat
    floor_log = math.floor(math.log10(t_hat))
    dt_sched = 10.0 ** floor_log
    return min(dt_sched, dt_max_hat)


# ══════════════════════════════════════════════════════════════════════════════
# _march_to_steady — the core transient march loop
# ══════════════════════════════════════════════════════════════════════════════

def _march_to_steady(sysm, state, *,
                     dt0_hat: float,
                     dt_max_hat: float,
                     max_steps: int = 2000,
                     time_stepping_tol: float = 1e-5,
                     flux_floor: float = 1e-6,
                     bdf2: bool = False,
                     log=None,
                     stage_name: str = "",
                     h_axis: int = 1,
                     newton_kw: Optional[dict] = None,
                     post_process: bool = False):
    """Core BDF transient march to steady state.

    Parameters
    ----------
    sysm : XDDSystem
    state : dict  — initial full-field state
    dt0_hat : float  — starting dt̂
    dt_max_hat : float  — maximum dt̂
    max_steps : int  — cap on accepted steps
    time_stepping_tol : float  — steady criterion field-norm threshold
    flux_floor : float  — steady criterion flux floor
    bdf2 : bool  — use BDF2 after the first step (BDF1 for step 0)
    log : _XDDRunLog or None
    stage_name : str  — label for the stage record
    h_axis : int  — coordinate axis for flux pair
    newton_kw : dict or None  — extra kwargs for step_bdf / solve_newton

    Returns
    -------
    (final_state, march_info)
        march_info keys: steps_accepted, steps_rejected, criterion_fired,
                         step_history, jny, jpy, t_hat
    """
    nk = newton_kw or {}
    dt_floor = 1e-20
    t_hat = 0.0
    dt_hat = dt0_hat

    prev_state = {f: state[f].copy() for f in range(NDOF)}
    prev2_state = None
    step_history: List[dict] = []
    steps_accepted = 0
    steps_rejected = 0
    jny, jpy = 0.0, 0.0
    t0_wall = time.time()

    for _ in range(max_steps):
        # Try BDF step
        order = 2 if (bdf2 and steps_accepted >= 1 and prev2_state is not None) else 1
        try:
            new_state, info = sysm.step_bdf(
                state, dt_hat,
                order=order,
                prev=prev_state,
                prev2=prev2_state,
                **nk
            )
        except Exception as exc:
            # Treat Newton exception as non-convergence
            info = {"converged": False, "reason": str(exc),
                    "dnorms": [], "rnorms": [], "iters": 0}
            new_state = state

        if not info["converged"]:
            # Halve dt and retry
            steps_rejected += 1
            dt_hat *= 0.5
            if dt_hat < dt_floor:
                if log is not None:
                    log.log_stage({
                        "stage": stage_name, "steps": steps_accepted,
                        "wall_s": time.time() - t0_wall,
                        "criterion_fired": False,
                        "final_jny": jny, "final_jpy": jpy,
                        "reason": "dt_floor_reached",
                    })
                return state, {
                    "steps_accepted": steps_accepted,
                    "steps_rejected": steps_rejected,
                    "criterion_fired": False,
                    "step_history": step_history,
                    "jny": jny, "jpy": jpy,
                    "t_hat": t_hat,
                    "reason": "dt_floor",
                }
            step_rec = {
                "t_hat": t_hat, "dt_hat": dt_hat,
                "newton_its": info.get("iters", 0),
                "dd_norm": float("inf"), "ex_norm": float("inf"),
                "jny": jny, "jpy": jpy, "accepted": False,
            }
            if log is not None:
                log.log_step(step_rec)
            continue

        # Accepted step — advance time
        t_hat += dt_hat
        steps_accepted += 1

        # Compute step-to-step field norms for the steady criterion.
        # dd fields: IPHI, IN, IP; ex fields: IXD, IXA.
        #
        # The exciton relative norm carries an ABSOLUTE floor so that a
        # genuinely-negligible exciton field (deep-depletion configs where
        # X̂_ss ≈ γ̂ n̂ p̂ / (1/τ̂) can be tiny) does not inflate ‖ΔX̂‖/‖X̂‖ during
        # the transient (X̂ grows from 0, so early Δ/X̂ ≈ 1).  The floor is a
        # small PER-NODE density (X̂_FLOOR = 1e-8), DECOUPLED from `flux_floor`:
        # they are unrelated quantities (flux_floor scales a current, X̂_FLOOR a
        # density).  1e-8 sits far below any physically-active exciton density
        # (lit or dark-recombination, O(1e-3) here) so those states use a TRUE
        # relative norm, and far above the deep-depletion floor so those
        # states are correctly treated as steady.  (The earlier
        # max(1e-10, flux_floor)·√n_f form = 9e-3 at flux_floor=1e-3 was
        # ABOVE the O(1e-3) lit exciton norm — it deflated the lit exciton
        # steadiness test; fixed per Block C review.)
        _X_FLOOR = 1e-8
        n_f = sysm.dm.n_nodes

        def _rel_norm_dd(fields):
            num2 = sum(np.linalg.norm(sysm.T.T @ (new_state[f] - state[f])) ** 2
                       for f in fields)
            den2 = sum(np.linalg.norm(sysm.T.T @ new_state[f]) ** 2
                       for f in fields)
            return math.sqrt(num2) / max(math.sqrt(den2), 1e-30)

        def _rel_norm_ex(fields):
            num2 = sum(np.linalg.norm(sysm.T.T @ (new_state[f] - state[f])) ** 2
                       for f in fields)
            den2 = sum(np.linalg.norm(sysm.T.T @ new_state[f]) ** 2
                       for f in fields)
            # max(actual_norm, X̂_FLOOR·√n_f): only a negligibly-small exciton
            # field is floored; active fields use the true relative norm.
            return math.sqrt(num2) / max(math.sqrt(den2), _X_FLOOR * math.sqrt(n_f))

        dd_norm = _rel_norm_dd([IPHI, IN, IP])
        ex_norm = _rel_norm_ex([IXD, IXA])

        # Flux pair
        jny, jpy = _flux_pair(sysm, new_state, h_axis=h_axis)

        step_rec = {
            "t_hat": t_hat, "dt_hat": dt_hat,
            "newton_its": info.get("iters", 0),
            "dd_norm": dd_norm, "ex_norm": ex_norm,
            "jny": jny, "jpy": jpy, "accepted": True,
        }
        if post_process:
            # CPU post_process.txt integral set, per accepted step (nondim).
            step_rec["post_process"] = _post_process(sysm, new_state, t_hat)
        step_history.append(step_rec)
        if log is not None:
            log.log_step(step_rec)

        # Update BDF history
        prev2_state = {f: prev_state[f].copy() for f in range(NDOF)}
        prev_state  = {f: state[f].copy()      for f in range(NDOF)}
        state       = new_state

        # Steady criterion
        converged, flux_imbal = _steady_criterion(
            step_history, time_stepping_tol, flux_floor)
        if converged:
            wall_s = time.time() - t0_wall
            if log is not None:
                log.log_stage({
                    "stage": stage_name, "steps": steps_accepted,
                    "wall_s": wall_s,
                    "criterion_fired": True,
                    "final_jny": jny, "final_jpy": jpy,
                    "flux_imbalance": flux_imbal,
                })
            return state, {
                "steps_accepted": steps_accepted,
                "steps_rejected": steps_rejected,
                "criterion_fired": True,
                "step_history": step_history,
                "jny": jny, "jpy": jpy,
                "t_hat": t_hat,
                "flux_imbalance": flux_imbal,
            }

        # Advance dt (log-space schedule, never decrease below current dt)
        dt_sched = _dt_schedule(t_hat, dt_hat, dt0_hat, dt_max_hat)
        dt_hat = max(dt_sched, dt_hat)

    # max_steps reached
    wall_s = time.time() - t0_wall
    if log is not None:
        log.log_stage({
            "stage": stage_name, "steps": steps_accepted,
            "wall_s": wall_s,
            "criterion_fired": False,
            "final_jny": jny, "final_jpy": jpy,
            "reason": "max_steps",
        })
    return state, {
        "steps_accepted": steps_accepted,
        "steps_rejected": steps_rejected,
        "criterion_fired": False,
        "step_history": step_history,
        "jny": jny, "jpy": jpy,
        "t_hat": t_hat,
        "reason": "max_steps",
    }


# ══════════════════════════════════════════════════════════════════════════════
# _post_process — volume-integrated diagnostic quantities
# ══════════════════════════════════════════════════════════════════════════════

def _post_process(sysm, state, t_hat: float = 0.0) -> dict:
    """Compute element-volume-weighted integrals of key diagnostic quantities.

    Uses GP quadrature (element Jacobian × GP weight × field value) to compute
    volume integrals without requiring direct access to the mass matrix.

    Returns
    -------
    dict with keys:
        int_n, int_p, int_xd, int_xa,
        int_gd, int_ga,
        int_kd_xd, int_ka_xa,
        int_xd_tau, int_xa_tau,
        int_gamma_np,
        t_hat
    """
    dm = sysm.dm
    h_all = dm.mesh.tree.h()

    cl = sysm._closures(state)
    gd, ga = sysm._gen()

    int_n = 0.0; int_p = 0.0; int_xd = 0.0; int_xa = 0.0
    int_gd = 0.0; int_ga = 0.0
    int_kd_xd = 0.0; int_ka_xa = 0.0
    int_xd_tau = 0.0; int_xa_tau = 0.0
    int_gamma_np = 0.0

    for pv, b in dm.bins.items():
        eids = dm.mesh.bins[pv]
        he = h_all[eids]
        jac = (0.5 * he) ** dm.dim   # [ne]
        nqp = b["nqp"]
        wt = dm.tables_by_p[pv].w    # [nqp]
        ne = len(eids)
        jac_gp = np.repeat(jac, nqp)
        wt_gp  = np.tile(wt, ne)
        dV     = wt_gp * jac_gp       # GP quadrature weight [ne*nqp]

        n_vals  = cl["n_gp"][pv]
        p_vals  = cl["p_gp"][pv]
        xd_vals = cl["xd_gp"][pv]
        xa_vals = cl["xa_gp"][pv]
        kd_vals = cl["kd"][pv]
        ka_vals = cl["ka"][pv]
        R_vals  = cl["R"][pv]        # γ̂ n̂ p̂ per GP

        gd_vals = gd[pv] if pv in gd else np.zeros_like(n_vals)
        ga_vals = ga[pv] if pv in ga else np.zeros_like(p_vals)

        int_n         += float(np.sum(dV * n_vals))
        int_p         += float(np.sum(dV * p_vals))
        int_xd        += float(np.sum(dV * xd_vals))
        int_xa        += float(np.sum(dV * xa_vals))
        int_gd        += float(np.sum(dV * gd_vals))
        int_ga        += float(np.sum(dV * ga_vals))
        int_kd_xd     += float(np.sum(dV * kd_vals * xd_vals))
        int_ka_xa     += float(np.sum(dV * ka_vals * xa_vals))
        int_xd_tau    += float(np.sum(dV * sysm.tau_inv_d * xd_vals))
        int_xa_tau    += float(np.sum(dV * sysm.tau_inv_a * xa_vals))
        int_gamma_np  += float(np.sum(dV * R_vals))

    return {
        "int_n":        int_n,
        "int_p":        int_p,
        "int_xd":       int_xd,
        "int_xa":       int_xa,
        "int_gd":       int_gd,
        "int_ga":       int_ga,
        "int_kd_xd":    int_kd_xd,
        "int_ka_xa":    int_ka_xa,
        "int_xd_tau":   int_xd_tau,
        "int_xa_tau":   int_xa_tau,
        "int_gamma_np": int_gamma_np,
        "t_hat":        t_hat,
    }


# ══════════════════════════════════════════════════════════════════════════════
# XDDRun — the driver class
# ══════════════════════════════════════════════════════════════════════════════

class XDDRun:
    """Continuation/marching protocol driver for XDDSystem.

    Drives XDDSystem via BDF transient marching to reach steady states.
    Steady states are NEVER reached by direct Newton on device configs —
    always by transient marching (BDF σ=1/Δt̂ regularises each step's Newton).

    Parameters
    ----------
    sysm : XDDSystem
        Configured XDDSystem (closures, GP coefficients, constraints all set).
    mesh : mesh object
        The device mesh (for bilayer_electrode_bcs / log_linear_ic).
    cons : constraints object
    params : XDDParams or None
        For scale information.  If None, defaults are used.
    morphology : array or None
        Unused in R0; reserved for future morphology-conditioned strategies.
    strategy : int
        Solver strategy.  One of PULSE_GENERATION(0), STEADY_STATE_JV(1),
        TRANSIENT_JV(2), STEADY_PULSE(3).
    Eg_hat : float or None
        Nondim band gap Ê_g = E_g / phi0.  Required for bilayer BCs.
    V_sweep : list of float or None
        Nondim applied voltages V̂_app for the JV sweep.
    G_max_hat : float or None
        Max nondim generation rate for the G-ramp.
    G_levels : int
        Number of G-ramp rungs (geometric from G_max/5^levels to G_max).
    generation : Generation or None
        A3 Generation closure for PULSE/STEADY_PULSE strategies.
    pulse_duration_hat : float or None
        Nondim pulse duration for PULSE_GENERATION / STEADY_PULSE.
    dt0_hat : float or None
        Starting dt̂.  Default: 1e-8.
    dt_max_hat : float
        Maximum dt̂.  Default: 0.1.
    time_stepping_tol : float
        Steady criterion field-norm threshold.
    flux_floor : float
        Steady criterion flux floor.
    max_steps_per_stage : int
        Cap on accepted steps per stage.
    bdf2 : bool
        Use BDF2 after the first step.
    minority_ln : float
        log(minority carrier BC floor).  Default: −60.
    h_axis : int
        Device height coordinate axis.  Default: 1 (y).
    outdir : str or None
        Output directory for log files.
    newton_kw : dict or None
        Extra kwargs passed to XDDSystem.solve_newton via step_bdf.
    """

    def __init__(
        self,
        sysm,
        mesh,
        cons,
        *,
        params=None,
        morphology=None,
        strategy: int = STEADY_STATE_JV,
        Eg_hat: Optional[float] = None,
        V_sweep: Optional[List[float]] = None,
        G_max_hat: Optional[float] = None,
        G_levels: int = 3,
        generation=None,
        pulse_duration_hat: Optional[float] = None,
        dt0_hat: Optional[float] = None,
        dt_max_hat: float = 0.1,
        time_stepping_tol: float = 1e-5,
        flux_floor: float = 1e-6,
        max_steps_per_stage: int = 500,
        bdf2: bool = False,
        minority_ln: float = -60.0,
        h_axis: int = 1,
        outdir: Optional[str] = None,
        newton_kw: Optional[dict] = None,
        eg_ramp_levels: Optional[List[float]] = None,
    ):
        self.sysm = sysm
        self.mesh = mesh
        self.cons = cons
        self.params = params
        self.morphology = morphology
        self.strategy = strategy
        self.Eg_hat = Eg_hat
        self.V_sweep = V_sweep if V_sweep is not None else [0.0]
        self.G_max_hat = G_max_hat
        self.G_levels = G_levels
        self.generation = generation
        self.pulse_duration_hat = pulse_duration_hat
        self.dt0_hat = dt0_hat if dt0_hat is not None else 1e-8
        self.dt_max_hat = dt_max_hat
        self.time_stepping_tol = time_stepping_tol
        self.flux_floor = flux_floor
        self.max_steps_per_stage = max_steps_per_stage
        self.bdf2 = bdf2
        self.minority_ln = minority_ln
        self.h_axis = h_axis
        self.outdir = outdir
        self.newton_kw = newton_kw or {}
        self.eg_ramp_levels = eg_ramp_levels  # None → auto ladder from Eg_hat
        self.log = _XDDRunLog(outdir=outdir)

    def _march_kw(self):
        return dict(
            dt0_hat=self.dt0_hat,
            dt_max_hat=self.dt_max_hat,
            max_steps=self.max_steps_per_stage,
            time_stepping_tol=self.time_stepping_tol,
            flux_floor=self.flux_floor,
            bdf2=self.bdf2,
            log=self.log,
            h_axis=self.h_axis,
            newton_kw=self.newton_kw,
        )

    def _apply_bcs(self, V_app_hat: float = 0.0):
        """Apply bilayer electrode BCs at the given applied voltage."""
        if self.Eg_hat is None:
            raise ValueError("XDDRun: Eg_hat is required for bilayer BCs")
        return bilayer_electrode_bcs(
            self.sysm, self.mesh, self.cons,
            Eg_hat=self.Eg_hat,
            V_app_hat=V_app_hat,
            h_axis=self.h_axis,
            minority_ln=self.minority_ln,
        )

    def _get_ic(self, V_app_hat: float = 0.0):
        """Return the marchable log-linear initial condition at given voltage.

        Uses the B5 log-linear IC (u=ln n̂, v=ln p̂ linear in height) rather than
        the PRIMAL Boltzmann `continuation_ic`: the latter's e^{φ̂} carriers give
        an it-1 residual ~1e17 at Ê_g≳20 (B4's documented divergence) and stall
        the BDF march at the dt floor.  The log-linear IC starts small (r0~1) so
        the σ-regularised BDF march advances from step 1.  (This is the same IC
        the dark-equilibrium march uses; the V̂-dependent φ̂ tilt enters through
        the Dirichlet BCs, which the march relaxes the interior toward.)
        """
        if self.Eg_hat is None:
            raise ValueError("XDDRun: Eg_hat is required for the initial condition")
        # log_linear_ic sets φ̂ = +Eg/2 − Eg·ξ (V̂=0 tilt); the applied-bias tilt
        # is imposed by the electrode BCs, so the interior IC uses the Eg tilt.
        return log_linear_ic(
            self.mesh, self.Eg_hat, self.minority_ln, h_axis=self.h_axis)

    # ── Eg-continuation (bootstraps the conditioning wall) ───────────────────

    def run_eg_ramp(self) -> tuple:
        """Continuation in Ê_g from a small value up to self.Eg_hat, marching
        each rung to steady with the previous rung's state as IC.

        Motivation.  A cold BDF march to a DEEP-drive bilayer (large Ê_g, tiny
        minority floor e^{minority_ln}) can fail on the FIRST Newton step: in
        log mode the reduced Jacobian carrier columns scale by diag(n̂) with
        n̂≈e^{minority_ln}, so the Newton direction is ill-conditioned (‖δu‖
        can blow up → exp overflow → every backtracked step is rejected).  The
        Ê_g-ladder keeps each rung's carrier span (hence the conditioning)
        bounded: rung k uses minority_ln = max(−Ê_g^{(k)}, floor_cap), so the
        Boltzmann IC is O(1) and Newton converges; carrying the state forward
        walks the solution to the target drive.

        Returns
        -------
        (final_state, march_info, ramp_history)
            ramp_history : list of (Ê_g^{(k)}, march_info) per rung.
        """
        if self.Eg_hat is None:
            raise ValueError("XDDRun: Eg_hat is required for run_eg_ramp")

        if self.eg_ramp_levels is not None:
            levels = list(self.eg_ramp_levels)
        else:
            levels = []
            eg = min(4.0, self.Eg_hat)
            while eg < self.Eg_hat * 0.999:
                levels.append(eg)
                eg = min(eg * 2.0, self.Eg_hat)
            levels.append(self.Eg_hat)

        # dark
        z = {pv: np.zeros(len(self.sysm.dist_gp[pv])) for pv in self.sysm.dm.bins}
        self.sysm.set_generation(z, z)

        state = None
        ramp_history = []
        final_info = None
        for k, Eg_cur in enumerate(levels):
            is_final = (k == len(levels) - 1)
            m_cur = self.minority_ln if is_final else max(-Eg_cur, -16.0)
            bilayer_electrode_bcs(
                self.sysm, self.mesh, self.cons,
                Eg_hat=Eg_cur, V_app_hat=0.0,
                h_axis=self.h_axis, minority_ln=m_cur,
            )
            if state is None:
                ic = log_linear_ic(self.mesh, Eg_cur, m_cur, h_axis=self.h_axis)
            else:
                ic = {f: state[f].copy() for f in range(NDOF)}
            self.log.log_header({
                "stage": f"eg_ramp_{k + 1}/{len(levels)}",
                "Eg_hat": Eg_cur, "minority_ln": m_cur,
            })
            state, info = _march_to_steady(
                self.sysm, ic,
                stage_name=f"eg_{Eg_cur:.1f}",
                **self._march_kw()
            )
            ramp_history.append((Eg_cur, {"accepted": info["steps_accepted"],
                                          "criterion_fired": info["criterion_fired"]}))
            final_info = info
        return state, final_info, ramp_history

    # ── Dark equilibrium ────────────────────────────────────────────────────

    def run_dark_equilibrium(self, use_eg_ramp: bool = False):
        """March to dark equilibrium (V̂=0, no generation).

        Parameters
        ----------
        use_eg_ramp : bool
            If True, bootstrap the march with an Ê_g-continuation ladder
            (run_eg_ramp) — for deep-drive configs whose cold-start BDF is
            ill-conditioned.  Default False: march directly from the
            log-linear IC, which suffices for resolvable-Debye configs.

        Returns
        -------
        (state, march_info)
        """
        z = {pv: np.zeros(len(self.sysm.dist_gp[pv]))
             for pv in self.sysm.dm.bins}
        self.sysm.set_generation(z, z)

        self.log.log_header({
            "stage": "dark_equilibrium",
            "V_app_hat": 0.0,
            "generation": "none",
            "dt0_hat": self.dt0_hat,
            "dt_max_hat": self.dt_max_hat,
            "eg_ramp_used": bool(use_eg_ramp),
        })

        if use_eg_ramp:
            state, info, ramp_hist = self.run_eg_ramp()
            info = dict(info)
            info["eg_ramp_history"] = ramp_hist
            return state, info

        self._apply_bcs(V_app_hat=0.0)
        ic = log_linear_ic(self.mesh, self.Eg_hat, self.minority_ln,
                           h_axis=self.h_axis)
        state, info = _march_to_steady(
            self.sysm, ic,
            stage_name="dark_eq",
            **self._march_kw()
        )
        return state, info

    # ── G-ramp continuation ─────────────────────────────────────────────────

    def run_generation_ramp(self, ic_state):
        """G-ramp: geometric ramp of generation from G_max/5^levels to G_max.

        Parameters
        ----------
        ic_state : dict
            Starting state (e.g. from run_dark_equilibrium).

        Returns
        -------
        (final_state, ramp_history)
            ramp_history is a list of (G_hat, march_info) tuples.
        """
        if self.G_max_hat is None or self.generation is None:
            raise ValueError("XDDRun: G_max_hat and generation are required for G-ramp")

        # Build the ramp levels: G_max/5^(levels) → G_max
        levels = self.G_levels
        G_levels_list = [self.G_max_hat / (5 ** (levels - i))
                         for i in range(1, levels + 1)]

        state = ic_state
        ramp_history = []

        for rung, G_hat in enumerate(G_levels_list):
            # Set generation at this rung — SHARED-peak scaling (preserves the
            # donor:acceptor ratio), the same convention as the PULSE /
            # STEADY_PULSE paths so a ramp's final rung and a pulse's lit steady
            # at the same G_max_hat are the identical generation field.
            self._set_generation_scaled(G_hat)

            self.log.log_header({
                "stage": f"G_ramp_{rung + 1}/{levels}",
                "G_hat": G_hat,
            })

            state, info = _march_to_steady(
                self.sysm, state,
                stage_name=f"G_ramp_{rung + 1}",
                **self._march_kw()
            )
            ramp_history.append((G_hat, info))

        return state, ramp_history

    # ── Voltage sweep ───────────────────────────────────────────────────────

    def run_voltage_sweep(self, ic_state, *, use_continuation: bool = True):
        """Sweep over V_sweep voltages, marching to steady at each bias.

        Parameters
        ----------
        ic_state : dict
            Starting state for the first bias point.
        use_continuation : bool
            If True (STEADY_STATE_JV): carry the previous voltage's final
            state as IC for the next.  If False (TRANSIENT_JV): reset to
            ic_state at each voltage.

        Returns
        -------
        (states, sweep_history)
            states : list of final states per voltage
            sweep_history : list of (V_hat, march_info) tuples
        """
        states = []
        sweep_history = []
        state = ic_state

        for vi, V_hat in enumerate(self.V_sweep):
            self._apply_bcs(V_app_hat=V_hat)

            if not use_continuation:
                state = self._get_ic(V_app_hat=V_hat)

            self.log.log_header({
                "stage": f"V_sweep_{vi + 1}/{len(self.V_sweep)}",
                "V_app_hat": V_hat,
            })

            final_state, info = _march_to_steady(
                self.sysm, state,
                stage_name=f"V_{V_hat:.4f}",
                **self._march_kw()
            )
            states.append(final_state)
            sweep_history.append((V_hat, info))

            if use_continuation:
                state = final_state

        return states, sweep_history

    # ── PULSE_GENERATION strategy ───────────────────────────────────────────

    def run_pulse(self, ic_state):
        """March under a generation pulse (strategy 0).

        Two-phase pulse (a simplified square waveform, not a per-step
        `generation.amplitude(t̂)` scaling): phase 1 sets full generation
        (peak-scaled to G_max_hat via `_set_generation_scaled`) and marches
        with dt̂_max capped to pulse_duration/10; phase 2 zeros generation
        (light off) and marches the dark relaxation.  Post-process integrals
        are recorded per accepted step in both phases.

        Returns
        -------
        (final_state, pulse_history)
        """
        if self.generation is None:
            raise ValueError("XDDRun: generation is required for PULSE strategy")

        pulse_dur = self.pulse_duration_hat or 1e-3
        state = ic_state
        mk = dict(self._march_kw()); mk["post_process"] = True

        # Phase 1: pulse on (generation scaled to G_max_hat if set)
        self._set_generation_scaled(self.G_max_hat)
        self.log.log_header({"stage": "pulse_on",
                              "duration_hat": pulse_dur})

        mk_on = dict(mk); mk_on["dt_max_hat"] = min(pulse_dur / 10.0,
                                                    self.dt_max_hat)
        state, info_on = _march_to_steady(
            self.sysm, state, stage_name="pulse_on", **mk_on)

        # Phase 2: pulse off (dark relaxation)
        z = {pv: np.zeros(len(self.sysm.dist_gp[pv]))
             for pv in self.sysm.dm.bins}
        self.sysm.set_generation(z, z)

        self.log.log_header({"stage": "pulse_off"})
        state, info_off = _march_to_steady(
            self.sysm, state, stage_name="pulse_off", **mk)

        return state, {"pulse_on": info_on, "pulse_off": info_off}

    # ── STEADY_PULSE strategy ───────────────────────────────────────────────

    def run_steady_pulse(self, ic_state):
        """Lit steady → light off → dark relaxation (strategy 3).

        Returns
        -------
        (final_state, history)
        """
        if self.generation is None:
            raise ValueError("XDDRun: generation is required for STEADY_PULSE strategy")

        # Phase 1: march to lit steady (generation scaled to G_max_hat if set)
        self._set_generation_scaled(self.G_max_hat)

        self.log.log_header({"stage": "lit_steady"})
        mk = dict(self._march_kw()); mk["post_process"] = True
        state, info_lit = _march_to_steady(
            self.sysm, ic_state,
            stage_name="lit_steady",
            **mk
        )

        # Phase 2: light off → dark relaxation (∫Ĝ → exactly 0 here)
        z = {pv: np.zeros(len(self.sysm.dist_gp[pv]))
             for pv in self.sysm.dm.bins}
        self.sysm.set_generation(z, z)

        self.log.log_header({"stage": "dark_relax"})
        state, info_dark = _march_to_steady(
            self.sysm, state,
            stage_name="dark_relax",
            **mk
        )

        return state, {"lit_steady": info_lit, "dark_relax": info_dark}

    def _set_generation_scaled(self, G_max_hat):
        """Set GP generation from self.generation.spatial, optionally rescaled
        so its peak magnitude equals G_max_hat (None → use raw spatial values)."""
        dist_gp = self.sysm.dist_gp
        gd_gp = {}; ga_gp = {}
        for pv in self.sysm.dm.bins:
            Gd, Ga = self.generation.spatial(dist_gp[pv], 0.0)
            if G_max_hat is not None:
                peak = max(float(np.max(np.abs(Gd))),
                           float(np.max(np.abs(Ga))), 1e-30)
                Gd = Gd * (G_max_hat / peak)
                Ga = Ga * (G_max_hat / peak)
            gd_gp[pv] = Gd
            ga_gp[pv] = Ga
        self.sysm.set_generation(gd_gp, ga_gp)

    # ── Top-level run() dispatcher ──────────────────────────────────────────

    def run(self) -> dict:
        """Run the simulation according to self.strategy.

        Returns
        -------
        dict with keys: strategy, dark_eq_info, histories, final_state, log
        """
        result = {"strategy": self.strategy}

        if self.strategy == STEADY_STATE_JV:
            # dark_eq → G_ramp (if generation) → V_sweep with continuation
            dark_state, dark_info = self.run_dark_equilibrium()
            result["dark_eq_info"] = dark_info

            if self.generation is not None and self.G_max_hat is not None:
                lit_state, ramp_hist = self.run_generation_ramp(dark_state)
                result["ramp_history"] = ramp_hist
            else:
                lit_state = dark_state

            states, sweep_hist = self.run_voltage_sweep(
                lit_state, use_continuation=True)
            result["sweep_history"] = sweep_hist
            result["final_state"] = states[-1] if states else lit_state
            result["histories"] = sweep_hist

        elif self.strategy == TRANSIENT_JV:
            # dark_eq → G_ramp (if generation) → V_sweep WITHOUT continuation
            dark_state, dark_info = self.run_dark_equilibrium()
            result["dark_eq_info"] = dark_info

            if self.generation is not None and self.G_max_hat is not None:
                lit_state, ramp_hist = self.run_generation_ramp(dark_state)
                result["ramp_history"] = ramp_hist
            else:
                lit_state = dark_state

            states, sweep_hist = self.run_voltage_sweep(
                lit_state, use_continuation=False)
            result["sweep_history"] = sweep_hist
            result["final_state"] = states[-1] if states else lit_state
            result["histories"] = sweep_hist

        elif self.strategy == PULSE_GENERATION:
            # dark_eq → pulse march
            dark_state, dark_info = self.run_dark_equilibrium()
            result["dark_eq_info"] = dark_info
            final_state, pulse_hist = self.run_pulse(dark_state)
            result["final_state"] = final_state
            result["histories"] = pulse_hist

        elif self.strategy == STEADY_PULSE:
            # dark_eq → lit_steady → light_off → dark relaxation
            dark_state, dark_info = self.run_dark_equilibrium()
            result["dark_eq_info"] = dark_info
            final_state, hist = self.run_steady_pulse(dark_state)
            result["final_state"] = final_state
            result["histories"] = hist

        else:
            raise ValueError(f"XDDRun: unknown strategy {self.strategy}")

        result["log"] = self.log.finalize()
        self.log.close()
        return result
