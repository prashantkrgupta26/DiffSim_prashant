"""OrgElMorph course - Computational C6: nonlinear-solver diagnostics.

Every implicit phase-field step is a *nonlinear* solve: the production
``CahnHilliardStepper.step`` runs a monolithic Newton iteration whose health
decides whether a run is trustworthy, slow, or garbage.  This module drives
that SAME production stepper (``src/diffsim/physics/cahn_hilliard.py`` -- the
brick of Physics P1 and Computational C1) and instruments it into a repeatable
DIAGNOSTIC workflow:

  1. ANATOMY of a healthy Newton solve -- residual norm ||r|| and update norm
     ||dx||_inf per iteration; a correct analytic tangent gives QUADRATIC
     convergence (the residual is squared each step).
  2. STOPPING criteria -- absolute ||dx||, relative ||dx||/||x||, and residual
     ||r||; when they agree, when they disagree, and which to trust.
  3. SAFEGUARDS -- the stepper's trust clamp (a c-increment > 2 units is a
     diverging transient, so the whole update is rescaled) and the
     Flory-Huggins box projection (Newton iterates clipped into the physical
     (0,1) so f'' never hits the 1/eps regularization cap).  We report the
     projection bookkeeping (proj_dofs, proj_max) the stepper already keeps.
  4. FAILURE MODES that GENUINELY FAIL, each with its diagnostic signature:
       - Newton STAGNATION: a deep quench at too-large dt -- the raw Newton
         direction diverges, the trust clamp caps every step at 2.0, so the
         solve crawls linearly and never reaches tol within the iteration
         budget.  The fix is a smaller dt, not more iterations.
       - MIN-DT failure: the adaptive controller cannot meet its LTE tolerance
         and is pinned at dt_min -- every step is a forced accept.
       - ILL-CONDITIONED Jacobian: a Flory-Huggins state driven toward the
         wall, where f''(c) ~ A/c blows up and the tangent's condition number
         explodes.
     On non-convergence we write a FAILURE CHECKPOINT (state + diagnostics) so
     the failure is reproducible and inspectable.

We do NOT edit the production stepper.  The per-iteration residual/update
trajectory -- which ``step()`` does not log -- is reconstructed through the
PUBLIC API: re-solving the same step with ``newton_max = 1, 2, 3, ...`` from
the identical state and reading ``last_newton`` and the captured
``last_system`` (``capture_system=True``).  Newton from a fixed state is
deterministic, so this exactly recovers the internal trajectory.

The mixed (c, mu) system solved each step is
    c_t = M div(grad mu),          (conserved transport)
    mu  = f'(c) - kappa lap c,     (chemical potential)
with f(c) = 1/4 (c^2-1)^2 (poly) or the regularized Flory-Huggins log (fh).
"""
from __future__ import annotations

import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.cahn_hilliard import CahnHilliardStepper, adaptive_march

# Model constants (the C1/P1 values).
M, KAP = 1.0, 5e-4
CLAMP = 2.0                 # the stepper's trust-clamp threshold on |dc|


def build_dm(level, p=1, device="cuda:0"):
    """Uniform 2-D box, 2^level cells per side, degree-p elements."""
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2), device)
    return dm, mesh, cons


# ---------------------------------------------------------------------------
# initial states used by the diagnostics (each is a deterministic function of
# the free-node coordinates + a fixed seed, so every run is reproducible)
# ---------------------------------------------------------------------------
def ic_gentle(x):
    """A smooth, small cosine perturbation of a uniform blend -- the regime
    where Newton is textbook well-behaved."""
    return 0.0 + 0.1 * np.cos(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])


def ic_quench(x, seed=0, amp=0.6):
    """A high-amplitude random field -- a deep spinodal quench that stresses
    the nonlinear solve (the raw Newton direction overshoots)."""
    rng = np.random.default_rng(seed)
    return 0.0 + amp * rng.standard_normal(len(x))


def _make_stepper(dm, dt, newton_max, energy="poly", fh_B=3.0,
                  newton_tol=1e-12, capture=True):
    return CahnHilliardStepper(
        dm, M, KAP, dt, order=1, newton_tol=newton_tol,
        newton_max=newton_max, energy=energy, fh_B=fh_B,
        linsolver="splu", capture_system=capture)


# ---------------------------------------------------------------------------
# 1. Newton trajectory reconstruction (public-API only)
# ---------------------------------------------------------------------------
def newton_trajectory(dm, ic, dt, kmax, energy="poly", fh_B=3.0,
                      newton_tol=1e-12, mu_init="consistent"):
    """Reconstruct ||r|| and ||dx||_inf per Newton iteration for ONE step.

    Re-solves the identical step capped at ``newton_max = 1..kmax`` and reads
    the captured system (``||r||`` before the k-th linear solve) and
    ``last_newton['dx_inf']`` (``||dx||_inf`` of the k-th update).  Because the
    step is deterministic, the k-th entry is exactly the k-th internal Newton
    iterate.  Returns per-iteration arrays + convergence flags + a clamp mask
    (an update capped at the trust threshold 2.0).
    """
    res, dxi, conv, clamp = [], [], [], []
    for k in range(1, kmax + 1):
        st = _make_stepper(dm, dt, k, energy, fh_B, newton_tol)
        st.set_initial(ic, mu_init=mu_init)
        st.step()
        A, r = st.last_system
        res.append(float(np.linalg.norm(r)))
        dxi.append(float(st.last_newton["dx_inf"]))
        conv.append(bool(st.last_newton["converged"]))
        clamp.append(bool(abs(st.last_newton["dx_inf"] - CLAMP) < 1e-9))
    # first converged iteration (1-based), else -1
    first = next((i + 1 for i, c in enumerate(conv) if c), -1)
    return dict(k=list(range(1, kmax + 1)), res=res, dx_inf=dxi,
                converged=conv, clamp=clamp, iters_to_converge=first,
                any_converged=first > 0, clamp_fired=any(clamp))


def stopping_criteria(traj, x_scale=1.0, tol=1e-8):
    """Iteration at which each stopping test would fire on a trajectory.

    Three tests at the SAME tol: residual ``||r|| < tol``, absolute update
    ``||dx||_inf < tol``, relative update ``||dx||_inf / x_scale < tol``.
    Returns the 1-based iteration each fires (``-1`` = never within the
    trajectory).  The point: a residual test and an update test can stop at
    different iterations; on a well-scaled O(1) field they agree, on a
    rescaled field the ABSOLUTE test silently changes meaning.
    """
    def first_below(seq, thr):
        return next((i + 1 for i, v in enumerate(seq) if v < thr), -1)
    return dict(
        tol=tol, x_scale=float(x_scale),
        by_residual=first_below(traj["res"], tol),
        by_abs_update=first_below(traj["dx_inf"], tol),
        by_rel_update=first_below([d / x_scale for d in traj["dx_inf"]], tol))


# ---------------------------------------------------------------------------
# 3. Safeguards: FH projection bookkeeping + trust-clamp activity
# ---------------------------------------------------------------------------
def safeguard_report(dm, dt=1.0, fh_B=8.0, newton_max=15, amp=0.3):
    """Run ONE deep Flory-Huggins quench step with the safeguards live and
    report what they did: the box-projection dof count + largest correction
    (kept by the stepper as ``proj_dofs`` / ``proj_max``), whether the trust
    clamp fired, and whether the step still converged.  The lesson: with the
    projection + clamp a deep FH quench whose early Newton iterates overshoot
    the physical (0,1) box -- driving f'' toward the 1/eps cap -- is pulled
    back onto the wall and converges cleanly."""
    st = _make_stepper(dm, dt, newton_max, energy="fh", fh_B=fh_B,
                       newton_tol=1e-10)
    st.set_initial(lambda x: 0.5 + amp * np.random.default_rng(0)
                   .standard_normal(len(x)), mu_init="consistent")
    st.step()
    return dict(proj_dofs=int(st.proj_dofs), proj_max=float(st.proj_max),
                converged=bool(st.last_newton["converged"]),
                iters=int(st.last_newton["iters"]),
                dx_inf=float(st.last_newton["dx_inf"]))


# ---------------------------------------------------------------------------
# 4a. Newton STAGNATION (a genuine non-convergence)
# ---------------------------------------------------------------------------
def stagnation_case(dm, dt=0.05, newton_max=15, seed=0):
    """A deep poly quench at too-large dt.  The raw Newton step is a diverging
    transient; the trust clamp caps every update at 2.0, so the residual
    crawls DOWN roughly linearly instead of being squared -- and after the
    stepper's whole iteration budget it is still many orders from tol.
    GENUINELY does not converge (``converged == False``); the diagnostic
    signature is a persistently clamped update with a slow, non-quadratic
    residual.  The fix is a smaller dt, not more iterations."""
    traj = newton_trajectory(dm, lambda x: ic_quench(x, seed), dt, newton_max,
                             energy="poly", newton_tol=1e-10)
    return dict(name="Newton stagnation (deep quench, dt too large)",
                dt=dt, newton_max=newton_max, traj=traj,
                converged=traj["converged"][-1],
                res_final=traj["res"][-1],
                clamp_fraction=float(np.mean(traj["clamp"])),
                diagnosis="clamped diverging direction; reduce dt")


def stagnation_fixed(dm, dt=2e-3, newton_max=15, seed=0):
    """The SAME deep quench with dt reduced -- Newton now converges.  Proof the
    stagnation above is a step-size problem, not a code bug."""
    traj = newton_trajectory(dm, lambda x: ic_quench(x, seed), dt, newton_max,
                             energy="poly", newton_tol=1e-10)
    return dict(dt=dt, converged=traj["converged"][-1],
                iters=traj["iters_to_converge"], res_final=traj["res"][-1])


# ---------------------------------------------------------------------------
# 4b. MIN-DT failure (adaptive controller pinned at the floor)
# ---------------------------------------------------------------------------
def min_dt_case(dm, dt0=0.02, dt_min=0.02, dt_max=0.1, tol=1e-9, t_end=0.15,
                seed=1):
    """Drive ``adaptive_march`` with an LTE tolerance so tight (and a dt_min so
    high) that the step-doubling estimator can NEVER meet it: every step is a
    forced accept at the floor.  Returns the dt trajectory (all at dt_min) and
    an explicit LTE-at-floor measurement showing LTE > tol -- the honest
    signature of a min-dt failure (the controller proposes, the floor
    disposes)."""
    st = CahnHilliardStepper(dm, M, KAP, dt0, order=2, newton_tol=1e-10,
                             linsolver="splu")
    st.set_initial(lambda x: ic_quench(x, seed, amp=0.4), mu_init="consistent")
    ts, dts = adaptive_march(st, t_end, tol=tol, dt_min=dt_min, dt_max=dt_max)
    all_floor = bool(np.allclose(dts, dt_min))
    # explicit LTE at the floor: one dt_min step vs two dt_min/2 steps
    lte = _measure_lte(dm, dt_min, seed)
    return dict(name="min-dt failure (controller pinned at floor)",
                dt_min=dt_min, tol=tol, n_steps=len(dts),
                dt_min_seen=float(min(dts)), dt_max_seen=float(max(dts)),
                all_at_floor=all_floor, lte_at_floor=float(lte),
                lte_exceeds_tol=bool(lte > tol),
                diagnosis="LTE tol unreachable at dt_min; loosen tol or "
                          "accept the floor's accuracy")


def _measure_lte(dm, dt, seed):
    """Step-doubling LTE at step size dt from a fresh quench state (BDF1)."""
    def march(nsub):
        st = CahnHilliardStepper(dm, M, KAP, dt / nsub, order=1,
                                 newton_tol=1e-10, linsolver="splu")
        st.set_initial(lambda x: ic_quench(x, seed, amp=0.4),
                       mu_init="consistent")
        for _ in range(nsub):
            st.step()
        return st.x[0::2].copy()
    c1, c2 = march(1), march(2)
    num = float(np.linalg.norm(c1 - c2))
    den = max(float(np.linalg.norm(c2)), 1e-30)
    return (num / den) / (2 ** 1 - 1)


# ---------------------------------------------------------------------------
# 4c. ILL-CONDITIONED Jacobian (Flory-Huggins near the wall)
# ---------------------------------------------------------------------------
def _fpp_fh(c, A=1.0, B=6.0, eps=1e-4):
    """f''(c) for the regularized FH energy: A(1/c + 1/(1-c)) - 2B, clamped by
    the same 1/eps cap the kernel uses.  Diverges as c -> wall."""
    c = np.clip(c, eps, 1.0 - eps)
    return A * (1.0 / c + 1.0 / (1.0 - c)) - 2.0 * B


def conditioning_case(dm3, fh_B=6.0, dt=0.5,
                      compositions=(0.5, 0.2, 0.05, 0.01, 0.003)):
    """Assemble ONE Flory-Huggins Newton Jacobian from a FROZEN uniform state
    at each mean composition and measure its (exact, dense) 2-norm condition
    number.  As the state approaches the wall c -> 0, f''(c) ~ A/c grows toward
    the 1/eps cap and the tangent's condition number climbs by orders of
    magnitude -- the reason deep-quench FH steps are hard and why the box
    projection (keeping iterates off the wall) matters.  ``dm3`` is a SMALL
    (level-3) mesh so the dense condition number is exact and cheap."""
    conds, fpp = [], []
    for cm in compositions:
        st = _make_stepper(dm3, dt, 1, energy="fh", fh_B=fh_B,
                           newton_tol=1e-14)
        st.set_initial(lambda x: np.full(len(x), cm), mu_init="zero")
        st.step()
        A, _ = st.last_system
        conds.append(float(np.linalg.cond(A.toarray())))
        fpp.append(float(_fpp_fh(cm, B=fh_B)))
    return dict(name="ill-conditioned FH Jacobian (state near the wall)",
                compositions=list(compositions), cond=conds, fpp=fpp,
                cond_ratio=float(conds[-1] / conds[0]),
                diagnosis="f'' -> 1/eps cap near the wall inflates cond(J); "
                          "keep iterates off the wall (projection)")


# ---------------------------------------------------------------------------
# failure checkpoint
# ---------------------------------------------------------------------------
def write_failure_checkpoint(path, tag, dm, ic, dt, energy, newton_max,
                             traj, fh_B=3.0):
    """Persist a reproducible failure checkpoint: the last Newton state, the
    residual/update trajectory, and the exact config needed to re-run.  A
    real run should NEVER silently drop a failed step -- it should checkpoint
    it for post-mortem (spec Phase 3)."""
    st = _make_stepper(dm, dt, newton_max, energy, fh_B, newton_tol=1e-10)
    c0 = st.set_initial(ic, mu_init="consistent")
    st.step()
    np.savez(path,
             tag=tag, dt=dt, energy=energy, newton_max=newton_max, fh_B=fh_B,
             c_initial=c0, c_last=st.x[0::2], mu_last=st.x[1::2],
             res_history=np.asarray(traj["res"]),
             dx_history=np.asarray(traj["dx_inf"]),
             converged=bool(traj["converged"][-1]))
    return path
