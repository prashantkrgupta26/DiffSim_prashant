"""NS-SBM course — Chapter 00 smoke core.

A tiny lid-driven cavity marched through BOTH production engines
(``LinearizedMonolithicStepper`` and ``LerayProjectionStepper``) for a handful
of pseudo-time steps. The point is not the physics (the mesh is far too coarse)
but to prove, end to end, that:

  * the equal-order VMS assembly + saddle solve runs (monolithic), and
  * the predictor -> SPD-PPE -> correction split runs (projection), and
  * the two engines produce finite, bounded, mutually-close velocity fields.

This is the real interface every later chapter uses — no toy re-implementation.
``run.py`` drives this through the course harness (config -> provenance ->
results.json -> tolerance check).
"""
from __future__ import annotations

import numpy as np

U_IN = 1.0


def build_cavity(level, Re, device="cpu"):
    """Unit-square cavity mesh + viscosity. Re via nu = U_IN * L / Re, L = 1."""
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh

    nu = U_IN * 1.0 / Re
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)                 # equal-order P1/P1
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    return dict(dm=dm, mesh=mesh, cons=cons, nu=nu)


def lid_g(x, t):
    """Dirichlet trace: lid (y=1) drives +x at U_IN; other walls no-slip."""
    g = np.zeros((len(x), 2))
    g[np.abs(x[:, 1] - 1.0) < 1e-12, 0] = U_IN
    return g


def march_monolithic(fx, dt, nsteps):
    from diffsim.steppers.linearized import LinearizedMonolithicStepper
    st = LinearizedMonolithicStepper(
        fx["dm"], fx["nu"], dt, f_fn=lambda x, t: np.zeros((len(x), 2)),
        g_fn=lid_g, order=1)
    st.set_initial(lambda x: np.zeros((len(x), 2)))
    x = None
    for _ in range(nsteps):
        x = st.step()
    return np.asarray(x)[:, :2], float(st.divergence_l2())


def march_projection(fx, dt, nsteps):
    from diffsim.steppers.leray import LerayProjectionStepper
    st = LerayProjectionStepper(
        fx["dm"], fx["nu"], dt, f_fn=lambda x, t: np.zeros((len(x), 2)),
        g_fn=lid_g, order=1, picard_iters=1)
    st.set_initial(lambda x: np.zeros((len(x), 2)))
    u = None
    for _ in range(nsteps):
        u, p = st.step()
    return np.asarray(u), float(st.divergence_l2())


def smoke(level=3, Re=100, dt=0.05, nsteps=20, device="cpu"):
    """Return the smoke result dict (both engines, finite/bounded checks)."""
    fx = build_cavity(level, Re, device)
    u_m, div_m = march_monolithic(fx, dt, nsteps)
    u_p, div_p = march_projection(fx, dt, nsteps)
    # max velocity magnitude and the same-mesh engine agreement
    umag_m = float(np.abs(u_m).max())
    umag_p = float(np.abs(u_p).max())
    d_pm = float(np.abs(u_p - u_m).max())
    return dict(
        level=level, side=2 ** level, n_steps=nsteps,
        all_finite=bool(np.all(np.isfinite(u_m)) and np.all(np.isfinite(u_p))),
        umax_mono=umag_m, umax_proj=umag_p,
        div_mono=div_m, div_proj=div_p,
        max_proj_minus_mono=d_pm)
