"""NS-SBM course — lid-driven cavity core (Chapters 01 & 02).

The BASE-soundness driver for both engines. A closed unit-square cavity, no
immersed body: the top lid (y=1) slides at U=1, the other three walls no-slip.
We march the SAME mesh with the requested engine(s) and compare the x=0.5
centerline u(y) against Ghia, Ghia & Shin (1982), Table I.

Chapter 01 (`01_monolithic_vms`) reads `march_monolithic`; Chapter 02
(`02_pressure_projection`) reads `march_projection`. The `run_cavity` driver
runs whichever engine(s) are asked for and returns the same result dict, so the
two chapters share one verified core — no toy re-implementation. Both call the
production steppers directly.
"""
from __future__ import annotations

import numpy as np

U_IN = 1.0

# Ghia, Ghia & Shin (1982), Table I: u(y) on the vertical centerline x=0.5.
GHIA_Y = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344,
                   0.6172, 0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703,
                   0.0625, 0.0547, 0.0000])
GHIA_U = {
    100: np.array([1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151,
                   0.00332, -0.13641, -0.20581, -0.21090, -0.15662, -0.10150,
                   -0.06434, -0.04775, -0.04192, -0.03717, 0.00000]),
    400: np.array([1.00000, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093,
                   0.16256, 0.02135, -0.11477, -0.17119, -0.32726, -0.24299,
                   -0.14612, -0.10338, -0.09266, -0.08186, 0.00000]),
}


def build_cavity(level, Re, device="cpu"):
    """Unit-square cavity mesh + BC masks. Re via nu = U_IN * L / Re, L=1."""
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
    """Dirichlet trace: lid (y=1) drives +x at U_IN; all other walls no-slip."""
    g = np.zeros((len(x), 2))
    g[np.abs(x[:, 1] - 1.0) < 1e-12, 0] = U_IN
    return g


def _centerline_u(fx, u_node):
    """Sample u_x(y) at x=0.5 at the Ghia y-stations via nodal interpolation."""
    from diffsim.mesh.pointeval import point_eval_weights
    W = point_eval_weights(fx["mesh"], np.stack(
        [np.full_like(GHIA_Y, 0.5), GHIA_Y], axis=1))
    T = fx["dm"].constraints.T.tocsr()
    return np.asarray(W @ np.asarray(T @ u_node[:, 0]))


def march_monolithic(fx, dt=0.05, nsteps=200):
    """Monolithic saddle march to steady (the oracle)."""
    from diffsim.steppers.linearized import LinearizedMonolithicStepper
    st = LinearizedMonolithicStepper(
        fx["dm"], fx["nu"], dt, f_fn=lambda x, t: np.zeros((len(x), 2)),
        g_fn=lid_g, order=1)
    st.set_initial(lambda x: np.zeros((len(x), 2)))
    x = None
    for _ in range(nsteps):
        x = st.step()
    return np.asarray(x)[:, :2], float(st.divergence_l2())


def march_projection(fx, dt=0.05, nsteps=200):
    """Pressure-projection march to steady (the scalable engine)."""
    from diffsim.steppers.leray import LerayProjectionStepper
    st = LerayProjectionStepper(
        fx["dm"], fx["nu"], dt, f_fn=lambda x, t: np.zeros((len(x), 2)),
        g_fn=lid_g, order=1, picard_iters=1)
    st.set_initial(lambda x: np.zeros((len(x), 2)))
    u = None
    for _ in range(nsteps):
        u, p = st.step()
    return np.asarray(u), float(st.divergence_l2())


def run_cavity(level=4, Re=100, dt=0.05, nsteps=200, device="cpu",
               engines=("monolithic", "projection")):
    """March the requested engine(s) and return a Ghia-comparison dict.

    ``engines`` selects which engine(s) to run. Both are marched by default so
    the same-mesh faithfulness comparison is available; Chapter 01 can pass
    ``engines=("monolithic",)`` and Chapter 02 ``("projection",)`` to focus,
    but both still report the Ghia comparison for their engine.
    """
    fx = build_cavity(level, Re, device)
    gu = GHIA_U[Re]
    out = dict(level=level, side=2 ** level, Re=Re, ghia=gu.tolist(),
               ghia_y=GHIA_Y.tolist())

    um = up = None
    if "monolithic" in engines:
        u_m, div_m = march_monolithic(fx, dt, nsteps)
        um = _centerline_u(fx, u_m)
        out.update(u_mono=um.tolist(), div_mono=div_m,
                   d_mono_ghia=float(np.abs(um - gu).max()))
    if "projection" in engines:
        u_p, div_p = march_projection(fx, dt, nsteps)
        up = _centerline_u(fx, u_p)
        out.update(u_proj=up.tolist(), div_proj=div_p,
                   d_proj_ghia=float(np.abs(up - gu).max()))
    if um is not None and up is not None:
        out["d_proj_mono"] = float(np.abs(up - um).max())
    return out
