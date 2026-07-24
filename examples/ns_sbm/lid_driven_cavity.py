#!/usr/bin/env python
r"""Worked example 7.1 — Lid-driven cavity (Ghia benchmark).

The BASE-soundness control for both DiffSim incompressible-NS engines. A closed
unit-square cavity, no immersed body, no SBM: the top lid (y=1) slides at
U=1 while the other three walls are no-slip. We march the SAME mesh with BOTH
engines and compare the x=0.5 centerline u(y) profile against Ghia, Ghia & Shin
(1982), Table I.

  * MONOLITHIC  — LinearizedMonolithicStepper: one coupled (u, p) saddle solve
    per step, equal-order P1/P1 stabilized by residual-based VMS (SUPG + PSPG +
    grad-div). This is the "oracle": the reference the projection must match.
  * PROJECTION  — LerayProjectionStepper: predictor -> SPD pressure-Poisson ->
    L2 velocity correction. The scalable engine (its PPE admits AMG at scale).

Enclosed flow => pressure is defined only up to a constant, so BOTH engines pin
one pressure DOF (free-node 0, the stepper default `pressure_outflow_nodes=None`).

WHAT TO EXPECT (verified on gpubox, level 4 = 16x16, Re=100, dt=0.05, 200 steps):

    projection  u(y=0.9766) = +0.8376   monolithic +0.8529   Ghia +0.84123
    projection  u(y=0.5000) = -0.1894   monolithic -0.1462   Ghia -0.20581
    max|proj - mono| over the Ghia stations = 0.0486  (same-mesh agreement)
    max|proj - Ghia|                        = 0.0164  (coarse mesh vs 129^2 table)
    max|mono - Ghia|                        = 0.0620

The coarse level-4 mesh (16x16 cells) cannot resolve Ghia's 129^2 table to the
third digit — that is expected. The decisive checks are (a) projection tracks
the monolithic on the SAME mesh and (b) both track Ghia to within coarse-mesh
tolerance. (Note: the pointwise ||div u|| of the equal-order VMS projection is
NOT driven to zero — the scheme controls the WEAK/PPE-space divergence; it is
finite and bounded, ~2 here, not a blow-up.)

KNOBS TO EXPLORE (bottom of file):
  * LEVEL   — mesh refinement (5 or 6 tightens toward Ghia; slower).
  * RE      — Reynolds number (100 or 400; 400 needs more steps to steady).
  * NSTEPS  — pseudo-time steps to steady.

Run:
    PYTHONPATH=src:examples python examples/ns_sbm/lid_driven_cavity.py
"""
import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.mesh.pointeval import point_eval_weights
from diffsim.steppers.leray import LerayProjectionStepper
from diffsim.steppers.linearized import LinearizedMonolithicStepper

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
    dim = 2
    nu = U_IN * 1.0 / Re
    tree = build_uniform(level, dim=dim)
    mesh = build_mesh(tree, p=1)                     # equal-order P1/P1
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim), device)
    coords = mesh.node_coords[cons.free_nodes]
    return dict(dm=dm, mesh=mesh, cons=cons, coords=coords, nu=nu, dim=dim)


def lid_g(x, t):
    """Dirichlet trace: lid (y=1) drives +x at U_IN; all other walls no-slip."""
    g = np.zeros((len(x), 2))
    g[np.abs(x[:, 1] - 1.0) < 1e-12, 0] = U_IN
    return g


def centerline_u(fx, u_node, Re):
    """Sample u(y) at x=0.5 at the Ghia y-stations via nodal interpolation."""
    W = point_eval_weights(fx["mesh"], np.stack(
        [np.full_like(GHIA_Y, 0.5), GHIA_Y], axis=1))
    T = fx["dm"].constraints.T.tocsr()
    return np.asarray(W @ np.asarray(T @ u_node[:, 0]))


def march_projection(fx, dt=0.05, nsteps=200):
    """Pressure-projection march to steady. Single Picard pass, order-1 BDF
    pseudo-time; enclosed pin (free-node 0, the default)."""
    st = LerayProjectionStepper(
        fx["dm"], fx["nu"], dt, f_fn=lambda x, t: np.zeros((len(x), 2)),
        g_fn=lid_g, order=1, picard_iters=1)
    st.set_initial(lambda x: np.zeros((len(x), 2)))
    u = None
    for _ in range(nsteps):
        u, p = st.step()
    return u, float(st.divergence_l2())


def march_monolithic(fx, dt=0.05, nsteps=200):
    """Monolithic saddle march to steady. Same strong lid/wall Dirichlet, same
    single pressure-DOF pin (node 0)."""
    st = LinearizedMonolithicStepper(
        fx["dm"], fx["nu"], dt, f_fn=lambda x, t: np.zeros((len(x), 2)),
        g_fn=lid_g, order=1)
    st.set_initial(lambda x: np.zeros((len(x), 2)))
    x = None
    for _ in range(nsteps):
        x = st.step()
    return x[:, :2], float(st.divergence_l2())


def run(level=4, Re=100, dt=0.05, nsteps=200, device="cpu"):
    fx = build_cavity(level, Re, device)
    u_p, div_p = march_projection(fx, dt, nsteps)
    u_m, div_m = march_monolithic(fx, dt, nsteps)
    up = centerline_u(fx, u_p, Re)
    um = centerline_u(fx, u_m, Re)
    gu = GHIA_U[Re]
    d_pm = float(np.abs(up - um).max())
    d_pg = float(np.abs(up - gu).max())
    d_mg = float(np.abs(um - gu).max())

    print(f"\n=== Lid-driven cavity  Re={Re}  level={level} "
          f"({2**level}x{2**level} cells)  dt={dt} nsteps={nsteps} ===")
    print(f" projection  ||div u|| = {div_p:.3e}")
    print(f" monolithic  ||div u|| = {div_m:.3e}")
    print(f" centerline u(y) @ x=0.5:  proj / mono / Ghia")
    for i, y in enumerate(GHIA_Y):
        print(f"   y={y:.4f}  {up[i]:+.4f} / {um[i]:+.4f} / {gu[i]:+.4f}")
    print(f" max|proj - mono| = {d_pm:.4f}   (same-mesh agreement)")
    print(f" max|proj - Ghia| = {d_pg:.4f}   (coarse mesh vs 129^2 table)")
    print(f" max|mono - Ghia| = {d_mg:.4f}")
    return dict(u_proj=up.tolist(), u_mono=um.tolist(), ghia=gu.tolist(),
                d_proj_mono=d_pm, d_proj_ghia=d_pg, d_mono_ghia=d_mg,
                div_proj=div_p, div_mono=div_m)


if __name__ == "__main__":
    import sys
    # KNOBS: python lid_driven_cavity.py [LEVEL] [RE] [NSTEPS]
    level = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    Re = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    nsteps = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    run(level=level, Re=Re, nsteps=nsteps)
