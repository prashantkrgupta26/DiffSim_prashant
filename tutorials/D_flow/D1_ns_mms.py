"""D1 — Incompressible Navier–Stokes: saddle points, stabilization, MMS.

LEARNING OUTCOME. You understand why velocity-pressure FEM is harder than
everything before it (the pressure has no time derivative and no diffusion
— it is a Lagrange multiplier enforcing div u = 0, making the system a
SADDLE POINT), why equal-order Q1/Q1 elements are illegal without help
(the inf-sup/LBB condition), and how PSPG/VMS stabilization buys them
back. You verify with a solenoidal MMS: velocity order 2, and you learn to
read pressure orders with realistic expectations.

BACKGROUND. Steady Oseen (convection frozen at the exact field a = u*):
    a.grad(u) + grad(p) - nu lap(u) = f,   div u = 0.
The library brick (diffsim.api.ns_bricks) assembles the monolithic (u,p)
block with: s=1/2 skew convection (energy-stable — see the vms tests),
SUPG + PSPG on the linearized strong residual, grad-div (tau_C). The MMS
is the classic vortex u* = (sin^2(pi x) sin(2 pi y), -sin(2 pi x)
sin^2(pi y)) — divergence-free with u* = 0 on the box — plus
p* = sin(pi x) cos(pi y), with hand-derived forcing.

EXPECTED RESULTS (nu = 0.1):
    velocity: errors ~ 2.05e-2 / 5.45e-3 / 1.39e-3, orders 1.91 / 1.97
    pressure: errors ~ 9.2e-2 / 3.3e-2 / 1.2e-2, orders 1.47 / 1.53
    (equal-order pressure converges at ~O(h^1.5) — the price of Q1/Q1
    + stabilization; velocity is unharmed).
    And at nu = 1e-3 (advection-dominated) the solve stays clean — SUPG.

Run:  python tutorials/D_flow/D1_ns_mms.py
"""
import os
import sys

import numpy as np
from scipy.sparse.linalg import splu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _viz as _viz  # guarded viz helper (no-ops when [viz] not installed)

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.api.ns_bricks import assemble_linear_ns
from diffsim.physics.poisson import gauss_points
from diffsim import default_device

DEVICE = default_device()
PI = np.pi


def u_star(x):
    return np.stack([np.sin(PI * x[:, 0]) ** 2 * np.sin(2 * PI * x[:, 1]),
                     -np.sin(2 * PI * x[:, 0]) * np.sin(PI * x[:, 1]) ** 2],
                    axis=1)


def p_star(x):
    return np.sin(PI * x[:, 0]) * np.cos(PI * x[:, 1])


def f_star(x, nu):
    sx, sy = np.sin(PI * x[:, 0]), np.sin(PI * x[:, 1])
    s2x, s2y = np.sin(2 * PI * x[:, 0]), np.sin(2 * PI * x[:, 1])
    c2x, c2y = np.cos(2 * PI * x[:, 0]), np.cos(2 * PI * x[:, 1])
    u1, u2 = sx ** 2 * s2y, -s2x * sy ** 2
    du1x, du1y = PI * s2x * s2y, 2 * PI * sx ** 2 * c2y
    du2x, du2y = -2 * PI * c2x * sy ** 2, -PI * s2x * s2y
    lap1 = 2 * PI ** 2 * c2x * s2y - 4 * PI ** 2 * sx ** 2 * s2y
    lap2 = 4 * PI ** 2 * s2x * sy ** 2 - 2 * PI ** 2 * s2x * c2y
    dpx = PI * np.cos(PI * x[:, 0]) * np.cos(PI * x[:, 1])
    dpy = -PI * np.sin(PI * x[:, 0]) * np.sin(PI * x[:, 1])
    return np.stack([u1 * du1x + u2 * du1y + dpx - nu * lap1,
                     u1 * du2x + u2 * du2y + dpy - nu * lap2], axis=1)


def solve(level, nu):
    ndof = 3
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), DEVICE)
    xq = gauss_points(mesh, dm.tables_by_p)
    # Oseen: advecting field = exact velocity (div a = 0 analytically)
    aq = {pv: u_star(xq[pv]) for pv in xq}
    dq = {pv: np.zeros(len(xq[pv])) for pv in xq}
    fq = {pv: f_star(xq[pv], nu) for pv in xq}
    A, b = assemble_linear_ns(dm, aq, dq, fq, nu)
    # strong u = 0 on the box; pressure pinned at one node (A2's lesson!)
    nfree = cons.T.shape[1]
    coords = mesh.node_coords[cons.free_nodes]
    bdry = np.where(mesh.boundary_nodes[cons.free_nodes])[0]
    A = A.tolil()
    for i in bdry:
        for c in range(2):
            r = i * ndof + c
            A.rows[r] = [int(r)]
            A.data[r] = [1.0]
            b[r] = 0.0
    A.rows[2] = [2]
    A.data[2] = [1.0]
    b[2] = p_star(coords[0:1])[0]
    x = splu(A.tocsr().tocsc()).solve(b).reshape(nfree, ndof)
    eu = np.sqrt(((x[:, :2] - u_star(coords)) ** 2).sum(1).mean())
    ep = np.sqrt(((x[:, 2] - p_star(coords)) ** 2).mean())
    return eu, ep


def main(levels=(3, 4, 5), nu=0.1):
    """Run the NS-MMS convergence study and emit convergence figures."""
    res = [solve(lv, nu) for lv in levels]
    eu = [r[0] for r in res]
    ep = [r[1] for r in res]
    print("velocity: errors " + "  ".join(f"{e:.3e}" for e in eu)
          + "   orders "
          + "  ".join(f"{np.log2(eu[i] / eu[i + 1]):.2f}" for i in range(len(eu) - 1)))
    print("pressure: errors " + "  ".join(f"{e:.3e}" for e in ep)
          + "   orders "
          + "  ".join(f"{np.log2(ep[i] / ep[i + 1]):.2f}" for i in range(len(ep) - 1)))
    if len(levels) == 3:
        eu_lo, _ = solve(4, 1e-3)
        print(f"advection-dominated nu=1e-3, level 4: velocity err {eu_lo:.3e} "
              "(finite and small => SUPG is doing its job)")
    # --- viz (additive; no-ops on base venv) ---
    _viz.convergence(__file__, list(levels), eu, "convergence_velocity",
                     slope=2, xlabel="refinement level", ylabel="L2 velocity error",
                     label="velocity")
    _viz.convergence(__file__, list(levels), ep, "convergence_pressure",
                     slope=1, xlabel="refinement level", ylabel="L2 pressure error",
                     label="pressure")
    return eu, ep


if __name__ == "__main__":
    main()
    print("""
EXPLORE
  (a) Remove the pressure pin. What does splu report, and why? (A2's
      pure-Neumann trap, now wearing flow clothing: pressure is defined up
      to a constant.)
  (b) Read make_linear_ns_Ae in src/diffsim/api/ns_bricks.py and find the
      four stabilization blocks (SUPG, PSPG u-coupling, PSPG p-Laplacian,
      grad-div). Set tauC's contribution to zero by editing a copy — what
      happens to div(u_h) (compute it from the solution)?
  (c) s-forms: assemble with s_skew=0 and s_skew=1 (a runtime scalar).
      Errors barely move at nu = 0.1 — but the ENERGY analysis differs;
      run tests/test_vms.py's identity checks and connect the dots.
  (d) PERFORMANCE CORNER: the (u,p) block at level 5 has ~3300 unknowns
      and splu eats it. Push to level 7 and record memory + time for the
      factorization. Extrapolate level 9. This wall — not accuracy — is
      what P2's matrix-free machinery removes.
""")
