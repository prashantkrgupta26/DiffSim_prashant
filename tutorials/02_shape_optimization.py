"""Tutorial 02 — Differentiable simulation: find the hidden circle.

THE GAME. Someone solved the Poisson problem of tutorial 01 on a SECRET disk
(center and radius unknown to you) and handed you only nine probe readings
u(x_i) from inside. Recover the disk.

THE POINT. This is the smallest instance of the loop that motivates all of
DiffSim: a QoI defined on simulation output, differentiated with respect to
GEOMETRY, driving gradient descent — with the mesh re-carved from scratch
every iteration. Nothing here is finite-differenced; the gradient chain is

    J -> dJ/du            (probe least squares, sparse point-evaluation)
      -> adjoint solve    A^T lam = dJ/du        (one transposed solve)
      -> -lam^T dR/dtheta (a Warp TAPE over the SBM face-residual kernels:
                           cotangents for the distance vectors d and the
                           mapped boundary data g(x+d))
      -> theta            (a torch graph through the Newton closest-point
                           projection, differentiated by the implicit
                           function theorem)

Within an iteration the element classification is FROZEN (it is piecewise
constant in theta — asserted by tests), which is what makes the gradient
well-defined. Between iterations the world is rebuilt: new retained set, new
surrogate, new constraints. Gradients survive because they never depended on
mesh topology, only on the smooth geometric quantities.

Run:  python tutorials/02_shape_optimization.py     (~2 min)
"""
import numpy as np
import torch
from scipy.sparse.linalg import splu

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.mesh.faces import face_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.geometry.csg import Sphere
from diffsim.sbm.surrogate import classify_lambda, extract_surrogate, GeometryData
from diffsim.sbm.poisson import SBMPoisson
from diffsim.sbm.adjoint import solve_adjoint, shape_gradient, probe_qoi

DEVICE = "cuda:0"
LEVEL = 4

u_star = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
u_star_t = lambda x: torch.sin(np.pi * x[:, 0]) * torch.sin(np.pi * x[:, 1])
f_star = lambda x: 2 * np.pi ** 2 * u_star(x)

# nine probes on a small ring — all safely inside every candidate disk
PROBES = 0.5 + 0.12 * np.array(
    [[np.cos(t), np.sin(t)]
     for t in np.linspace(0, 2 * np.pi, 9, endpoint=False)])


def forward(theta):
    """theta = (cx, cy, r) -> everything the adjoint needs."""
    oracle = Sphere((float(theta[0]), float(theta[1])), float(theta[2]))
    tree = build_uniform(LEVEL, dim=2)
    retained, _ = classify_lambda(tree, oracle, lam=0.0)
    sf = extract_surrogate(retained)
    mesh = build_mesh(retained, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), DEVICE)
    geo = GeometryData.evaluate(oracle, retained, sf, face_tables(1, 2))
    prob = SBMPoisson(dm, geo, sf, g_fn=u_star, kappa=1.3)
    A, b, meta = prob.assemble(f_star)
    u_free = splu(A.tocsc()).solve(b)
    u_all = np.asarray(dm.constraints.T @ u_free)
    return oracle, prob, dm, A, meta, u_all


if __name__ == "__main__":
    truth = np.array([0.52, 0.47, 0.31])
    # synthesize the "sensor data" from the secret geometry
    from diffsim.mesh.pointeval import point_eval_weights
    _, _, dm_t, _, _, u_all_t = forward(truth)
    targets = np.asarray(point_eval_weights(dm_t.mesh, PROBES) @ u_all_t)

    theta = torch.tensor([0.50, 0.50, 0.25], dtype=torch.float64,
                         requires_grad=True)
    opt = torch.optim.Adam([theta], lr=2e-2)
    print(f"{'iter':>4} {'J':>12} {'cx':>8} {'cy':>8} {'r':>8}")
    for it in range(100):
        if it == 60:
            for g in opt.param_groups:
                g["lr"] = 4e-3                    # settle Adam's oscillation
        oracle, prob, dm, A, meta, u_all = forward(theta.detach().numpy())
        evalJ, dJdu_fn = probe_qoi(dm, PROBES, targets)
        J = evalJ(u_all)
        if it % 10 == 0:
            t = theta.detach().numpy()
            print(f"{it:>4} {J:>12.3e} {t[0]:>8.4f} {t[1]:>8.4f} {t[2]:>8.4f}")
        if J < 1e-14:
            break
        lam = solve_adjoint(A, dJdu_fn(u_all))            # adjoint solve
        shape_gradient(prob, u_all, lam, oracle, meta,    # tape + IFT chain
                       g_fn_torch=u_star_t)
        opt.zero_grad()
        theta.grad = torch.tensor(np.concatenate(
            [oracle.center.grad.numpy(), [float(oracle.radius.grad)]]))
        opt.step()
        with torch.no_grad():
            theta[2].clamp_(0.15, 0.45)

    found = theta.detach().numpy()
    print(f"\nrecovered theta = {found.round(5)}")
    print(f"true      theta = {truth}")
    print(f"|error|         = {np.abs(found - truth).round(6)}")
    print("""
EXERCISES
  (a) Delete probes until recovery fails. How few readings determine three
      geometric unknowns, and which configurations are degenerate?
  (b) The conductivity was kappa = 1.3. Add it as a fourth unknown using
      diffsim.sbm.adjoint.kappa_gradient. (The test suite does exactly
      this in tests/test_ad_gradients.py::test_kappa_gradient.)
  (c) Swap the CSG circle for a GridSDF and optimize the VOXELS with the
      same machinery (see test_gradient_gridsdf_voxels) — congratulations,
      you are doing level-set topology optimization.
""")
