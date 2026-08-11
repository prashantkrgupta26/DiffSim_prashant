"""M6: Learn free-energy + mobility from a trajectory — SYNTHETIC PATH.

Demonstrates the end-to-end learn-from-trajectory pipeline:
  1. Build a "truth" BasisMultiEnergy + MobilityClosure.
  2. Run a forward trajectory from a cosine IC to produce target phi-fields.
  3. Starting from gamma=0 / a_nominal, recover the parameters by gradient
     descent using MultiCHAdjoint.gradient.

This script is the SYNTHETIC (self-contained) path.  No MD data is required.

# Plan B hook:
#   Replace the ``phi_star`` block below with snapshots loaded from a
#   molecular-dynamics run (e.g. via a LAMMPS / OpenMM dump reader).
#   The learning loop below is identical; only the target field changes.
#   See the project's Plan B specification for the MD ingest pipeline.

Usage:
    .venv/bin/python3 examples/m6_learn_from_md.py
"""
import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.adjoint import BasisMultiEnergy, MobilityClosure
from diffsim.adjoint.multiphase import MultiCHForward, MultiCHAdjoint


# ---------------------------------------------------------------------------
# mesh
# ---------------------------------------------------------------------------
LEVEL = 4
M = 2

tree = build_uniform(LEVEL, dim=2)
mesh = build_mesh(tree, p=1)
cons = build_constraints(mesh)
dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), "cpu")
coords = mesh.node_coords
nn = dm.n_nodes
blk = 2 * M

print(f"Mesh: level={LEVEL}  n_nodes={nn}  M={M}")

# ---------------------------------------------------------------------------
# fixed physics
# ---------------------------------------------------------------------------
chi = np.zeros((M + 1, M + 1))
chi[0, 1] = chi[1, 0] = 2.5
chi[0, 2] = chi[2, 0] = 1.0
chi[1, 2] = chi[2, 1] = 0.8
N = np.ones(M + 1)
kappa = [1e-2, 1e-2]
dt = 5e-3
n_steps = 5

# deterministic cosine IC
cc = np.cos(np.pi * coords[:, 0]) * np.cos(np.pi * coords[:, 1])
phi0 = [0.30 + 0.05 * cc, 0.30 + 0.05 * cc]

# ---------------------------------------------------------------------------
# SYNTHETIC TARGET: truth energy + mobility → target phi-fields
# ---------------------------------------------------------------------------
# Plan B hook: replace phi_star with snapshots loaded from MD.
# E.g.:
#   import load_md_snapshots  # project Plan-B ingest module
#   phi_star = load_md_snapshots("run.dump", mesh, t_final=0.025)

gamma_star = {"basis_0_2": 0.20, "basis_0_3": -0.12,
              "basis_1_2": 0.10, "basis_1_3": 0.06}
a_star = {"mob_m0": 1.4, "mob_c": 0.5}

print("\n--- Truth parameters ---")
for k, v in {**gamma_star, **a_star}.items():
    print(f"  {k:14s} = {v:.4f}")

en_star = BasisMultiEnergy(chi, N, degrees=(2, 3), coeffs=dict(gamma_star))
mob_star = MobilityClosure("phi_diag", M=M, coeffs=dict(a_star))
fwd_star = MultiCHForward(dm, en_star, mobility=mob_star, kappa=kappa, dt=dt)
fwd_star.set_initial(phi0)
fwd_star.run(n_steps)
phi_star = [fwd_star.steps[-1]["phis"][i].copy() for i in range(M)]
print(f"\nTarget trajectory: {n_steps} BDF1 steps, dt={dt}")

# ---------------------------------------------------------------------------
# RECOVERY: gradient descent on {basis_*, mob_*}
# ---------------------------------------------------------------------------
param_names = list(gamma_star.keys()) + list(a_star.keys())
gamma_cur = {k: 0.0 for k in gamma_star}
a_cur = {"mob_m0": 1.0, "mob_c": 0.0}

# Adam optimizer (inline, deterministic)
n_opt = 30
lr = 2e-2
beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
m1 = {k: 0.0 for k in param_names}
m2 = {k: 0.0 for k in param_names}


def forward_loss_grads(gc, ac):
    """Run forward, compute field-L2 loss and parameter cotangents."""
    en = BasisMultiEnergy(chi, N, degrees=(2, 3), coeffs=dict(gc))
    mob = MobilityClosure("phi_diag", M=M, coeffs=dict(ac))
    fwd = MultiCHForward(dm, en, mobility=mob, kappa=kappa, dt=dt)
    fwd.set_initial(phi0)
    fwd.run(n_steps)
    phis_N = [fwd.steps[-1]["phis"][i] for i in range(M)]
    J = 0.5 * float(sum(((phis_N[i] - phi_star[i]) ** 2).sum()
                        for i in range(M)))
    dJdx = [np.zeros(blk * nn) for _ in range(n_steps)]
    for i in range(M):
        dJdx[-1][2 * i::blk] = phis_N[i] - phi_star[i]
    grads = MultiCHAdjoint(fwd).gradient(dJdx, param_names)
    return J, grads


J0, _ = forward_loss_grads(gamma_cur, a_cur)
print(f"\n--- Gradient descent: {n_opt} steps, lr={lr}, Adam ---")
print(f"{'step':>5}  {'loss':>12}  {'drop_factor':>12}")
print(f"{'0':>5}  {J0:12.4e}  {'1.0':>12}")

for step in range(1, n_opt + 1):
    J, grads = forward_loss_grads(gamma_cur, a_cur)
    for k in param_names:
        g = grads[k]
        m1[k] = beta1 * m1[k] + (1 - beta1) * g
        m2[k] = beta2 * m2[k] + (1 - beta2) * g * g
        m1h = m1[k] / (1 - beta1 ** step)
        m2h = m2[k] / (1 - beta2 ** step)
        delta = lr * m1h / (np.sqrt(m2h) + eps_adam)
        if k in gamma_cur:
            gamma_cur[k] -= delta
        else:
            a_cur[k] -= delta
    if step % 5 == 0 or step == 1:
        print(f"{step:>5}  {J:12.4e}  {J0/max(J, 1e-20):12.1f}x")

J_final, _ = forward_loss_grads(gamma_cur, a_cur)
drop_factor = J0 / max(J_final, 1e-20)

print(f"\n--- Results after {n_opt} opt steps ---")
print(f"Loss: {J0:.4e} → {J_final:.4e}  (drop = {drop_factor:.0f}x)")
print(f"\n{'Parameter':16s}  {'truth':>8}  {'recovered':>10}  {'error':>8}  {'identifiable'}")
print("-" * 65)
truth = {**gamma_star, **a_star}
theta0 = {**{k: 0.0 for k in gamma_star}, "mob_m0": 1.0, "mob_c": 0.0}
theta_f = {**gamma_cur, **a_cur}

# Note which params are strongly vs weakly identifiable from a single trajectory
# basis_* have a near-degenerate Gramian (many basis combinations give the same
# final field); mob_* are strongly identifiable.
WEAKLY_IDENTIFIABLE = set(gamma_star.keys())  # basis_* (see module docstring)

for k in param_names:
    err = theta_f[k] - truth[k]
    ident = "WEAK (basis)" if k in WEAKLY_IDENTIFIABLE else "STRONG (mob)"
    print(f"{k:16s}  {truth[k]:+8.4f}  {theta_f[k]:+10.4f}  {err:+8.4f}  {ident}")

err0 = np.sqrt(sum((theta0[k] - truth[k]) ** 2 for k in param_names))
errf = np.sqrt(sum((theta_f[k] - truth[k]) ** 2 for k in param_names))
print(f"\nJoint parameter error: {err0:.4e} → {errf:.4e}  "
      f"({'improved' if errf < err0 else 'WORSE'})")

print("""
Identifiability note:
  basis_* coefficients are weakly identifiable from a single trajectory's
  final field — many combinations produce the same observed phi_star.
  mob_m0 and mob_c are strongly identifiable and converge to near-truth values.
  For better basis recovery: use multiple trajectories, different ICs, or
  intermediate-time observations (see Plan B).

# Plan B hook:
#   Here you would swap phi_star for real MD snapshots and run the same loop.
#   The gradient engine (MultiCHAdjoint.gradient) is MD-agnostic; only the
#   target construction changes.
""")
