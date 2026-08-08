"""Sub-project ②: NeuralCrystalEnergy — recover planted coupling from a synthetic
trajectory by matching the final phi+psi fields through the hand adjoint.

Demonstrates the K>0 "learn-it" payoff: given target fields phi*, psi* produced
by a planted NeuralCrystalEnergy, we recover the coupling coefficients cpl_{k,b}
by gradient descent through CrystalCHAdjoint.

Identifiability note
--------------------
cpl_{k,1}  (the linear-in-ψ mode, Legendre degree 1) is strongly identifiable
from a single final snapshot because it drives the dominant Allen-Cahn flux.

Higher-ψ-degree modes cpl_{k,b>=2} couple more weakly to the observed final field
from a single snapshot — they are weakly identifiable in the single-snapshot
regime. For robust recovery of higher modes, use multiple snapshots at different
times or multiple initial conditions (motivates sub-project ③ multi-snapshot +
descriptor loss), mirroring M6 Plan A Task 5 for the chi/N / basis_* modes.

Usage (CLI smoke-test)::

    .venv/bin/python examples/crystal_learn_from_synthetic.py

(Guard under ``if __name__ == '__main__':`` so the module is importable by tests.)
"""
import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.adjoint import (
    MobilityClosure, CrystalCHForward, CrystalCHAdjoint, NeuralCrystalEnergy)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_dm(level, dim=2):
    """Build a DeviceMesh at a given uniform refinement level."""
    tree = build_uniform(level, dim=dim)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim), "cpu")
    return dm, mesh


def _make_energy(chi, N, crystallizable, deg_psi, coeffs=None,
                 basis_coeffs=None):
    """Construct a NeuralCrystalEnergy with given coupling (and optional basis)
    coefficients. ``coeffs`` is a dict of ``cpl_{k}_{b}`` keys; absent keys
    default to zero."""
    return NeuralCrystalEnergy(
        chi, N, crystallizable, deg_psi=deg_psi,
        coeffs=coeffs or {},
        basis_coeffs=basis_coeffs or {})


def _make_fwd(dm, energy, crystallizable, M, onsager, kappa, eps2, L, dt,
              order, phi0, psi0):
    """Build and run CrystalCHForward from given initial fields."""
    mob = MobilityClosure("const", M=M, onsager=np.asarray(onsager))
    fwd = CrystalCHForward(dm, energy, crystallizable=crystallizable,
                           mobility=mob, kappa=kappa, eps2=eps2, L=L,
                           dt=dt, order=order)
    fwd.set_initial([p.copy() for p in phi0],
                    psi0_list=[p.copy() for p in psi0])
    return fwd


def _compute_loss(phis_N, psis_N, phi_star, psi_star):
    """0.5 * (||phi_N - phi*||^2 + ||psi_N - psi*||^2)."""
    loss = 0.5 * sum(float(((phis_N[i] - phi_star[i]) ** 2).sum())
                     for i in range(len(phis_N)))
    loss += 0.5 * sum(float(((psis_N[j] - psi_star[j]) ** 2).sum())
                      for j in range(len(psis_N)))
    return loss


def _build_dJdx(phis_N, psis_N, phi_star, psi_star, n_steps, blk, M, K):
    """Build the adjoint terminal condition dJ/dx as a list of n_steps vectors.
    Only the final step carries a non-zero entry (field mismatch at step N)."""
    nn = len(phis_N[0])
    dJdx = [np.zeros(blk * nn) for _ in range(n_steps)]
    for i in range(M):
        dJdx[-1][2 * i::blk] = phis_N[i] - phi_star[i]
    for j in range(K):
        dJdx[-1][2 * M + j::blk] = psis_N[j] - psi_star[j]
    return dJdx


# ---------------------------------------------------------------------------
# public API (importable by tests)
# ---------------------------------------------------------------------------

def recover_coupling(dm, coords, planted, names, n_steps, order,
                     n_iter, lr):
    """Recover coupling coefficients of a NeuralCrystalEnergy from a synthetic
    trajectory by gradient descent through the hand adjoint.

    Parameters
    ----------
    dm : DeviceMesh
        The computational mesh.
    coords : ndarray, shape (n_nodes, dim)
        Node coordinates (used to build a smooth initial condition).
    planted : dict
        True ``cpl_{k}_{b}`` coefficients, e.g. ``{"cpl_0_1": 0.15}``.
    names : list[str]
        Parameter names to recover (subset of ``planted`` keys).
    n_steps : int
        Number of BDF time steps in each forward solve.
    order : int
        BDF order (1 or 2).
    n_iter : int
        Number of gradient-descent iterations.
    lr : float
        Gradient-descent step size (plain gradient descent, not Adam).

    Returns
    -------
    loss_hist : list[float]
        Loss value at each iteration (including the re-evaluated final loss).
    theta_hat : dict[str, float]
        Recovered coupling coefficients after ``n_iter`` steps.
    theta_true : dict[str, float]
        Planted (true) coupling coefficients.

    Notes
    -----
    The recovery uses CrystalCHAdjoint (the hand adjoint), not the twin.
    This demonstrates the production gradient path.

    Identifiability: cpl_{k,1} (degree-1, dominant mode) recovers well from a
    single final snapshot.  Higher-ψ-degree modes (b>=2) are weakly identifiable
    from a single snapshot due to their weaker coupling to the observed fields.
    For robust higher-degree recovery, use multiple snapshots at different times
    (sub-project ③ multi-snapshot + descriptor loss, mirroring M6 Plan A Task 5).
    """
    # ---- problem setup -------------------------------------------------------
    # infer M, crystallizable, deg_psi from planted keys
    crystallizable = tuple(sorted(set(
        int(nm.split("_")[1]) for nm in planted)))
    deg_psi = tuple(sorted(set(
        int(nm.split("_")[2]) for nm in planted)))
    M = max(crystallizable) + 1           # minimal M that covers all k
    K = len(crystallizable)
    nn = dm.n_nodes
    blk = 2 * M + K

    chi = np.zeros((M + 1, M + 1))
    chi[0, 1] = chi[1, 0] = 2.5
    for a in range(M + 1):
        for b in range(a + 1, M + 1):
            if (a, b) != (0, 1):
                chi[a, b] = chi[b, a] = 1.0
    N = 1.0 + 0.3 * np.arange(M + 1, dtype=float)
    onsager = np.eye(M)
    kappa = [0.01 * (i + 1) for i in range(M)]
    eps2 = {k: 0.015 for k in crystallizable}
    # L=2.0 drives the Allen-Cahn equation hard enough that psi spans a wide
    # range of [0,1], making Legendre modes of different degrees (L_1 vs L_2)
    # sufficiently decorrelated for simultaneous identification from a single
    # final snapshot.  With L=1.2 the psi field barely moves and L_1, L_2 are
    # nearly co-linear over the narrow psi range, causing a near-degenerate
    # recovery Hessian.
    L_vals = {k: 2.0 for k in crystallizable}
    dt = 1e-2

    # Small non-zero basis correction to make the planted energy non-trivially
    # different from pure FH (optional but makes the problem more realistic)
    basis_coeffs = {"basis_0_2": 0.05} if M >= 1 else {}

    # Cosine IC: phi near 0.28, psi with amplitude 0.15 so the field spans
    # roughly [0.15, 0.45] after a few steps — enough to decorrelate L_1, L_2.
    cc = np.cos(np.pi * coords[:, 0]) * np.cos(np.pi * coords[:, 1])
    phi0 = [0.28 + 0.03 * cc * (-1) ** i for i in range(M)]
    psi0 = [0.30 + 0.15 * cc for _ in range(K)]

    # ---- PLANTED: generate target fields phi*, psi* -------------------------
    planted_energy = _make_energy(chi, N, crystallizable, deg_psi,
                                  coeffs=dict(planted),
                                  basis_coeffs=dict(basis_coeffs))
    fwd_planted = _make_fwd(dm, planted_energy, crystallizable, M,
                            onsager, kappa, eps2, L_vals, dt, order,
                            phi0, psi0)
    fwd_planted.run(n_steps)
    rec_planted = fwd_planted.steps[-1]
    phi_star = [p.copy() for p in rec_planted["phis"]]
    psi_star = [p.copy() for p in rec_planted["psis"]]

    # ---- GUESS: initialise coupling to zero ----------------------------------
    guess_coeffs = {nm: 0.0 for nm in names}
    # keep basis_coeffs the same as planted (only coupling is unknown)
    guess_energy = _make_energy(chi, N, crystallizable, deg_psi,
                                coeffs=dict(guess_coeffs),
                                basis_coeffs=dict(basis_coeffs))

    # ---- GRADIENT DESCENT LOOP ----------------------------------------------
    loss_hist = []

    for _it in range(n_iter):
        # Forward pass
        fwd = _make_fwd(dm, guess_energy, crystallizable, M,
                        onsager, kappa, eps2, L_vals, dt, order,
                        phi0, psi0)
        fwd.run(n_steps)
        rec = fwd.steps[-1]
        phis_N = rec["phis"]
        psis_N = rec["psis"]

        # Loss at CURRENT coeffs
        loss = _compute_loss(phis_N, psis_N, phi_star, psi_star)
        loss_hist.append(loss)

        # Adjoint: dJ/dtheta
        dJdx = _build_dJdx(phis_N, psis_N, phi_star, psi_star,
                            n_steps, blk, M, K)
        grads = CrystalCHAdjoint(fwd).gradient(dJdx, names)

        # Gradient-descent update of coupling coefficients
        for nm in names:
            _, sk, sb = nm.split("_")
            k, b = int(sk), int(sb)
            guess_energy.c[(k, b)] -= lr * grads[nm]

    # ---- RE-EVALUATE loss at the FINAL coeffs (avoid M6 off-by-one) ---------
    # The loop records loss BEFORE the update at step _it, so after n_iter
    # updates we need one more forward pass to get the loss at the final params.
    fwd_final = _make_fwd(dm, guess_energy, crystallizable, M,
                          onsager, kappa, eps2, L_vals, dt, order,
                          phi0, psi0)
    fwd_final.run(n_steps)
    rec_final = fwd_final.steps[-1]
    loss_final = _compute_loss(rec_final["phis"], rec_final["psis"],
                               phi_star, psi_star)
    loss_hist.append(loss_final)

    # ---- build return dicts --------------------------------------------------
    theta_hat = {}
    for nm in names:
        _, sk, sb = nm.split("_")
        theta_hat[nm] = float(guess_energy.c[(int(sk), int(sb))])

    theta_true = {nm: float(v) for nm, v in planted.items()}

    return loss_hist, theta_hat, theta_true


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("crystal_learn_from_synthetic.py — synthetic coupling recovery demo")
    dm, mesh = _make_dm(2)
    coords = mesh.node_coords
    planted = {"cpl_0_1": 0.15, "cpl_0_2": -0.1}
    names = ["cpl_0_1", "cpl_0_2"]

    print(f"\nMesh: n_nodes={dm.n_nodes}  dim=2")
    print(f"Planted: {planted}")
    print(f"n_steps=4, order=1, n_iter=40, lr=0.5\n")

    loss_hist, theta_hat, theta_true = recover_coupling(
        dm, coords, planted=planted, names=names,
        n_steps=4, order=1, n_iter=40, lr=0.5)

    print(f"{'iter':>5}  {'loss':>12}")
    for i, l in enumerate(loss_hist):
        if i % 5 == 0 or i == len(loss_hist) - 1:
            print(f"{i:>5}  {l:12.4e}")

    drop = loss_hist[0] / max(loss_hist[-1], 1e-30)
    print(f"\nLoss drop: {loss_hist[0]:.4e} → {loss_hist[-1]:.4e}  ({drop:.1f}x)")

    print(f"\n{'Parameter':14s}  {'planted':>10}  {'recovered':>10}  {'error%':>8}")
    print("-" * 52)
    for nm in names:
        tr = theta_true[nm]
        rc = theta_hat[nm]
        err_pct = 100.0 * abs(rc - tr) / max(abs(tr), 1e-14)
        print(f"{nm:14s}  {tr:+10.4f}  {rc:+10.4f}  {err_pct:7.1f}%")
