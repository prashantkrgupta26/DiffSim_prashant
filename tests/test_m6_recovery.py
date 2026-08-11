"""Synthetic-recovery gate (M6 Task 5).

Proves the learn-from-trajectory pipeline can recover a KNOWN energy + mobility
from a single forward trajectory.

Pipeline:
  1. Build "truth" BasisMultiEnergy(coeffs=gamma_star) + MobilityClosure("phi_diag",
     coeffs=a_star), run MultiCHForward from a deterministic cosine IC, record
     the final phi-fields as phi_star.
  2. Starting from gamma=0 (pure FH) and a_nominal=(m0=1.0, c=0.0), run gradient
     descent on {basis_*, mob_*} using J = 0.5*sum_i ||phi_i,N - phi_star_i||^2
     and MultiCHAdjoint.gradient for the cotangents.
  3. Assert:
     PRIMARY:   loss drops >= 10x from step 0 to final.
     SECONDARY: recovered parameter error shrinks (||theta_final - theta*|| < ||theta_0 - theta*||).

Identifiability note:
  The field-L2 loss on a single short trajectory is invariant to certain
  linear combinations of basis coefficients (the Gramian has rank < n_basis in
  the composition-field subspace spanned by the observed trajectory).  In
  practice, on a level-4 mesh with 5 forward steps, the basis_0_* and
  basis_1_* coefficients do NOT individually converge to their true values --
  a different combination of basis modes produces an identical final field.
  The mobility parameters (mob_m0, mob_c) are more strongly identifiable and
  converge reliably.  Because of this:
    - The PRIMARY assertion (loss >=10x drop) is robust and always passes.
    - The SECONDARY assertion is on the JOINT parameter vector norm
      ||theta_final - theta*|| < ||theta_0 - theta*||, which captures the
      component(s) that ARE identifiable without over-asserting on degenerate
      directions.
  The per-parameter diagnostics are printed to make the identifiable vs.
  degenerate directions visible.

All operations are deterministic (fixed IC, fixed optimizer, no randomness).
"""
import numpy as np
import pytest

pytestmark = pytest.mark.ad


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _dm(level, dim=2):
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh
    tree = build_uniform(level, dim=dim)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim), "cpu")
    return dm, mesh


# ---------------------------------------------------------------------------
# core gradient-descent helper shared by test + example script
# ---------------------------------------------------------------------------
def _run_recovery(dm, mesh, n_steps=5, n_opt=30, lr=2e-2, dt=5e-3, M=2):
    """Synthetic-recovery loop.  Returns (loss_hist, theta_hist, truth, param_names).

    Adam optimizer (beta1=0.9, beta2=0.999), deterministic, no randomness.
    n_steps=5, n_opt=30, lr=2e-2, level=4 achieves >100x loss drop in ~12s.

    Returns
    -------
    loss_hist  : list[float]   length n_opt+2 (step 0 = initial, steps 1..n_opt = after updates, final re-eval)
    theta_hist : list[dict]    parameter dicts at each opt step (same length as loss_hist)
    truth      : dict          ground truth parameter values
    param_names: list[str]     ordered list of all learnable parameter names
    """
    from diffsim.adjoint import BasisMultiEnergy, MobilityClosure
    from diffsim.adjoint.multiphase import MultiCHForward, MultiCHAdjoint

    coords = mesh.node_coords
    nn = dm.n_nodes
    blk = 2 * M

    # ------------------------------------------------------------------
    # shared physics (chi, N, kappa) — fixed throughout, not learned
    # ------------------------------------------------------------------
    chi = np.zeros((M + 1, M + 1))
    chi[0, 1] = chi[1, 0] = 2.5
    chi[0, 2] = chi[2, 0] = 1.0
    chi[1, 2] = chi[2, 1] = 0.8
    N = np.ones(M + 1)
    kappa = [1e-2, 1e-2]

    # ------------------------------------------------------------------
    # deterministic cosine IC (same as the existing parity tests)
    # ------------------------------------------------------------------
    cc = np.cos(np.pi * coords[:, 0]) * np.cos(np.pi * coords[:, 1])
    phi0 = [0.30 + 0.05 * cc, 0.30 + 0.05 * cc]

    # ------------------------------------------------------------------
    # TRUTH: non-zero basis and non-trivial mobility
    # ------------------------------------------------------------------
    gamma_star = {"basis_0_2": 0.20, "basis_0_3": -0.12,
                  "basis_1_2": 0.10, "basis_1_3": 0.06}
    a_star = {"mob_m0": 1.4, "mob_c": 0.5}

    en_star = BasisMultiEnergy(chi, N, degrees=(2, 3), coeffs=dict(gamma_star))
    mob_star = MobilityClosure("phi_diag", M=M, coeffs=dict(a_star))
    fwd_star = MultiCHForward(dm, en_star, mobility=mob_star, kappa=kappa, dt=dt)
    fwd_star.set_initial(phi0)
    fwd_star.run(n_steps)
    phi_star = [fwd_star.steps[-1]["phis"][i].copy() for i in range(M)]

    # ------------------------------------------------------------------
    # learnable parameters: all basis_*/mob_* names to be recovered
    # ------------------------------------------------------------------
    param_names = list(gamma_star.keys()) + list(a_star.keys())

    # ------------------------------------------------------------------
    # initial guess: gamma=0 (pure FH), a_nominal=(m0=1.0, c=0.0)
    # ------------------------------------------------------------------
    gamma_cur = {k: 0.0 for k in gamma_star}
    a_cur = {"mob_m0": 1.0, "mob_c": 0.0}

    # ------------------------------------------------------------------
    # Adam optimizer state (inline, deterministic, no randomness)
    # ------------------------------------------------------------------
    m1 = {k: 0.0 for k in param_names}  # first moment
    m2 = {k: 0.0 for k in param_names}  # second moment
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    def _forward_loss_and_grads(gc, ac):
        """Build forward with current params, run, compute J and cotangents."""
        en = BasisMultiEnergy(chi, N, degrees=(2, 3), coeffs=dict(gc))
        mob = MobilityClosure("phi_diag", M=M, coeffs=dict(ac))
        fwd = MultiCHForward(dm, en, mobility=mob, kappa=kappa, dt=dt)
        fwd.set_initial(phi0)
        fwd.run(n_steps)
        phis_N = [fwd.steps[-1]["phis"][i] for i in range(M)]
        # J = 0.5 * sum_i ||phi_i,N - phi_star_i||^2
        J = 0.5 * float(sum(((phis_N[i] - phi_star[i]) ** 2).sum()
                            for i in range(M)))
        # cotangent: dJ/dx at the last step, zero elsewhere
        dJdx = [np.zeros(blk * nn) for _ in range(n_steps)]
        for i in range(M):
            dJdx[-1][2 * i::blk] = phis_N[i] - phi_star[i]
        grads = MultiCHAdjoint(fwd).gradient(dJdx, param_names)
        return J, grads

    # ------------------------------------------------------------------
    # initial loss (step 0 before any update)
    # ------------------------------------------------------------------
    loss_hist = []
    theta_hist = []

    J0, _ = _forward_loss_and_grads(gamma_cur, a_cur)
    loss_hist.append(J0)
    theta_hist.append({**gamma_cur, **a_cur})

    # ------------------------------------------------------------------
    # gradient descent loop
    # ------------------------------------------------------------------
    for step in range(1, n_opt + 1):
        J, grads = _forward_loss_and_grads(gamma_cur, a_cur)
        # Adam update
        for k in param_names:
            g = grads[k]
            m1[k] = beta1 * m1[k] + (1 - beta1) * g
            m2[k] = beta2 * m2[k] + (1 - beta2) * g * g
            m1h = m1[k] / (1 - beta1 ** step)
            m2h = m2[k] / (1 - beta2 ** step)
            delta = lr * m1h / (np.sqrt(m2h) + eps_adam)
            if k in gamma_cur:
                gamma_cur[k] = gamma_cur[k] - delta
            else:
                a_cur[k] = a_cur[k] - delta
        loss_hist.append(J)
        theta_hist.append({**gamma_cur, **a_cur})

    # ------------------------------------------------------------------
    # final re-evaluation at final parameters to align loss_hist[-1] with theta_hist[-1]
    # ------------------------------------------------------------------
    J_final, _ = _forward_loss_and_grads(gamma_cur, a_cur)
    loss_hist.append(J_final)
    theta_hist.append({**gamma_cur, **a_cur})

    truth = {**gamma_star, **a_star}
    return loss_hist, theta_hist, truth, param_names


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------
def test_synthetic_recovery_of_energy_and_mobility():
    """Synthetic-recovery gate: recover known energy + mobility from a trajectory.

    Truth: BasisMultiEnergy(gamma_star) + phi_diag(a_star) with nonzero coeffs.
    Target: final phi-field after n_steps BDF1 steps from a cosine IC.
    Start:  gamma=0 (pure FH), mob_m0=1.0, mob_c=0.0.
    Optimizer: Adam (inline, deterministic), 30 steps, lr=2e-2, ~12s wall time.

    PRIMARY assertion:  loss drops >= 10x.
    SECONDARY assertion: joint parameter error shrinks
      ||theta_final - theta*|| < ||theta_0 - theta*||.

    Identifiability note:
      basis_* coefficients are weakly identifiable from a single trajectory's
      final field (the field-L2 loss is invariant to certain linear combinations
      of basis modes).  The mobility parameters (mob_m0, mob_c) are strongly
      identifiable and converge to near-truth values.  The JOINT norm assertion
      is satisfied even with basis degeneracy because the identifiable mob_*
      directions dominate the improvement.  We do NOT assert individual basis
      convergence; only the loss drop (PRIMARY) and joint-norm improvement
      (SECONDARY) are enforced.
    """
    dm, mesh = _dm(level=4)
    loss_hist, theta_hist, truth, param_names = _run_recovery(
        dm, mesh, n_steps=5, n_opt=30, lr=2e-2, dt=5e-3, M=2)

    J0 = loss_hist[0]
    J_final = loss_hist[-1]
    drop_factor = J0 / max(J_final, 1e-20)

    theta0 = theta_hist[0]
    theta_final = theta_hist[-1]

    err0 = np.sqrt(sum((theta0[k] - truth[k]) ** 2 for k in param_names))
    err_final = np.sqrt(sum((theta_final[k] - truth[k]) ** 2 for k in param_names))

    print(f"\nloss[0]={J0:.4e}  loss[final]={J_final:.4e}  "
          f"drop_factor={drop_factor:.2f}x")
    print(f"||theta_0 - theta*||={err0:.4e}  "
          f"||theta_final - theta*||={err_final:.4e}")
    print("Per-parameter recovery:")
    for k in param_names:
        delta = theta_final[k] - truth[k]
        moved = theta_final[k] - theta0[k]
        print(f"  {k:14s}: truth={truth[k]:+.3f}  final={theta_final[k]:+.3f}  "
              f"err={delta:+.4f}  moved={moved:+.4f}")

    # PRIMARY: loss must drop by at least 10x
    assert drop_factor >= 10.0, (
        f"Loss did not drop 10x: J0={J0:.3e} J_final={J_final:.3e} "
        f"factor={drop_factor:.2f}")

    # SECONDARY: joint parameter error must shrink
    # (Note: individual basis_* coefficients may not converge due to identifiability
    # degeneracy — the joint norm captures the identifiable mob_* directions)
    assert err_final < err0, (
        f"Parameter error did not decrease: "
        f"err0={err0:.4e} err_final={err_final:.4e}")
